"""Run a local detector or replay saved detections through FLAME-inspired logic."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time

from flame import Config, FlameProcessor


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name+'.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def frames(path):
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        origin, previous = None, None
        for index, decoded in enumerate(container.decode(stream)):
            if decoded.time is None:
                raise ValueError('Missing presentation timestamp; do not invent timing')
            if origin is None:
                origin = float(decoded.time)
            t = float(decoded.time) - origin
            if previous is not None and t <= previous:
                raise ValueError('Non-increasing presentation timestamps')
            previous = t
            yield index, t, decoded.to_ndarray(format='bgr24')


def prefetched_frames(path, capacity=3):
    """Overlap sequential decoding with processing; preserve exact order/PTS."""
    pending=queue.Queue(maxsize=capacity)
    stop=threading.Event()
    end=object()
    def put(value):
        while not stop.is_set():
            try:
                pending.put(value,timeout=.1)
                return
            except queue.Full:
                pass
    def produce():
        source=frames(path)
        try:
            for row in source:
                if stop.is_set():
                    break
                put(row)
        except BaseException as error:
            put(error)
        finally:
            source.close()
            put(end)
    worker=threading.Thread(target=produce,name='sequential-video-decoder',daemon=True)
    worker.start()
    try:
        while True:
            value=pending.get()
            if value is end:
                break
            if isinstance(value,BaseException):
                raise value
            yield value
    finally:
        stop.set()
        worker.join(timeout=5)


def run(args):
    import cv2
    cv2.setNumThreads(args.threads)
    video = Path(args.video).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new output directory; existing evidence is preserved')
    cfg = Config(confidence=args.confidence, confirmation_sec=args.confirmation_sec,
                 occlusion_sec=args.occlusion_sec, fire_class=args.fire_class, smoke_class=args.smoke_class)
    expected = None
    if args.predictions:
        predictions = Path(args.predictions).resolve()
        summary = Path(args.cache_summary).resolve() if args.cache_summary else None
        if summary is None:
            raise ValueError('Replay requires --cache-summary to verify source and prediction identities')
        expected = json.loads(summary.read_text(encoding='utf-8'))
        if expected['status'] != 'complete' or expected['video_sha256'] != sha(video) or expected['raw_sha256'] != sha(predictions):
            raise ValueError('Cache/source hash verification failed')
        floor = args.cache_confidence_floor
        if floor is None or floor > cfg.confidence or floor < 0:
            raise ValueError('Declare --cache-confidence-floor no higher than the requested threshold')
        opener = gzip.open if predictions.suffix == '.gz' else open
        stream = opener(predictions, 'rt', encoding='utf-8')
        model_info = dict(mode='verified_prediction_replay', model=expected.get('model'),
                          prediction_sha256=expected['raw_sha256'], cache_summary_sha256=sha(summary),
                          confidence_floor=floor)
        model = None
    else:
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        weights = Path(args.weights).resolve()
        if not weights.is_file():
            raise FileNotFoundError(weights)
        from ultralytics import YOLO
        model = YOLO(str(weights), task='detect')
        names = {int(k): str(v).lower() for k,v in model.names.items()}
        if names.get(cfg.fire_class) not in ('fire', 'flame', 'flames') or names.get(cfg.smoke_class) != 'smoke':
            raise ValueError(f'Class mapping must be explicitly fire/flame and smoke; got {names}')
        model_info = dict(mode='new_detector_inference', checkpoint=str(weights), checkpoint_sha256=sha(weights), classes=names)
        stream = None
    output.mkdir(parents=True)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), method='independent FLAME-inspired implementation',
                    source_video=str(video), video_sha256=sha(video), config=asdict(cfg), model=model_info,
                    code_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('flame.py')]},
                    no_training=True, no_ground_truth_used_for_decisions=True,
                    all_decoded_frames=True, original_resolution=True, decoder_prefetch_frames=3,
                    opencv_threads=args.threads)
    write(output/'protocol.json', protocol)
    processor = FlameProcessor(cfg)
    counters = dict(frames=0, raw_candidate_frames=0, moving_candidate_frames=0,
                    observed_confirmed_frames=0, confirmed_alarm_frames=0, confirmation_events=0,
                    first_confirmation_sec=None)
    started = time.perf_counter()
    frame_stream=prefetched_frames(video)
    try:
        with (output/'frames.jsonl').open('x', encoding='utf-8') as target:
            for index, t, image in frame_stream:
                if stream is not None:
                    line = stream.readline()
                    if not line:
                        raise ValueError('Prediction cache ends before decoded video')
                    saved = json.loads(line)
                    if (saved['frame_index'] != index or abs(saved['timestamp_s']-t) > 1e-4
                            or (saved['height'],saved['width']) != image.shape[:2]):
                        raise ValueError(f'Cache/video timeline or dimension mismatch at frame {index}')
                    boxes = saved['boxes_xyxy_conf_cls']
                else:
                    result = model.predict(image, imgsz=640, rect=False, conf=cfg.confidence,
                                           iou=0.7, max_det=300, device=args.device, verbose=False, save=False)[0]
                    boxes = result.boxes.data.detach().cpu().tolist()
                result = processor.update(image, boxes, t)
                result['frame_index'] = index
                target.write(json.dumps(result, separators=(',',':'), allow_nan=False)+'\n')
                counters['frames'] += 1
                counters['raw_candidate_frames'] += bool(result['raw_boxes'])
                counters['moving_candidate_frames'] += bool(result['moving_boxes'])
                counters['observed_confirmed_frames'] += bool(result['observed_confirmed_boxes'])
                counters['confirmed_alarm_frames'] += result['alarm']
                counters['confirmation_events'] += len(result['new_confirmations'])
                if result['new_confirmations'] and counters['first_confirmation_sec'] is None:
                    counters['first_confirmation_sec'] = t
                if index % 600 == 0:
                    state = dict(status='running', timestamp_sec=t, elapsed_sec=time.perf_counter()-started, **counters)
                    write(output/'status.json', state)
                    print(json.dumps(state), flush=True)
            if stream is not None and stream.readline():
                raise ValueError('Prediction cache contains extra frames after video EOF')
        if expected and counters['frames'] != expected['frames']:
            raise ValueError('Decoded frame count differs from verified cache')
        if not counters['frames']:
            raise ValueError('Video contains no decoded frames')
        final = dict(status='complete', last_timestamp_sec=t, elapsed_sec=time.perf_counter()-started,
                     output_sha256=sha(output/'frames.jsonl'), **counters)
        write(output/'summary.json', final)
        write(output/'status.json', final)
        return final
    except BaseException as error:
        write(output/'status.json',dict(status='failed', error=repr(error), **counters))
        raise
    finally:
        frame_stream.close()
        if stream is not None:
            stream.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--weights', help='Existing locally loadable fire/smoke checkpoint; no training')
    source.add_argument('--predictions', help='Existing all-frame JSONL or JSONL.GZ cache')
    p.add_argument('--cache-summary')
    p.add_argument('--cache-confidence-floor', type=float)
    p.add_argument('--output', required=True)
    p.add_argument('--confidence', type=float, default=.4)
    p.add_argument('--confirmation-sec', type=float, default=1.)
    p.add_argument('--occlusion-sec', type=float, default=.1)
    p.add_argument('--fire-class', type=int, default=0)
    p.add_argument('--smoke-class', type=int, default=1)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    return p


if __name__ == '__main__':
    print(json.dumps(run(parser().parse_args()), indent=2))
