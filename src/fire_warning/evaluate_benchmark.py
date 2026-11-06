"""Five-row event-timestamp proxy evaluation; reviewed box labels are optional."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from .auto_annotate_videos import clip_id
from .common import (DATA, RESULTS, Detector, Pipeline, VideoSource, accepted_number,
    detector_arguments, read_csv, resolve_video, run_main, sha256, stats, write_csv, write_json)
from .fsm_engine import FSMConfig, FireSmokeFSM, scaled_size

VARIANTS = [
    ('raw_025', 'Raw 0.25'), ('raw_014', 'Raw 0.14'),
    ('persistence', 'Detector + Tracking + Persistence'),
    ('full_off', 'Full 4-Stage FSM [Glare Filter OFF]'),
    ('full_on', 'Full 4-Stage FSM [Glare Filter ON]'),
]


def iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0


def match_boxes(predictions, truth):
    pairs = sorted([(iou(p, g), pi, gi) for pi, p in enumerate(predictions) for gi, g in enumerate(truth)
                    if int(p[5]) == int(g[5]) and iou(p, g) >= 0.5], reverse=True)
    ps, gs = set(), set()
    for _, pi, gi in pairs:
        if pi not in ps and gi not in gs: ps.add(pi); gs.add(gi)
    return len(ps), len(predictions)-len(ps), len(truth)-len(gs)


def reviewed_truth(folder, index, image_size):
    if not folder: return None
    folder = Path(folder)
    names = [f'frame_{index:08d}.txt', f'frame_{index:06d}.txt', f'{index:08d}.txt']
    path = next((folder / name for name in names if (folder / name).is_file()), None)
    if path is None: return None
    w, h = image_size
    out = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        parts = line.split()
        if len(parts) != 5: raise ValueError(f'Invalid reviewed YOLO row in {path}')
        cls, x, y, bw, bh = map(float, parts)
        if cls not in (0, 1) or not all(0 <= v <= 1 for v in (x,y,bw,bh)) or bw <= 0 or bh <= 0:
            raise ValueError(f'Invalid reviewed box in {path}')
        out.append([(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h, 1.0, int(cls)])
    return out


def evaluate_records(row, records, fps, labels_dir=None, partial=False):
    """Pure CPU replay; labels do not participate in primary event metrics."""
    ignition = accepted_number(row, 't_ignition_start_sec')
    aftermath = accepted_number(row, 't_aftermath_start_sec')
    target_alert = accepted_number(row, 'should_alert_human')
    fsms = dict(persistence=FireSmokeFSM(FSMConfig(fps=fps, persistence_only=True, glare_filter=False)),
                full_off=FireSmokeFSM(FSMConfig(fps=fps, glare_filter=False)),
                full_on=FireSmokeFSM(FSMConfig(fps=fps, glare_filter=True)))
    metrics = {key: dict(variant=key, method=name, video_path=row['video_path'], frames=0,
        negative_frames=0, false_positive_frames=0, negative_duration_sec=0.0,
        false_alarm_incidents=0, first_positive_sec=None, t_stage2_onset_sec=None, ttd_sec=None,
        predicted_size_stratum=None, preconfirmed=False, first_progression_sec=None,
        progression_delay_sec=None, alarm=False, first_alarm_sec=None, actual_stage2_transitions=[],
        box_review_frames=0, box_tp=0, box_fp=0, box_fn=0, fsm_times=[], first_actual_confirmation=None,
        positive_frames=0, timestamp_quality_ok=True) for key, name in VARIANTS}
    last_alarm = {key: None for key, _ in VARIANTS}
    previous_t, last_t = None, None
    for record in records:
        t = record['timestamp_sec']
        dt = 1 / fps if previous_t is None else t - previous_t
        # Clip intervals at ignition instead of charging the first positive frame to negatives.
        negative_dt = dt if ignition == -1 else max(0, min(t + 1/fps, ignition) - t) if ignition is not None else 0
        negative = ignition is not None and (ignition == -1 or t < ignition)
        truth = reviewed_truth(labels_dir, record['frame_index'], record['image_size'])
        for key, _ in VARIANTS:
            m = metrics[key]
            m['frames'] += 1
            m['timestamp_quality_ok'] &= record.get('timestamp_quality', 'presentation_timestamp') == 'presentation_timestamp'
            state = None
            if key.startswith('raw_'):
                threshold = 0.25 if key == 'raw_025' else 0.14
                candidates = [b for b in record['boxes'] if b[4] >= threshold]
                positive, alarm = bool(candidates), bool(candidates)
                onset_candidate = max(candidates, key=lambda b:b[4]) if candidates else None
                size = scaled_size(onset_candidate, record['image_size']) if onset_candidate else None
                confirmed_at = None
            else:
                start = time.perf_counter()
                state = fsms[key].update(record['tracks'], t, tuple(record['image_size']))
                m['fsm_times'].append((time.perf_counter() - start) * 1000)
                current = state['confirmed_tracks']
                positive, alarm = bool(current), state['alarm']
                onset_candidate = max(current, key=lambda s:s['ema_conf']) if current else None
                size = onset_candidate if onset_candidate else None
                confirmed_at = onset_candidate['confirmed_at'] if onset_candidate else None
                candidates = [[*s['box'], s['ema_conf'], s['cls_id']] for s in current]
                for tr in state['transitions']:
                    if tr['stage'] == 2:
                        m['actual_stage2_transitions'].append(tr)
                        if m['first_actual_confirmation'] is None: m['first_actual_confirmation'] = t
                if (key != 'persistence' and aftermath is not None and aftermath >= 0 and t >= aftermath
                        and any(s['progression_at'] is not None for s in current)
                        and m['first_progression_sec'] is None):
                    m['first_progression_sec'] = t
                    m['progression_delay_sec'] = t - aftermath
            m['positive_frames'] += int(positive)
            if ignition is not None and ignition >= 0 and t >= ignition and positive and m['first_positive_sec'] is None:
                m['first_positive_sec'] = t
                if not key.startswith('raw_'): m['t_stage2_onset_sec'] = t
                m['ttd_sec'] = t - ignition
                m['predicted_size_stratum'] = size['size_tag']
                m['preconfirmed'] = confirmed_at is not None and confirmed_at < ignition
            if negative:
                m['negative_frames'] += 1
                m['negative_duration_sec'] += negative_dt
                m['false_positive_frames'] += int(positive)
                if alarm:
                    if last_alarm[key] is None or t - last_alarm[key] >= 1.0:
                        m['false_alarm_incidents'] += 1
                    last_alarm[key] = t
            if alarm:
                m['alarm'] = True
                if m['first_alarm_sec'] is None: m['first_alarm_sec'] = t
            if truth is not None:
                tp, fp, fn = match_boxes(candidates, truth)
                m['box_review_frames'] += 1
                m['box_tp'] += tp; m['box_fp'] += fp; m['box_fn'] += fn
        previous_t = last_t = t
    out = []
    for key, _ in VARIANTS:
        m = metrics[key]
        supported = ignition is not None and ignition >= 0 and last_t is not None and last_t >= ignition and not partial
        m['event_observable'] = supported
        m['event_detected'] = m['first_positive_sec'] is not None if supported else None
        m['should_alert_human'] = int(target_alert) if target_alert is not None and not partial else None
        m['aftermath_observable'] = key in ('full_off','full_on') and aftermath is not None and aftermath >= 0 and last_t is not None and last_t >= aftermath and not partial
        m['fsm_latency_ms'] = stats(m.pop('fsm_times'))
        m['false_positive_frame_pct'] = 100*m['false_positive_frames']/m['negative_frames'] if m['negative_frames'] else None
        m['specificity'] = 1-m['false_positive_frames']/m['negative_frames'] if m['negative_frames'] else None
        m['false_alarms_per_hour'] = 3600*m['false_alarm_incidents']/m['negative_duration_sec'] if m['negative_duration_sec'] else None
        m['reason'] = ('partial_run_outcome_metrics_unavailable' if partial else
                       'event_timestamp_missing' if ignition is None else
                       'onset_outside_observed_window' if ignition >= 0 and not supported else '')
        if not m['timestamp_quality_ok']:
            m['ttd_sec'] = m['progression_delay_sec'] = None
            m['reason'] += ';exact_timing_unavailable'
        out.append(m)
    return out


def aggregate(per_clip):
    rows = []
    for key, name in VARIANTS:
        clips = [r for r in per_clip if r['variant'] == key]
        events = [r for r in clips if r['event_observable']]
        delays = [r['ttd_sec'] for r in events if r['ttd_sec'] is not None]
        labeled = [r for r in clips if r['should_alert_human'] is not None]
        tp = sum(r['alarm'] and r['should_alert_human'] == 1 for r in labeled)
        fp = sum(r['alarm'] and r['should_alert_human'] == 0 for r in labeled)
        fn = sum(not r['alarm'] and r['should_alert_human'] == 1 for r in labeled)
        negative_frames = sum(r['negative_frames'] for r in clips)
        negative_sec = sum(r['negative_duration_sec'] for r in clips)
        false_frames = sum(r['false_positive_frames'] for r in clips)
        alarms = sum(r['false_alarm_incidents'] for r in clips)
        row = dict(variant=key, method=name, evaluation_mode='event_timestamp_proxy', clips=len(clips),
            frames=sum(r['frames'] for r in clips), events=len(events),
            event_recall=sum(r['event_detected'] for r in events)/len(events) if events else None,
            missed_events=sum(not r['event_detected'] for r in events),
            mean_ttd_sec=stats(delays)['mean'], median_ttd_sec=stats(delays)['median'],
            detected_le_1s=sum(r['ttd_sec'] is not None and r['ttd_sec'] <= 1 for r in events)/len(events) if events else None,
            detected_le_3s=sum(r['ttd_sec'] is not None and r['ttd_sec'] <= 3 for r in events)/len(events) if events else None,
            false_positive_frame_pct=100*false_frames/negative_frames if negative_frames else None,
            specificity=1-false_frames/negative_frames if negative_frames else None,
            false_alarm_incidents=alarms, negative_hours=negative_sec/3600,
            false_alarms_per_hour=3600*alarms/negative_sec if negative_sec else None,
            alert_labeled_clips=len(labeled), alert_tp=tp, alert_fp=fp, alert_fn=fn,
            alert_precision=tp/(tp+fp) if tp+fp else None, alert_recall=tp/(tp+fn) if tp+fn else None,
            alert_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
        for tag in ('TINY', 'SMALL', 'MEDIUM_LARGE'):
            # Sizes of missed events are unknown; do not invent size-specific recall denominators.
            sized = [r for r in events if r['predicted_size_stratum'] == tag]
            row[tag.lower()+'_detected_events'] = len(sized)
            row[tag.lower()+'_mean_ttd_sec'] = stats([r['ttd_sec'] for r in sized if r['ttd_sec'] is not None])['mean']
            for boundary in (1,3):
                row[tag.lower()+f'_detected_le_{boundary}s_fraction_of_sized_detections'] = (
                    sum(r['ttd_sec'] is not None and r['ttd_sec']<=boundary for r in sized)/len(sized) if sized else None)
        progression = [r for r in clips if r['aftermath_observable']]
        row['progression_event_recall'] = sum(r['first_progression_sec'] is not None for r in progression)/len(progression) if progression else None
        row['mean_progression_delay_sec'] = stats([r['progression_delay_sec'] for r in progression if r['progression_delay_sec'] is not None])['mean']
        row['fire_first_progression_accuracy'] = None
        row['smoke_first_progression_accuracy'] = None
        rows.append(row)
    return rows


def cached_video(row, detector, cache_dir, max_frames=None):
    path = resolve_video(row['video_path'])
    signature = dict(schema=1, source_sha256=sha256(path), detector=detector.identity(),
        common_sha256=sha256(Path(__file__).with_name('common.py')), max_frames=max_frames)
    signature_key = __import__('hashlib').sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir) / (clip_id(path) + '-' + signature_key[:12])
    meta_path, frames_path = cache.with_suffix('.json'), cache.with_suffix('.jsonl')
    if meta_path.exists() and frames_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['signature'] == signature and sha256(frames_path) == meta['frames_sha256']:
            return frames_path, meta
    frames_path.parent.mkdir(parents=True, exist_ok=True)
    temp = frames_path.with_suffix('.partial')
    count = 0
    with VideoSource(path) as source, temp.open('w', encoding='utf-8') as output:
        pipeline = Pipeline(detector, source.fps)
        first = source.read()
        if first is None: raise RuntimeError('Video contains no decodable frames')
        for _ in range(30): detector.predict(first.image)
        pipeline.reset()
        frame = first
        start = time.perf_counter()
        while frame is not None and (max_frames is None or count < max_frames):
            record = pipeline.process(frame)
            record.pop('state')
            record['decoded_sha256'] = __import__('hashlib').sha256(frame.image.tobytes()).hexdigest()
            output.write(json.dumps(record, allow_nan=False) + '\n')
            count += 1
            frame = source.read() if max_frames is None or count < max_frames else None
        meta = dict(signature=signature, fps=source.fps, duration_sec=source.duration_sec,
            decoded_frames=count, partial=max_frames is not None and count >= max_frames,
            accuracy_cache_generation_sec=time.perf_counter()-start,
            timing_is_native_cache_generation_not_five_pipeline_comparison=True)
    temp.replace(frames_path)
    meta['frames_sha256'] = sha256(frames_path)
    write_json(meta_path, meta)
    return frames_path, meta


def frame_records(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f: yield json.loads(line)


def generate_report(output, summary, per_clip, errors):
    lines = ['# Four-stage benchmark', '',
        'Primary results are **event-timestamp detection proxies**. A post-onset indication does not prove spatial correctness.',
        'Size strata use predicted boxes. Missed-event sizes are unknown, so size-specific ground-truth recall is unavailable.',
        'Raw rows treat any qualifying detection as an alarm; temporal rows use their stated decision rules.', '',
        '| Method | Events | Recall | Mean TTD (s) | ≤1s | ≤3s | FP frames (%) | Alarms/hour | Alert F1 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt = lambda v: 'unavailable' if v is None else f'{v:.4f}' if isinstance(v, float) else str(v)
    for row in summary:
        lines.append('| ' + ' | '.join(fmt(row[k]) for k in ('method','events','event_recall','mean_ttd_sec',
            'detected_le_1s','detected_le_3s','false_positive_frame_pct','false_alarms_per_hour','alert_f1')) + ' |')
    lines += ['', 'Predicted-size counts and TTD are in 4stage_metrics.csv; actual confirmation transitions are in per_clip_results.json.',
        'Optional reviewed-box results use class-matched IoU ≥0.5 and are separate from the primary tables.',
        'Performance requires an actual runtime benchmark. Shared accuracy-cache generation is not a five-row speed comparison.',
        'No onset timestamp or alert label was inferred as accepted ground truth.', '']
    lines.append('Fire-first/smoke-first correctness is unavailable without independent class-onset annotations; automatic class order is not ground truth.')
    for row in per_clip:
        if row['reason']: lines.append(f'- {row["method"]}, {row["video_path"]}: {row["reason"]}')
    for error in errors: lines.append(f'- Failed clip: {error["video_path"]}: {error["error"]}')
    (Path(output) / '4stage_benchmark_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = Path(output) / 'figures'
    figures.mkdir(exist_ok=True)
    for metric, ylabel in [('event_recall','Event-timestamp recall proxy'), ('mean_ttd_sec','Successful-event mean TTD (seconds)'),
                            ('false_alarms_per_hour','False alarm incidents / hour'), ('alert_f1','Event alert F1')]:
        fig, ax = plt.subplots(figsize=(11, 5))
        for index, row in enumerate(summary):
            value = row[metric]
            if value is not None: ax.bar(index, value)
            else: ax.text(index, 0, 'N/A', ha='center')
        ax.set_xticks(range(5), ['Raw .25','Raw .14','Persistence','Full OFF','Full ON'])
        ax.set_ylabel(ylabel); fig.tight_layout()
        fig.savefig(figures / (metric+'.png'), dpi=300)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    detector_arguments(parser)
    parser.add_argument('--worksheet', type=Path, default=DATA/'video_stages.csv')
    parser.add_argument('--output-dir', type=Path, default=RESULTS)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--reviewed-labels-root', type=Path)
    parser.add_argument('--benchmark-source', help='Optional recorded video or webcam 0 for five-row actual runtime tests')
    parser.add_argument('--benchmark-seconds', type=float, default=300)
    parser.add_argument('--benchmark-rounds', type=int, default=3)
    args = parser.parse_args(argv)
    rows = read_csv(args.worksheet)
    if not rows: raise RuntimeError(f'No worksheet rows in {args.worksheet}; run auto_annotate_videos or create a video_path CSV')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detector = Detector(args.weights, args.device, args.precision)
    per_clip, identities, errors = [], [], []
    for row in rows:
        try:
            if row.get('review_status') == 'source_changed_review_needed':
                raise ValueError('Source changed; update its identity and reverify event timestamps')
            path, meta = cached_video(row, detector, args.cache_dir or args.output_dir/'cache', args.max_frames)
            labels = row.get('reviewed_labels_dir') or (args.reviewed_labels_root / clip_id(row['video_path']) if args.reviewed_labels_root else None)
            labels = resolve_video(labels) if labels else None
            metrics = evaluate_records(row, frame_records(path), meta['fps'], labels, meta['partial'])
            per_clip.extend(metrics)
            identities.append(dict(video_path=row['video_path'], cache=str(path), **meta))
            print(f'Evaluated all five rows: {row["video_path"]}')
        except Exception as e:
            errors.append(dict(video_path=row['video_path'], error=f'{type(e).__name__}: {e}'))
            print(f'Skipped clip: {e}')
    summary = aggregate(per_clip)
    write_csv(args.output_dir/'4stage_metrics.csv', summary)
    write_json(args.output_dir/'per_clip_results.json', per_clip)
    write_json(args.output_dir/'run_manifest.json', dict(detector=detector.identity(), sources=identities, errors=errors))
    secondary = [{k:r[k] for k in ('variant','method','video_path','box_review_frames','box_tp','box_fp','box_fn')}
                 for r in per_clip if r['box_review_frames']]
    write_csv(args.output_dir/'optional_iou_metrics.csv', secondary,
        ['variant','method','video_path','box_review_frames','box_tp','box_fp','box_fn'])
    generate_report(args.output_dir, summary, per_clip, errors)
    if args.benchmark_source:
        from .demo_app import benchmark_stream
        benchmark_stream(detector, args.benchmark_source, args.output_dir/'performance',
            seconds=args.benchmark_seconds, rounds=args.benchmark_rounds, variants=[k for k,_ in VARIANTS])
        performance=json.loads((args.output_dir/'performance'/'benchmark.json').read_text())
        with (args.output_dir/'4stage_benchmark_report.md').open('a',encoding='utf-8') as f:
            f.write('\n## Actual recorded/live runtime measurements\n\n| Method | Pace | Round | Completed FPS | Mean latency ms | Mean FSM ms |\n|---|---|---:|---:|---:|---:|\n')
            for r in performance['results']:
                latency=r['latency_and_stage_ms']
                f.write(f'| {r["variant"]} | {r["pace"]} | {r["round"]} | {r["completed_fps"]:.3f} | {latency["latency_ms"]["mean"]:.3f} | {latency["fsm_ms"]["mean"] if latency["fsm_ms"]["mean"] is not None else "N/A"} |\n')
    if not per_clip: raise RuntimeError('No clips evaluated; reports contain unavailable metrics and error reasons')
    print(f'Report: {args.output_dir / "4stage_benchmark_report.md"}')


if __name__ == '__main__': run_main(main)

