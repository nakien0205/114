"""Create draft labels and timestamp suggestions without accepting them as truth."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .common import (DATA, ROOT, Detector, Pipeline, VideoSource, accepted_number,
    detector_arguments, discover_videos, read_csv, resolve_video, run_main, sha256, write_csv, write_json)

FIELDS = ['video_path', 'subset', 'experiment_group', 'split', 'fps', 'duration_sec',
    't_ignition_start_sec', 't_aftermath_start_sec', 'should_alert_human',
    'auto_t_ignition_start_sec', 'auto_t_aftermath_start_sec', 'auto_smoke_first',
    'source_sha256', 'auto_source_sha256', 'review_status', 'reviewed_labels_dir', 'label_id', 'annotation_status']


def clip_id(path):
    p = resolve_video(path)
    return p.stem + '-' + hashlib.sha256(str(p).encode()).hexdigest()[:10]


def merge_worksheet(existing, suggestions):
    """Only suggestions/metadata update; accepted event cells remain human-owned."""
    merged = {str(resolve_video(r['video_path'])): dict(r) for r in existing}
    for suggestion in suggestions:
        key = str(resolve_video(suggestion['video_path']))
        old = merged.get(key)
        row = dict(suggestion) if old is None else dict(old)
        if old is not None:
            for field, value in suggestion.items():
                if field.startswith('auto_') or field in ('fps', 'duration_sec', 'annotation_status', 'label_id',
                        'images_dir', 'labels_dir', 'inference_sampling', 'model_confidence'):
                    row[field] = value
            if old.get('source_sha256') and old['source_sha256'] != suggestion['auto_source_sha256']:
                row['review_status'] = 'source_changed_review_needed'
        for field in ('t_ignition_start_sec', 't_aftermath_start_sec', 'should_alert_human'):
            row.setdefault(field, '')
        if not row.get('source_sha256'): row['source_sha256'] = suggestion['auto_source_sha256']
        merged[key] = row
    return list(merged.values())


def annotate_video(row, detector, output_root, stride=5, max_frames=None):
    import cv2
    p = resolve_video(row['video_path'])
    folder = Path(output_root) / clip_id(p)
    images, labels = folder / 'images', folder / 'labels'
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    (folder / 'classes.txt').write_text('fire\nsmoke\n', encoding='utf-8')
    ignition = accepted_number(row, 't_ignition_start_sec')
    first_candidate, first_confirmation, first_progression = None, None, None
    class_first, mappings = {}, []
    count = 0
    with VideoSource(p) as source:
        pipeline = Pipeline(detector, source.fps)
        while (max_frames is None or count < max_frames):
            frame = source.read()
            if frame is None: break
            record = pipeline.process(frame)
            if record['boxes'] and first_candidate is None: first_candidate = frame.timestamp_sec
            for tr in record['state']['transitions']:
                if tr['stage'] == 2 and first_confirmation is None: first_confirmation = tr['timestamp_sec']
                if tr['stage'] == 3 and first_progression is None: first_progression = tr['timestamp_sec']
            for box in record['boxes']: class_first.setdefault(int(box[5]), frame.timestamp_sec)
            dense = (first_candidate is not None and frame.timestamp_sec - first_candidate <= 3)
            dense |= ignition is not None and ignition >= 0 and ignition - 1 <= frame.timestamp_sec <= ignition + 3
            if frame.index % stride == 0 or dense:
                stem = f'frame_{frame.index:08d}'
                image_path, label_path = images / (stem + '.jpg'), labels / (stem + '.txt')
                if not image_path.exists() and not cv2.imwrite(str(image_path), frame.image):
                    raise RuntimeError(f'Could not write {image_path}')
                # Never overwrite a label the owner may have edited.
                if not label_path.exists():
                    h, w = frame.image.shape[:2]
                    lines = []
                    for x1, y1, x2, y2, conf, cls in record['boxes']:
                        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
                        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
                        if x2 > x1 and y2 > y1:
                            lines.append(f'{int(cls)} {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}')
                    label_path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
                mappings.append(dict(image=stem + '.jpg', frame_index=frame.index,
                    timestamp_sec=frame.timestamp_sec, timestamp_quality=frame.timestamp_quality))
            count += 1
        if count == 0: raise RuntimeError('No frames decoded')
        duration, fps = source.duration_sec, source.fps
    previous = read_csv(folder / 'frame_map.csv')
    combined = {r['image']: r for r in previous}
    combined.update({r['image']: r for r in mappings})
    write_csv(folder / 'frame_map.csv', sorted(combined.values(), key=lambda r: int(r['frame_index'])))
    suggestion = dict(row, video_path=str(p), fps=fps, duration_sec=duration if duration is not None else '',
        auto_t_ignition_start_sec=first_confirmation if first_confirmation is not None else '',
        auto_t_aftermath_start_sec=first_progression if first_progression is not None else '',
        auto_smoke_first=int(class_first.get(1, float('inf')) < class_first.get(0, float('inf'))),
        auto_source_sha256=sha256(p), label_id=clip_id(p),
        annotation_status='partial_draft' if max_frames and count >= max_frames else 'draft')
    write_json(folder / 'draft_manifest.json', dict(suggestion=suggestion, processed_frames=count,
        detector=detector.identity(), labels_are_ground_truth=False,
        import_instructions='Open images in X-AnyLabeling; Import YOLO detection labels with classes.txt. Draft labels are optional; verify event CSV independently.'))
    return suggestion


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    detector_arguments(parser)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--worksheet', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--keyframes-only', action='store_true', help='Batch inference on exported keyframes, rather than infer all decoded frames')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--jpeg-quality', type=int, default=90)
    parser.add_argument('--resume', action='store_true', help='Skip completed keyframe drafts with matching source/model/sampling identity')
    args = parser.parse_args(argv)
    if args.stride < 1: parser.error('--stride must be >=1')
    if not 1<=args.batch_size<=16: parser.error('--batch-size must be 1..16')
    if not 1<=args.jpeg_quality<=100: parser.error('--jpeg-quality must be 1..100')
    worksheet = args.worksheet or args.data_dir / 'video_stages.csv'
    output = args.output_dir or args.data_dir / 'auto_labeled'
    rows = discover_videos(args.data_dir, args.manifest)
    if not rows:
        for folder in ('mmodelfire', 'negatives', 'custom_4stage'):
            (args.data_dir / 'videos' / folder).mkdir(parents=True, exist_ok=True)
        print(f'No videos yet. Drop files under {args.data_dir / "videos"}, or supply --manifest.')
        return
    detector = Detector(args.weights, args.device, args.precision)
    existing = read_csv(worksheet)
    old = {str(resolve_video(r['video_path'])): r for r in existing}
    suggestions, errors = [], []
    for row in rows:
        row = {**row, **old.get(str(resolve_video(row['video_path'])), {})}
        try:
            existing_manifest = output / clip_id(row['video_path']) / 'draft_manifest.json'
            suggestion = None
            if args.resume and args.keyframes_only and existing_manifest.exists():
                import json
                proof = json.loads(existing_manifest.read_text(encoding='utf-8'))
                prior = proof.get('suggestion', {})
                if (proof.get('annotation_mode') == 'keyframes_only' and prior.get('annotation_status') == 'draft'
                        and prior.get('auto_source_sha256') == sha256(resolve_video(row['video_path']))
                        and proof.get('detector',{}).get('checkpoint_sha256') == detector.sha
                        and prior.get('inference_sampling') == f'stride_{args.stride}_plus_dense_first_candidate_3sec'
                        and proof.get('detector',{}).get('precision') == detector.precision):
                    suggestion = prior
                    print(f'Resume: completed draft retained: {row["video_path"]}',flush=True)
            if suggestion is None:
                if args.keyframes_only:
                    from .annotation_batch import annotate_keyframes
                    suggestion = annotate_keyframes(row, detector, output, args.stride, args.max_frames,
                        args.batch_size, args.jpeg_quality)
                else:
                    suggestion = annotate_video(row, detector, output, args.stride, args.max_frames)
            suggestions.append(suggestion)
            # Re-read immediately before writing so edits made while processing a
            # previous clip are preserved as well as edits made between runs.
            merged = merge_worksheet(read_csv(worksheet), [suggestion])
            fields = list(dict.fromkeys(FIELDS + [k for r in merged for k in r]))
            write_csv(worksheet, merged, fields)
            print(f'Draft complete: {suggestion["video_path"]}',flush=True)
            write_json(output / 'annotation_summary.json', dict(completed=len(suggestions),total=len(rows),errors=errors,
                worksheet=str(worksheet),status='running',accepted_timestamps_overwritten=False))
        except Exception as e:
            errors.append(dict(video_path=row['video_path'], error=f'{type(e).__name__}: {e}'))
            print(f'Skipped {row["video_path"]}: {e}',flush=True)
    write_json(output / 'annotation_summary.json', dict(completed=len(suggestions),total=len(rows), errors=errors,
        worksheet=str(worksheet),status='completed' if not errors else 'completed_with_errors',accepted_timestamps_overwritten=False))
    write_json(output / 'progress.json', dict(status='completed' if not errors else 'completed_with_errors',
        completed=len(suggestions),total=len(rows),worksheet=str(worksheet),errors=errors))
    if errors and not suggestions: raise RuntimeError('No video could be annotated; see annotation_summary.json')
    from .review_annotations import build_review
    review_report = build_review(worksheet, output, worksheet.parent)
    print(f'Detection review: {worksheet.parent / "review_detections.html"} ({review_report["status"]})')
    print(f'Verify the three event cells in {worksheet}. Box review is optional.')


if __name__ == '__main__': run_main(main)

