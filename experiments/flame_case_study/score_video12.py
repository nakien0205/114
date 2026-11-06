"""Read-only single-video evaluation using the owner's frozen annotations."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from flame import Config, CentroidConfirmation
from run import sha, write

HERE = Path(__file__).resolve().parent
ARMS = ('raw', 'motion_filtered', 'confirmed_1s', 'confirmed_3s')


def iou(a, b):
    intersection = max(0., min(a[2],b[2])-max(a[0],b[0])) * max(0., min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.


def match_count(boxes, truth):
    """Confidence-ordered, one-to-one IoU>=.50 fire matching."""
    remaining = set(range(len(truth)))
    matches = 0
    for box in sorted(boxes, key=lambda b: (-b[4], *b[:4])):
        if not remaining:
            break
        score, index = max((iou(box,truth[i]),-i) for i in remaining)
        if score >= .5:
            remaining.remove(-index)
            matches += 1
    return matches


def visibility_statistics(times, flags, start_frame, end_frame):
    """Left-held sampled indication on [start PTS,end PTS), no GT interpolation."""
    start, stop = start_frame, end_frame
    deltas = np.diff(times[start:stop+1])
    selected = np.asarray(flags[start:stop],dtype=bool)
    gaps = []
    opening = None
    for index in range(start,stop):
        if not flags[index] and opening is None:
            opening = index
        elif flags[index] and opening is not None:
            gaps.append(dict(start_sec=times[opening],end_sec=times[index],duration_sec=times[index]-times[opening]))
            opening = None
    if opening is not None:
        gaps.append(dict(start_sec=times[opening],end_sec=times[stop],duration_sec=times[stop]-times[opening]))
    after_hit = next((i for i in range(start,stop+1) if flags[i]),None)
    internal = [g for g in gaps if after_hit is not None and g['start_sec'] > times[after_hit] and g['end_sec'] < times[stop]]
    return dict(visible_duration_sec=float(sum(deltas)),
                fire_box_presence_fraction=float(np.dot(deltas,selected)/sum(deltas)),
                total_missing_sec=float(np.dot(deltas,~selected)),
                missing_runs=len(gaps),longest_missing_sec=max((g['duration_sec'] for g in gaps),default=0.),
                internal_gap_count=len(internal),internal_gap_total_sec=sum(g['duration_sec'] for g in internal),
                gaps=gaps)


def score(directory):
    directory = Path(directory).resolve()
    study = json.loads((HERE/'video12-study.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    protocol = json.loads((directory/'protocol.json').read_text())
    if summary['status'] != 'complete' or sha(directory/'frames.jsonl') != summary['output_sha256']:
        raise ValueError('Incomplete or changed output')
    if protocol['video_sha256'] != study['source']['video_sha256'] or protocol['model'].get('model') != 'plain_combined_60_s2026':
        raise ValueError('Unexpected source/model')
    for kind in ('truth','references'):
        if sha(study[kind+'_path']) != study[kind+'_sha256']:
            raise ValueError(f'Frozen {kind} changed')
    reviews = json.loads(Path(study['truth_path']).read_text())['reviews']['12']
    interval, = reviews['intervals']
    units = [u for u in json.loads(Path(study['references_path']).read_text())['primary_units'] if u['video_id']=='12']
    wanted = {c['frame_index'] for u in units for c in u['candidates']}
    cfg = Config(**study['config'])
    three = CentroidConfirmation(replace(cfg,confirmation_sec=3.0))
    times, flags = [], {arm:[] for arm in ARMS}
    kept = {}
    events = {'confirmed_1s':[], 'confirmed_3s':[]}
    generic_alarm = {'confirmed_1s':[], 'confirmed_3s':[]}
    with (directory/'frames.jsonl').open(encoding='utf-8') as stream:
        for index,line in enumerate(stream):
            row = json.loads(line)
            if row['frame_index'] != index:
                raise ValueError('Frame timeline is not contiguous')
            t = row['timestamp_sec']
            result3 = three.update(row['moving_boxes'],t,(row['width'],row['height']))
            boxes = dict(raw=row['raw_boxes'],motion_filtered=row['moving_boxes'],
                         confirmed_1s=row['observed_confirmed_boxes'],confirmed_3s=result3['observed_confirmed_boxes'])
            fire = {arm:[b for b in value if int(b[5])==cfg.fire_class] for arm,value in boxes.items()}
            times.append(t)
            for arm in ARMS:
                flags[arm].append(bool(fire[arm]))
            if index in wanted:
                kept[index] = fire
            for arm,result in [('confirmed_1s',row),('confirmed_3s',result3)]:
                events[arm].extend(result['new_confirmations'])
                generic_alarm[arm].append(bool(result['alarm']))
    if len(times) != summary['frames']:
        raise ValueError('Summary frame count mismatch')
    per_arm = {}
    for arm in ARMS:
        measurements, earliest_possible, earliest_definite = [], None, None
        for u in units:
            counts = [match_count(kept[c['frame_index']][arm],u['truth']) for c in u['candidates']]
            measurements.append(dict(unit=u['unit'],truth_boxes=len(u['truth']),minimum_matches=min(counts),maximum_matches=max(counts)))
            inside = all(interval['start_frame'] <= c['frame_index'] <= interval['end_frame'] for c in u['candidates'])
            if u['truth'] and inside:
                hits = [c['timestamp_s'] for c,n in zip(u['candidates'],counts) if n]
                if hits:
                    v=min(hits);earliest_possible=v if earliest_possible is None else min(earliest_possible,v)
                if all(counts):
                    v=max(c['timestamp_s'] for c in u['candidates'])
                    earliest_definite=v if earliest_definite is None else min(earliest_definite,v)
        denominator = sum(m['truth_boxes'] for m in measurements)
        result = visibility_statistics(times,flags[arm],interval['start_frame'],interval['end_frame'])
        result.update(annotated_truth_boxes=denominator,
            matched_gt_box_fraction_lower=sum(m['minimum_matches'] for m in measurements)/denominator,
            matched_gt_box_fraction_upper=sum(m['maximum_matches'] for m in measurements)/denominator,
            first_possible_correct_sample_sec=earliest_possible,first_definite_correct_sample_sec=earliest_definite,
            first_possible_correct_sample_delay_sec=None if earliest_possible is None else earliest_possible-interval['start_s'],
            first_definite_correct_sample_delay_sec=None if earliest_definite is None else earliest_definite-interval['start_s'],
            first_fire_box_sec=next((t for t,positive in zip(times,flags[arm]) if positive),None))
        per_arm[arm]=result
        write(directory/f'{arm}-reference-matches.json',measurements)
    for arm in events:
        first = events[arm][0]['timestamp_sec'] if events[arm] else None
        post = sum(times[i+1]-times[i] for i in range(interval['end_frame']+1,len(times)-1) if generic_alarm[arm][i])
        per_arm[arm].update(generic_first_alarm_sec=first,
            generic_first_alarm_relative_to_flame_sec=None if first is None else first-interval['start_s'],
            generic_confirmation_events=len(events[arm]),alarm_active_after_visible_flame_end_sec=post)
    report=dict(study=study['study'],status='complete_exploratory_single_video',interval=interval,
        source_video=protocol['source_video'],video_sha256=protocol['video_sha256'],model=study['model'],
        frames=len(times),reference_samples=len(units),positive_reference_samples=sum(bool(u['truth']) for u in units),
        results=per_arm,processed_output_sha256=summary['output_sha256'],
        limits=study['limitations']+['Fire-box presence is not spatially validated on unannotated frames.',
        'Sampled first correct hits are not exact first-correct alarm times.',
        'One previously examined fire clip cannot establish generalization or false alarms/hour on no-fire videos.',
        'No ten-second physical-danger rule is implemented or validated.'])
    write(directory/'evaluation.json',report)
    lines=['# FLAME-inspired test on 12.mp4','',
        f"Frozen visible flame: **{interval['start_s']:.3f}–{interval['end_s']:.3f} s**. {len(times):,} decoded frames.",
        '','Detector: existing combined-data plain nano, trained for 60 epochs, seed 2026, validation-selected best checkpoint.',
        'Saved verified detector predictions were replayed; MOG2 processed every original-resolution decoded frame. No new neural inference or training.',
        '', '| Output | Fire-box presence during visible flame | Longest missing span (s) | Internal gaps | Matched annotated GT boxes | First definite correct sampled hit delay (s) |',
        '|---|---:|---:|---:|---:|---:|']
    for arm,r in per_arm.items():
        delay=r['first_definite_correct_sample_delay_sec']
        lines.append(f"| {arm} | {r['fire_box_presence_fraction']:.2%} | {r['longest_missing_sec']:.3f} | {r['internal_gap_count']} | {r['matched_gt_box_fraction_lower']:.2%}–{r['matched_gt_box_fraction_upper']:.2%} | {'none' if delay is None else f'{delay:.3f}'} |")
    lines += ['', 'Fire-box presence includes spatially wrong boxes; use annotated matching to judge localization. Missing spans include onset/trailing misses; internal gaps exclude those boundaries.',
        '', '## Alarm state versus observed evidence','']
    for arm in events:
        r=per_arm[arm]
        lines.append(f"- {arm}: first generic flame/smoke alarm at {r['generic_first_alarm_sec']} s; {r['generic_confirmation_events']} confirmation events; alarm active for {r['alarm_active_after_visible_flame_end_sec']:.3f} s after visible flame ended.")
    lines += ['', 'Algorithm 1 leaves confirmed tracks confirmed during missing observations. This is not recovered visual evidence. Flame/smoke are associated together, so a pre-flame alarm is not automatically a false alarm; smoke onset is unavailable.',
              '', '## Limits',''] + ['- '+s for s in report['limits']]
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Export a standalone scientific figure, with no temporal smoothing of flags.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(5,1,figsize=(13,7),sharex=True,layout='constrained')
    axes[0].broken_barh([(interval['start_s'],interval['end_s']-interval['start_s'])],(.1,.8),facecolors='#26734d')
    axes[0].set_ylabel('Human\nvisible flame')
    for ax,arm in zip(axes[1:],ARMS):
        ax.fill_between(times,0,np.asarray(flags[arm],dtype=int),step='post',color='#d47822')
        ax.set_ylabel(arm.replace('_','\n')); ax.set_ylim(0,1.1)
    for ax in axes:
        ax.set_yticks([]);ax.axvline(interval['start_s'],color='#26734d',linestyle='--',linewidth=.7)
    axes[-1].set_xlabel('Source video time (seconds)')
    fig.suptitle('12.mp4 — fire-box presence; only sampled annotations establish correct localization')
    fig.savefig(directory/'timeline.png',dpi=180)
    plt.close(fig)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory')
    result=score(p.parse_args().directory)
    print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2))
