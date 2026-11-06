"""Finish a detached annotation job: bounded retries and a saved output audit.

This is a one-shot completion step, not a scheduled task. Original videos and
existing human-owned CSV cells/label files are never changed by the audit.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from .common import BASELINE_SHA,DEFAULT_WEIGHTS,read_csv,sha256,write_json


def alive(pid):
    import psutil
    try:
        process=psutil.Process(pid)
        return process.is_running() and 'fire_warning.auto_annotate_videos' in process.cmdline()
    except (psutil.NoSuchProcess,psutil.AccessDenied):return False


def audit(root,worksheet):
    rows=read_csv(worksheet);issues=[];videos=[]
    for row in rows:
        p=root/row['label_id']/'draft_manifest.json'
        if row.get('annotation_status')!='draft' or not p.exists():
            issues.append(dict(video_path=row['video_path'],error='No completed draft'))
            continue
        manifest=json.loads(p.read_text(encoding='utf-8'))
        folder=p.parent
        with (folder/'frame_map.csv').open(encoding='utf-8',newline='') as f:maps=list(csv.DictReader(f))
        images={p.stem for p in (folder/'images').iterdir() if p.suffix=='.jpg'}
        labels={p.stem for p in (folder/'labels').iterdir() if p.suffix=='.txt'}
        expected={Path(r['image']).stem for r in maps}
        if not expected.issubset(images) or not expected.issubset(labels):
            issues.append(dict(video_path=row['video_path'],error='Missing image/label pairs'))
        if len(expected)!=len(maps):issues.append(dict(video_path=row['video_path'],error='Duplicate frame-map rows'))
        if (folder/'classes.txt').read_text().splitlines()!=['fire','smoke']:
            issues.append(dict(video_path=row['video_path'],error='Class map changed'))
        # Check spread-out artifacts without loading hundreds of GB of JPEG data.
        checked_boxes=0
        for entry in maps[::max(1,len(maps)//16)]:
            image=folder/'images'/entry['image'];label=folder/'labels'/(Path(entry['image']).stem+'.txt')
            if not image.exists() or not label.exists():continue
            with image.open('rb') as f:
                first=f.read(2);f.seek(-2,2);last=f.read(2)
            if first!=b'\xff\xd8' or last!=b'\xff\xd9':
                issues.append(dict(video_path=row['video_path'],error=f'Incomplete JPEG: {image.name}'))
            for line in label.read_text(encoding='utf-8').splitlines():
                fields=line.split()
                try:
                    valid=len(fields)==5 and int(fields[0]) in (0,1) and all(0<=float(v)<=1 for v in fields[1:]) and float(fields[3])>0 and float(fields[4])>0
                except (ValueError,IndexError):valid=False
                if not valid:issues.append(dict(video_path=row['video_path'],error=f'Invalid draft YOLO row: {label.name}'))
                checked_boxes+=1
        videos.append(dict(video_path=row['video_path'],subset=row['subset'],decoded_frames=manifest['processed_frames'],
            keyframes=len(expected),images=len(images),label_files=len(labels),sampled_boxes_checked=checked_boxes,
            source_sha256=manifest['suggestion']['auto_source_sha256']))
    if sha256(DEFAULT_WEIGHTS)!=BASELINE_SHA:issues.append(dict(error='Frozen source checkpoint changed'))
    return dict(status='completed' if not issues and len(videos)==len(rows) else 'completed_with_issues',
        completed_utc=datetime.now(timezone.utc).isoformat(),expected_videos=len(rows),completed_videos=len(videos),
        datasets=dict(Counter(r['subset'] for r in videos)),total_source_frames=sum(v['decoded_frames'] for v in videos),
        total_keyframes=sum(v['keyframes'] for v in videos),total_label_files=sum(v['label_files'] for v in videos),
        worksheet=str(worksheet),output_directory=str(root),issues=issues,videos=videos,
        box_validation='Pair completeness for all mapped frames; sampled normalized-YOLO syntax/JPEG integrity. No semantic box correctness claim.',
        event_times='Automatic proposal columns populated; human verification fields preserved')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--pid',type=int,required=True)
    args=p.parse_args();root=args.root;worksheet=root.parent/'video_stages.csv'
    while alive(args.pid):time.sleep(10)
    for retry in range(2):
        rows=read_csv(worksheet)
        summary_path=root/'annotation_summary.json'
        summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if all(r.get('annotation_status')=='draft' for r in rows) and not summary.get('errors'):break
        with (root/f'retry_{retry+1}.log').open('w',encoding='utf-8') as log:
            command=[sys.executable,'-u','-m','fire_warning.auto_annotate_videos',
                '--manifest',str(root/'input_manifest.json'),'--worksheet',str(worksheet),'--output-dir',str(root),
                '--keyframes-only','--batch-size','4','--jpeg-quality','90','--stride','5','--precision','fp32','--device','0','--resume']
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False)
    result=audit(root,worksheet)
    write_json(root/'completion_report.json',result)
    lines=['Annotation run '+result['status'],'',f'Videos: {result["completed_videos"]}/{result["expected_videos"]}',
        f'Datasets: {result["datasets"]}',f'Decoded source frames: {result["total_source_frames"]}',
        f'JPEG keyframes: {result["total_keyframes"]}',f'YOLO label files: {result["total_label_files"]}',
        f'Worksheet: {worksheet}','',result['event_times'],result['box_validation']]
    lines += [str(issue) for issue in result['issues']]
    (root/'COMPLETED.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
