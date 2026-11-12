"""Apply the owner's negative-folder flag rule without changing model outputs.

Uses the running review API for revision-safe updates. Batch flags are not
framewise visual verification, accepted event labels, or empty training boxes.
"""
import argparse
import copy
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request,urlopen

from .review_annotations import read_rows,write_index,firesense_partition as partition


FLAG_BASIS='owner_negative_folder_rule_not_framewise_visual_review'


def batch_state(row,frames,state,updated_utc):
    """Pure transformation: preserve all prior flags, notes, and annotations."""
    if partition(row) not in (('fire','neg'),('smoke','neg')):
        raise ValueError('Batch flagging is restricted to FireSense negative folders')
    if state['clip_id']!=row['label_id'] or state['video_path']!=row['video_path']:
        raise ValueError('State/source mismatch')
    result=copy.deepcopy(state);existing={f['id'] for f in result['flags']};added=[]
    target_class=partition(row)[0]
    for frame_index,timestamp,image,boxes in frames:
        for box_index,box in enumerate(boxes):
            identifier=f'{frame_index}:{box_index}:false_positive'
            if identifier in existing:continue
            predicted='fire' if box[0]==0 else 'smoke'
            note=(f'Batch flag per owner instruction: FIRESENSE/{target_class}/neg. '
                  f'Prediction: {predicted}. Basis: {FLAG_BASIS}. '
                  'The published negative sets are class-specific; this owner rule flags both predicted classes. '
                  'Not a visually verified empty training annotation.')
            flag=dict(id=identifier,frame_index=frame_index,timestamp_sec=timestamp,box_index=box_index,
                      issue='false_positive',notes=note,updated_utc=updated_utc)
            result['flags'].append(flag);existing.add(identifier);added.append(flag)
    return result,added


def request_json(url,payload=None):
    request=Request(url,data=json.dumps(payload).encode('utf-8') if payload is not None else None,
                    headers={'Content-Type':'application/json'} if payload is not None else {})
    with urlopen(request,timeout=30) as response:return json.load(response)


def snapshot(worksheet,root,rows):
    """Hash unchanged review artifacts; source video metadata requires no decode."""
    protected={Path(worksheet),root.parent/'detection_index.csv',root.parent/'detection_summary.csv'}
    sources={}
    for row in rows:
        if row.get('subset')=='mmodelfire' or partition(row) in (('fire','pos'),('smoke','pos')):
            folder=root/row['label_id']
            protected.update([folder/'frame_map.csv',folder/'draft_manifest.json',folder/'classes.txt',
                              root/'review'/(row['label_id']+'.html'),root/'manual_reviews'/(row['label_id']+'.json')])
        source=Path(row['video_path'])
        if source.exists():
            stat=source.stat();sources[str(source)]=[stat.st_size,stat.st_mtime_ns]
    return dict(files={str(path):hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
                       for path in sorted(protected)},source_video_metadata=sources)


def run_batch(worksheet,root,api='http://127.0.0.1:8766'):
    worksheet,root=Path(worksheet).resolve(),Path(root).resolve();rows=read_rows(worksheet)
    before=snapshot(worksheet,root,rows)
    batch_id=datetime.now(timezone.utc).strftime('firesense_negative_%Y%m%dT%H%M%S%fZ')
    folder=root/'review_batches'/batch_id;folder.mkdir(parents=True)
    (folder/'preservation_before.json').write_text(json.dumps(before,indent=2),encoding='utf-8')
    stats=[];flag_rows=[];timestamp=datetime.now(timezone.utc).isoformat()
    for row in rows:
        if partition(row) not in (('fire','neg'),('smoke','neg')):continue
        clip=row['label_id'];page=root/'review'/(clip+'.html')
        text=page.read_text(encoding='utf-8')
        data=json.loads(text.split('<script id="data" type="application/json">')[1].split('</script>')[0])
        if data['clip_id']!=clip or data['video_path']!=row['video_path']:
            raise ValueError('Review page/source mismatch')
        frames=data['frames'];positive=[f for f in frames if f[3]]
        total=sum(len(f[3]) for f in positive)
        state=request_json(api+'/api/review?clip='+clip)
        (folder/(clip+'.before.json')).write_text(json.dumps(state,indent=2),encoding='utf-8')
        updated,added=batch_state(row,positive,state,timestamp)
        if added:
            for retry in range(3):
                try:
                    saved=request_json(api+'/api/review',updated)
                    assert saved['annotations']==state['annotations'],'Manual training annotations changed'
                    break
                except HTTPError as error:
                    if error.code!=409 or retry==2:raise
                    state=request_json(api+'/api/review?clip='+clip)
                    updated,added=batch_state(row,positive,state,timestamp)
            print(f'Flagged {clip}: {len(positive)} positive frames, {total} boxes ({len(added)} new flags)',flush=True)
        stat=dict(label_id=clip,video_path=row['video_path'],category=partition(row)[0],sampled_frames=len(frames),
                  prediction_frames=len(positive),predicted_boxes=total,new_flags=len(added),
                  status='batch_flagged_by_owner_rule' if total else 'no_saved_predictions')
        stats.append(stat)
        for frame_index,t,image,boxes in positive:
            for box_index,box in enumerate(boxes):
                flag_rows.append(dict(video_path=row['video_path'],label_id=clip,frame_index=frame_index,timestamp_sec=t,
                    box_index=box_index,predicted_class='fire' if box[0]==0 else 'smoke',issue='false_positive',basis=FLAG_BASIS))
    after=snapshot(worksheet,root,rows)
    if before!=after:raise RuntimeError('Protected artifacts changed during the batch; inspect preservation snapshot')
    report=dict(status='completed',batch_id=batch_id,completed_utc=datetime.now(timezone.utc).isoformat(),basis=FLAG_BASIS,
        negative_clips=len(stats),negative_clips_with_predictions=sum(s['predicted_boxes']>0 for s in stats),
        prediction_frames=sum(s['prediction_frames'] for s in stats),predicted_boxes=sum(s['predicted_boxes'] for s in stats),
        new_flags=sum(s['new_flags'] for s in stats),remaining_positive_clips=sum(partition(r) in (('fire','pos'),('smoke','pos')) for r in rows),
        protected_artifacts_unchanged=True,accepted_event_cells_modified=False,training_annotations_created=False,
        videos=stats,provenance_note='Folder-rule flags, not individual visual confirmation. FireSense publishes class-specific negative sets.',
        source='https://zenodo.org/records/836749')
    (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fields=['video_path','label_id','frame_index','timestamp_sec','box_index','predicted_class','issue','basis']
    with (root.parent/'firesense_negative_flags.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(flag_rows)
    (root/'firesense_review_queue.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    summaries=read_rows(root.parent/'detection_summary.csv')
    write_index(summaries,sum(int(s['sampled_frames']) for s in summaries),root.parent)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worksheet',type=Path,required=True);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--api',default='http://127.0.0.1:8766')
    args=parser.parse_args();report=run_batch(args.worksheet,args.root,args.api)
    print(json.dumps({k:v for k,v in report.items() if k!='videos'},indent=2))


if __name__=='__main__':main()
