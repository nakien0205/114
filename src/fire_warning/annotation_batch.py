"""Bounded batched keyframe annotation with a dense first-candidate window.

All source frames are decoded for coverage and PTS validation. Only exported
keyframes are inferred. Original decoded video frames are held for one batch,
allowing retroactive dense inference after the first candidate in that batch.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import math
import time

from .common import Pipeline, accepted_number, read_csv, resolve_video, sha256, write_csv, write_json


def write_frame(images, labels, index, image, boxes, quality):
    import cv2
    stem=f'frame_{index:08d}'
    image_path,label_path=images/(stem+'.jpg'),labels/(stem+'.txt')
    if not image_path.exists() and not cv2.imwrite(str(image_path),image,[cv2.IMWRITE_JPEG_QUALITY,quality]):
        raise RuntimeError(f'Could not write {image_path}')
    if not label_path.exists():
        h,w=image.shape[:2]; lines=[]
        for x1,y1,x2,y2,conf,cls in boxes:
            x1,x2=max(0,min(w,x1)),max(0,min(w,x2))
            y1,y2=max(0,min(h,y1)),max(0,min(h,y2))
            if x2>x1 and y2>y1:
                lines.append(f'{int(cls)} {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}')
        label_path.write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')


def annotate_keyframes(row,detector,output_root,stride=5,max_frames=None,batch_size=8,jpeg_quality=90):
    import av
    from .auto_annotate_videos import clip_id
    path=resolve_video(row['video_path']);folder=Path(output_root)/clip_id(path)
    images,labels=folder/'images',folder/'labels'
    images.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
    classes=folder/'classes.txt'
    if classes.exists() and classes.read_text().splitlines()!=['fire','smoke']:
        raise ValueError('Existing class mapping differs from frozen fire/smoke mapping')
    if not classes.exists():classes.write_text('fire\nsmoke\n',encoding='utf-8')
    ignition=accepted_number(row,'t_ignition_start_sec')
    mappings=[];class_first={};first_candidate=None;first_confirmation=None;first_progression=None
    any_alert=False;inferred=0;decoded_count=0;started=time.perf_counter();last_progress=started
    pending_writes=deque();buffer=[];selected=[]
    source_hash=sha256(path)
    with av.open(str(path)) as container,ThreadPoolExecutor(max_workers=2) as pool:
        stream=container.streams.video[0]
        stream.codec_context.thread_count=2;stream.codec_context.thread_type='AUTO'
        fps=float(stream.average_rate or stream.guessed_rate or 0)
        if not math.isfinite(fps) or fps<=0:raise ValueError('Source FPS unavailable')
        duration=float(stream.duration*stream.time_base) if stream.duration else container.duration/1e6 if container.duration else None
        pipeline=Pipeline(detector,max(fps/stride,1e-3))
        origin=None;last_time=None

        def flush():
            nonlocal first_candidate,first_confirmation,first_progression,any_alert,inferred,last_progress
            if not selected:return
            items={index:(index,t,frame,quality) for index,t,frame,quality in selected}
            decoded_images={index:frame.to_ndarray(format='bgr24') for index,t,frame,quality in selected}
            predictions=detector.predict_batch([decoded_images[r[0]] for r in selected])
            predictions_by_index={r[0]:boxes for r,boxes in zip(selected,predictions)}
            if first_candidate is None:
                first_candidate=next((r[1] for r,boxes in zip(selected,predictions) if boxes),None)
            if first_candidate is not None:
                extras=[r for r in buffer if first_candidate<=r[1]<=first_candidate+3 and r[0] not in items]
                for begin in range(0,len(extras),batch_size):
                    chunk=extras[begin:begin+batch_size]
                    arrays=[r[2].to_ndarray(format='bgr24') for r in chunk]
                    for r,image,boxes in zip(chunk,arrays,detector.predict_batch(arrays)):
                        items[r[0]]=r;decoded_images[r[0]]=image;predictions_by_index[r[0]]=boxes
            for index in sorted(items):
                _,t,_,quality=items[index];image=decoded_images[index];boxes=predictions_by_index[index]
                dense=(first_candidate is not None and first_candidate<=t<=first_candidate+3)
                dense|=ignition is not None and ignition>=0 and ignition-1<=t<=ignition+3
                cadence=fps if dense else fps/stride
                pipeline.fsm.config=replace(pipeline.fsm.config,fps=max(cadence,1e-3))
                for tracker in pipeline.trackers.trackers:tracker.max_time_lost=max(1,round(cadence))
                tracks=pipeline.trackers.update(boxes,image)
                state=pipeline.fsm.update(tracks,t,(image.shape[1],image.shape[0]))
                for tr in state['transitions']:
                    if tr['stage']==2 and first_confirmation is None:first_confirmation=tr['timestamp_sec']
                    if tr['stage']==3 and first_progression is None:first_progression=tr['timestamp_sec']
                any_alert|=state['alarm']
                for box in boxes:class_first.setdefault(int(box[5]),t)
                pending_writes.append(pool.submit(write_frame,images,labels,index,image,boxes,jpeg_quality))
                if len(pending_writes)>=batch_size*2:pending_writes.popleft().result()
                mappings.append(dict(image=f'frame_{index:08d}.jpg',frame_index=index,timestamp_sec=t,timestamp_quality=quality))
                inferred+=1
            if time.perf_counter()-last_progress>=20:
                write_json(Path(output_root)/'progress.json',dict(video_path=str(path),subset=row.get('subset'),
                    decoded_source_frames=decoded_count,inferred_keyframes=inferred,elapsed_sec=time.perf_counter()-started,
                    media_timestamp_sec=items[max(items)][1],duration_sec=duration,status='running'))
                print(f'Progress: {path.name}: source frame {decoded_count}, {inferred} labeled keyframes',flush=True)
                last_progress=time.perf_counter()
            buffer.clear();selected.clear()

        for index,frame in enumerate(container.decode(video=0)):
            if max_frames is not None and index>=max_frames:break
            pts=frame.time
            if pts is None:t=index/fps;quality='nominal_fps_fallback'
            else:
                if origin is None:origin=float(pts)
                t=float(pts)-origin;quality='presentation_timestamp'
            if last_time is not None and t<=last_time:raise ValueError('Non-increasing source timestamps')
            last_time=t;decoded_count+=1
            item=(index,t,frame,quality);buffer.append(item)
            dense=first_candidate is not None and first_candidate<=t<=first_candidate+3
            dense|=ignition is not None and ignition>=0 and ignition-1<=t<=ignition+3
            if index%stride==0 or dense:selected.append(item)
            if len(selected)>=batch_size:flush()
        flush()
        for pending in pending_writes:pending.result()
    if not decoded_count:raise ValueError('No frames decoded')
    previous={r['image']:r for r in read_csv(folder/'frame_map.csv')}
    previous.update({r['image']:r for r in mappings})
    write_csv(folder/'frame_map.csv',sorted(previous.values(),key=lambda r:int(r['frame_index'])))
    suggestion=dict(row,video_path=str(path),fps=fps,duration_sec=duration if duration is not None else '',
        auto_t_ignition_start_sec=first_confirmation if first_confirmation is not None else '',
        auto_t_aftermath_start_sec=first_progression if first_progression is not None else '',
        auto_should_alert_human=int(any_alert),auto_smoke_first=int(class_first.get(1,float('inf'))<class_first.get(0,float('inf'))),
        auto_source_sha256=source_hash,label_id=clip_id(path),images_dir=str(images),labels_dir=str(labels),
        annotation_status='partial_draft' if max_frames is not None and decoded_count>=max_frames else 'draft',
        inference_sampling=f'stride_{stride}_plus_dense_first_candidate_3sec',model_confidence=0.14)
    write_json(folder/'draft_manifest.json',dict(suggestion=suggestion,processed_frames=decoded_count,inferred_frames=inferred,
        keyframes_written=len(mappings),elapsed_sec=time.perf_counter()-started,detector=detector.identity(),
        labels_are_ground_truth=False,batch_size=batch_size,jpeg_quality=jpeg_quality,annotation_mode='keyframes_only',
        sampling_note='All source frames decoded. Inference on every fifth frame and the first candidate dense window; automatic times are proposals, not full-frame benchmark measurements.'))
    return suggestion
