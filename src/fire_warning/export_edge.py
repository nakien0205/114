"""Export the frozen detector without altering its checkpoint; validate and record fallback."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import shutil
import time

from .common import (DEFAULT_WEIGHTS, RESULTS, ROOT, Detector, VideoSource, detector_arguments,
    read_csv, resolve_video, run_main, sha256, write_json)
from .evaluate_benchmark import cached_video, evaluate_records, frame_records, iou


def numeric_agreement(reference, candidate):
    """Near-threshold candidates are logged separately, never silently ignored in parity."""
    reference_strong=[r for r in reference if r[4]>=0.16]
    used=set(); failures=[]
    for ref in reference_strong:
        candidates=[(iou(ref,p),index,p) for index,p in enumerate(candidate)
                    if index not in used and int(p[5])==int(ref[5])]
        if not candidates:
            failures.append(dict(reference=ref,reason='no_class_match')); continue
        overlap,index,pred=max(candidates,key=lambda x:x[0]); used.add(index)
        if overlap<0.90 or abs(ref[4]-pred[4])>0.02:
            failures.append(dict(reference=ref,candidate=pred,iou=overlap,reason='numeric_tolerance'))
    extra=[r for index,r in enumerate(candidate) if index not in used and r[4]>=0.16]
    return dict(passed=not failures and not extra,reference_strong_boxes=len(reference_strong),
        failures=failures,extra_strong_boxes=extra,
        near_threshold_reference=[r for r in reference if r[4]<0.16],
        near_threshold_candidate=[r for r in candidate if r[4]<0.16])


def sample_frames(sample,limit=60):
    import cv2
    if sample is None:
        manifest=ROOT/'research/hardware-benchmark-20261010/laptop-inputs/manifest.json'
        if not manifest.exists(): raise ValueError('Provide --sample image/video for export validation')
        row=json.loads(manifest.read_text())['images'][0]
        sample=manifest.parent/row['path']
    sample=resolve_video(sample)
    if sample.suffix.lower() in ('.jpg','.jpeg','.png','.bmp'):
        image=cv2.imread(str(sample))
        if image is None: raise RuntimeError('Sample image cannot be read')
        return [image],dict(path=str(sample),sha256=sha256(sample),type='image')
    frames=[]
    with VideoSource(sample) as source:
        for _ in range(limit):
            frame=source.read()
            if frame is None: break
            frames.append(frame.image)
    if not frames: raise RuntimeError('No sample video frames')
    return frames,dict(path=str(sample),sha256=sha256(sample),type='video',frames=len(frames))


def event_parity(reference,candidate,worksheet,output):
    if not worksheet: return dict(status='unavailable',reason='No development event worksheet supplied')
    rows=read_csv(worksheet)
    if any(row.get('split','development') not in ('development','dev','') for row in rows):
        raise ValueError('Export acceptance uses development clips only')
    ref_metrics=[]; candidate_metrics=[]
    for row in rows:
        p,m=cached_video(row,reference,Path(output)/'reference_cache')
        q,n=cached_video(row,candidate,Path(output)/'candidate_cache')
        ref_metrics.extend(r for r in evaluate_records(row,frame_records(p),m['fps']) if r['variant']=='full_on')
        candidate_metrics.extend(r for r in evaluate_records(row,frame_records(q),n['fps']) if r['variant']=='full_on')
    failures=[]; observable=0
    for ref,cand in zip(ref_metrics,candidate_metrics):
        if ref['event_observable']:
            observable+=1
            if ref['event_detected'] and not cand['event_detected']: failures.append('additional onset miss: '+ref['video_path'])
            for boundary in (1,3):
                if ref['ttd_sec'] is not None and ref['ttd_sec']<=boundary and (cand['ttd_sec'] is None or cand['ttd_sec']>boundary):
                    failures.append(f'lost <= {boundary}s detection: '+ref['video_path'])
        if cand['false_alarm_incidents']>ref['false_alarm_incidents']: failures.append('additional negative alarm: '+ref['video_path'])
        if ref['should_alert_human']==1 and ref['alarm'] and not cand['alarm']: failures.append('additional alert miss: '+ref['video_path'])
    return dict(status='passed' if observable and not failures else 'failed' if failures else 'unavailable',
        observable_events=observable,failures=failures,reference=ref_metrics,candidate=candidate_metrics)


def export_models(args):
    import torch, onnx
    from ultralytics import YOLO
    reference=Detector(args.weights,args.device,'fp32')
    before=sha256(args.weights)
    frames,sample=sample_frames(args.sample,args.validation_frames)
    expected=[reference.predict(image)[0] for image in frames]
    output=args.output_dir/platform.system().lower()
    output.mkdir(parents=True,exist_ok=True)
    staging=output/'staging'; staging.mkdir(exist_ok=True)
    staged=staging/'best.pt'
    shutil.copy2(args.weights,staged)
    existing_proof=json.loads((output/'export_manifest.json').read_text()) if (output/'export_manifest.json').exists() else {}
    manifest=dict(source_checkpoint_sha256=before,runtime=reference.identity(),sample=sample,
        validation_is_accuracy_proof=False,artifacts={},fallback=None)
    for fmt,suffix in [('onnx','.onnx'),('engine','.engine')]:
        if fmt not in args.formats: continue
        try:
            if not torch.cuda.is_available(): raise RuntimeError('FP16 CUDA export requires an available GPU')
            if fmt=='engine':
                import tensorrt
                manifest['tensorrt_version']=tensorrt.__version__
            target=output/('best_fp16'+suffix)
            if args.reuse_existing and target.exists():
                proof=existing_proof
                artifact=proof.get('artifacts',{}).get(fmt,{})
                if proof.get('source_checkpoint_sha256')!=before or artifact.get('sha256')!=sha256(target):
                    raise RuntimeError('Existing export identity is not verified; rebuild without --reuse-existing')
                print(f'Revalidating existing {fmt} artifact...',flush=True)
            else:
                print(f'Exporting {fmt} FP16...',flush=True)
                model=YOLO(str(staged))
                produced=Path(model.export(format=fmt,imgsz=640,batch=1,dynamic=False,
                    half=True,device=reference.device,opset=17,nms=False,workspace=2,verbose=False))
                shutil.copy2(produced,target)
            precision={}
            if fmt=='onnx':
                graph=onnx.load(target); onnx.checker.check_model(graph)
                precision=dict(fp16_initializers=sum(t.data_type==onnx.TensorProto.FLOAT16 for t in graph.graph.initializer),
                    input_types=[v.type.tensor_type.elem_type for v in graph.graph.input])
                if not precision['fp16_initializers']: raise RuntimeError('Export contains no FP16 initializers')
            backend=Detector(target,reference.device,'fp16')
            comparisons=[numeric_agreement(ref,backend.predict(image)[0]) for ref,image in zip(expected,frames)]
            parity=event_parity(reference,backend,args.validation_worksheet,output/f'{fmt}_parity')
            passed=all(r['passed'] for r in comparisons) and parity['status']!='failed'
            manifest['artifacts'][fmt]=dict(path=str(target),sha256=sha256(target),precision=precision,
                numeric_validation_passed=all(r['passed'] for r in comparisons),comparisons=comparisons,
                reference_strong_boxes=sum(r['reference_strong_boxes'] for r in comparisons),
                event_parity=parity,accepted_for_sample_inference=passed,
                event_validated_for_deployment=passed and parity['status']=='passed',runtime=backend.identity())
        except Exception as e:
            manifest['artifacts'][fmt]=dict(error=f'{type(e).__name__}: {e}',accepted_for_sample_inference=False,
                event_validated_for_deployment=False)
            print(f'{fmt} unavailable/failed: {e}',flush=True)
        write_json(output/'export_manifest.json',manifest)
    # Explicit PyTorch fallback also checked, rather than silently relabelling a failed export.
    engine=manifest['artifacts'].get('engine',{})
    if not engine.get('event_validated_for_deployment',False):
        for precision in ('fp16','fp32'):
            try:
                candidate=Detector(args.weights,args.device,precision)
                comparisons=[numeric_agreement(ref,candidate.predict(image)[0]) for ref,image in zip(expected,frames)]
                if not all(r['passed'] for r in comparisons): continue
                parity=event_parity(reference,candidate,args.validation_worksheet,output/f'pytorch_{precision}_parity')
                if parity['status']=='failed': continue
                manifest['fallback']=dict(backend='pytorch',precision=candidate.precision,
                    reason='No event-validated compatible TensorRT engine',runtime=candidate.identity(),
                    numeric_comparisons=comparisons,event_parity=parity)
                break
            except Exception as e: manifest.setdefault('fallback_errors',[]).append(str(e))
    if sha256(args.weights)!=before: raise RuntimeError('Source checkpoint changed during export')
    manifest['source_checkpoint_unchanged']=True
    write_json(output/'export_manifest.json',manifest)
    if not any(a.get('accepted_for_sample_inference') for a in manifest['artifacts'].values()) and not manifest['fallback']:
        raise RuntimeError('No usable export or validated PyTorch fallback; see export_manifest.json')
    print(f'Export evidence: {output/"export_manifest.json"}')
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    detector_arguments(p)
    p.add_argument('--output-dir',type=Path,default=RESULTS/'exports')
    p.add_argument('--formats',nargs='+',choices=['onnx','engine'],default=['onnx','engine'])
    p.add_argument('--sample',type=Path)
    p.add_argument('--validation-frames',type=int,default=60)
    p.add_argument('--validation-worksheet',type=Path)
    p.add_argument('--reuse-existing',action='store_true',help='Revalidate artifacts only if their saved source/artifact hashes match')
    args=p.parse_args(argv)
    if args.validation_frames<1: p.error('--validation-frames must be >=1')
    export_models(args)


if __name__=='__main__': run_main(main)

