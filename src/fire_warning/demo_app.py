"""Native/Streamlit demonstration and three-round recorded/live performance tests."""
from __future__ import annotations

import argparse
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import wave

from .common import (DATA, RESULTS, ROOT, Detector, Pipeline, VideoSource, VideoWriter,
    accepted_number, detector_arguments, read_csv, resolve_video, run_main, sha256, stats, write_csv, write_json)


def alarm_wave():
    import numpy as np
    rate = 22050
    samples = np.arange(int(rate*0.5)) / rate
    signal = (np.sin(2*np.pi*1100*samples) * 12000).astype('<i2').tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(rate); f.writeframes(signal)
    return buffer.getvalue()


class AlarmOutput:
    def __init__(self, output, silent=False):
        self.output, self.silent = Path(output), silent
        self.cancel = threading.Event()
        self.count = 0
        self.warning = None

    def acknowledge(self): self.cancel.set()

    def trigger(self, frame, state):
        import cv2
        self.output.mkdir(parents=True, exist_ok=True)
        self.count += 1
        name = self.output / f'incident_{time.time_ns()}_{self.count}'
        if not cv2.imwrite(str(name.with_suffix('.jpg')), frame):
            raise RuntimeError('Incident snapshot could not be saved')
        write_json(name.with_suffix('.json'), state)
        if not self.silent:
            self.cancel.clear()
            threading.Thread(target=self._sound, daemon=True).start()

    def _sound(self):
        try:
            if sys.platform == 'win32':
                import winsound
                for _ in range(3):
                    if self.cancel.is_set(): return
                    winsound.Beep(1100, 250)
                    if self.cancel.wait(0.1): return
            else:
                import shutil
                executable = shutil.which('aplay')
                if not executable:
                    self.warning = 'Native audio unavailable; Streamlit can play the browser alarm.'
                    return
                subprocess.run([executable, '-q'], input=alarm_wave(), timeout=5, check=False,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e: self.warning = f'Audio unavailable: {e}'


def draw_hud(image, record, fps=0, raw=False):
    import cv2
    out = image.copy()
    h, w = out.shape[:2]
    state = record['state']
    stage = state['stage']
    color = {1:(60,150,50),2:(0,200,230),3:(0,130,240),4:(210,110,40)}[stage]
    if state['alarm']: color = (30,30,245) if int(time.time()*3)%2 else (20,20,150)
    raw_threshold = record.get('raw_threshold',0.25)
    label = (f'RAW DETECTOR {raw_threshold:.2f}' if raw else f'STAGE {stage}: ' +
        {1:'PRE-IGNITION / MONITORING',2:'ONSET CONFIRMED',3:'PROGRESSION',4:state['disposition']}[stage])
    cv2.rectangle(out, (0,0),(w,55),color,-1)
    cv2.putText(out,label,(10,33),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)
    if raw:
        boxes = [dict(box=b[:4],cls_id=int(b[5]),ema_conf=b[4],track_id=0) for b in record['boxes'] if b[4]>=raw_threshold]
    else: boxes = state['confirmed_tracks']
    from .fsm_engine import scaled_size
    for track in boxes:
        x1,y1,x2,y2 = map(int,track['box'])
        tag = scaled_size(track['box'], (w,h))
        text = f'#{track["track_id"]} {"FIRE" if track["cls_id"]==0 else "SMOKE"} {track["ema_conf"]:.2f} | {tag["size_tag"]} {tag["scaled_width"]:.0f}x{tag["scaled_height"]:.0f}'
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,80,255) if track['cls_id']==0 else (230,170,70),2)
        cv2.putText(out,text,(max(0,x1),max(70,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)
    ms = record['timings'].get('detector_ms',0)+record['timings'].get('tracking_ms',0)+record['timings'].get('fsm_ms',0)
    cv2.rectangle(out,(0,h-45),(w,h),(35,35,35),-1)
    score = state['hazard_score']
    timing_text = f'TTD {state["event_ttd_sec"]:.2f}s' if state.get('event_ttd_sec') is not None else (
        f'Evidence {max(0,record["timestamp_sec"]-state["evidence_onset_sec"]):.2f}s' if state.get('evidence_onset_sec') is not None else '')
    cv2.putText(out,f'{fps:.1f} processed FPS | {ms:.2f} ms | Hazard {score:.1f} | {timing_text} {state.get("alert_reason") or "monitoring"}',
        (10,h-17),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    cv2.rectangle(out,(0,h-5),(int(w*score/100),h),(0,0,255),-1)
    return out


def raw_record(frame, detector, variant):
    boxes, timings = detector.predict(frame.image)
    threshold = 0.25 if variant == 'raw_025' else 0.14
    positive = any(b[4]>=threshold for b in boxes)
    return dict(frame_index=frame.index,timestamp_sec=frame.timestamp_sec,boxes=boxes,raw_threshold=threshold,
        timings=dict(decode_ms=frame.read_ms,tracking_ms=0.0,**timings),
        state=dict(stage=2 if positive else 1,disposition='RAW_INDICATION',alarm=positive,
            hazard_score=0,confirmed_tracks=[],alert_reason='single_frame' if positive else None))


def benchmark_stream(detector, source, output, seconds=300, rounds=3, warmups=30,
                     variants=None, paces=None, record=False):
    """Paced files submit frames at 30Hz; late slots skip input, never duplicate inference."""
    import cv2, torch, psutil
    from .common import Frame
    if seconds <= 0 or rounds < 1 or warmups < 0: raise ValueError('Invalid benchmark duration/rounds/warmups')
    variants = variants or ['full_on']
    paces = paces or ['paced','unpaced']
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    identity = dict(source=str(source), source_sha256=None if str(source).isdecimal() else sha256(source),
        recorded_stream=not str(source).isdecimal(), detector=detector.identity(),
        requested_source_size=[1280,720], target_submission_fps=30,
        repeats_are_recorded_stream_tests_not_physical_camera_evidence=True,
        latency_boundary='host read/decode through render and optional encode; sensor capture latency unmeasured')
    results = []
    for round_no in range(1, rounds+1):
        order = variants if round_no%2 else list(reversed(variants))
        for pace in paces:
            for variant in order:
                source_obj = VideoSource(source,width=1280,height=720)
                writer = None
                rows, loops, dropped, resized = [],0,0,False
                actual_size = [source_obj.width,source_obj.height]
                resized = actual_size != [1280,720]
                try:
                    first = source_obj.read()
                    if first is None: raise RuntimeError('No benchmark frames')
                    first = replace(first,image=cv2.resize(first.image,(1280,720)))
                    for _ in range(warmups): detector.predict(first.image)
                    pipeline = Pipeline(detector,source_obj.fps,glare_filter=variant=='full_on',persistence_only=variant=='persistence')
                    if detector.device != 'cpu': torch.cuda.reset_peak_memory_stats()
                    started, slot = time.perf_counter(),0
                    frame = first
                    while time.perf_counter()-started < seconds:
                        if pace=='paced':
                            deadline=started+slot/30
                            if deadline>time.perf_counter(): time.sleep(deadline-time.perf_counter())
                            late=max(0,int((time.perf_counter()-deadline)*30))
                            for _ in range(late):
                                skip=source_obj.read()
                                if skip is None:
                                    if source_obj.camera: raise RuntimeError('Camera disconnected')
                                    source_obj.close(); source_obj=VideoSource(source,width=1280,height=720); pipeline.reset(); loops+=1
                                    skip=source_obj.read()
                                    if skip is None: raise RuntimeError('Empty file cannot be looped')
                            dropped+=late; slot+=late
                        begin=time.perf_counter()
                        if frame is None: frame=source_obj.read()
                        if frame is None:
                            if source_obj.camera: raise RuntimeError('Camera disconnected')
                            source_obj.close(); source_obj=VideoSource(source,width=1280,height=720); pipeline.reset(); loops+=1
                            frame=source_obj.read()
                            if frame is None: raise RuntimeError('Empty file cannot be looped')
                        resized |= frame.image.shape[:2] != (720,1280)
                        if frame.image.shape[:2] != (720,1280): frame=replace(frame,image=cv2.resize(frame.image,(1280,720)))
                        rec=raw_record(frame,detector,variant) if variant.startswith('raw_') else pipeline.process(frame)
                        before_draw=time.perf_counter()
                        rendered=draw_hud(frame.image,rec,raw=variant.startswith('raw_'))
                        draw_ms=(time.perf_counter()-before_draw)*1000
                        encode_ms=0.0
                        if record:
                            if writer is None:
                                writer=VideoWriter(output/f'{variant}_{pace}_round{round_no}.mp4',1280,720,30)
                            before_encode=time.perf_counter()
                            writer.write(rendered,time.perf_counter()-started)
                            encode_ms=(time.perf_counter()-before_encode)*1000
                        rows.append(dict(frame_index=frame.index,media_timestamp_sec=frame.timestamp_sec,
                            loop=loops,latency_ms=(time.perf_counter()-begin)*1000,draw_ms=draw_ms,encode_ms=encode_ms,
                            **rec['timings']))
                        frame=None; slot+=1
                    elapsed=time.perf_counter()-started
                    totals={key:stats([r[key] for r in rows if key in r]) for key in
                        ['latency_ms','decode_ms','preprocess_ms','inference_ms','postprocess_ms','detector_ms','tracking_ms','fsm_ms','draw_ms','encode_ms']}
                    result=dict(variant=variant,pace=pace,round=round_no,seconds_requested=seconds,
                        elapsed_sec=elapsed,completed_frames=len(rows),completed_fps=len(rows)/elapsed,
                        completed_fps_ge_30=len(rows)/elapsed>=30,dropped_submission_frames=dropped,
                        physical_camera_dropped_frames=None,loop_count=loops,input_resized_to_720p=resized,
                        reported_original_source_size=actual_size,source_fps=source_obj.fps,warmups=warmups,
                        latency_and_stage_ms=totals,mean_fsm_under_0_5ms=(totals['fsm_ms']['mean']<0.5 if totals['fsm_ms']['mean'] is not None else None),
                        process_rss_bytes=psutil.Process().memory_info().rss,
                        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if detector.device!='cpu' else None,
                        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved() if detector.device!='cpu' else None,
                        recording=record,display_included=False,
                        full_five_minute_protocol=seconds>=300 and rounds>=3)
                    results.append(result)
                    write_csv(output/f'{variant}_{pace}_round{round_no}_frames.csv',rows)
                    write_json(output/'benchmark.json',dict(identity=identity,results=results))
                    print(f'{variant} {pace} round {round_no}: {result["completed_fps"]:.2f} FPS, {loops} loops')
                finally:
                    source_obj.close()
                    if writer: writer.close()
    write_csv(output/'benchmark_summary.csv',[dict(variant=r['variant'],pace=r['pace'],round=r['round'],
        fps=r['completed_fps'],mean_latency_ms=r['latency_and_stage_ms']['latency_ms']['mean'],
        p95_latency_ms=r['latency_and_stage_ms']['latency_ms']['p95'],
        mean_fsm_ms=r['latency_and_stage_ms']['fsm_ms']['mean'],dropped=r['dropped_submission_frames'],
        loops=r['loop_count'],five_minute_protocol=r['full_five_minute_protocol']) for r in results])
    return results


class DemoWorker:
    """One resource-owning worker; UI never touches inference state concurrently."""
    def __init__(self,args):
        self.args=args; self.stop=threading.Event(); self.reset_event=threading.Event(); self.ack=threading.Event()
        self.lock=threading.Lock(); self.latest=None; self.error=None; self.done=False
        self.thread=threading.Thread(target=self.run,daemon=True)
        self.alarm=None

    def start(self): self.thread.start()
    def stop_and_wait(self): self.stop.set(); self.thread.join(timeout=5)

    def run(self):
        import cv2
        writer=None
        try:
            detector=Detector(self.args.weights,self.args.device,self.args.precision)
            self.alarm=AlarmOutput(self.args.output_dir/'incidents',self.args.silent)
            ignition=None
            if not str(self.args.source).isdecimal():
                for row in read_csv(self.args.worksheet):
                    if resolve_video(row['video_path'])==resolve_video(self.args.source):
                        ignition=accepted_number(row,'t_ignition_start_sec'); break
            with VideoSource(self.args.source,width=1280,height=720) as source:
                pipeline=Pipeline(detector,source.fps,glare_filter=not self.args.glare_off)
                started=time.perf_counter(); playback_start=started
                count=0; was_alarm=False; event_ttd=None
                while not self.stop.is_set():
                    frame=source.read()
                    if frame is None: break
                    if not source.camera and not self.args.unpaced:
                        delay=playback_start+frame.timestamp_sec-time.perf_counter()
                        if delay>0 and self.stop.wait(delay): break
                    if self.reset_event.is_set():
                        pipeline.reset(); was_alarm=False; self.reset_event.clear()
                    if self.ack.is_set():
                        pipeline.fsm.acknowledge(); self.alarm.acknowledge(); self.ack.clear()
                    rec=pipeline.process(frame)
                    if ignition is not None and ignition>=0 and frame.timestamp_sec>=ignition and rec['state']['confirmed_tracks'] and event_ttd is None:
                        event_ttd=frame.timestamp_sec-ignition
                    rec['state']['event_ttd_sec']=event_ttd
                    count+=1
                    fps=count/max(time.perf_counter()-started,1e-9)
                    rendered=draw_hud(frame.image,rec,fps)
                    if self.args.comparison:
                        import numpy as np
                        rendered=np.hstack((draw_hud(frame.image,rec,fps,raw=True),rendered))
                    if rec['state']['alarm'] and not was_alarm: self.alarm.trigger(frame.image,rec['state'])
                    was_alarm=rec['state']['alarm']
                    if self.args.record:
                        if writer is None: writer=VideoWriter(self.args.record,rendered.shape[1],rendered.shape[0],source.fps)
                        writer.write(rendered,frame.timestamp_sec)
                    with self.lock: self.latest=(rendered,rec,fps,self.alarm.count)
                    if self.args.max_frames and count>=self.args.max_frames: break
        except Exception as e: self.error=f'{type(e).__name__}: {e}'
        finally:
            if writer:
                try: writer.close()
                except Exception as e: self.error=f'Recording finalization failed: {e}'
            if self.alarm: self.alarm.acknowledge()
            self.done=True


def streamlit_page(args):
    import streamlit as st
    st.set_page_config(page_title='Fire / Smoke Four-Stage Demo',layout='wide')
    st.title('Fire / Smoke Four-Stage Demo')
    st.caption('Local research demonstration. Browser refresh FPS and completed inference FPS are separate.')
    files=sorted(str(p) for p in (DATA/'videos').glob('**/*') if p.suffix.lower() in ('.mp4','.avi','.mkv'))
    source=st.text_input('Source: local video path or host webcam 0',value=str(args.source))
    if files:
        selected=st.selectbox('Files in data/videos', ['Use source field']+files)
        if selected!='Use source field': source=selected
    compare=st.checkbox('Side-by-side raw .25 / full FSM',value=args.comparison)
    silent=st.checkbox('Mute audio',value=args.silent)
    columns=st.columns(4)
    worker=st.session_state.get('worker')
    if columns[0].button('Start',disabled=worker is not None and not worker.done):
        options=argparse.Namespace(**vars(args)); options.source=source; options.comparison=compare; options.silent=silent
        worker=DemoWorker(options); st.session_state.worker=worker; st.session_state.played_incident=0; worker.start()
    if columns[1].button('Stop') and worker: worker.stop_and_wait()
    if columns[2].button('Reset state') and worker: worker.reset_event.set()
    if columns[3].button('Acknowledge / silence') and worker: worker.ack.set()
    @st.fragment(run_every=0.1)
    def refresh():
        active=st.session_state.get('worker')
        if not active: return
        if active.error: st.error(active.error)
        with active.lock: latest=active.latest
        if latest:
            image,rec,fps,incident=latest
            st.image(image,channels='BGR',width='stretch')
            st.write(dict(processed_fps=round(fps,2),stage=rec['state']['stage'],
                hazard=rec['state']['hazard_score'],alert_reason=rec['state']['alert_reason']))
            if incident>st.session_state.get('played_incident',0) and not silent:
                st.audio(alarm_wave(),format='audio/wav',autoplay=True)
                st.session_state.played_incident=incident
        if active.done: st.caption('Stopped / end of file')
    refresh()


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    detector_arguments(p)
    p.add_argument('--mode',choices=['cv2','streamlit'],default='cv2')
    p.add_argument('--source',default='0')
    p.add_argument('--worksheet',type=Path,default=DATA/'video_stages.csv')
    p.add_argument('--output-dir',type=Path,default=RESULTS/'demo')
    p.add_argument('--record',type=Path)
    p.add_argument('--comparison',action='store_true')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--silent',action='store_true')
    p.add_argument('--glare-off',action='store_true')
    p.add_argument('--unpaced',action='store_true')
    p.add_argument('--max-frames',type=int)
    p.add_argument('--benchmark',action='store_true')
    p.add_argument('--seconds',type=float,default=300)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--warmups',type=int,default=30)
    p.add_argument('--all-variants',action='store_true')
    p.add_argument('--pace',choices=['both','paced','unpaced'],default='both')
    return p.parse_args(argv)


def main(argv=None):
    args=parse_args(argv)
    if args.benchmark:
        from .evaluate_benchmark import VARIANTS
        detector=Detector(args.weights,args.device,args.precision)
        benchmark_stream(detector,args.source,args.output_dir,args.seconds,args.rounds,args.warmups,
            variants=[k for k,_ in VARIANTS] if args.all_variants else ['full_on'],
            paces=None if args.pace=='both' else [args.pace],record=bool(args.record))
        return
    if args.mode=='streamlit':
        command=[sys.executable,'-m','streamlit','run',str(Path(__file__).with_name('streamlit_app.py')),
            '--server.headless=true','--',*(argv if argv is not None else sys.argv[1:])]
        raise SystemExit(subprocess.call(command,cwd=ROOT))
    import cv2
    worker=DemoWorker(args); worker.start()
    try:
        while not worker.done:
            with worker.lock: latest=worker.latest
            if latest and not args.headless:
                cv2.imshow('Four-stage fire/smoke | Q stop | R reset | A acknowledge',latest[0])
                key=cv2.waitKey(1)&255
                if key in (ord('q'),27): break
                if key==ord('r'): worker.reset_event.set()
                if key==ord('a'): worker.ack.set()
            time.sleep(0.01)
    finally:
        worker.stop_and_wait()
        if not args.headless: cv2.destroyAllWindows()
    if worker.error: raise RuntimeError(worker.error)
    if worker.latest:
        _,rec,fps,_=worker.latest
        write_json(args.output_dir/'demo_summary.json',dict(source=args.source,processed_fps=fps,last_record=rec,
            playback_paced=not args.unpaced,physical_camera=str(args.source).isdecimal()))
        print(f'Demo completed: {fps:.2f} processed FPS')


if __name__=='__main__': run_main(main)

