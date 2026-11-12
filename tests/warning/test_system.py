from dataclasses import replace
import json
from pathlib import Path

import pytest

from fire_warning.fsm_engine import FireSmokeFSM,FSMConfig,scaled_size
from fire_warning.common import ClassTrackers, VideoSource, VideoWriter, Frame
from fire_warning.auto_annotate_videos import merge_worksheet
from fire_warning.evaluate_benchmark import evaluate_records,aggregate,VARIANTS,match_boxes
from fire_warning.export_edge import numeric_agreement


def detection(conf=.2,side=14,track_id=1,cls=0,x=100):
    return [x,100,x+side,100+side,track_id,conf,cls]


def sequence(fsm,seconds=7,fps=30,conf=.2,side=14):
    return [fsm.update([detection(conf,side)],i/fps,(640,640)) for i in range(int(seconds*fps))]


def record(i,boxes=None,tracks=None,fps=30):
    return dict(frame_index=i,timestamp_sec=i/fps,image_size=[640,640],
        timestamp_quality='presentation_timestamp',boxes=boxes or [],tracks=tracks or [])


def test_stage4_glare_ablations_preserve_onset():
    off=sequence(FireSmokeFSM(glare_filter=False))
    on=sequence(FireSmokeFSM(glare_filter=True))
    assert [r['confirmed_tracks']!=[] for r in off]==[r['confirmed_tracks']!=[] for r in on]
    assert off[-1]['alarm'] and off[-1]['alert_reason']=='five_second_persistence'
    assert not on[-1]['alarm'] and on[-1]['confirmed_tracks'][0]['persistence_blocked']


@pytest.mark.parametrize('cls',[0,1])
def test_high_conf_single_class_persistence_alert(cls):
    fsm=FireSmokeFSM()
    states=[fsm.update([detection(.7,cls=cls)],i/30,(640,640)) for i in range(210)]
    assert states[-1]['alarm']
    assert states[-1]['alert_reason']=='five_second_persistence'


def test_short_match_does_not_alert_and_missing_evidence_is_not_extinction():
    fsm=FireSmokeFSM()
    states=sequence(fsm,seconds=1,conf=.7)
    assert any(s['stage']==2 for s in states)
    assert not any(s['alarm'] for s in states)
    lost=fsm.update([],1.6,(640,640))
    assert lost['stage']==4 and lost['disposition']=='EVIDENCE_LOST_MONITORING'
    assert fsm.update([],2.1,(640,640))['stage']==1


@pytest.mark.parametrize('first_cls',[0,1])
def test_both_progression_orders(first_cls):
    fsm=FireSmokeFSM()
    for i in range(60):
        boxes=[detection(.8,track_id=1,cls=first_cls)]
        if i>=20: boxes.append(detection(.8,track_id=2,cls=1-first_cls,x=110))
        state=fsm.update(boxes,i/30,(640,640))
    assert state['stage']==3
    assert all(s['cooccurrence'] for s in state['confirmed_tracks'])


def test_score_alert_bypasses_guard():
    fsm=FireSmokeFSM()
    for i in range(150):
        state=fsm.update([detection(.9,track_id=1),detection(.9,track_id=2,cls=1,x=110)],i/30,(640,640))
    assert state['alarm'] and state['alert_reason']=='hazard_score'
    assert state['hazard_score']>=65


def test_growth_pathway_without_other_class():
    fsm=FireSmokeFSM()
    for i in range(125):
        side=14 if i<40 else 20
        state=fsm.update([detection(.7,side=side)],i/30,(640,640))
    assert state['stage']==3
    assert state['confirmed_tracks'][0]['area_growth']>=1.35


def test_second_nonstatic_track_can_alert():
    fsm=FireSmokeFSM()
    for i in range(210):
        state=fsm.update([detection(.2),detection(.7,track_id=2,x=500)],i/30,(640,640))
    assert state['alarm']
    assert state['confirmed_tracks'][0]['persistence_blocked']


@pytest.mark.parametrize('confidence,blocked',[(.299,True),(.30,False)])
def test_glare_confidence_boundary(confidence,blocked):
    last=sequence(FireSmokeFSM(),conf=confidence)[-1]
    assert last['confirmed_tracks'][0]['persistence_blocked']==blocked
    assert last['alarm']!=blocked


def test_insufficient_history_does_not_flag_rigidity():
    state=sequence(FireSmokeFSM(),seconds=.4)[-1]
    assert state['confirmed_tracks'] and not state['confirmed_tracks'][0]['static_low_confidence']


@pytest.mark.parametrize('fps',[25,30,60])
def test_confirmation_scales_with_fps(fps):
    states=sequence(FireSmokeFSM(fps=fps),seconds=1,fps=fps,conf=.7)
    first=next(s['timestamp_sec'] for s in states if s['confirmed_tracks'])
    assert .266 <= first <= .31


def test_intermittent_missing_frames_do_not_confirm():
    fsm=FireSmokeFSM()
    for i in range(90):
        state=fsm.update([detection(.7)] if i%3==0 else [],i/30,(640,640))
        assert not state['confirmed_tracks']


def test_latch_acknowledge_and_reset():
    fsm=FireSmokeFSM()
    sequence(fsm,conf=.7)
    assert fsm.update([],8,(640,640))['alarm']
    fsm.acknowledge()
    assert fsm.update([],9,(640,640))['audio_acknowledged']
    fsm.reset()
    assert not fsm.update([],0,(640,640))['alarm']


@pytest.mark.parametrize('side,tag',[(16,'TINY'),(16.01,'SMALL'),(32,'SMALL'),(32.01,'MEDIUM_LARGE')])
def test_size_boundaries(side,tag):
    assert scaled_size([0,0,side,side],(640,640))['size_tag']==tag


def test_timestamps_and_invalid_tracks():
    fsm=FireSmokeFSM()
    fsm.update([],0,(640,640))
    with pytest.raises(ValueError): fsm.update([],0,(640,640))
    with pytest.raises(ValueError): fsm.update([[0,0,10,10,1,float('nan'),0]],1,(640,640))


def test_weak_bytetrack_boxes_keep_identity():
    import numpy as np
    trackers=ClassTrackers(30)
    image=np.zeros((640,640,3),dtype=np.uint8)
    ids=[]
    for _ in range(20):
        tracks=trackers.update([[100,100,114,114,.18,0]],image)
        if tracks: ids.append(tracks[0][4])
    assert len(ids)==20 and len(set(ids))==1


def test_class_tracking_does_not_merge_classes():
    import numpy as np
    trackers=ClassTrackers(30)
    image=np.zeros((640,640,3),dtype=np.uint8)
    for _ in range(5): tracks=trackers.update([[100,100,114,114,.7,0],[100,100,114,114,.7,1]],image)
    assert len(tracks)==2 and len({t[4] for t in tracks})==2


def test_timestamp_evaluation_runs_without_labels_and_uses_predicted_size():
    rows=[record(i,[[100,100,114,114,.7,0]],[detection(.7)]) for i in range(210)]
    metrics=evaluate_records(dict(video_path='test.mp4',t_ignition_start_sec='1',should_alert_human='1'),rows,30)
    assert len(metrics)==5
    assert all(m['event_detected'] for m in metrics)
    assert all(m['ttd_sec']==0 for m in metrics)
    assert all(m['predicted_size_stratum']=='TINY' for m in metrics)
    assert all(m['box_review_frames']==0 for m in metrics)
    assert next(m for m in metrics if m['variant']=='full_on')['preconfirmed']


def test_missing_annotations_produce_all_five_unavailable_rows():
    metrics=evaluate_records(dict(video_path='test.mp4'),[record(i) for i in range(10)],30)
    summary=aggregate(metrics)
    assert len(summary)==5
    assert all(m['event_recall'] is None and m['alert_f1'] is None for m in summary)


def test_real_glare_ablation_counts():
    rows=[record(i,[[100,100,114,114,.2,0]],[detection(.2)]) for i in range(210)]
    metrics=evaluate_records(dict(video_path='test.mp4',t_ignition_start_sec='-1',should_alert_human='0'),rows,30)
    off=next(m for m in metrics if m['variant']=='full_off')
    on=next(m for m in metrics if m['variant']=='full_on')
    assert off['alarm'] and not on['alarm']
    assert off['false_positive_frames']==on['false_positive_frames']
    assert off['false_alarm_incidents']==1 and on['false_alarm_incidents']==0


def test_optional_iou_table(tmp_path):
    (tmp_path/'frame_00000000.txt').write_text('0 0.1671875 0.1671875 0.021875 0.021875\n')
    metrics=evaluate_records(dict(video_path='test.mp4'),[record(0,[[100,100,114,114,.7,0]])],30,tmp_path)
    raw=metrics[0]
    assert raw['box_review_frames']==1 and raw['box_tp']==1 and raw['box_fp']==0
    assert match_boxes([[0,0,10,10,.9,1]],[[0,0,10,10,1,0]])==(0,1,1)


def test_manual_cells_and_extra_columns_preserved():
    old=[dict(video_path='test.mp4',t_ignition_start_sec='2.3',t_aftermath_start_sec='4.5',
        should_alert_human='0',notes='owner',auto_t_ignition_start_sec='3',source_sha256='old')]
    new=[dict(video_path='test.mp4',auto_t_ignition_start_sec=1,auto_source_sha256='old')]
    row=merge_worksheet(old,new)[0]
    assert (row['t_ignition_start_sec'],row['t_aftermath_start_sec'],row['should_alert_human'],row['notes'])==('2.3','4.5','0','owner')
    assert row['auto_t_ignition_start_sec']==1


def test_partial_run_outcomes_unavailable():
    metrics=evaluate_records(dict(video_path='test.mp4',t_ignition_start_sec='0',should_alert_human='1'),[record(0)],30,partial=True)
    assert all(m['event_detected'] is None and m['should_alert_human'] is None for m in metrics)


def test_export_numeric_validation_detects_missing_or_extra_box():
    box=[0,0,10,10,.7,0]
    assert numeric_agreement([box],[box])['passed']
    assert not numeric_agreement([box],[])['passed']
    assert not numeric_agreement([],[box])['passed']


def test_video_pts_and_three_round_loop_benchmark(tmp_path):
    import numpy as np
    from fire_warning.demo_app import benchmark_stream
    path=tmp_path/'short.mp4'
    writer=VideoWriter(path,1280,720,30)
    for i in range(2): writer.write(np.zeros((720,1280,3),dtype=np.uint8),i/30)
    writer.close()
    with VideoSource(path) as source:
        first,second=source.read(),source.read()
        assert first.timestamp_sec==0 and second.timestamp_sec==pytest.approx(1/30,abs=1e-6)
        assert first.timestamp_quality=='presentation_timestamp'
    class FakeDetector:
        device='cpu'
        def identity(self): return dict(backend='synthetic_fixture')
        def predict(self,image):
            return [],dict(detector_ms=.01,preprocess_ms=0,inference_ms=.01,postprocess_ms=0)
    results=benchmark_stream(FakeDetector(),str(path),tmp_path/'perf',seconds=.12,rounds=3,warmups=1,paces=['unpaced'])
    assert len(results)==3 and all(r['loop_count']>=1 for r in results)
    assert all(not r['full_five_minute_protocol'] for r in results)


def test_unreadable_video_and_odd_writer_dimensions(tmp_path):
    with pytest.raises(Exception): VideoSource(tmp_path/'missing.mp4')
    with pytest.raises(ValueError): VideoWriter(tmp_path/'bad.mp4',101,100,30)


def test_nonrigid_area_variation_allows_persistence():
    fsm=FireSmokeFSM()
    for i in range(210):
        state=fsm.update([detection(.2,side=14 if i%2 else 15)],i/30,(640,640))
    assert state['alarm'] and not state['confirmed_tracks'][0]['static_low_confidence']


def test_nonrigid_aspect_variation_allows_persistence():
    fsm=FireSmokeFSM()
    for i in range(210):
        w=14 if i%2 else 15
        state=fsm.update([[100,100,100+w,100+196/w,1,.2,0]],i/30,(640,640))
    assert state['alarm'] and state['confirmed_tracks'][0]['aspect_ratio_std']>=.015


def test_camera_unavailable_without_opening_real_camera(monkeypatch):
    import cv2
    class ClosedCapture:
        released=False
        def isOpened(self):return False
        def release(self):self.released=True
    fake=ClosedCapture()
    monkeypatch.setattr(cv2,'VideoCapture',lambda *_:fake)
    with pytest.raises(RuntimeError,match='unavailable'):VideoSource('0')
    assert fake.released


def test_snapshot_failure_is_reported(tmp_path,monkeypatch):
    import cv2,numpy as np
    from fire_warning.demo_app import AlarmOutput
    monkeypatch.setattr(cv2,'imwrite',lambda *_:False)
    with pytest.raises(RuntimeError,match='snapshot'):AlarmOutput(tmp_path,silent=True).trigger(np.zeros((10,10,3)),{})


def test_wrong_runtime_fails_clearly(monkeypatch):
    import ultralytics
    from fire_warning.common import Detector
    monkeypatch.setattr(ultralytics,'__version__','8.4.21')
    with pytest.raises(RuntimeError,match='8.4.24 required'):Detector()


def test_box_class_confidence_and_id_validation():
    for row in ([0,0,1,1,1,1.1,0],[0,0,1,1,1,.7,2],[0,0,0,1,1,.7,0]):
        with pytest.raises(ValueError):FireSmokeFSM().update([row],0,(640,640))
    with pytest.raises(ValueError):FireSmokeFSM().update([detection(),detection()],0,(640,640))


def test_glare_history_growth_threshold():
    from fire_warning.fsm_engine import TrackState
    from collections import deque
    track=TrackState(1,0,0,1,.2,110,[0,0,10,11],{})
    track.onset_area=100
    track.history=deque([(0,100,1),(1,100,1)])
    assert not track.rigidity()['static_low_confidence']  # equality 1.10 is not <1.10
    track.smoothed_area=109.9
    assert track.rigidity()['static_low_confidence']


def test_worker_failure_terminates_without_gpu_or_camera(tmp_path,monkeypatch):
    from fire_warning import demo_app
    from argparse import Namespace
    monkeypatch.setattr(demo_app,'Detector',lambda *_:object())
    def fail_source(*_,**__):raise RuntimeError('fixture unavailable')
    monkeypatch.setattr(demo_app,'VideoSource',fail_source)
    worker=demo_app.DemoWorker(Namespace(weights=None,device='cpu',precision='fp32',output_dir=tmp_path,
        silent=True,source='file.mp4',worksheet=tmp_path/'missing.csv'))
    worker.run()
    assert worker.done and 'fixture unavailable' in worker.error


def test_streamlit_initial_render(monkeypatch):
    import sys
    monkeypatch.setattr(sys,'argv',['streamlit_app.py'])
    from streamlit.testing.v1 import AppTest
    import fire_warning
    at=AppTest.from_file(str(Path(fire_warning.__file__).with_name('streamlit_app.py')),default_timeout=15).run()
    assert not at.exception
    assert {'Start','Stop','Reset state','Acknowledge / silence'}<={b.label for b in at.button}


@pytest.mark.parametrize('positive',[False,True])
def test_batched_keyframes_dense_window_and_empty_labels(tmp_path,positive):
    import numpy as np
    from fire_warning.annotation_batch import annotate_keyframes
    path=tmp_path/'input.mp4'
    writer=VideoWriter(path,128,128,10)
    for i in range(40):writer.write(np.zeros((128,128,3),dtype=np.uint8),i/10)
    writer.close()
    class FakeDetector:
        def identity(self):return dict(backend='unit_fixture')
        def predict_batch(self,images):
            assert 1<=len(images)<=8
            return [[[10,10,20,20,.7,0]] if positive else [] for _ in images]
    result=annotate_keyframes(dict(video_path=str(path)),FakeDetector(),tmp_path/'out',batch_size=4)
    mappings=__import__('csv')
    with (Path(result['images_dir']).parent/'frame_map.csv').open() as f:rows=list(mappings.DictReader(f))
    expected=list(range(31))+[35] if positive else list(range(0,40,5))
    assert [int(r['frame_index']) for r in rows]==expected
    assert all((Path(result['labels_dir'])/(Path(r['image']).stem+'.txt')).exists() for r in rows)
    if not positive:assert all(not p.read_text() for p in Path(result['labels_dir']).glob('*.txt'))
    assert result['annotation_status']=='draft'
    # Reruns preserve edited files, even if model predictions disagree.
    label=next(Path(result['labels_dir']).glob('*.txt'));label.write_text('owner edit')
    annotate_keyframes(dict(video_path=str(path)),FakeDetector(),tmp_path/'out',batch_size=4)
    assert label.read_text()=='owner edit'


def test_annotation_completion_audit(tmp_path,monkeypatch):
    import numpy as np
    from fire_warning import finish_annotation_run as finish
    from fire_warning.annotation_batch import annotate_keyframes
    from fire_warning.common import write_csv,sha256
    path=tmp_path/'input.mp4'
    writer=VideoWriter(path,128,128,10)
    for i in range(10):writer.write(np.zeros((128,128,3),dtype=np.uint8),i/10)
    writer.close()
    class FakeDetector:
        def identity(self):return dict(backend='fixture')
        def predict_batch(self,images):return [[] for _ in images]
    row=annotate_keyframes(dict(video_path=str(path),subset='fixture'),FakeDetector(),tmp_path/'out')
    sheet=tmp_path/'events.csv';write_csv(sheet,[row])
    monkeypatch.setattr(finish,'DEFAULT_WEIGHTS',path)
    monkeypatch.setattr(finish,'BASELINE_SHA',sha256(path))
    audited=finish.audit(tmp_path/'out',sheet)
    assert audited['status']=='completed' and audited['total_keyframes']==2
    assert audited['total_label_files']==2 and not audited['issues']

