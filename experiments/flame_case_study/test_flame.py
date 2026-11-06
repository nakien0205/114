import gzip
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from flame import Config, CentroidConfirmation, FlameProcessor
from run import frames, prefetched_frames, run, sha
from score_video12 import match_count, visibility_statistics


BOX = [2, 2, 8, 8, .8, 0]


def test_confirmation_uses_elapsed_time_and_strict_boundary():
    tracker = CentroidConfirmation()
    assert not tracker.update([BOX], 0, (10,10))['alarm']
    assert not tracker.update([BOX], 1., (10,10))['alarm']
    out = tracker.update([BOX], 1.01, (10,10))
    assert out['alarm'] and out['new_confirmations'][0]['timestamp_sec'] == 1.01


def test_short_gap_restarts_confirmation_without_backdating():
    tracker = CentroidConfirmation()
    for t, boxes in [(0,[BOX]),(.6,[BOX]),(.7,[]),(.75,[BOX]),(1.7,[BOX])]:
        out = tracker.update(boxes,t,(10,10))
        assert not out['alarm']
    assert tracker.update([BOX],1.76,(10,10))['alarm']


def test_occlusion_timeout_and_rearming():
    tracker = CentroidConfirmation()
    tracker.update([BOX],0,(10,10))
    tracker.update([],.01,(10,10))
    assert tracker.update([],.12,(10,10))['rejected_track_ids'] == [1]
    assert tracker.update([BOX],.13,(10,10))['tracks'][0]['id'] == 2


def test_confirmed_alarm_does_not_fabricate_observed_boxes():
    tracker = CentroidConfirmation()
    tracker.update([BOX],0,(10,10))
    tracker.update([BOX],1.1,(10,10))
    out = tracker.update([],50.,(10,10))
    assert out['alarm'] and out['observed_confirmed_boxes'] == []
    assert out['tracks'][0]['state'] == 'Confirmed'
    tracker.reset()
    assert not tracker.update([],0,(10,10))['alarm']


def test_class_agnostic_greedy_one_to_one_association():
    tracker = CentroidConfirmation()
    other = [0,0,1,1,.7,0]
    tracker.update([BOX,other],0,(10,10))
    smoke = [2,2,8,8,.8,1]
    out = tracker.update([smoke],.1,(10,10))
    assert sum(t['observed'] for t in out['tracks']) == 1
    assert next(t for t in out['tracks'] if t['observed'])['box'][5] == 1


def test_foreground_fraction_strict_boundary_and_shadow_exclusion():
    processor = FlameProcessor()
    image = np.zeros((10,10,3),np.uint8)
    box = [0,0,10,10,.4,0]
    mask = np.zeros((10,10),np.uint8)
    mask.flat[:20] = 255
    assert not processor.update(image,[box],0,mask)['moving_boxes']
    mask.flat[20] = 255
    assert processor.update(image,[box],.1,mask)['moving_boxes'] == [box]
    mask[:] = 127
    assert not processor.update(image,[box],.2,mask)['moving_boxes']


def test_stationary_candidate_is_filtered_after_background_adaptation():
    processor = FlameProcessor()
    image = np.zeros((20,20,3),np.uint8)
    for i in range(25):
        out = processor.update(image,[BOX],i/30)
    assert out['raw_boxes'] and not out['moving_boxes']


def test_foreground_updates_even_without_detections():
    processor = FlameProcessor()
    image = np.zeros((20,20,3),np.uint8)
    for i in range(25):
        processor.update(image,[],i/30)
    assert not processor.update(image,[BOX],1.)['moving_boxes']


@pytest.mark.parametrize('bad', [[1,1,1,2,.5,0], [0,0,2,2,float('nan'),0], [0,0,2,2,.5,.2]])
def test_invalid_detections_do_not_mutate_stream(bad):
    p = FlameProcessor()
    with pytest.raises(ValueError):
        p.update(np.zeros((10,10,3),np.uint8),[bad],0.)
    assert p.previous_time is None


def test_zero_width_box_below_confidence_is_discarded_before_geometry():
    p=FlameProcessor()
    image=np.zeros((10,10,3),np.uint8)
    out=p.update(image,[[10,0,10,8,.0018,1]],0.)
    assert not out['raw_boxes'] and not out['alarm']


def test_source_boundary_requires_reset():
    p = FlameProcessor()
    image = np.zeros((10,10,3),np.uint8)
    p.update(image,[],1.)
    with pytest.raises(ValueError):
        p.update(image,[],1.)
    with pytest.raises(ValueError):
        p.update(np.zeros((20,20,3),np.uint8),[],2.)
    p.reset()
    p.update(image,[],0.)


def test_localization_matching_does_not_reuse_one_box_for_two_targets():
    truth = [BOX[:4],BOX[:4]]
    assert match_count([BOX],truth) == 1
    assert match_count([[0,0,1,1,.99,0],BOX],truth) == 1


def test_missing_runs_separate_internal_gaps_from_onset_and_tail():
    stats = visibility_statistics([0.,1.,2.,3.,4.,5.],[False,True,False,True,False,False],0,5)
    assert stats['missing_runs'] == 3
    assert stats['longest_missing_sec'] == 1.
    assert stats['internal_gap_count'] == 1
    assert stats['fire_box_presence_fraction'] == .4


def test_confirmation_never_emits_boxes_on_a_missing_frame():
    processor = FlameProcessor(Config(confirmation_sec=.1))
    image=np.zeros((10,10,3),np.uint8)
    foreground=np.full((10,10),255,np.uint8)
    processor.update(image,[BOX],0.,foreground)
    assert processor.update(image,[BOX],.2,foreground)['observed_confirmed_boxes']
    out=processor.update(image,[],.3,foreground)
    assert out['alarm'] and not out['raw_boxes'] and not out['observed_confirmed_boxes']


def test_replay_end_to_end_and_identity_checks(tmp_path):
    video = tmp_path/'synthetic.avi'
    writer = cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(32,32))
    assert writer.isOpened()
    for i in range(20):
        image = np.zeros((32,32,3),np.uint8)
        image[5:15,5:15] = 60 + (i%2)*180
        writer.write(image)
    writer.release()
    plain=list(frames(video));prefetched=list(prefetched_frames(video))
    assert [(i,t) for i,t,_ in plain]==[(i,t) for i,t,_ in prefetched]
    assert all(np.array_equal(a[2],b[2]) for a,b in zip(plain,prefetched))
    cache = tmp_path/'predictions.jsonl.gz'
    with gzip.open(cache,'wt',encoding='utf-8') as stream:
        for index,t,image in frames(video):
            stream.write(json.dumps(dict(frame_index=index,timestamp_s=t,height=32,width=32,
                boxes_xyxy_conf_cls=[[5,5,15,15,.9,0]]))+'\n')
    summary = tmp_path/'cache.json'
    summary.write_text(json.dumps(dict(status='complete',video_sha256=sha(video),raw_sha256=sha(cache),frames=20,model='synthetic')))
    args = SimpleNamespace(video=str(video),output=str(tmp_path/'out'),confidence=.4,confirmation_sec=.3,
        occlusion_sec=.1,fire_class=0,smoke_class=1,predictions=str(cache),cache_summary=str(summary),
        cache_confidence_floor=.001,threads=1)
    out = run(args)
    assert out['status'] == 'complete' and out['frames'] == 20
    with pytest.raises(FileExistsError):
        run(args)
    args.output = str(tmp_path/'second')
    summary.write_text(summary.read_text().replace(sha(video),'0'*64))
    with pytest.raises(ValueError,match='hash'):
        run(args)
