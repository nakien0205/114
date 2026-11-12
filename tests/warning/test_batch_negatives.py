import csv
import json
import threading

import pytest

from fire_warning.batch_flag_firesense_negatives import batch_state,partition,run_batch
from fire_warning.review_annotations import build_review
from fire_warning.review_server import make_server


def row(subset='firesense',group='fire',sign='neg'):
    return dict(subset=subset,video_path=f'C:/Downloads/FIRESENSE/{group}/{sign}/test.avi',label_id='clip')


def state(r):
    return dict(schema_version=1,clip_id=r['label_id'],video_path=r['video_path'],revision=0,flags=[],annotations={})


def test_owner_rule_flags_both_classes_without_creating_training_labels():
    r=row();s=state(r);s['annotations']={'5':{'reviewed':True,'boxes':[[0,.5,.5,.1,.1]]}}
    frames=[[5,1.25,'frame.jpg',[[0,.5,.5,.1,.1],[1,.6,.6,.1,.1]]],[10,2.5,'empty.jpg',[]]]
    changed,added=batch_state(r,frames,s,'2026-10-15T00:00:00Z')
    assert len(added)==2 and changed['annotations']==s['annotations'] and not s['flags']
    assert all(f['issue']=='false_positive' and 'not_framewise_visual_review' in f['notes'] for f in added)
    changed['flags'][0]['notes']='Original owner note'
    again,extra=batch_state(r,frames,changed,'later')
    assert not extra and len(again['flags'])==2 and again['flags'][0]['notes']=='Original owner note'


@pytest.mark.parametrize('r',[row(sign='pos'),row(subset='mmodelfire'),row(sign='negative'),row(group='unknown')])
def test_nonnegative_and_other_datasets_are_excluded(r):
    with pytest.raises(ValueError):batch_state(r,[],state(r),'now')


def test_live_batch_is_scoped_idempotent_and_preserves_other_data(tmp_path):
    root=tmp_path/'auto_labeled';rows=[]
    definitions=[('neg','firesense','FIRESENSE/fire/neg/negative.avi'),('pos','firesense','FIRESENSE/fire/pos/positive.avi'),
                 ('mmodel','mmodelfire','MmodelFire/Video/example.mp4')]
    for clip,subset,path in definitions:
        video=tmp_path/path;video.parent.mkdir(parents=True,exist_ok=True);video.write_bytes(b'original video fixture')
        folder=root/clip;(folder/'labels').mkdir(parents=True);(folder/'images').mkdir()
        (folder/'labels'/'frame_00000005.txt').write_text('0 .5 .5 .2 .2\n1 .5 .5 .1 .1\n')
        (folder/'frame_map.csv').write_text('image,frame_index,timestamp_sec,timestamp_quality\nframe_00000005.jpg,5,1.25,presentation_timestamp\n')
        rows.append(dict(label_id=clip,subset=subset,video_path=str(video),width=640,height=480,source_sha256='source-'+clip))
    worksheet=tmp_path/'video_stages.csv'
    with worksheet.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    build_review(worksheet,root,tmp_path)
    server=make_server(worksheet,root,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    before={p:p.read_bytes() for p in [worksheet,root/'review'/'mmodel.html',root/'review'/'pos.html',tmp_path/'detection_index.csv']}
    try:
        report=run_batch(worksheet,root,f'http://127.0.0.1:{server.server_port}')
        assert report['negative_clips']==1 and report['predicted_boxes']==2 and report['new_flags']==2
        assert report['remaining_positive_clips']==1 and report['protected_artifacts_unchanged']
        assert not server.review_store.load('neg')['annotations']
        assert not (root/'manual_reviews'/'pos.json').exists() and not (root/'manual_reviews'/'mmodel.json').exists()
        assert before=={p:p.read_bytes() for p in before}
        again=run_batch(worksheet,root,f'http://127.0.0.1:{server.server_port}')
        assert again['new_flags']==0 and len(server.review_store.load('neg')['flags'])==2
        index=(tmp_path/'review_detections.html').read_text(encoding='utf-8')
        assert 'scope.value="firesense_positive"' in index and 'MmodelFire — unchanged' in index
    finally:server.shutdown();server.server_close();thread.join()
