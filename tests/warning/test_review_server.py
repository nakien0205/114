import csv
import hashlib
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request,urlopen
import zipfile

import pytest

from fire_warning.review_server import ReviewStore,RevisionConflict,make_server
from fire_warning.review_annotations import refresh_pages,build_review,viewer_page


@pytest.fixture
def store(tmp_path):
    root=tmp_path/'auto_labeled';folder=root/'clip';(folder/'images').mkdir(parents=True)
    (folder/'labels').mkdir()
    for frame in (0,5,10):
        (folder/'images'/f'frame_{frame:08}.jpg').write_bytes(b'original JPEG fixture')
        (folder/'labels'/f'frame_{frame:08}.txt').write_text('0 .5 .5 .1 .1\n')
    (folder/'frame_map.csv').write_text('image,frame_index,timestamp_sec,timestamp_quality\n'
        'frame_00000000.jpg,0,0,presentation_timestamp\nframe_00000005.jpg,5,1.25,presentation_timestamp\n'
        'frame_00000010.jpg,10,2.5,presentation_timestamp\n')
    worksheet=tmp_path/'video_stages.csv'
    row=dict(label_id='clip',video_path=str(tmp_path/'original.mp4'),source_sha256='abc',subset='test',width=640,height=480,
             t_ignition_start_sec='1.0',t_aftermath_start_sec='',should_alert_human='1')
    with worksheet.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerow(row)
    return ReviewStore(worksheet,root)


def annotation(frame=5,boxes=None):
    return dict(frame_index=frame,timestamp_sec=999,reviewed=True,
                reviewed_utc='2026-10-15T08:00:00Z',boxes=boxes if boxes is not None else [[0,.5,.5,.2,.3]])


def test_disk_save_flags_and_manual_boxes_preserves_sources(store):
    worksheet=store.root.parent/'video_stages.csv';before=worksheet.read_bytes()
    label=store.root/'clip'/'labels'/'frame_00000005.txt';raw=label.read_bytes()
    state=store.load('clip')
    state['annotations']['5']=annotation()
    state['flags']=[dict(id='5:frame:false_positive',frame_index=5,issue='false_positive',box_index=None,
                         notes='Bright lamp, not fire',updated_utc='2026-10-15T08:00:00Z')]
    saved=store.save('clip',state)
    assert saved['revision']==1 and saved['annotations']['5']['timestamp_sec']==1.25
    assert store.load('clip')==saved
    assert worksheet.read_bytes()==before and label.read_bytes()==raw


def test_stale_tab_cannot_overwrite_review(store):
    original=store.load('clip');store.save('clip',original)
    with pytest.raises(RevisionConflict):store.save('clip',original)
    assert store.load('clip')['revision']==1


@pytest.mark.parametrize('boxes',[[[0,.5,.5,0,.1]],[[0,.05,.5,.2,.1]],[[1,.5,.5,float('nan'),.1]],[[2,.5,.5,.1,.1]],[[True,.5,.5,.1,.1]]])
def test_invalid_boxes_cannot_enter_training(store,boxes):
    state=store.load('clip');state['annotations']['5']=annotation(boxes=boxes)
    with pytest.raises(ValueError):store.save('clip',state)
    assert store.load('clip')['revision']==0


def test_unknown_frames_and_unreviewed_drafts_are_rejected(store):
    state=store.load('clip');state['annotations']['99']=annotation(frame=99)
    with pytest.raises(ValueError):store.save('clip',state)
    state['annotations']={'5':{**annotation(),'reviewed':False}}
    with pytest.raises(ValueError):store.save('clip',state)


def test_export_contains_only_reviewed_pairs_and_explicit_empty_background(store):
    with pytest.raises(ValueError,match='No reviewed'):store.export_training()
    state=store.load('clip');state['annotations']={'5':annotation(),'10':annotation(frame=10,boxes=[])}
    store.save('clip',state)
    archive,count=store.export_training()
    with archive,zipfile.ZipFile(archive) as z:
        assert count==2
        assert len([n for n in z.namelist() if n.startswith('images/')])==2
        assert 'images/clip_frame_00000000.jpg' not in z.namelist()
        assert z.read('labels/clip_frame_00000010.txt')==b''
        assert z.read('labels/clip_frame_00000005.txt').startswith(b'0 0.50000000')
        manifest=json.loads(z.read('review_manifest.json'))
        assert manifest[0]['provenance']=='human_reviewed_manual_annotation'
        assert 'val: images' not in z.read('data.yaml').decode()


def test_changed_source_does_not_reuse_saved_review(store):
    state=store.load('clip');state['source_sha256']='other-source'
    with pytest.raises(ValueError):store.save('clip',state)


def test_http_save_reload_export_and_cross_origin_rejection(store):
    server=make_server(store.root.parent/'video_stages.csv',store.root,0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        state=json.load(urlopen(base+'/api/review?clip=clip'))
        state['annotations']['5']=annotation()
        request=Request(base+'/api/review',data=json.dumps(state).encode(),headers={'Content-Type':'application/json'},method='POST')
        assert json.load(urlopen(request))['revision']==1
        assert json.load(urlopen(base+'/api/review-status'))['reviewed_frames']==1
        with urlopen(base+'/api/export-training?clip=clip') as result:
            assert result.headers['Content-Type']=='application/zip' and result.read(2)==b'PK'
        request=Request(base+'/api/review',data=json.dumps(state).encode(),headers={'Content-Type':'application/json','Origin':'https://outside.example'},method='POST')
        with pytest.raises(HTTPError) as error:urlopen(request)
        assert error.value.code==403
        with pytest.raises(HTTPError) as error:urlopen(base+'/unrelated-private-file.txt')
        assert error.value.code==403
    finally:server.shutdown();server.server_close();thread.join()


def test_refresh_ui_preserves_frame_data_csvs_and_manual_reviews(store):
    worksheet=store.root.parent/'video_stages.csv'
    build_review(worksheet,store.root,store.root.parent)
    store.save('clip',store.load('clip'))
    paths=[worksheet,store.root.parent/'detection_index.csv',store.root.parent/'detection_summary.csv',store.directory/'clip.json']
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report=refresh_pages(worksheet,store.root,store.root.parent)
    assert report['videos']==1 and report['sampled_frames']==3
    assert before=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    text=(store.root/'review'/'clip.html').read_text()
    assert 'Save flag' in text and 'Pre-label' in text and 'Draw rectangle' in text


def test_explicit_dataset_boxes_are_distinct_from_model_and_manual_labels(store):
    folder=store.root.parent/'provided';folder.mkdir()
    (folder/'frame_00000005.txt').write_text('1 .4 .5 .2 .3\n')
    row={**store.rows['clip'],'dataset_labels_dir':str(folder)}
    page=viewer_page(row,[[5,1.25,'frame_00000005.jpg',[[0,.5,.5,.1,.1]]]],store.root/'clip',store.root.parent/'review_detections.html')
    data=json.loads(page.split('<script id="data" type="application/json">')[1].split('</script>')[0])
    assert data['dataset_labels']['5'][0][0]==1 and data['frames'][0][3][0][0]==0
    assert 'Dataset YOLO labels explicitly supplied' in data['dataset_label_status']
