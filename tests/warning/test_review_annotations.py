import csv
import hashlib
import json
from pathlib import Path

import pytest

from fire_warning.review_annotations import build_review, clock_time, parse_labels,viewer_page


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def test_index_maps_boxes_to_source_time_without_changing_worksheet(tmp_path):
    root = tmp_path/'auto_labeled'; folder=root/'clip'; folder.mkdir(parents=True)
    (folder/'images').mkdir(); (folder/'labels').mkdir()
    (folder/'labels'/'frame_00000005.txt').write_text('0 0.5 0.6 0.1 0.2\n1 0.2 0.3 0.2 0.2\n')
    (folder/'labels'/'frame_00000010.txt').write_text('')
    csv_write(folder/'frame_map.csv', [dict(image='frame_00000005.jpg',frame_index=5,timestamp_sec=1.25,timestamp_quality='pts'),
                                      dict(image='frame_00000010.jpg',frame_index=10,timestamp_sec=2.5,timestamp_quality='pts')])
    worksheet=tmp_path/'video_stages.csv'
    csv_write(worksheet,[dict(label_id='clip',video_path=str(tmp_path/'unsafe<video>.mp4'),subset='test',
                             width=352,height=288,t_ignition_start_sec='0.9',auto_t_ignition_start_sec='1.5')])
    before=hashlib.sha256(worksheet.read_bytes()).hexdigest()
    report=build_review(worksheet,root,tmp_path)
    assert report['status']=='completed' and report['sampled_frames']==2 and report['prediction_frames']==1
    assert hashlib.sha256(worksheet.read_bytes()).hexdigest()==before
    with (tmp_path/'detection_index.csv').open(encoding='utf-8-sig') as stream: rows=list(csv.DictReader(stream))
    assert rows[0]['frame_index']=='5' and rows[0]['timestamp_sec']=='1.25'
    assert rows[0]['fire_boxes']=='1' and rows[0]['smoke_boxes']=='1'
    page=(root/'review'/'clip.html').read_text(encoding='utf-8')
    data=json.loads(page.split('<script id="data" type="application/json">')[1].split('</script>')[0])
    assert len(data['frames'])==2 and data['frames'][1][3]==[]
    assert 'unsafe&lt;video&gt;' in page
    assert 'Individual box confidence is unavailable' in page
    assert '../../review_detections.html' in page
    assert data['images']=='../clip/images/'
    cached=build_review(worksheet,root,tmp_path,reuse_pages=True)
    assert cached['reused_existing_page_boxes'] and cached['prediction_frames']==1
    assert hashlib.sha256(worksheet.read_bytes()).hexdigest()==before


def test_missing_label_is_reported_without_claiming_no_predictions(tmp_path):
    root=tmp_path/'auto_labeled'; folder=root/'clip';folder.mkdir(parents=True)
    csv_write(folder/'frame_map.csv',[dict(image='missing.jpg',frame_index=0,timestamp_sec=0)])
    worksheet=tmp_path/'video_stages.csv'
    csv_write(worksheet,[dict(label_id='clip',video_path='missing.mp4',width=640,height=640)])
    report=build_review(worksheet,root,tmp_path)
    assert report['status']=='completed_with_errors' and report['videos']==0 and len(report['errors'])==1


@pytest.mark.parametrize('row',['2 .5 .5 .1 .1','0 nan .5 .1 .1','0 .5 .5 0 .1','0 .5 .5 .1 .1 .9'])
def test_invalid_yolo_labels_are_rejected(tmp_path,row):
    path=tmp_path/'label.txt';path.write_text(row)
    with pytest.raises(ValueError):parse_labels(path)


def test_timestamp_format():
    assert clock_time(3661.234)=='01:01:01.234'


def test_older_worksheet_uses_saved_image_dimensions(tmp_path):
    import cv2
    import numpy as np
    folder=tmp_path/'auto_labeled'/'clip';(folder/'images').mkdir(parents=True)
    assert cv2.imwrite(str(folder/'images'/'frame.jpg'),np.zeros((20,30,3),dtype=np.uint8))
    page=viewer_page(dict(video_path=str(tmp_path/'video.mp4')),[[0,0,'frame.jpg',[]]],folder,tmp_path/'review_detections.html')
    data=json.loads(page.split('<script id="data" type="application/json">')[1].split('</script>')[0])
    assert data['width']==30 and data['height']==20
