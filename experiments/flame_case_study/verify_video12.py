"""Independent output invariants; no detector or MOG2 rerun."""
import argparse
import gzip
import json
from pathlib import Path

from run import sha, write


def verify(directory):
    directory=Path(directory).resolve()
    protocol=json.loads((directory/'protocol.json').read_text())
    summary=json.loads((directory/'summary.json').read_text())
    root=Path(__file__).resolve().parents[2]
    cache=root/'research/demo-recovery-20261105/predictions/plain_combined_60_s2026/12.jsonl.gz'
    if summary['status']!='complete' or sha(cache)!=protocol['model']['prediction_sha256']:
        raise ValueError('Unexpected run/cache identity')
    if sha(directory/'frames.jsonl')!=summary['output_sha256']:
        raise ValueError('Output hash mismatch')
    for name,digest in protocol['code_sha256'].items():
        if sha(Path(__file__).with_name(name))!=digest:
            raise ValueError(f'Processing source changed: {name}')
    cfg=protocol['config'];seen_confirmed=set();previous_time=None
    total=0;confirmation_count=0
    with gzip.open(cache,'rt',encoding='utf-8') as original,(directory/'frames.jsonl').open(encoding='utf-8') as output:
        for index,line in enumerate(output):
            saved=json.loads(original.readline());out=json.loads(line)
            assert out['frame_index']==saved['frame_index']==index
            assert abs(out['timestamp_sec']-saved['timestamp_s'])<1e-4
            if previous_time is not None:
                assert out['timestamp_sec']>previous_time
            previous_time=out['timestamp_sec']
            expected=[]
            for b in saved['boxes_xyxy_conf_cls']:
                if b[4]>=cfg['confidence'] and int(b[5]) in (cfg['fire_class'],cfg['smoke_class']):
                    expected.append([max(0.,b[0]),max(0.,b[1]),min(float(saved['width']),b[2]),min(float(saved['height']),b[3]),float(b[4]),float(b[5])])
            assert out['raw_boxes']==expected
            assert len(out['foreground_fractions'])==len(expected)
            assert out['moving_boxes']==[b for b,r in zip(expected,out['foreground_fractions']) if r>cfg['foreground_fraction']]
            raw=[tuple(b) for b in out['raw_boxes']];moving=[tuple(b) for b in out['moving_boxes']]
            assert all(b in raw for b in moving)
            assert len(moving)<=len(raw)
            observed=[tuple(t['box']) for t in out['tracks'] if t['observed'] and t['state']=='Confirmed']
            assert observed==[tuple(b) for b in out['observed_confirmed_boxes']]
            assert all(b in moving for b in observed)
            current_confirmed={t['id'] for t in out['tracks'] if t['state']=='Confirmed'}
            assert seen_confirmed<=current_confirmed
            new={e['track_id'] for e in out['new_confirmations']}
            assert new==current_confirmed-seen_confirmed
            assert all(e['timestamp_sec']==out['timestamp_sec'] for e in out['new_confirmations'])
            assert out['alarm']==bool(current_confirmed)
            seen_confirmed=current_confirmed
            confirmation_count+=len(new);total+=1
        assert not original.readline()
    assert total==summary['frames']==54410
    assert confirmation_count==summary['confirmation_events']
    result=dict(status='passed',frames_verified=total,confirmation_events=confirmation_count,
        raw_predictions_exactly_match_frozen_cache=True,no_fabricated_boxes=True,
        confirmed_state_absorbing=True,source_code_hashes_unchanged=True,
        output_sha256=summary['output_sha256'])
    write(directory/'independent-verification.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory')
    print(json.dumps(verify(p.parse_args().directory),indent=2))
