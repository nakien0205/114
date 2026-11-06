"""Two independent seeds, exactly one process per T4; no DDP."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from data_runtime import write


def job_spec(root, assets, vendor, seed, gpu):
    root,assets,vendor = Path(root),Path(assets),Path(vendor)
    folder = root / 'runs' / f'seed{seed}'
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'settings').mkdir()
    arm = json.loads((assets/'experiment-spec.json').read_text())['arm']
    assert arm == 'adaptive'
    cfg = dict(seed=seed,arm=arm,physical_gpu=gpu,run_root=str(folder),suite_root=str(root),assets_root=str(assets))
    write(folder/'configuration.json',cfg)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONHASHSEED=str(seed), PYTHONUNBUFFERED='1',
        YOLO_CONFIG_DIR=str(folder/'settings'), YOLO_AUTOINSTALL='false',
        CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        WANDB_MODE='disabled', COMET_MODE='DISABLED', PYTHONPATH=os.pathsep.join([str(assets),str(vendor)]))
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE'):
        env.pop(key,None)
    return dict(seed=seed,gpu=gpu,folder=folder,env=env,
        command=[sys.executable,'-u',str(assets/'runner.py'),str(folder/'configuration.json')])


def package(root, assets):
    root,assets = Path(root),Path(assets)
    target = root.parent / (root.name+'_results.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for current,dirs,files in os.walk(root,followlinks=False):
            dirs[:] = [d for d in dirs if d not in ('assets','vendor','data','settings','__pycache__')]
            for name in files:
                if not name.endswith('.partial'):
                    path = Path(current)/name
                    archive.write(path,path.relative_to(root).as_posix())
        for path in assets.iterdir():
            if path.suffix in ('.py','.json','.yaml','.md'):
                archive.write(path,'reproduction/'+path.name)
        for path in (root/'data').glob('*/labels/*.txt'):
            archive.write(path,'combined-labels/'+path.relative_to(root/'data').as_posix())
    print('Results archive:',target,flush=True)
    return target


def summarize(root, results):
    rows = [dict(seed=r['seed'],tiny_fire_AP=r['tiny_fire_AP'],small_fire_AP=r['small_fire_AP'],
        small_smoke_AP=r['small_smoke_AP'],mAP50=r['normal']['metrics/mAP50(B)'],mAP50_95=r['normal']['metrics/mAP50-95(B)']) for r in results]
    with (Path(root)/'review-table.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    means = {k:sum(row[k] for row in rows)/len(rows) if all(row[k] is not None for row in rows) else None for k in rows[0] if k!='seed'}
    write(Path(root)/'all-results.json',dict(complete=len(results)==2,per_seed=results,descriptive_mean=means,
        warning='Two seeds, same dataset and ungrouped source splits; compare matching adaptive/plain seeds; no independent-event claim. AP JSON/CSV units are 0-1.'))


def run_all(root,assets,vendor,launcher=None,poll_seconds=2,check_gpus=True):
    root,assets,vendor = Path(root),Path(assets),Path(vendor)
    assert json.loads((root/'data-preparation.json').read_text())['complete']
    assert not (root/'runs').exists(), 'Run folder exists; preserve it. No automatic resume/retraining.'
    if check_gpus:
        code = ('import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; '
            'assert "T4" in torch.cuda.get_device_name(0); print(torch.cuda.get_device_name(0))')
        for gpu in (0,1):
            subprocess.run([sys.executable,'-c',code],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),check=True)
    active = []
    try:
        for seed,gpu in ((42,0),(2026,1)):
            spec = job_spec(root,assets,vendor,seed,gpu)
            stream = (spec['folder']/'console.log').open('w')
            try:
                process = subprocess.Popen(spec['command'],env=spec['env'],stdout=stream,stderr=subprocess.STDOUT) if launcher is None else launcher(spec,stream)
            except BaseException:
                stream.close()
                raise
            active.append((spec,process,stream))
            print(f'Started seed {seed}, physical T4 {gpu}, 60 epochs, batch8',flush=True)
        last = time.monotonic()
        while any(p.poll() is None for _,p,_ in active):
            if time.monotonic()-last >= 45:
                for spec,p,_ in active:
                    path = spec['folder']/'latest-progress.json'
                    progress = json.loads(path.read_text()) if path.exists() else {}
                    print('Progress:',dict(seed=spec['seed'],epoch=progress.get('epoch'),remaining_hours=progress.get('remaining_hours_estimate'),exit=p.poll()),flush=True)
                last = time.monotonic()
            time.sleep(poll_seconds)
        results,failures = [],[]
        for spec,p,_ in active:
            path = spec['folder']/'summary.json'
            if p.returncode == 0 and path.exists() and json.loads(path.read_text()).get('complete'):
                results.append(json.loads(path.read_text()))
            else:
                failures.append(spec['seed'])
        if results:
            summarize(root,results)
        write(root/'suite-status.json',dict(successful_seeds=[r['seed'] for r in results],failed_seeds=failures))
        if failures:
            raise RuntimeError(f'Seeds failed: {failures}. Other seed was allowed to finish. Download partial results and inspect console.log.')
        return results
    finally:
        for _,p,stream in active:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill();p.wait(timeout=10)
            stream.close()
        package(root,assets)
