"""One independent Kaggle seed worker. No training on import."""
import csv
import json
import importlib.metadata
import os
from pathlib import Path
import shutil
import sys
import time

import torch
import yaml
from ultralytics import YOLO, settings
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import init_seeds
from dense_model import build, digest, DenseSpatialModel
from data_runtime import write, file_hash, capture_validator
import operating_points
import size_evaluator as size

CONFIG = None


def root():
    return Path(CONFIG['run_root'])


def assets():
    return Path(CONFIG['assets_root'])


def completed_training_epochs(path):
    """Use the trainer's results, not callbacks also fired by final validation."""
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(float(row['epoch'])) for row in rows]
    assert epochs == list(range(1,61)), f'Expected completed training epochs 1..60, received {epochs}'
    return len(epochs)


def training_args():
    args = json.loads((assets() / 'baseline-train-settings.json').read_text())
    args.update(seed=CONFIG['seed'], epochs=60, imgsz=640, batch=8, workers=2, pretrained=False,
        resume=False, device=0, patience=0, save_period=10, project=str(root()), name='train',
        data=str(root() / 'data.yaml'), model=str(assets() / 'yolo26n-p2.yaml'), exist_ok=False)
    return args


class DenseTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        assert weights is None and self.data['nc'] == 2
        model, audit = build(assets(), CONFIG['seed'])
        write(root() / 'initialization.json', audit)
        return model

    def _build_train_pipeline(self):
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)
        return super()._build_train_pipeline()

    def _setup_train(self):
        super()._setup_train()
        audit = json.loads((root() / 'initialization.json').read_text())
        assert isinstance(self.model,DenseSpatialModel)
        assert digest(self.model.state_dict()) == audit['initial_digest']
        assert not self.optimizer.state and self.amp
        ids = {id(p) for g in self.optimizer.param_groups for p in g['params']}
        assert all(id(p) in ids for p in self.model.model[11].parameters() if p.requires_grad)
        write(root() / 'trainer-start-audit.json', dict(passed=True, initial=audit,
            actual_args=vars(self.args), optimizer=type(self.optimizer).__name__, AMP=bool(self.amp),
            fresh_optimizer=True, all_neck_parameters_in_optimizer=True,
            pipeline_seed=self.args.seed+1+RANK))


def own_data():
    source = yaml.safe_load((Path(CONFIG['suite_root']) / 'data.yaml').read_text())
    data = Path(source['path'])
    for split in ('train','val'):
        folder = root() / 'data' / split
        folder.mkdir(parents=True, exist_ok=False)
        shutil.copytree(data / split / 'labels', folder / 'labels')
        (folder / 'images').symlink_to((data / split / 'images').resolve(), target_is_directory=True)
    (root() / 'data.yaml').write_text(yaml.safe_dump(dict(source,path=str((root()/'data').resolve()))))


def preflight():
    init_seeds(CONFIG['seed'], deterministic=True)
    model,audit = build(assets(),CONFIG['seed'])
    model.eval().cuda()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        output = model(torch.zeros(8,3,640,640,device='cuda'))[0]
    assert torch.isfinite(output).all()
    write(root() / 'preflight.json',dict(passed=True, batch8_640_eval_AMP_forward=True,
        optimizer_steps=0, training_memory_fit_verified=False, initialization=audit))
    del model,output
    torch.cuda.empty_cache()


def train():
    own_data()
    preflight()
    args = training_args()
    record = dict(status='running', training_completed=False, training_args=args, seed=CONFIG['seed'],
        runtime={k:importlib.metadata.version(k) for k in ('torch','torchvision','numpy','pycocotools')},
        python=sys.version, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        ultralytics=__import__('ultralytics').__version__, GPU=torch.cuda.get_device_name(0),
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'), test_used=False,
        checkpoint_selection='Standard overall validation fitness; no tiny-AP checkpoint selection')
    write(root() / 'run-record.json',record)
    wrapper = YOLO(args['model'],task='detect')
    started_epoch = [None]
    logged_epoch = [None]
    def on_start(trainer):
        started_epoch[0] = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
    def on_end(trainer):
        if logged_epoch[0] == trainer.epoch:
            return  # final_eval invokes this callback again for the last epoch
        logged_epoch[0] = trainer.epoch
        elapsed = time.perf_counter()-started_epoch[0]
        progress = dict(seed=CONFIG['seed'], epoch=trainer.epoch+1, epoch_seconds=elapsed,
            remaining_hours_estimate=elapsed*(trainer.epochs-trainer.epoch-1)/3600,
            peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
            metrics={k:float(v) for k,v in trainer.metrics.items()})
        write(root() / 'latest-progress.json',progress)
        with (root() / 'epoch-progress.jsonl').open('a') as stream:
            stream.write(json.dumps(progress)+'\n')
    wrapper.add_callback('on_train_epoch_start',on_start)
    wrapper.add_callback('on_fit_epoch_end',on_end)
    started = time.perf_counter()
    try:
        wrapper.train(trainer=DenseTrainer,**args)
        completed = completed_training_epochs(root()/'train/results.csv')
        assert completed == 60, f'Expected 60 completed epochs, received {completed}'
        record.update(status='complete',training_completed=True, completed_epochs=completed,
            training_seconds=time.perf_counter()-started,
            best_checkpoint_sha256=file_hash(root()/'train/weights/best.pt'))
    except BaseException as exc:
        record.update(status='failed_or_interrupted',error=repr(exc))
        raise
    finally:
        write(root()/'run-record.json',record)


def evaluate():
    assert json.loads((root()/'run-record.json').read_text())['training_completed']
    reference = {r['image']:r for r in json.loads((assets()/'validation-ground-truth.json').read_text())}
    checkpoint = root()/'train/weights/best.pt'
    model = YOLO(str(checkpoint),task='detect')
    assert isinstance(model.model,DenseSpatialModel) and model.model.dense_ready
    metric = model.val(validator=capture_validator(reference),data=str(root()/'data.yaml'),split='val',
        imgsz=640,batch=8,workers=0,device=0,conf=.001,iou=.7,max_det=300,half=False,augment=False,
        rect=True,nms=False,plots=True,verbose=False,project=str(root()/'evaluation'),name='combined',exist_ok=True)
    directory = root()/'evaluation/combined'
    normal = {k:float(v) for k,v in metric.results_dict.items()}
    write(directory/'normal-metrics.json',normal)
    rows = json.loads((directory/'native-predictions.json').read_text())
    assert len(rows) == len(reference) == len({r['image'] for r in rows})
    assert {r['image'] for r in rows} == set(reference)
    all_scores = {}
    all_points = {}
    for label,subset in [('combined',rows),('home',[r for r in rows if r['source']=='home']),('dfire',[r for r in rows if r['source']=='dfire'])]:
        write(directory/f'{label}-geometry-audit.json',size.prediction_audit(subset))
        gt,dt = size.convert(subset,True)
        ev = size.run_coco(gt,dt,size.CUSTOM_RANGES,size.CUSTOM_NAMES)
        scores = size.extract(ev,gt,'input640')
        all_scores[label] = scores
        all_points[label] = operating_points.evaluate(subset)
    write(directory/'size-metrics.json',all_scores)
    write(directory/'operating-points.json',all_points)
    def get(group,category='fire'):
        return next(r['AP50_95'] for r in all_scores['combined'] if r['category']==category and
            r['area_group']==group and r['max_dets_per_image_category']==100)
    summary = dict(complete=True,seed=CONFIG['seed'],model='Dense spatial adaptive P2-P5 nano',
        tiny_fire_AP=get('tiny_lt16'),small_fire_AP=get('small_lt32_including_tiny'),
        small_smoke_AP=get('small_lt32_including_tiny','smoke'),normal=normal,
        operating_points=all_points,checkpoint_sha256=file_hash(checkpoint),
        limits='No baseline trained; original source splits may share scenes; two seeds describe optimization variability, not independent-event generalization')
    write(root()/'summary.json',summary)
    return summary


def worker(path,evaluate_only=False):
    global CONFIG
    CONFIG = json.loads(Path(path).read_text())
    assert CONFIG['seed'] in (42,2026)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert 'T4' in torch.cuda.get_device_name(0), 'Select Kaggle T4 x2'
    torch.cuda.set_device(0)
    expected = json.loads((assets()/'expected-runtime.json').read_text())
    actual = {k:importlib.metadata.version(k) for k in expected}
    write(root()/'runtime-check.json',dict(expected=expected,actual=actual,passed=actual==expected))
    assert actual == expected, 'Runtime differs from the established 60-epoch recipe; inspect before training.'
    assert __import__('ultralytics').__version__=='8.4.24'
    settings.update(dict(sync=False,wandb=False,comet=False,mlflow=False,clearml=False,neptune=False,dvc=False,hub=False,tensorboard=False))
    try:
        if not evaluate_only:
            train()
        result = evaluate()
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        write(root()/'worker-status.json',dict(status='failed',error=repr(exc)))
        raise
    write(root()/'worker-status.json',dict(status='complete',seed=CONFIG['seed']))


if __name__ == '__main__':
    worker(sys.argv[1], '--evaluate-only' in sys.argv[2:])
