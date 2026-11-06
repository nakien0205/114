"""Plain YOLO26n-P2 and standard P2 SRC; no training on import."""
from pathlib import Path
import torch
from ultralytics.nn.tasks import DetectionModel
from src_models import restore_nano_start, install_src, validate_src, shared_digest, state_digest

ARMS = {'plain':'YOLO26n-P2 baseline','src':'YOLO26n-P2 baseline + standard SRC'}


def validate(model,arm):
    assert arm in ARMS and model.baseline_arm == arm
    assert model.yaml['scale']=='n' and model.stride.tolist()==[4,8,16,32]
    assert model.end2end and model.model[-1].nc==2 and model.names=={0:'fire',1:'smoke'}
    if arm=='src':
        validate_src(model,'n')
    else:
        assert len(model.model)==30 and model.model[-1].f==[19,22,25,28]
        assert not hasattr(model,'src_scale')
    return model


def extra_parameters(model,arm):
    return list(model.model[20].parameters()) if arm=='src' else []


def build(assets,seed,arm):
    assert seed in (42,2026) and arm in ARMS
    assets = Path(assets)
    model = DetectionModel(str(assets/'yolo26n-p2.yaml'),nc=2,verbose=False)
    restored = restore_nano_start(model,assets/'paired-initial-state.pt')
    before = shared_digest(model)
    src = None
    if arm=='src':
        # Added-module initialization varies by seed without consuming the
        # baseline pipeline RNG. All common starting tensors remain identical.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            src = install_src(model,'n')
        assert shared_digest(model)==before
        assert float(model.model[20].gamma.detach())==0
    model.baseline_arm = arm
    model.names = {0:'fire',1:'smoke'}
    validate(model,arm)
    return model,dict(arm=arm,model=ARMS[arm],seed=seed,restored=restored,SRC=src,
        initial_digest=state_digest(model.state_dict()),shared_start_digest=before,
        extra_start_digest=state_digest(model.model[20].state_dict()) if arm=='src' else None,
        parameters=sum(p.numel() for p in model.parameters()),
        seed_scope='Fixed saved pre-training P2 start for all arms/seeds; seed-specific SRC parameters plus data order/augmentation; fresh optimizer',
        trained_fire_weights_used=False)
