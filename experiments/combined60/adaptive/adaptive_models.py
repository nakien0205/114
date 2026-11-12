"""Legacy P3-to-P2 coarse-context gate; no dense neck or training on import."""
import copy
import hashlib
from pathlib import Path
import torch
from ultralytics.nn.tasks import DetectionModel
from coarse_gate import CoarseContextConcat

ARMS={'adaptive':'YOLO26n-P2 + legacy adaptive coarse-context gate'}


def state_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        for part in (name,str(value.dtype),str(tuple(value.shape))): h.update(part.encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def shared_state(model):
    return {k:v for k,v in model.state_dict().items() if not k.startswith('model.18.')}


def validate(model,arm='adaptive'):
    assert arm=='adaptive' and model.baseline_arm==arm
    assert len(model.model)==30 and model.stride.tolist()==[4,8,16,32]
    assert model.model[-1].f==[19,22,25,28] and model.end2end and model.model[-1].nc==2
    gate=model.model[18]
    assert isinstance(gate,CoarseContextConcat) and gate.mode=='p3p4'
    assert gate.i==18 and gate.f==[-1,16,13,2] and gate.np==6209
    assert (gate.c3,gate.c4,gate.fuse.out_channels)==(64,128,32)
    assert {2,13,16}.issubset(model.save)
    assert model.yaml['head'][7]==[[-1,16,13,2],1,'CoarseContextConcat',[32,'p3p4']]
    return model


def extra_parameters(model,arm='adaptive'):
    assert arm=='adaptive'
    return list(model.model[18].parameters())


def build(assets,seed,arm='adaptive'):
    assert arm=='adaptive' and seed in (42,2026)
    assets=Path(assets)
    path=assets/'paired-initial-state.pt'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='bdeaf442ae3096d401bc1ae700487b6157bef2fbadde32f1da299fe957d71f7b'
    paired=torch.load(path,map_location='cpu',weights_only=True)
    model=DetectionModel(str(assets/'yolo26n-p2.yaml'),nc=2,verbose=False)
    model.load_state_dict(paired['shared'],strict=True)
    before=state_digest(model.state_dict())
    rng=torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        gate=CoarseContextConcat(c3=64,c4=128,hidden=32,mode='p3p4')
    assert torch.equal(rng,torch.get_rng_state())
    gate.i,gate.f,gate.type=18,[-1,16,13,2],'CoarseContextConcat'
    gate.np=sum(p.numel() for p in gate.parameters())
    gate.train(model.training)
    model.model[18]=gate
    model.save=sorted(set(model.save)|{2,13,16})
    model.yaml=copy.deepcopy(model.yaml)
    model.yaml['head'][7]=[[-1,16,13,2],1,'CoarseContextConcat',[32,'p3p4']]
    model.baseline_arm=arm
    model.names={0:'fire',1:'smoke'}
    assert state_digest(shared_state(model))==before
    assert all(torch.equal(shared_state(model)[k],v) for k,v in paired['shared'].items())
    validate(model,arm)
    return model,dict(arm=arm,model=ARMS[arm],seed=seed,initial_digest=state_digest(model.state_dict()),
        shared_start_digest=before,extra_start_digest=state_digest(gate.state_dict()),
        parameters=sum(p.numel() for p in model.parameters()),extra_parameters=gate.np,
        shared_start_tensors_equal=True,added_module_RNG_preserved=True,initial_gain=1.,trained_fire_weights_used=False,
        seed_scope='Fixed saved pre-training P2 start; seed-specific gate initialization and data order/augmentation; fresh optimizer')
