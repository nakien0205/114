"""Dense spatial P2-P5 neck. No frame state; importing never trains."""
import copy
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.nn.modules import Conv, C3k2
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import initialize_weights


def digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


class SpatialFusion(nn.Module):
    def __init__(self, inputs, channels):
        super().__init__()
        self.align = nn.ModuleList(Conv(c, channels, 1) for c in inputs)
        self.reduce = nn.ModuleList(nn.Conv2d(channels, 8, 1) for _ in inputs)
        self.logits = nn.Conv2d(8 * len(inputs), len(inputs), 1)
        # Equal initial weights; learn content dependence during training.
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)
        self.refine = C3k2(channels, channels, n=1, c3k=True)

    def branches(self, values, shape):
        aligned = []
        for value, projection in zip(values, self.align):
            if value.shape[-2] > shape[0] or value.shape[-1] > shape[1]:
                value = F.adaptive_avg_pool2d(value, shape)
            value = projection(value)
            if value.shape[-2:] != shape:
                value = F.interpolate(value, size=shape, mode='nearest')
            aligned.append(value)
        return aligned

    def weights(self, aligned):
        compressed = [F.silu(project(x)) for project, x in zip(self.reduce, aligned)]
        return self.logits(torch.cat(compressed, 1)).float().softmax(1).to(aligned[0].dtype)

    def forward(self, values, shape):
        aligned = self.branches(values, shape)
        weights = self.weights(aligned)
        mixture = sum(value * weights[:, k:k+1] for k, value in enumerate(aligned))
        return self.refine(mixture)


class DenseSpatialNeck(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone X2/X3/X4/X5 = 64/128/128/256 channels.
        self.t4 = SpatialFusion([128, 256], 128)
        self.t3 = SpatialFusion([128, 128, 256], 64)
        self.t2 = SpatialFusion([64, 64, 128, 256], 32)
        self.u3 = SpatialFusion([64, 32], 64)
        self.u4 = SpatialFusion([128, 32, 64], 128)
        self.u5 = SpatialFusion([256, 32, 64, 128], 256)

    def forward(self, values):
        x2,x3,x4,x5 = values
        t4 = self.t4([x4,x5], x4.shape[-2:])
        t3 = self.t3([x3,t4,x5], x3.shape[-2:])
        t2 = self.t2([x2,t3,t4,x5], x2.shape[-2:])
        u3 = self.u3([t3,t2], t3.shape[-2:])
        u4 = self.u4([t4,t2,u3], t4.shape[-2:])
        u5 = self.u5([x5,t2,u3,u4], x5.shape[-2:])
        return [t2,u3,u4,u5]


class DenseSpatialModel(DetectionModel):
    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        if not getattr(self, 'dense_ready', False):
            return super()._predict_once(x, profile, visualize, embed)
        if profile or visualize or embed is not None:
            raise NotImplementedError('Use ordinary forward; custom per-layer profiling/embeddings are not defined.')
        features = []
        for index in range(11):
            x = self.model[index](x)
            if index in (2,4,6,10):
                features.append(x)
        return self.model[-1](self.model[11](features))


def build(assets, seed):
    assert seed in (42,2026)
    assets = Path(assets)
    # Both seeds share the same saved pre-training backbone/head from the COCO-initialized P2 model
    # tensors. The newly defined neck and training randomness vary by seed.
    start = torch.load(assets / 'paired-initial-state.pt', map_location='cpu', weights_only=True)
    cfg = str(assets / 'yolo26n-p2.yaml')
    model = DenseSpatialModel(cfg, nc=2, verbose=False)
    model.load_state_dict(start['shared'], strict=True)
    original = copy.deepcopy(model.state_dict())
    head = model.model[-1]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        neck = DenseSpatialNeck()
        initialize_weights(neck)
    model.model = nn.Sequential(*list(model.model[:11]), neck, head)
    head.i, head.f = 12, [11]
    model.save = []
    model.dense_ready = True
    model.yaml['dense_spatial'] = dict(version=1, seed=seed, memory=False,
        alignment='average pool down / nearest up + 1x1 Conv BN SiLU', fusion='per-pixel branch softmax, weighted sum, C3k2')
    model.names = {0:'fire',1:'smoke'}
    # Ensure all shared backbone and detect tensors exactly match saved start.
    state = model.state_dict()
    shared = [k for k in state if int(k.split('.')[1]) <= 10]
    assert all(torch.equal(state[k], original[k]) for k in shared)
    for k,v in head.state_dict().items():
        assert torch.equal(v, original['model.29.' + k]), k
    assert model.stride.tolist() == [4,8,16,32] and model.end2end
    return model, dict(seed=seed, initialization='Saved pre-training backbone/head from the COCO-initialized P2 model; fresh seed-specific dense neck; fresh optimizer',
        initial_digest=digest(state), backbone_digest=digest({k:state[k] for k in shared}),
        neck_digest=digest(neck.state_dict()), shared_backbone_head_verified=True,
        parameters=sum(p.numel() for p in model.parameters()), no_trained_fire_checkpoint_used=True,
        seed_scope='Dense-neck initialization, data order and augmentation; common saved backbone/head start')
