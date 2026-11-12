"""Insert SRC after P2 fusion while preserving stock detector initialization."""
import copy
import hashlib
import torch
from torch import nn
from ultralytics.utils.torch_utils import initialize_weights
from src_attention import SpatialRelationalCoAttention

PAIRED_SHA = 'bdeaf442ae3096d401bc1ae700487b6157bef2fbadde32f1da299fe957d71f7b'


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode()); digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def canonical_shared_state(model):
    if not hasattr(model, 'src_scale'):
        return model.state_dict()
    result = {}
    for name, tensor in model.state_dict().items():
        parts = name.split('.')
        index = int(parts[1])
        if index == 20:
            continue
        parts[1] = str(index - 1 if index > 20 else index)
        result['.'.join(parts)] = tensor
    return result


def shared_digest(model):
    return state_digest(canonical_shared_state(model))


def restore_nano_start(model, paired_path):
    assert hashlib.sha256(paired_path.read_bytes()).hexdigest() == PAIRED_SHA
    paired = torch.load(paired_path, map_location='cpu', weights_only=True)
    assert paired['shared_digest'] == 'a5a5e09d95794939e1d31a52b1a248c3f54de115a3d4237259bbe2b710d5d386'
    model.load_state_dict(paired['shared'], strict=True)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in paired['shared'].items())
    return {'method': 'Exact saved legacy nano starting tensors, already COCO-initialized; before Home Fire training',
            'paired_state_sha256': PAIRED_SHA, 'all_shared_start_tensors_verified_equal': True,
            'historical_shared_digest': paired['shared_digest'], 'trained_Home_Fire_weights_used': False}


def install_src(model, scale):
    assert scale in {'n', 'm'} and model.yaml['scale'] == scale
    assert len(model.model) == 30 and model.model[-1].f == [19, 22, 25, 28]
    before = shared_digest(model)
    rng = torch.get_rng_state().clone()
    channels = model.model[19].cv2.conv.out_channels
    assert channels == {'n': 32, 'm': 128}[scale]
    with torch.random.fork_rng(devices=[]):
        src = SpatialRelationalCoAttention(c1=channels, reduction=8, sr_ratio=8)
        # The original YAML build applies these standard BN/activation settings
        # to every module. Apply them to SRC alone after post-build insertion.
        initialize_weights(src)
    assert torch.equal(rng, torch.get_rng_state())
    src.i, src.f, src.type = 20, -1, 'SpatialRelationalCoAttention'
    src.np = sum(p.numel() for p in src.parameters())
    src.train(model.training)
    def reroute(value):
        if isinstance(value, list):
            return [reroute(v) for v in value]
        return value + 1 if value >= 19 else value
    layers = list(model.model)
    for old_index, layer in enumerate(layers):
        if old_index >= 20:
            layer.i = old_index + 1
            layer.f = reroute(layer.f)
    model.model = nn.Sequential(*layers[:20], src, *layers[20:])
    model.save = sorted({reroute(i) for i in model.save})
    cfg = copy.deepcopy(model.yaml)
    graph = copy.deepcopy(cfg['head'])
    for j in range(9, len(graph)):
        graph[j][0] = reroute(graph[j][0])
    graph.insert(9, [-1, 1, 'SpatialRelationalCoAttention', [channels, 8, 8]])
    cfg['head'] = graph
    model.yaml, model.src_scale = cfg, scale
    validate_src(model, scale)
    assert shared_digest(model) == before
    return {'scale': scale, 'SRC_index': 20, 'channels': channels, 'added_parameters': src.np,
            'total_training_parameters': sum(p.numel() for p in model.parameters()),
            'shared_start_digest': before, 'shared_tensors_preserved': True, 'CPU_RNG_preserved': True,
            'rng_after_insertion_sha256': hashlib.sha256(rng.numpy().tobytes()).hexdigest(),
            'identity_initial_gamma': float(src.gamma.detach())}


def validate_src(model, scale):
    assert model.src_scale == scale and model.yaml['scale'] == scale
    assert len(model.model) == 31 and model.model[-1].f == [20, 23, 26, 29]
    assert model.model[-1].i == 30 and model.stride.tolist() == [4., 8., 16., 32.]
    module = model.model[20]
    assert isinstance(module, SpatialRelationalCoAttention) and module.c1 == {'n': 32, 'm': 128}[scale]
    assert module.sr_ratio == 8 and module.inter_c == 16
    assert all(m.eps == 1e-3 and m.momentum == .03 for m in module.modules() if isinstance(m, nn.BatchNorm2d))
    assert {20, 23, 26, 29}.issubset(model.save)
    return model
