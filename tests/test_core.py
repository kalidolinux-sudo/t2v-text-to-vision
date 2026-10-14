"""Fast CPU checks of the core pieces (no downloads, no GPU): python -m pytest -q"""
import torch

from t2v import target
from t2v.generator import T2VGenerator
from t2v.neighbors import class_row_table, clip_mix, mix_rows
from t2v.tokenizer import ParamTokenizer

ARCH = "tina_cnn"


def zoo(n=6, seed=0):
    torch.manual_seed(seed)
    nets = [target.build(ARCH) for _ in range(n)]
    return target.param_spec(nets[0]), torch.stack([target.flatten(m) for m in nets])


def test_flatten_roundtrip():
    spec, flat = zoo(1)
    net = target.load_flat(target.build(ARCH), flat[0].clone())
    assert torch.allclose(target.flatten(net), flat[0])
    assert flat.shape[1] == sum(torch.Size(s).numel() for _, s in spec)


def test_head_rows_add_and_remove():
    _, flat = zoo(2)
    rows = torch.randn(2, 3, target.head_rows(ARCH, flat).shape[-1])
    back = target.add_head_rows(ARCH, target.add_head_rows(ARCH, flat, rows), rows, sign=-1.0)
    assert torch.allclose(back, flat, atol=1e-6)


def test_tokenizer_roundtrip_all_norms():
    spec, flat = zoo()
    for norm in ("tensor", "element", "zscore"):
        tok = ParamTokenizer(spec, 256, ParamTokenizer.compute_stats(flat, spec, norm), target.MAX_CLASSES)
        assert torch.allclose(tok.decode(tok.encode(flat)), flat, atol=1e-4), norm
        assert len(tok.head_pos) == target.MAX_CLASSES  # one token per classifier row


def test_generator_loss_and_sample_shapes():
    spec, flat = zoo()
    tok = ParamTokenizer(spec, 256, ParamTokenizer.compute_stats(flat, spec, "element"), target.MAX_CLASSES)
    gen = T2VGenerator(tok.layer_ids, tok.intra_ids, tok.slots, 256, clip_dim=16, d_model=32, heads=4,
                       denoiser="mlp", mlp_hidden=64, joint_layers=1, clip=3.0, valid=tok.valid)
    K = target.MAX_CLASSES
    emb = torch.nn.functional.normalize(torch.randn(len(flat), K, 16), dim=-1)
    mask = torch.ones(len(flat), K, dtype=torch.bool)
    row_valid = torch.ones(len(flat), tok.num_tokens, dtype=torch.bool)
    loss = gen.loss(tok.encode(flat), tok.valid, emb, mask, row_valid, head_weight=10.0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    with torch.no_grad():
        tokens = gen.sample(emb[:2], mask[:2], steps=3, body_noise=0.0)
    assert tokens.shape == (2, tok.num_tokens, 256)
    assert tok.decode(tokens).shape == (2, flat.shape[1])


def test_neighbour_rows():
    rows = torch.arange(24, dtype=torch.float).view(2, 3, 4)       # 2 models x 3 slots x (feat+1)
    classes = torch.tensor([[0, 1, -1], [1, 2, -1]])
    table = class_row_table(rows, classes, num_classes=4)
    assert torch.allclose(table[1], (rows[0, 1] + rows[1, 0]) / 2)  # class 1 averaged over its two models
    assert torch.all(table[3] == 0)                                 # never trained: zero row
    names = {"a": 0, "b": 1}
    assert torch.allclose(mix_rows({"a": 0.25, "b": 0.75}, table, names), 0.25 * table[0] + 0.75 * table[1])
    assert mix_rows({"unknown": 1.0}, table, names) is None
    emb = torch.eye(3)
    mixed = clip_mix(emb[:1], emb, [0, 1, 2], table, k=1, exclude=[0])  # its own class is excluded
    assert mixed.shape == (1, 4)
