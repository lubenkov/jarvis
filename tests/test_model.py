import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from jarvis.model.block import SwiGLU
from jarvis.model.config import ModelConfig
from jarvis.model.norm import RMSNorm
from jarvis.model.rope import RoPE
from jarvis.model.transformer import TransformerLM


def tiny_config(**overrides):
    cfg = ModelConfig(
        vocab_size=128,
        context_length=16,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
    )
    return replace(cfg, **overrides)


def test_rmsnorm_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 8)
    norm = RMSNorm(8)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, 8))

    out = norm(x)
    reference = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps) * norm.weight

    assert out.shape == x.shape
    assert torch.allclose(out, reference, atol=1e-6, rtol=1e-6)
    unweighted = out / norm.weight
    assert torch.allclose(unweighted.pow(2).mean(-1), torch.ones(2, 5), atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rmsnorm_preserves_dtype(dtype):
    x = torch.randn(2, 3, 8, dtype=dtype)
    out = RMSNorm(8)(x)
    assert out.dtype == dtype


def test_swiglu_shapes_and_gradients():
    cfg = tiny_config()
    ff = SwiGLU(cfg)
    x = torch.randn(2, 7, cfg.d_model, requires_grad=True)

    out = ff(x)

    assert out.shape == x.shape
    assert ff.gate.out_features == cfg.d_ff
    assert ff.up.out_features == cfg.d_ff
    assert ff.down.out_features == cfg.d_model
    out.square().mean().backward()
    for name, param in ff.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_rope_extends_frequency_cache():
    rope = RoPE(head_dim=8, max_seq_len=4)
    q = torch.randn(1, 2, 6, 8)
    k = torch.randn(1, 2, 6, 8)

    q_rot, k_rot = rope.rotate(q, k)

    assert rope.cos_cached.shape[0] >= 6
    reference = RoPE(head_dim=8, max_seq_len=6)
    q_ref, k_ref = reference.rotate(q, k)
    assert torch.allclose(q_rot, q_ref, atol=1e-6)
    assert torch.allclose(k_rot, k_ref, atol=1e-6)


def test_rope_offset_matches_longer_sequence_slice():
    rope = RoPE(head_dim=8, max_seq_len=32)
    q = torch.randn(1, 2, 8, 8)

    q_full, _ = rope.rotate(q, q, offset=0)
    q_tail, _ = rope.rotate(q[:, :, 3:], q[:, :, 3:], offset=3)

    assert torch.allclose(q_tail, q_full[:, :, 3:], atol=1e-6)


def test_rope_preserves_vector_norms():
    rope = RoPE(head_dim=8, max_seq_len=16)
    q = torch.randn(2, 3, 7, 8)

    q_rot, _ = rope.rotate(q, q)

    assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-5)


def test_future_tokens_do_not_change_past_logits():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = TransformerLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    changed_future = ids.clone()
    changed_future[0, -1] = (changed_future[0, -1] + 1) % cfg.vocab_size
    changed_middle = ids.clone()
    changed_middle[0, 6] = (changed_middle[0, 6] + 1) % cfg.vocab_size

    with torch.no_grad():
        base = model(ids)
        future = model(changed_future)
        middle = model(changed_middle)

    assert torch.allclose(base[:, :-1], future[:, :-1], atol=1e-6)
    assert torch.allclose(base[:, :6], middle[:, :6], atol=1e-6)
    assert not torch.allclose(base[:, -1], future[:, -1])


def test_incremental_kv_cache_matches_full_pass():
    torch.manual_seed(0)
    cfg = tiny_config(context_length=12)
    model = TransformerLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 12))

    with torch.no_grad():
        full = model(ids)
        caches = None
        steps = []
        for t in range(ids.shape[1]):
            logits, caches = model(ids[:, t:t + 1], caches=caches, use_cache=True)
            steps.append(logits)
        incremental = torch.cat(steps, dim=1)

    assert (full - incremental).abs().max().item() < 1e-4


def test_embeddings_are_tied():
    model = TransformerLM(tiny_config())
    assert model.lm_head.weight is model.token_embedding.weight


@pytest.mark.parametrize("scaled", [True, False])
def test_depth_scaled_residual_init(scaled):
    cfg = tiny_config(depth_scaled_init=scaled)
    model = TransformerLM(cfg)
    residual_std = 0.02 / math.sqrt(2 * cfg.n_layers) if scaled else 0.02

    for block in model.blocks:
        for projection in (block.attn.out_proj, block.ff.down):
            std = projection.weight.detach().std().item()
            assert std == pytest.approx(residual_std, rel=0.15)
        assert block.attn.qkv_proj.weight.detach().std().item() == pytest.approx(0.02, rel=0.15)


def test_backward_produces_finite_gradients():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = TransformerLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))

    logits = model(ids)
    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), ids.reshape(-1))
    loss.backward()

    assert torch.isfinite(loss)
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def test_parameter_count_excludes_embeddings_on_request():
    cfg = tiny_config()
    model = TransformerLM(cfg)
    total = model.n_params()
    non_embedding = model.n_params(non_embedding=True)
    assert total - non_embedding == cfg.vocab_size * cfg.d_model
