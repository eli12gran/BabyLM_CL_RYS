"""Regression test: per-doc jvp must not crash with sdpa attention."""
import pytest
import torch
from influence_curriculum.score import InfluenceConfig, compute_influence_matrix


def test_per_doc_jvp_uses_eager_not_sdpa():
    """attn_impl condition must use use_jvp (not use_vmap_jvp) so per-doc jvp also gets eager."""
    import inspect
    src = inspect.getsource(compute_influence_matrix)
    assert 'attn_impl = "eager" if use_jvp else "sdpa"' in src, (
        "attn_impl must be 'eager' whenever use_jvp=True — "
        "found the wrong condition (use_vmap_jvp or other) in compute_influence_matrix"
    )


def test_default_pass2_batch_size_enables_vmap():
    """InfluenceConfig() default must trigger the vmap(jvp) path."""
    cfg = InfluenceConfig()
    assert cfg.pass2_batch_size > 1, (
        f"pass2_batch_size default is {cfg.pass2_batch_size}; "
        "must be >1 to enable vmap(jvp) path"
    )
