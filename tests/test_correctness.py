import json
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from influence_curriculum.data import DataConfig, load_documents
from influence_curriculum.curriculum import CurriculumConfig, build_curriculum
from influence_curriculum.score import (
    InfluenceConfig, _fallback_grad, _unit, _unit_batch, compute_influence_matrix,
)
try:
    from influence_curriculum.score import _vmap_grads, _HAS_VMAP
except ImportError:
    _HAS_VMAP = False


def test_line_segmentation(tmp_path):
    (tmp_path / "src.txt").write_text("hello\nworld\n\nskip me\n")
    texts, ids = load_documents(str(tmp_path), DataConfig(doc_boundary="line", min_doc_tokens=0))
    assert texts == ["hello", "world", "skip me"]
    assert ids == ["src#0", "src#1", "src#2"]


def test_blank_line_segmentation(tmp_path):
    (tmp_path / "src.txt").write_text("para one\nstill one\n\npara two\n")
    texts, ids = load_documents(str(tmp_path), DataConfig(doc_boundary="blank_line", min_doc_tokens=0))
    assert texts == ["para one\nstill one", "para two"]
    assert ids == ["src#0", "src#1"]


def test_doc_id_stable(tmp_path):
    (tmp_path / "a.txt").write_text("x\ny\n")
    (tmp_path / "b.txt").write_text("z\n")
    _, ids = load_documents(str(tmp_path), DataConfig(min_doc_tokens=0))
    assert "a#0" in ids and "a#1" in ids and "b#0" in ids


def test_mean_gradient_identity():
    """dot-with-mean == (1/D)*sum_of_pairwise for any unit-normalized gradient matrix."""
    rng = np.random.default_rng(0)
    D, V = 8, 50
    raw = rng.random((D, V)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    grads = raw / norms  # unit-normalized

    efficient = grads @ grads.mean(axis=0)                              # O(D)
    pairwise = np.array([(1 / D) * (grads[i] @ grads.T).sum()          # O(D^2)
                         for i in range(D)])
    np.testing.assert_allclose(efficient, pairwise, rtol=1e-5)


def test_permutation_validity(tmp_path):
    D, T = 20, 3
    rng = np.random.default_rng(42)
    Phi = rng.random((D, T)).astype(np.float32)
    texts = [f"doc {i}" for i in range(D)]
    doc_ids = [f"fake#{i}" for i in range(D)]
    build_curriculum(Phi, texts, doc_ids, CurriculumConfig(segment_size=5), str(tmp_path), seed=0)
    for e in range(T):
        path = tmp_path / "curriculum" / f"epoch_{e:02d}.jsonl"
        assert path.exists()
        ids = [json.loads(l)["id"] for l in path.read_text().splitlines()]
        assert sorted(ids) == sorted(doc_ids), f"epoch {e} is not a permutation"


@pytest.mark.skipif(not _HAS_VMAP, reason="torch.func not available")
def test_vmap_matches_fallback():
    """_vmap_grads must produce the same per-doc gradients as _fallback_grad.

    Both approaches use the same padded input so results are directly comparable.
    CPU uses a slower-but-correct SDPA fallback; tolerance is relaxed to 1e-3.
    """
    MODEL = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager")
    model.eval()

    texts = ["The cat sat.", "Dogs are great.", "I love ML."]
    encodings = [tokenizer(t, return_tensors="pt") for t in texts]

    emb_name = next(n for n, p in model.named_parameters()
                    if p is model.get_input_embeddings().weight)

    # Pad all to same length — both approaches use the same padded ids
    max_len = max(enc["input_ids"].shape[-1] for enc in encodings)
    pad_id  = tokenizer.pad_token_id
    padded  = torch.stack([
        torch.nn.functional.pad(
            enc["input_ids"].squeeze(0), (0, max_len - enc["input_ids"].shape[-1]), value=pad_id
        )
        for enc in encodings
    ])  # (3, max_len)

    masks  = torch.ones_like(padded)   # all tokens real (no padding in this test)
    vmap_g = _vmap_grads(model, emb_name, padded, masks).float()

    for i in range(len(texts)):
        fallback = torch.tensor(_fallback_grad(model, padded[i], masks[i], "cpu"))
        diff = (vmap_g[i] - fallback).abs().max().item()
        assert diff < 1e-3, f"doc {i}: vmap vs fallback max diff = {diff:.2e}"


@pytest.mark.skipif(not _HAS_VMAP, reason="torch.func not available")
def test_unit_batch_matches_unit():
    """_unit_batch (row-wise) must match applying _unit to each row individually."""
    rng = np.random.default_rng(1)
    G = rng.random((8, 50)).astype(np.float32)
    batch_result = _unit_batch(torch.tensor(G)).numpy()
    per_row = np.stack([_unit(G[i]) for i in range(len(G))])
    np.testing.assert_allclose(batch_result, per_row, rtol=1e-5)


def test_determinism(tmp_path):
    D, T = 10, 2
    rng = np.random.default_rng(7)
    Phi = rng.random((D, T)).astype(np.float32)
    texts = [f"doc {i}" for i in range(D)]
    doc_ids = [f"fake#{i}" for i in range(D)]
    cfg = CurriculumConfig()
    build_curriculum(Phi, texts, doc_ids, cfg, str(tmp_path / "r1"), seed=42)
    build_curriculum(Phi, texts, doc_ids, cfg, str(tmp_path / "r2"), seed=42)
    for e in range(T):
        f1 = (tmp_path / "r1" / "curriculum" / f"epoch_{e:02d}.jsonl").read_text()
        f2 = (tmp_path / "r2" / "curriculum" / f"epoch_{e:02d}.jsonl").read_text()
        assert f1 == f2, f"epoch {e} differs between identical runs"
