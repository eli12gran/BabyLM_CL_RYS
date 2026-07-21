from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .curriculum import CurriculumConfig, build_curriculum
from .data import DataConfig, load_documents
from .score import InfluenceConfig, compute_influence_matrix
from .train import TrainingConfig, train_surrogate


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sort_by_influence(
    model,
    dataset_dir: str,
    output_dir: str,
    *,
    checkpoints: list[str] | None = None,
    tokenizer=None,
    data_config: DataConfig | None = None,
    training_config: TrainingConfig | None = None,
    influence_config: InfluenceConfig | None = None,
    curriculum_config: CurriculumConfig | None = None,
    seed: int = 0,
    device: str = "auto",
) -> str:
    data_config = data_config or DataConfig()
    training_config = training_config or TrainingConfig()
    influence_config = influence_config or InfluenceConfig()
    curriculum_config = curriculum_config or CurriculumConfig()
    device = _resolve_device(device)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if isinstance(model, str):
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(model)
    elif tokenizer is None:
        raise ValueError("tokenizer is required when model is an nn.Module")

    # Phase 0: load + tokenize
    texts, doc_ids = load_documents(dataset_dir, data_config, tokenizer)
    batch_out = tokenizer(texts, truncation=True, max_length=training_config.max_seq_len, padding=False, return_tensors=None)
    encodings = [{k: torch.tensor([v[i]]) for k, v in batch_out.items()} for i in range(len(texts))]

    # Phase 1: surrogate
    if checkpoints is None:
        checkpoints = train_surrogate(model, tokenizer, texts, str(out), training_config, seed, device)
    else:
        if len(checkpoints) == 0:
            raise ValueError("checkpoints list is empty; T must be >= 1")
        if len(checkpoints) == 1:
            warnings.warn("T=1: per-epoch curriculum collapses to a single ordering.", stacklevel=2)
        elif len(checkpoints) != training_config.epochs:
            warnings.warn(
                "Supplied checkpoint count doesn't match training_config.epochs. "
                "If checkpoints were not produced by a random-order run on this data, "
                "results will diverge from the paper.",
                stacklevel=2,
            )

    # Phase 2: influence matrix
    Phi = compute_influence_matrix(checkpoints, encodings, influence_config, device)
    np.save(str(out / "influence_matrix.npy"), Phi)
    (out / "doc_ids.json").write_text(json.dumps(doc_ids))

    # Phase 3: curriculum
    build_curriculum(Phi, texts, doc_ids, curriculum_config, str(out), seed)

    # Persist resolved config
    (out / "config.json").write_text(json.dumps({
        "data_config": {k: str(v) for k, v in data_config.__dict__.items()},
        "training_config": training_config.__dict__,
        "influence_config": influence_config.__dict__,
        "curriculum_config": curriculum_config.__dict__,
        "seed": seed,
        "device": device,
        "num_docs": len(texts),
        "T": len(checkpoints),
    }, indent=2))

    return str(out)
