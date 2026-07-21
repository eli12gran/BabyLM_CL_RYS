from __future__ import annotations
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass
class CurriculumConfig:
    strategy: Literal["epoch_wise", "cumulative"] = "epoch_wise"
    # epoch_wise options
    aggregation: str = "per_epoch_raw"   # "per_epoch_raw" | "lognormal"
    direction: str = "asc"               # "asc" | "desc"
    segment_size: int = 1000
    lognormal_window: int = 10
    lognormal_mu: float = 0.0
    lognormal_sigma: float = 1.0
    # cumulative options
    n_groups: int = 2                    # number of difficulty groups (= number of output epochs)


def _lognormal_kernel(window: int, mu: float, sigma: float) -> np.ndarray:
    from scipy.stats import lognorm
    k = np.arange(1, window + 1, dtype=float)
    h = lognorm.pdf(k, s=sigma, scale=np.exp(mu))
    return h / h.sum()


def build_curriculum(
    Phi: np.ndarray,
    texts: list[str],
    doc_ids: list[str],
    config: CurriculumConfig,
    output_dir: str,
    seed: int,
) -> None:
    D, T = Phi.shape
    rng = random.Random(seed)
    out = Path(output_dir) / "curriculum"
    out.mkdir(parents=True, exist_ok=True)

    if config.strategy == "cumulative":
        _build_cumulative(Phi, texts, doc_ids, config, out, rng)
        return

    if config.aggregation == "lognormal":
        h = _lognormal_kernel(config.lognormal_window, config.lognormal_mu, config.lognormal_sigma)
        scores = np.zeros_like(Phi)
        for t in range(T):
            for k, hk in enumerate(h):
                lag = k + 1
                if t - lag >= 0:
                    scores[:, t] += Phi[:, t - lag] * hk
    else:
        scores = Phi.view()  # view only; build_curriculum never mutates scores

    for e in range(T):
        col = scores[:, e]
        order = np.argsort(col)
        if config.direction == "desc":
            order = order[::-1].copy()

        shuffled: list[int] = []
        for start in range(0, len(order), config.segment_size):
            seg = order[start : start + config.segment_size].tolist()
            rng.shuffle(seg)
            shuffled.extend(seg)

        with open(out / f"epoch_{e:02d}.jsonl", "w") as f:
            for idx in shuffled:
                f.write(json.dumps({"id": doc_ids[idx], "text": texts[idx]}) + "\n")


def _build_cumulative(
    Phi: np.ndarray,
    texts: list[str],
    doc_ids: list[str],
    config: CurriculumConfig,
    out: Path,
    rng: random.Random,
) -> None:
    D = Phi.shape[0]
    G = config.n_groups

    # aggregate across all checkpoints → one difficulty score per doc
    agg = Phi.mean(axis=1)
    order = np.argsort(agg)[::-1]  # descending: highest influence = easiest

    # split into G equal groups
    boundaries = [round(D * g / G) for g in range(G + 1)]
    groups = [order[boundaries[g] : boundaries[g + 1]].tolist() for g in range(G)]

    # epoch k includes groups 0..k (growing subset)
    for k in range(G):
        subset: list[int] = []
        for g in range(k + 1):
            seg = groups[g][:]
            rng.shuffle(seg)
            subset.extend(seg)

        with open(out / f"epoch_{k:02d}.jsonl", "w") as f:
            for idx in subset:
                f.write(json.dumps({"id": doc_ids[idx], "text": texts[idx]}) + "\n")
