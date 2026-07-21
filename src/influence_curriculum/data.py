from __future__ import annotations
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class DataConfig:
    doc_boundary: str | dict[str, str | Callable] = "line"
    default_boundary: str = "line"
    min_doc_tokens: int = 1
    chunking: str = "truncate"


def _segment(text: str, rule: str | Callable) -> list[str]:
    if callable(rule):
        return [s.strip() for s in rule(text) if s.strip()]
    if rule == "line":
        return [l.strip() for l in text.splitlines() if l.strip()]
    if rule == "blank_line":
        return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if rule == "whole_file":
        return [text.strip()] if text.strip() else []
    raise ValueError(f"Unknown doc_boundary: {rule!r}")


def load_documents(
    dataset_dir: str,
    config: DataConfig,
    tokenizer=None,
) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    doc_ids: list[str] = []

    for path in sorted(Path(dataset_dir).glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        rule = config.doc_boundary
        if isinstance(rule, dict):
            rule = rule.get(path.name, config.default_boundary)
        segs = _segment(raw, rule)
        for i, seg in enumerate(segs):
            if tokenizer is not None:
                n = len(tokenizer.encode(seg, add_special_tokens=False))
                if n < config.min_doc_tokens:
                    continue
            elif tokenizer is None and len(seg.split()) < config.min_doc_tokens:
                continue
            texts.append(seg)
            doc_ids.append(f"{path.stem}#{i}")

    if len(texts) < 2000:
        warnings.warn(
            f"Only {len(texts)} documents loaded from {dataset_dir}. "
            "Segmentation may not have fired correctly.",
            stacklevel=2,
        )
    return texts, doc_ids
