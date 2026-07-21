import argparse
import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import Unigram
from tokenizers.normalizers import Lowercase, NFD, Sequence, StripAccents
from tokenizers.trainers import UnigramTrainer


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


CONFIG = {
    "languages": ["eng", "nld", "zho"],
    "hf_datasets": {
        "eng": "BabyLM-community/BabyLM-2026-Strict",
        "nld": "BabyLM-community/babylm-nld",
        "zho": "BabyLM-community/babylm-zho",
    },
    # Keep below 100M for tokenizer training selection. The model-training script
    # keeps your previous 33,333,333 adjusted units per language by default.
    "total_tokens_target": 99_000_000,
    "vocab_size_per_lang": 24_000,
    "tokmix_vocab_size": 48_000,
    "special_tokens": ["<UNK>", "<PAD>", "<CLS>", "<SEP>", "<MASK>"],
    "unk_token": "<UNK>",
    "pad_token": "<PAD>",
    "cls_token": "<CLS>",
    "sep_token": "<SEP>",
    "mask_token": "<MASK>",
    "output_dir": "./tokenizers_tokmix",
    "tokmix_weights": None,
    "prob_floor": 1e-300,
    # SentencePiece convention: whitespace is represented by U+2581.
    "sentencepiece_replacement": "▁",
}


OFFICIAL_BYTE_PREMIUM = {
    "eng": 1.000000,
    "nld": 1.051606,
    "zho": 0.935966,
}


def count_content_units(text: str, lang: str) -> int:
    if not text:
        return 0
    if lang == "zho":
        return len(re.findall(r"[\u4e00-\u9fff]", text))
    return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))


def count_official_tokens(text: str, lang: str, zho_tokenizer=None) -> int:
    if not text:
        return 0
    if lang in {"eng", "nld"}:
        return len(text.split())
    if lang == "zho":
        if zho_tokenizer is None:
            raise ValueError("zho_tokenizer is required for Chinese official token counting.")
        return len(zho_tokenizer.encode(text, add_special_tokens=False))
    raise ValueError(f"Unknown language: {lang}")


def load_babylm_datasets() -> Dict[str, Dataset]:
    logger.info("=" * 80)
    logger.info("Loading BabyLM datasets")
    logger.info("=" * 80)

    datasets: Dict[str, Dataset] = {}
    for lang, hf_path in CONFIG["hf_datasets"].items():
        logger.info(f"Loading {lang} from {hf_path}...")
        dataset = load_dataset(hf_path, split="train", trust_remote_code=True)
        datasets[lang] = dataset
        logger.info(f"✓ Loaded {lang}: {len(dataset):,} examples")
    return datasets


def get_language_data_generators(
    datasets_dict: Dict[str, Dataset],
    tokenization_budget: int,
) -> Tuple[Dict[str, Callable[[], Iterable[str]]], Dict[str, float]]:
    """
    Create one generator per language, each capped by equal adjusted BabyLM budget.
    This is for tokenizer training, not model exposure counting.
    """
    n_langs = len(datasets_dict)
    adjusted_budget_per_lang = tokenization_budget / n_langs
    budgets = {lang: adjusted_budget_per_lang for lang in datasets_dict.keys()}

    zho_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    generators: Dict[str, Callable[[], Iterable[str]]] = {}

    for lang, dataset in datasets_dict.items():
        official_bpf = OFFICIAL_BYTE_PREMIUM[lang]

        def make_generator(lang_code: str, ds: Dataset, target_adjusted_budget: float, bpf: float):
            def gen() -> Iterable[str]:
                official_tokens = 0
                emitted = 0
                for example in ds:
                    text = example.get("text", "")
                    n_tokens = count_official_tokens(text, lang_code, zho_tokenizer=zho_tokenizer)
                    if n_tokens == 0:
                        continue
                    official_tokens += n_tokens
                    emitted += 1
                    yield text
                    if official_tokens * bpf >= target_adjusted_budget:
                        logger.info(
                            f"{lang_code}: tokenizer-training cap reached at "
                            f"{official_tokens:,} official tokens / "
                            f"{official_tokens * bpf:,.0f} adjusted units / "
                            f"{emitted:,} examples"
                        )
                        break
            return gen

        generators[lang] = make_generator(lang, dataset, budgets[lang], official_bpf)

    return generators, budgets


def audit_babylm_budget(
    datasets_dict: Dict[str, Dataset],
    budgets: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    logger.info("=" * 80)
    logger.info("Auditing tokenizer-training BabyLM adjusted budget")
    logger.info("=" * 80)

    zho_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    stats: Dict[str, Dict[str, Any]] = {}

    for lang, dataset in datasets_dict.items():
        target_adjusted_budget = budgets[lang]
        docs = 0
        custom_content_units = 0
        official_tokens = 0
        utf8_bytes = 0
        characters = 0

        for example in dataset:
            text = example.get("text", "")
            units = count_content_units(text, lang)
            if units == 0:
                continue
            docs += 1
            custom_content_units += units
            official_tokens += count_official_tokens(text, lang, zho_tokenizer=zho_tokenizer)
            utf8_bytes += len(text.encode("utf-8"))
            characters += len(text)
            if official_tokens * OFFICIAL_BYTE_PREMIUM[lang] >= target_adjusted_budget:
                break

        stats[lang] = {
            "documents": docs,
            "custom_content_units": custom_content_units,
            "official_tokens": official_tokens,
            "official_byte_premium": OFFICIAL_BYTE_PREMIUM[lang],
            "adjusted_budget": official_tokens * OFFICIAL_BYTE_PREMIUM[lang],
            "utf8_bytes": utf8_bytes,
            "characters": characters,
            "utf8_bytes_per_character": utf8_bytes / characters if characters else 0.0,
        }

    total_adjusted = sum(s["adjusted_budget"] for s in stats.values())
    for lang, s in stats.items():
        share = s["adjusted_budget"] / total_adjusted * 100 if total_adjusted else 0.0
        logger.info(
            f"{lang.upper()}: documents={s['documents']:,}, "
            f"official_tokens={s['official_tokens']:,}, "
            f"adjusted_budget={s['adjusted_budget']:,.0f} ({share:.2f}%)"
        )

    return stats


def create_sentencepiece_unigram_template(language: Optional[str] = None) -> Tokenizer:
    """
    Create a SentencePiece-style Unigram tokenizer using Hugging Face tokenizers.
    The normalizer is kept close to your earlier setup for comparability:
        - NFD + StripAccents for all languages.
        - Lowercase for non-Chinese languages.
    """
    tokenizer = Tokenizer(Unigram())

    normalizers = [NFD(), StripAccents()]
    if language != "zho":
        normalizers.append(Lowercase())
    tokenizer.normalizer = Sequence(normalizers)

    replacement = CONFIG["sentencepiece_replacement"]
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement=replacement, prepend_scheme="always")
    tokenizer.decoder = decoders.Metaspace(replacement=replacement, prepend_scheme="always")
    return tokenizer


def train_monolingual_unigram_tokenizer(
    language: str,
    data_generator: Callable[[], Iterable[str]],
    vocab_size: int,
) -> Tokenizer:
    logger.info("=" * 80)
    logger.info(f"Training {language.upper()} monolingual SentencePiece-style Unigram tokenizer")
    logger.info("=" * 80)

    tokenizer = create_sentencepiece_unigram_template(language)
    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=CONFIG["special_tokens"],
        unk_token=CONFIG["unk_token"],
        show_progress=True,
        max_piece_length=16,
        shrinking_factor=0.75,
    )
    tokenizer.train_from_iterator(data_generator(), trainer=trainer, length=None)

    logger.info(f"✓ Trained {language.upper()} tokenizer. Final vocab size: {len(tokenizer.get_vocab()):,}")
    return tokenizer


def extract_unigram_scores(
    tokenizer: Tokenizer,
    exclude_special_tokens: bool = True,
) -> Dict[str, float]:
    tokenizer_json = json.loads(tokenizer.to_str())
    model = tokenizer_json.get("model", {})
    if model.get("type") != "Unigram":
        raise ValueError(f"Expected Unigram tokenizer, got model type: {model.get('type')}")

    raw_vocab = model.get("vocab", [])
    special_set = set(CONFIG["special_tokens"])
    scores: Dict[str, float] = {}

    for piece, score in raw_vocab:
        if exclude_special_tokens and piece in special_set:
            continue
        scores[piece] = float(score)
    return scores


def normalize_tokmix_weights(
    languages: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    if weights is None:
        return {lang: 1.0 / len(languages) for lang in languages}

    missing = set(languages) - set(weights.keys())
    if missing:
        raise ValueError(f"Missing TOKMIX weights for languages: {sorted(missing)}")

    total = sum(float(weights[lang]) for lang in languages)
    if total <= 0:
        raise ValueError("TOKMIX weights must sum to a positive value.")
    return {lang: float(weights[lang]) / total for lang in languages}


def collect_base_pieces(monolingual_tokenizers: Dict[str, Tokenizer]) -> List[str]:

    replacement = CONFIG["sentencepiece_replacement"]
    base_pieces = set()

    for tokenizer in monolingual_tokenizers.values():
        for piece in tokenizer.get_vocab().keys():
            if piece in CONFIG["special_tokens"]:
                continue
            # Single character pieces, plus SentencePiece word-start + one character.
            if len(piece) == 1 or (len(piece) == 2 and piece.startswith(replacement)):
                base_pieces.add(piece)

    return sorted(base_pieces)


def compute_token_origin_summary(token_languages: Dict[str, set]) -> Dict[str, int]:
    summary = Counter()
    for _, langs in token_languages.items():
        summary[f"tokens_seen_in_{len(langs)}_language(s)"] += 1
    return dict(summary)


def build_tokmix_vocabulary(monolingual_tokenizers: Dict[str, Tokenizer], final_vocab_size: int, weights: Optional[Dict[str, float]] = None,) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:

    languages = list(monolingual_tokenizers.keys())
    norm_weights = normalize_tokmix_weights(languages, weights)

    logger.info("=" * 80)
    logger.info("Building shared TOKMIX vocabulary")
    logger.info("=" * 80)
    logger.info(f"Weights: {norm_weights}")

    mixed_probs: Dict[str, float] = defaultdict(float)
    token_languages: Dict[str, set] = defaultdict(set)
    per_lang_vocab_sizes: Dict[str, int] = {}

    for lang, tokenizer in monolingual_tokenizers.items():
        scores = extract_unigram_scores(tokenizer, exclude_special_tokens=True)
        per_lang_vocab_sizes[lang] = len(scores)
        weight = norm_weights[lang]

        for piece, score in scores.items():
            prob = math.exp(score)
            if prob <= 0:
                continue
            mixed_probs[piece] += weight * prob
            token_languages[piece].add(lang)

    special_tokens = list(CONFIG["special_tokens"])
    special_set = set(special_tokens)
    base_pieces = [p for p in collect_base_pieces(monolingual_tokenizers) if p not in special_set]

    protected = special_tokens + base_pieces
    protected_set = set(protected)

    if final_vocab_size <= len(special_tokens):
        raise ValueError(
            f"final_vocab_size={final_vocab_size} is too small for special tokens "
            f"({len(special_tokens)})."
        )

    # If base pieces exceed available slots, keep the highest-probability base pieces.
    max_base_slots = max(final_vocab_size - len(special_tokens), 0)
    if len(base_pieces) > max_base_slots:
        logger.warning(
            f"Base pieces ({len(base_pieces):,}) exceed available non-special slots "
            f"({max_base_slots:,}); trimming base pieces by mixed probability."
        )
        base_pieces = sorted(
            base_pieces,
            key=lambda p: mixed_probs.get(p, CONFIG["prob_floor"]),
            reverse=True,
        )[:max_base_slots]
        protected = special_tokens + base_pieces
        protected_set = set(protected)

    remaining_slots = final_vocab_size - len(protected)
    candidate_items = [
        (piece, prob)
        for piece, prob in mixed_probs.items()
        if piece not in protected_set
    ]
    candidate_items.sort(key=lambda x: x[1], reverse=True)

    final_vocab: List[Tuple[str, float]] = []
    final_vocab.append((CONFIG["unk_token"], 0.0))
    for special in CONFIG["special_tokens"]:
        if special != CONFIG["unk_token"]:
            final_vocab.append((special, 0.0))

    existing = {piece for piece, _ in final_vocab}

    # Add base pieces first, with their TOKMIX mixed probability when available.
    for piece in base_pieces:
        if piece in existing:
            continue
        final_vocab.append((piece, math.log(max(mixed_probs.get(piece, CONFIG["prob_floor"]), CONFIG["prob_floor"]))))
        existing.add(piece)

    for piece, prob in candidate_items:
        if piece in existing:
            continue
        final_vocab.append((piece, math.log(max(prob, CONFIG["prob_floor"]))))
        existing.add(piece)
        if len(final_vocab) >= final_vocab_size:
            break

    if len(final_vocab) != final_vocab_size:
        logger.warning(
            f"Final vocab size is {len(final_vocab):,}, not requested {final_vocab_size:,}. "
            "This can happen if the monolingual union is smaller than the requested size."
        )

    metadata = {
        "strategy": "TOKMIX",
        "description": (
            "Monolingual SentencePiece-style Unigram tokenizers were merged by "
            "weighted average of vocabulary unit probabilities, then sorted and trimmed."
        ),
        "languages": languages,
        "weights": norm_weights,
        "target_vocab_size": final_vocab_size,
        "actual_vocab_size": len(final_vocab),
        "special_tokens": CONFIG["special_tokens"],
        "sentencepiece_replacement": CONFIG["sentencepiece_replacement"],
        "pre_tokenizer": "Metaspace/SentencePiece-style, not ByteLevel",
        "per_language_non_special_vocab_sizes": per_lang_vocab_sizes,
        "union_non_special_vocab_size": len(mixed_probs),
        "protected_base_pieces_size": len(base_pieces),
        "selected_non_special_vocab_size": len(final_vocab) - len(CONFIG["special_tokens"]),
        "overlap_summary": compute_token_origin_summary(token_languages),
    }
    return final_vocab, metadata


def build_tokmix_tokenizer(
    monolingual_tokenizers: Dict[str, Tokenizer],
    final_vocab_size: int,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[Tokenizer, Dict[str, Any]]:
    final_vocab, metadata = build_tokmix_vocabulary(monolingual_tokenizers, final_vocab_size, weights)
    tokmix_tokenizer = Tokenizer(Unigram(final_vocab, unk_id=0))

    # Use the same normalizer family as the monolingual tokenizers.
    tokmix_tokenizer.normalizer = monolingual_tokenizers[list(monolingual_tokenizers.keys())[0]].normalizer

    replacement = CONFIG["sentencepiece_replacement"]
    tokmix_tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement=replacement, prepend_scheme="always")
    tokmix_tokenizer.decoder = decoders.Metaspace(replacement=replacement, prepend_scheme="always")
    return tokmix_tokenizer, metadata


def save_tokmix_outputs(
    tokmix_tokenizer: Tokenizer,
    monolingual_tokenizers: Dict[str, Tokenizer],
    metadata: Dict[str, Any],
    budget_stats: Dict[str, Any],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    tokmix_path = os.path.join(output_dir, "tokmix_tokenizer.json")
    tokmix_tokenizer.save(tokmix_path)

    monolingual_dir = os.path.join(output_dir, "monolingual_tokenizers")
    os.makedirs(monolingual_dir, exist_ok=True)
    for lang, tokenizer in monolingual_tokenizers.items():
        tokenizer.save(os.path.join(monolingual_dir, f"tokenizer_{lang}.json"))

    hf_dir = os.path.join(output_dir, "hf_tokmix_tokenizer")
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokmix_tokenizer,
        unk_token=CONFIG["unk_token"],
        pad_token=CONFIG["pad_token"],
        cls_token=CONFIG["cls_token"],
        sep_token=CONFIG["sep_token"],
        mask_token=CONFIG["mask_token"],
    )
    fast_tokenizer.model_max_length = 256
    fast_tokenizer.save_pretrained(hf_dir)

    vocab = fast_tokenizer.get_vocab()
    with open(os.path.join(output_dir, "tokmix_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    full_metadata = {
        **metadata,
        "config": CONFIG,
        "budget_stats": budget_stats,
        "output_files": {
            "tokmix_tokenizer_json": tokmix_path,
            "hf_tokenizer_dir": hf_dir,
            "monolingual_tokenizers_dir": monolingual_dir,
        },
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(full_metadata, f, indent=2, ensure_ascii=False)

    logger.info("=" * 80)
    logger.info("✓ TOKMIX tokenizer outputs saved")
    logger.info(f"Shared tokenizer JSON: {tokmix_path}")
    logger.info(f"HF tokenizer dir: {hf_dir}")
    logger.info(f"Metadata: {os.path.join(output_dir, 'metadata.json')}")
    logger.info("=" * 80)


def parse_weights(weights_str: Optional[str]) -> Optional[Dict[str, float]]:
    if not weights_str:
        return None
    # Format: eng=1,nld=1,zho=1
    weights: Dict[str, float] = {}
    for part in weights_str.split(","):
        if "=" not in part:
            raise ValueError("Weights must be formatted like eng=1,nld=1,zho=1")
        lang, value = part.split("=", 1)
        weights[lang.strip()] = float(value)
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a TOKMIX tokenizer.")
    parser.add_argument("--output_dir", default=CONFIG["output_dir"])
    parser.add_argument("--total_tokens_target", type=int, default=CONFIG["total_tokens_target"])
    parser.add_argument("--vocab_size_per_lang", type=int, default=CONFIG["vocab_size_per_lang"])
    parser.add_argument("--tokmix_vocab_size", type=int, default=CONFIG["tokmix_vocab_size"])
    parser.add_argument(
        "--tokmix_weights",
        default=None,
        help="Optional weights like eng=1,nld=1,zho=1. Default is uniform.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CONFIG["output_dir"] = args.output_dir
    CONFIG["total_tokens_target"] = args.total_tokens_target
    CONFIG["vocab_size_per_lang"] = args.vocab_size_per_lang
    CONFIG["tokmix_vocab_size"] = args.tokmix_vocab_size
    CONFIG["tokmix_weights"] = parse_weights(args.tokmix_weights)

    logger.info("=" * 80)
    logger.info("BabyLM 2026 TOKMIX tokenizer-building pipeline")
    logger.info("=" * 80)
    logger.info("Correction enforced: SentencePiece-style Metaspace, not ByteLevel.")

    datasets = load_babylm_datasets()
    generators, budgets = get_language_data_generators(
        datasets,
        tokenization_budget=CONFIG["total_tokens_target"],
    )
    budget_stats = audit_babylm_budget(datasets, budgets)

    monolingual_tokenizers: Dict[str, Tokenizer] = {}
    for lang in CONFIG["languages"]:
        monolingual_tokenizers[lang] = train_monolingual_unigram_tokenizer(
            language=lang,
            data_generator=generators[lang],
            vocab_size=CONFIG["vocab_size_per_lang"],
        )

    tokmix_tokenizer, metadata = build_tokmix_tokenizer(
        monolingual_tokenizers,
        final_vocab_size=CONFIG["tokmix_vocab_size"],
        weights=CONFIG["tokmix_weights"],
    )
    save_tokmix_outputs(tokmix_tokenizer, monolingual_tokenizers, metadata, budget_stats, CONFIG["output_dir"])

    logger.info("✓ TOKMIX tokenizer pipeline complete")


if __name__ == "__main__":
    main()
