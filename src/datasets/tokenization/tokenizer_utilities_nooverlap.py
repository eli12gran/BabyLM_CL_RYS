import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tokenizers import Tokenizer

# These are the only tokens shared across language vocabularies in NOOVERLAP.
SPECIAL_TOKENS = ["<UNK>", "<PAD>", "<CLS>", "<SEP>", "<MASK>"]


class NOOVERLAPTokenizer:

    def __init__(
        self,
        tokenizers_dict: Dict[str, Tokenizer],
        language_tags: Optional[Dict[str, str]] = None,
        special_tokens: Optional[List[str]] = None,
    ):
        self.tokenizers = tokenizers_dict
        self.languages = list(tokenizers_dict.keys())
        self.language_tags = language_tags or {lang: lang for lang in self.languages}
        self.special_tokens = special_tokens or SPECIAL_TOKENS

        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}

        # local_to_global[lang][local_id] -> global_nooverlap_id
        self.local_to_global: Dict[str, Dict[int, int]] = {}

        # global_to_local[global_id] -> (lang, local_id)
        self.global_to_local: Dict[int, Tuple[str, int]] = {}

        # Per-language non-special global ID intervals.
        self.language_id_ranges: Dict[str, Dict[str, int]] = {}

        self._build_combined_vocab()

        self.unk_token = "<UNK>"
        self.pad_token = "<PAD>"
        self.cls_token = "<CLS>"
        self.sep_token = "<SEP>"
        self.mask_token = "<MASK>"

        self.unk_token_id = self.token_to_id[self.unk_token]
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.cls_token_id = self.token_to_id[self.cls_token]
        self.sep_token_id = self.token_to_id[self.sep_token]
        self.mask_token_id = self.token_to_id[self.mask_token]

    def _add_global_token(self, token: str, global_id: int) -> None:
        if token in self.token_to_id:
            raise ValueError(f"Duplicate global token string: {token}")
        self.vocab[token] = global_id
        self.token_to_id[token] = global_id
        self.id_to_token[global_id] = token

    def _build_combined_vocab(self) -> None:
        current_id = 0

        # 1) Shared special tokens only.
        for token in self.special_tokens:
            self._add_global_token(token, current_id)
            current_id += 1

        # 2) Disjoint language-specific vocabulary segments.
        for lang in self.languages:
            tokenizer = self.tokenizers[lang]
            local_vocab = tokenizer.get_vocab()  # token -> local_id
            self.local_to_global[lang] = {}

            segment_start = current_id

            for token, local_id in sorted(local_vocab.items(), key=lambda x: x[1]):
                if token in self.special_tokens:
                    # Special tokens are shared across languages.
                    self.local_to_global[lang][local_id] = self.token_to_id[token]
                    continue

                # Prefix the token string only for metadata/readability.
                # The model receives integer IDs, not this literal string.
                visible_lang = self.language_tags.get(lang, lang)
                global_token = f"{visible_lang}::{token}"

                self._add_global_token(global_token, current_id)
                self.local_to_global[lang][local_id] = current_id
                self.global_to_local[current_id] = (lang, local_id)
                current_id += 1

            self.language_id_ranges[lang] = {
                "start_inclusive": segment_start,
                "end_exclusive": current_id,
                "size_excluding_shared_specials": current_id - segment_start,
            }

    def remap_ids(self, local_ids: List[int], language: str) -> List[int]:
        if language not in self.local_to_global:
            raise ValueError(f"Language {language} not supported. Available: {self.languages}")

        mapping = self.local_to_global[language]
        return [mapping.get(int(local_id), self.unk_token_id) for local_id in local_ids]

    def encode(
        self,
        text: str,
        language: str,
        add_special_tokens: bool = False,
    ) -> List[int]:
        if language not in self.tokenizers:
            raise ValueError(f"Language {language} not supported. Available: {self.languages}")

        encoded = self.tokenizers[language].encode(
            text,
            add_special_tokens=add_special_tokens,
        )
        return self.remap_ids(encoded.ids, language)

    def encode_batch(
        self,
        texts: List[str],
        language: str,
        add_special_tokens: bool = False,
    ) -> List[List[int]]:
        if language not in self.tokenizers:
            raise ValueError(f"Language {language} not supported. Available: {self.languages}")

        encoded_batch = self.tokenizers[language].encode_batch(
            texts,
            add_special_tokens=add_special_tokens,
        )
        return [self.remap_ids(encoded.ids, language) for encoded in encoded_batch]

    def decode(self, ids: List[int], language: Optional[str] = None) -> str:
        """
        Best-effort decode.

        If language is provided, only IDs belonging to that language segment are
        decoded. If language is None and the sequence mixes languages, contiguous
        runs are decoded with their corresponding monolingual tokenizer.
        """
        ids = [int(i) for i in ids if int(i) not in set(self.token_to_id[t] for t in self.special_tokens)]

        if language is not None:
            local_ids = [
                self.global_to_local[i][1]
                for i in ids
                if i in self.global_to_local and self.global_to_local[i][0] == language
            ]
            return self.tokenizers[language].decode(local_ids)

        pieces = []
        current_lang = None
        current_local_ids: List[int] = []

        def flush() -> None:
            nonlocal current_lang, current_local_ids
            if current_lang is not None and current_local_ids:
                pieces.append(self.tokenizers[current_lang].decode(current_local_ids))
            current_lang = None
            current_local_ids = []

        for global_id in ids:
            pair = self.global_to_local.get(global_id)
            if pair is None:
                flush()
                continue

            lang, local_id = pair
            if current_lang is None:
                current_lang = lang
            elif lang != current_lang:
                flush()
                current_lang = lang

            current_local_ids.append(local_id)

        flush()
        return "".join(pieces)

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab_size_per_language(self, include_shared_specials: bool = False) -> Dict[str, int]:
        sizes = {}
        for lang in self.languages:
            size = self.language_id_ranges[lang]["size_excluding_shared_specials"]
            if include_shared_specials:
                size += len(self.special_tokens)
            sizes[lang] = size
        return sizes

    def token_to_global_id(self, token: str, language: Optional[str] = None) -> Optional[int]:
        if token in self.token_to_id:
            return self.token_to_id[token]
        if language is not None:
            visible_lang = self.language_tags.get(language, language)
            return self.token_to_id.get(f"{visible_lang}::{token}")
        return None

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        with open(os.path.join(output_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "unk_token": self.unk_token,
                    "pad_token": self.pad_token,
                    "cls_token": self.cls_token,
                    "sep_token": self.sep_token,
                    "mask_token": self.mask_token,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tokenizer_class": "NOOVERLAPTokenizer",
                    "strategy": "NOOVERLAP",
                    "requires_language_for_encoding": True,
                    "special_tokens": self.special_tokens,
                    "languages": self.languages,
                    "language_tags": self.language_tags,
                    "pad_token_id": self.pad_token_id,
                    "unk_token_id": self.unk_token_id,
                    "model_max_length": 256,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        metadata = {
            "strategy": "NOOVERLAP",
            "description": (
                "Disjoint language-specific vocabulary segments. Only special tokens are shared. "
                "Monolingual local token IDs are remapped to global NOOVERLAP IDs before training."
            ),
            "languages": self.languages,
            "language_tags": self.language_tags,
            "special_tokens": self.special_tokens,
            "shared_special_token_ids": {
                token: self.token_to_id[token] for token in self.special_tokens
            },
            "vocab_size": self.get_vocab_size(),
            "vocab_per_language_excluding_shared_specials": self.get_vocab_size_per_language(
                include_shared_specials=False
            ),
            "vocab_per_language_including_shared_specials": self.get_vocab_size_per_language(
                include_shared_specials=True
            ),
            "language_id_ranges": self.language_id_ranges,
            "note": (
                "The visible vocab strings are prefixed with language tags for metadata only. "
                "The model input is global integer IDs."
            ),
        }

        with open(os.path.join(output_dir, "nooverlap_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def save_with_assets(self, output_dir: str, tokenizer_asset_dir: str = ".") -> List[str]:
        """
        Save metadata and copy the monolingual tokenizer JSON files needed to
        reconstruct this NOOVERLAP tokenizer later.
        """
        self.save(output_dir)

        copied = []
        for lang in self.languages:
            src = Path(tokenizer_asset_dir) / f"tokenizer_{lang}.json"
            dst = Path(output_dir) / f"tokenizer_{lang}.json"
            if src.exists():
                shutil.copy2(src, dst)
                copied.append(src.name)

        utility_src = Path(tokenizer_asset_dir) / "tokenizer_utilities.py"
        if utility_src.exists():
            shutil.copy2(utility_src, Path(output_dir) / "tokenizer_utilities.py")
            copied.append("tokenizer_utilities.py")

        return copied
