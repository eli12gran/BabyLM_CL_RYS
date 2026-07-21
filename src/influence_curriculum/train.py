from __future__ import annotations
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_scheduler

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **_):  # type: ignore[misc]
        return it


@dataclass
class TrainingConfig:
    epochs: int = 10
    effective_batch_size: int = 2048
    per_device_batch_size: int = 32
    learning_rate: float = 7e-4
    lr_scheduler: str = "cosine"
    adam_betas: tuple = (0.9, 0.98)
    adam_eps: float = 1e-6
    weight_decay: float = 0.01
    max_seq_len: int = 256
    fp16: bool = False


def train_surrogate(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    output_dir: str,
    config: TrainingConfig,
    seed: int,
    device: str,
) -> list[str]:
    torch.manual_seed(seed)
    model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)

    # Pre-pad to fixed length → TensorDataset with no per-batch collation overhead
    batch_out = tokenizer(
        texts,
        truncation=True,
        max_length=config.max_seq_len,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = batch_out["input_ids"]       # (N, L)
    attn_mask = batch_out["attention_mask"]  # (N, L)
    dataset = TensorDataset(input_ids, attn_mask)

    model = model.to(device)
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.adam_betas,
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )
    grad_accum = max(1, config.effective_batch_size // config.per_device_batch_size)
    steps_per_epoch = max(1, math.ceil(len(dataset) / config.per_device_batch_size))
    total_steps = max(1, config.epochs * math.ceil(steps_per_epoch / grad_accum))
    scheduler = get_scheduler(
        config.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=0, num_training_steps=total_steps,
    )

    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths: list[str] = []

    pin = device == "cuda"
    indices = list(range(len(dataset)))
    rng = random.Random(seed)

    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)
    pbar = tqdm(total=total_steps, desc="surrogate", unit="step")

    for epoch in range(config.epochs):
        rng.shuffle(indices)
        loader = DataLoader(
            dataset,
            batch_size=config.per_device_batch_size,
            sampler=indices,
            pin_memory=pin,
        )
        optimizer.zero_grad()
        running_loss, accum_count = 0.0, 0

        for step, (ids, mask) in enumerate(loader):
            ids  = ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels = ids.clone()
            labels[mask == 0] = -100   # ignore padding in loss
            with torch.cuda.amp.autocast(enabled=config.fp16):
                outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            scaler.scale(outputs.loss / grad_accum).backward()
            running_loss += outputs.loss.item()
            accum_count += 1

            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                pbar.update(1)
                pbar.set_postfix(epoch=epoch, loss=f"{running_loss/accum_count:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
                running_loss, accum_count = 0.0, 0

        if len(loader) % grad_accum != 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            pbar.update(1)

        path = str(ckpt_dir / f"epoch_{epoch:02d}")
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        checkpoint_paths.append(path)
        tqdm.write(f"epoch {epoch:02d} done — checkpoint saved to {path}")

    pbar.close()
    model.eval()
    optimizer.zero_grad()
    return checkpoint_paths


def train_curriculum(
    model: torch.nn.Module,
    tokenizer,
    phases: list[tuple[str, int]],
    output_dir: str,
    config: TrainingConfig,
    seed: int,
    device: str,
    start_phase: int = 0,
    start_epoch: int = 0,
    hub_repo: str | None = None,
    hub_token: str | None = None,
) -> tuple[list[str], list[dict]]:
    """Train a model through curriculum phases with per-epoch checkpointing.

    phases: [(jsonl_path, n_epochs), ...] — each phase trains on its JSONL for n_epochs.
    start_phase / start_epoch: resume by skipping already-completed work.
    hub_repo: if set, push each checkpoint after saving.
    Returns (checkpoint_paths, history) where history entries are
    {"phase": int, "epoch": int, "loss": float, "lr": float}.
    """
    import json as _json

    torch.manual_seed(seed)
    rng = random.Random(seed)
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.train()

    grad_accum = max(1, config.effective_batch_size // config.per_device_batch_size)
    pin = device == "cuda"

    # ── Compute total optimizer steps across ALL phases ───────────────────────
    total_steps = 0
    phase_steps: list[int] = []
    for jsonl_path, n_epochs in phases:
        texts = [_json.loads(l)["text"] for l in Path(jsonl_path).read_text().splitlines() if l.strip()]
        n_docs = len(texts)
        steps = max(1, n_epochs * math.ceil(math.ceil(n_docs / config.per_device_batch_size) / grad_accum))
        phase_steps.append(steps)
        total_steps += steps

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.adam_betas,
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )
    scheduler = get_scheduler(
        config.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=0, num_training_steps=max(1, total_steps),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)

    # ── Fast-forward scheduler to resume point ────────────────────────────────
    completed_steps = 0
    for p_idx in range(start_phase):
        for _ in range(phase_steps[p_idx]):
            scheduler.step()
            completed_steps += 1
    # fast-forward within start_phase for completed epochs
    if start_phase < len(phases):
        jsonl_path, _ = phases[start_phase]
        texts_fp = [_json.loads(l)["text"] for l in Path(jsonl_path).read_text().splitlines() if l.strip()]
        n_docs_fp = len(texts_fp)
        steps_per_epoch_fp = max(1, math.ceil(math.ceil(n_docs_fp / config.per_device_batch_size) / grad_accum))
        for _ in range(start_epoch * steps_per_epoch_fp):
            scheduler.step()
            completed_steps += 1

    pbar = tqdm(total=total_steps, initial=completed_steps, desc="curriculum", unit="step")

    checkpoint_paths: list[str] = []
    history: list[dict] = []

    for p_idx, (jsonl_path, n_epochs) in enumerate(phases):
        if p_idx < start_phase:
            continue

        texts = [_json.loads(l)["text"] for l in Path(jsonl_path).read_text().splitlines() if l.strip()]
        batch_out = tokenizer(
            texts,
            truncation=True,
            max_length=config.max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        dataset = TensorDataset(batch_out["input_ids"], batch_out["attention_mask"])
        indices = list(range(len(dataset)))

        epoch_start = start_epoch if p_idx == start_phase else 0

        for epoch in range(epoch_start, n_epochs):
            rng.shuffle(indices)
            loader = DataLoader(
                dataset,
                batch_size=config.per_device_batch_size,
                sampler=indices,
                pin_memory=pin,
            )
            optimizer.zero_grad()
            running_loss, accum_count = 0.0, 0
            epoch_total_loss, epoch_total_count = 0.0, 0

            for step, (ids, mask) in enumerate(loader):
                ids  = ids.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                labels = ids.clone()
                labels[mask == 0] = -100
                with torch.cuda.amp.autocast(enabled=config.fp16):
                    outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
                scaler.scale(outputs.loss / grad_accum).backward()
                running_loss += outputs.loss.item()
                accum_count += 1
                epoch_total_loss += outputs.loss.item()
                epoch_total_count += 1

                if (step + 1) % grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    pbar.update(1)
                    pbar.set_postfix(
                        phase=p_idx, epoch=epoch,
                        loss=f"{running_loss/accum_count:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )
                    running_loss, accum_count = 0.0, 0

            if len(loader) % grad_accum != 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                pbar.update(1)

            epoch_loss = epoch_total_loss / epoch_total_count if epoch_total_count else float("nan")
            current_lr = scheduler.get_last_lr()[0]
            history.append({"phase": p_idx, "epoch": epoch, "loss": epoch_loss, "lr": current_lr})

            ckpt_name = f"phase_{p_idx:02d}_epoch_{epoch:02d}"
            ckpt_path = str(ckpt_dir / ckpt_name)
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            checkpoint_paths.append(ckpt_path)
            tqdm.write(f"phase {p_idx:02d} epoch {epoch:02d} done — saved {ckpt_name}  loss={epoch_loss:.4f}")

            if hub_repo:
                try:
                    from huggingface_hub import HfApi
                    HfApi().upload_folder(
                        folder_path=ckpt_path,
                        repo_id=hub_repo,
                        path_in_repo=ckpt_name,
                        repo_type="model",
                        token=hub_token,
                        commit_message=f"checkpoint {ckpt_name}",
                    )
                    tqdm.write(f"  → pushed {ckpt_name} to {hub_repo}")
                except Exception as e:
                    tqdm.write(f"  ⚠ Hub push failed: {e}")

    pbar.close()
    model.eval()
    return checkpoint_paths, history


def train_word_aware(
    model: torch.nn.Module,
    tokenizer,
    phases: list[tuple[str, int]],
    output_dir: str,
    config: TrainingConfig,
    seed: int,
    device: str,
    word_checkpoints: list[int],
    checkpoint_names: dict[int, str] | None = None,
    hub_repo: str | None = None,
    hub_token: str | None = None,
    start_words: int = 0,
    start_phase: int = 0,
    start_epoch: int = 0,
    total_word_target: int | None = None,
) -> tuple[list[str], list[dict]]:
    """Train with per-epoch checkpoints + word-count milestone checkpoints.

    word_checkpoints: sorted list of cumulative word counts at which to save
        a revision checkpoint named chck_{N} (e.g. [1_000_000, ..., 10_000_000]).
    start_words: words already processed before this call (for resume).
    total_word_target: if set, stop when this many cumulative words have been
        processed. Extends the last phase with extra epochs if the scheduled
        epochs don't reach the target; stops early if they overshoot it.
    Returns (checkpoint_paths, history) where checkpoint_paths includes both
    per-epoch paths (phase_XX_epoch_YY) and milestone paths (chck_{N}).
    history entries: {phase, epoch, loss, lr, words_processed}.
    """
    import json as _json

    torch.manual_seed(seed)
    rng = random.Random(seed)
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.train()

    grad_accum = max(1, config.effective_batch_size // config.per_device_batch_size)
    pin = device == "cuda"

    # ── Count words per phase epoch (needed for target estimation) ────────────
    phase_words_per_epoch: list[int] = []
    phase_doc_counts: list[int] = []
    for jsonl_path, _ in phases:
        lines = [l for l in Path(jsonl_path).read_text().splitlines() if l.strip()]
        n_docs = len(lines)
        total_words = sum(len(_json.loads(l)["text"].split()) for l in lines)
        phase_words_per_epoch.append(total_words)
        phase_doc_counts.append(n_docs)

    # ── Compute effective epochs per phase ────────────────────────────────────
    specified_epochs = [n for _, n in phases]
    if total_word_target is not None and total_word_target > start_words:
        remaining = total_word_target - start_words
        # Count only words from epochs not yet completed at the resume point.
        scheduled_remaining = 0
        for p_idx, (wpe, ne) in enumerate(zip(phase_words_per_epoch, specified_epochs)):
            if p_idx < start_phase:
                continue
            epoch_from = start_epoch if p_idx == start_phase else 0
            scheduled_remaining += wpe * max(0, ne - epoch_from)
        if scheduled_remaining < remaining:
            extra = math.ceil((remaining - scheduled_remaining) / max(1, phase_words_per_epoch[-1]))
            effective_epochs = list(specified_epochs)
            # If resuming mid-last-phase (start_epoch > specified), the loop starts at
            # start_epoch, so effective_epochs[-1] must be start_epoch + extra, not
            # specified[-1] + extra (which would give fewer actual epochs than needed).
            last_epoch_start = start_epoch if start_phase == len(phases) - 1 else 0
            effective_epochs[-1] = max(specified_epochs[-1], last_epoch_start) + extra
        else:
            effective_epochs = list(specified_epochs)
    else:
        effective_epochs = list(specified_epochs)

    # ── Total optimizer steps across all phases (for LR scheduler) ───────────
    total_steps = 0
    phase_steps: list[int] = []
    for n_docs, n_eff in zip(phase_doc_counts, effective_epochs):
        steps = max(1, n_eff * math.ceil(math.ceil(n_docs / config.per_device_batch_size) / grad_accum))
        phase_steps.append(steps)
        total_steps += steps

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.adam_betas,
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )
    scheduler = get_scheduler(
        config.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=0, num_training_steps=max(1, total_steps),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)

    # Fast-forward scheduler to resume point
    completed_steps = 0
    for p_idx in range(start_phase):
        for _ in range(phase_steps[p_idx]):
            scheduler.step()
            completed_steps += 1
    if start_phase < len(phases):
        n_docs_fp = phase_doc_counts[start_phase]
        steps_per_epoch_fp = max(1, math.ceil(math.ceil(n_docs_fp / config.per_device_batch_size) / grad_accum))
        for _ in range(start_epoch * steps_per_epoch_fp):
            scheduler.step()
            completed_steps += 1

    # Milestones still to hit (skip those already passed by start_words)
    pending = sorted(m for m in word_checkpoints if m > start_words)

    pbar = tqdm(total=total_steps, initial=completed_steps, desc="train", unit="step")

    checkpoint_paths: list[str] = []
    history: list[dict] = []
    words_so_far = start_words

    for p_idx, (jsonl_path, _) in enumerate(phases):
        if p_idx < start_phase:
            continue

        lines = [l for l in Path(jsonl_path).read_text().splitlines() if l.strip()]
        texts = [_json.loads(l)["text"] for l in lines]
        word_counts = [len(t.split()) for t in texts]

        batch_out = tokenizer(
            texts,
            truncation=True,
            max_length=config.max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        dataset = TensorDataset(batch_out["input_ids"], batch_out["attention_mask"])
        indices = list(range(len(dataset)))

        epoch_start = start_epoch if p_idx == start_phase else 0
        n_eff = effective_epochs[p_idx]

        for epoch in range(epoch_start, n_eff):
            if total_word_target is not None and words_so_far >= total_word_target:
                break

            rng.shuffle(indices)
            loader = DataLoader(
                dataset,
                batch_size=config.per_device_batch_size,
                sampler=indices,
                pin_memory=pin,
            )
            optimizer.zero_grad()
            running_loss, accum_count = 0.0, 0
            epoch_total_loss, epoch_total_count = 0.0, 0
            B = config.per_device_batch_size

            for step, (ids, mask) in enumerate(loader):
                # Count words in this batch using original texts
                batch_indices = indices[step * B : step * B + ids.shape[0]]
                words_so_far += sum(word_counts[i] for i in batch_indices)

                ids  = ids.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                labels = ids.clone()
                labels[mask == 0] = -100
                with torch.cuda.amp.autocast(enabled=config.fp16):
                    outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
                scaler.scale(outputs.loss / grad_accum).backward()
                running_loss  += outputs.loss.item()
                accum_count   += 1
                epoch_total_loss  += outputs.loss.item()
                epoch_total_count += 1

                if (step + 1) % grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    pbar.update(1)
                    pbar.set_postfix(
                        phase=p_idx, epoch=epoch,
                        words=f"{words_so_far:,}",
                        loss=f"{running_loss/accum_count:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )
                    running_loss, accum_count = 0.0, 0

                # Save milestone checkpoints as we cross word thresholds
                while pending and words_so_far >= pending[0]:
                    milestone = pending.pop(0)
                    ckpt_name = (checkpoint_names or {}).get(milestone, f"chck_{milestone}")
                    ckpt_path = str(ckpt_dir / ckpt_name)
                    model.save_pretrained(ckpt_path)
                    tokenizer.save_pretrained(ckpt_path)
                    _fix_tokenizer_config(ckpt_path)
                    checkpoint_paths.append(ckpt_path)
                    tqdm.write(f"  milestone {ckpt_name} at {words_so_far:,} words")
                    if hub_repo:
                        _push_revision(hub_repo, hub_token, ckpt_path, ckpt_name)

            if len(loader) % grad_accum != 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                pbar.update(1)

            epoch_loss = epoch_total_loss / epoch_total_count if epoch_total_count else float("nan")
            current_lr = scheduler.get_last_lr()[0]
            history.append({
                "phase": p_idx,
                "epoch": epoch,
                "loss": epoch_loss,
                "lr": current_lr,
                "words_processed": words_so_far,
            })

            # Per-epoch checkpoint (folder on main, not a revision)
            ckpt_name = f"phase_{p_idx:02d}_epoch_{epoch:02d}"
            ckpt_path = str(ckpt_dir / ckpt_name)
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            _fix_tokenizer_config(ckpt_path)
            checkpoint_paths.append(ckpt_path)
            tqdm.write(f"phase {p_idx:02d} epoch {epoch:02d} done — {words_so_far:,} words  loss={epoch_loss:.4f}")

            if hub_repo:
                try:
                    from huggingface_hub import HfApi
                    HfApi().upload_folder(
                        folder_path=ckpt_path,
                        repo_id=hub_repo,
                        path_in_repo=ckpt_name,
                        repo_type="model",
                        token=hub_token,
                        commit_message=f"checkpoint {ckpt_name}  words={words_so_far:,}",
                    )
                except Exception as e:
                    tqdm.write(f"  ⚠ Hub push failed: {e}")

        if total_word_target is not None and words_so_far >= total_word_target:
            break

    pbar.close()
    model.eval()
    return checkpoint_paths, history


def _fix_tokenizer_config(checkpoint_path: str) -> None:
    """Patch tokenizer_config.json to use a standard HF tokenizer class.

    Some tokenizer wrappers (e.g. TokenizersBackend) save a non-standard
    tokenizer_class that AutoProcessor cannot resolve. Normalise it to
    PreTrainedTokenizerFast so the competition eval pipeline works.
    """
    import json as _json
    cfg_path = Path(checkpoint_path) / "tokenizer_config.json"
    if not cfg_path.exists():
        return
    cfg = _json.loads(cfg_path.read_text())
    standard = {"PreTrainedTokenizerFast", "GPT2Tokenizer", "GPT2TokenizerFast", None}
    if cfg.get("tokenizer_class") not in standard:
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    for k in ("backend", "is_local", "local_files_only"):
        cfg.pop(k, None)
    cfg_path.write_text(_json.dumps(cfg, indent=2))


def _push_revision(hub_repo: str, hub_token: str | None, local_path: str, revision: str) -> None:
    """Push a checkpoint as a named git revision (branch) on the HF model repo."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # Create the branch if it doesn't exist, then upload to it
        try:
            api.create_branch(repo_id=hub_repo, branch=revision,
                              repo_type="model", token=hub_token)
        except Exception:
            pass  # branch already exists — fine, we'll overwrite
        api.upload_folder(
            folder_path=local_path,
            repo_id=hub_repo,
            path_in_repo=".",
            repo_type="model",
            revision=revision,
            token=hub_token,
            commit_message=f"milestone {revision}",
        )
        tqdm.write(f"  → pushed revision {revision} to {hub_repo}")
    except Exception as e:
        tqdm.write(f"  ⚠ Hub revision push failed ({revision}): {e}")
