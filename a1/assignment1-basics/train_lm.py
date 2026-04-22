from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.Function import (
    AdamW,
    cross_entropy,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)
from cs336_basics.model import TransformerLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer LM for CS336 A1.")

    parser.add_argument("--train-data", type=Path, required=True, help="Path to tokenized train data (.npy or raw uint16).")
    parser.add_argument("--val-data", type=Path, required=True, help="Path to tokenized validation data (.npy or raw uint16).")
    parser.add_argument("--data-format", choices=["auto", "npy", "bin"], default="auto")
    parser.add_argument("--data-dtype", default="uint16", help="NumPy dtype for raw binary memmap mode.")

    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1360)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    parser.add_argument("--max-learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--cosine-cycle-iters", type=int, default=20_000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume-from", type=Path, default=None, help="Checkpoint path to resume from.")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="cs336-a1")
    parser.add_argument("--wandb-run-name", default=None)

    return parser.parse_args()


def _resolve_data_format(path: Path, mode: str) -> str:
    if mode != "auto":
        return mode
    return "npy" if path.suffix == ".npy" else "bin"


def load_token_array(path: Path, data_format: str, data_dtype: str) -> np.ndarray:
    fmt = _resolve_data_format(path, data_format)
    if fmt == "npy":
        return np.load(path, mmap_mode="r")
    return np.memmap(path, dtype=np.dtype(data_dtype), mode="r")


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def resolve_device(requested_device: str) -> str:
    if requested_device.startswith("cuda"):
        try:
            # Probe CUDA kernel usability, not just torch.cuda.is_available().
            _ = torch.zeros(1, device=requested_device)
            return requested_device
        except Exception as exc:
            print(f"[warn] CUDA device '{requested_device}' is not usable: {exc}")
            print("[warn] Falling back to CPU. Upgrade PyTorch CUDA build for this GPU to use CUDA training.")
            return "cpu"
    return requested_device


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    val_data: np.ndarray,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: str,
    num_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(num_batches):
        x, y = get_batch(val_data, batch_size=batch_size, context_length=context_length, device=device)
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(len(losses), 1))


def maybe_init_wandb(args: argparse.Namespace) -> Any | None:
    if not args.wandb:
        return None
    try:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
        return wandb
    except Exception as exc:
        print(f"[warn] wandb init failed, continue without wandb: {exc}")
        return None


def save_latest_checkpoint(
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    latest = out_dir / "latest.pt"
    numbered = out_dir / f"step_{step:08d}.pt"
    save_checkpoint(model=model, optimizer=optimizer, iteration=step, out=latest)
    save_checkpoint(model=model, optimizer=optimizer, iteration=step, out=numbered)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    train_data = load_token_array(args.train_data, args.data_format, args.data_dtype)
    val_data = load_token_array(args.val_data, args.data_format, args.data_dtype)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.max_learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume_from is not None and args.resume_from.exists():
        start_step = load_checkpoint(args.resume_from, model=model, optimizer=optimizer)
        print(f"[info] resumed from {args.resume_from}, start at step {start_step}")

    wandb = maybe_init_wandb(args)
    model.train()

    for step in range(start_step, args.steps):
        lr = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=args.max_learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )
        set_optimizer_lr(optimizer, lr)

        x, y = get_batch(train_data, batch_size=args.batch_size, context_length=args.context_length, device=device)

        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, args.vocab_size), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if step % args.log_interval == 0:
            print(f"step={step:>7d} train_loss={loss.item():.6f} lr={lr:.6e}")
            if wandb is not None:
                wandb.log({"step": step, "train_loss": loss.item(), "lr": lr})

        if step % args.eval_interval == 0:
            val_loss = evaluate(
                model=model,
                val_data=val_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                vocab_size=args.vocab_size,
                device=device,
                num_batches=args.eval_batches,
            )
            print(f"step={step:>7d} val_loss={val_loss:.6f}")
            if wandb is not None:
                wandb.log({"step": step, "val_loss": val_loss})

        if step % args.save_interval == 0:
            save_latest_checkpoint(model=model, optimizer=optimizer, step=step, out_dir=args.out_dir)

    final_step = max(args.steps - 1, 0)
    save_latest_checkpoint(model=model, optimizer=optimizer, step=final_step, out_dir=args.out_dir)
    print(f"[done] checkpoints saved in {args.out_dir}")

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
