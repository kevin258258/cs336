from __future__ import annotations

import json
import pickle
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

from .tokenizer import Tokenizer, train_bpe


DEFAULT_EOT = "<|endoftext|>"


def _ensure_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def save_tokenizer_artifacts(
    output_dir: str | Path,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
    metadata: dict[str, Any] | None = None,
) -> None:
    output_path = _ensure_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(output_path / "merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    summary = dict(metadata or {})
    summary["special_tokens"] = list(special_tokens)
    with open(output_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def load_tokenizer_artifacts(
    output_dir: str | Path,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]], list[str], dict[str, Any]]:
    output_path = _ensure_path(output_dir)
    with open(output_path / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(output_path / "merges.pkl", "rb") as f:
        merges = pickle.load(f)
    with open(output_path / "summary.json", encoding="utf-8") as f:
        metadata = json.load(f)
    special_tokens = list(metadata.get("special_tokens", []))
    return vocab, merges, special_tokens, metadata


def load_tokenizer(output_dir: str | Path) -> Tokenizer:
    vocab, merges, special_tokens, _ = load_tokenizer_artifacts(output_dir)
    return Tokenizer(vocab, merges, special_tokens=special_tokens)


def get_longest_token(vocab: dict[int, bytes]) -> bytes:
    return max(vocab.values(), key=lambda token: (len(token), token))


def sample_documents(
    input_path: str | Path,
    num_documents: int,
    delimiter: str = DEFAULT_EOT,
) -> list[str]:
    text = _ensure_path(input_path).read_text(encoding="utf-8")
    documents = [doc for doc in text.split(delimiter) if doc]
    return documents[:num_documents]


def compute_compression_stats(tokenizer: Tokenizer, documents: list[str]) -> dict[str, float]:
    total_bytes = 0
    total_tokens = 0
    for document in documents:
        total_bytes += len(document.encode("utf-8"))
        total_tokens += len(tokenizer.encode(document))
    bytes_per_token = total_bytes / total_tokens if total_tokens else 0.0
    return {
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "bytes_per_token": bytes_per_token,
    }


def compare_tokenizers_on_documents(
    tokenizers: dict[str, Tokenizer],
    documents: list[str],
) -> dict[str, dict[str, float]]:
    return {
        name: compute_compression_stats(tokenizer, documents)
        for name, tokenizer in tokenizers.items()
    }


def measure_encode_throughput(tokenizer: Tokenizer, text: str, repeat: int = 3) -> dict[str, float]:
    best_seconds = float("inf")
    total_bytes = len(text.encode("utf-8"))
    for _ in range(repeat):
        start = time.perf_counter()
        tokenizer.encode(text)
        elapsed = time.perf_counter() - start
        best_seconds = min(best_seconds, elapsed)
    bytes_per_second = total_bytes / best_seconds if best_seconds else float("inf")
    return {
        "bytes": total_bytes,
        "seconds": best_seconds,
        "bytes_per_second": bytes_per_second,
    }


def estimate_processing_time(bytes_per_second: float, total_bytes: int) -> dict[str, float]:
    if bytes_per_second <= 0:
        return {"bytes": total_bytes, "seconds": float("inf"), "hours": float("inf")}
    seconds = total_bytes / bytes_per_second
    return {"bytes": total_bytes, "seconds": seconds, "hours": seconds / 3600.0}


def export_token_ids(
    tokenizer: Tokenizer,
    input_path: str | Path,
    output_path: str | Path,
    dtype: np.dtype = np.uint16,
) -> int:
    text = _ensure_path(input_path).read_text(encoding="utf-8")
    token_ids = np.asarray(tokenizer.encode(text), dtype=dtype)
    output_file = _ensure_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, token_ids)
    return int(token_ids.size)


def train_and_save_tokenizer(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
    output_dir: str | Path,
) -> dict[str, Any]:
    input_file = _ensure_path(input_path)
    start = time.perf_counter()
    vocab, merges = train_bpe(input_file, vocab_size, special_tokens)
    elapsed = time.perf_counter() - start
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    longest_token = get_longest_token(vocab)

    metadata = {
        "input_path": str(input_file),
        "vocab_size": vocab_size,
        "special_tokens": list(special_tokens),
        "train_seconds": elapsed,
        "peak_rss_kb": peak_rss_kb,
        "actual_vocab_size": len(vocab),
        "num_merges": len(merges),
        "longest_token_utf8": longest_token.decode("utf-8", errors="replace"),
        "longest_token_repr": repr(longest_token),
    }
    save_tokenizer_artifacts(output_dir, vocab, merges, special_tokens, metadata)
    return metadata
