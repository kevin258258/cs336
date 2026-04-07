from __future__ import annotations

import json

import numpy as np

from cs336_basics.bpe_experiments import (
    compare_tokenizers_on_documents,
    compute_compression_stats,
    estimate_processing_time,
    export_token_ids,
    load_tokenizer_artifacts,
    save_tokenizer_artifacts,
)
from cs336_basics.tokenizer import Tokenizer


def test_save_and_load_tokenizer_artifacts(tmp_path):
    vocab = {0: b"<|endoftext|>", 1: b"a", 2: b"b", 3: b"ab"}
    merges = [(b"a", b"b")]
    special_tokens = ["<|endoftext|>"]

    save_tokenizer_artifacts(
        output_dir=tmp_path,
        vocab=vocab,
        merges=merges,
        special_tokens=special_tokens,
        metadata={"corpus": "tiny"},
    )

    loaded_vocab, loaded_merges, loaded_special_tokens, metadata = load_tokenizer_artifacts(tmp_path)
    assert loaded_vocab == vocab
    assert loaded_merges == merges
    assert loaded_special_tokens == special_tokens
    assert metadata["corpus"] == "tiny"

    with open(tmp_path / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["special_tokens"] == special_tokens


def test_export_token_ids_writes_uint16(tmp_path):
    vocab = {0: b"<|endoftext|>", 1: b"a", 2: b"b", 3: b"ab"}
    merges = [(b"a", b"b")]
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    input_path = tmp_path / "toy.txt"
    input_path.write_text("ab<|endoftext|>ab", encoding="utf-8")
    output_path = tmp_path / "toy.npy"

    token_count = export_token_ids(tokenizer, input_path, output_path)
    array = np.load(output_path)

    assert token_count == 3
    assert array.dtype == np.uint16
    assert array.tolist() == [3, 0, 3]


def test_compute_compression_stats():
    vocab = {0: b"<|endoftext|>", 1: b"a", 2: b"b", 3: b"ab"}
    merges = [(b"a", b"b")]
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    stats = compute_compression_stats(tokenizer, ["ab", "ab<|endoftext|>"])

    assert stats["total_bytes"] == len("ab".encode("utf-8")) + len("ab<|endoftext|>".encode("utf-8"))
    assert stats["total_tokens"] == 3
    assert stats["bytes_per_token"] > 0


def test_compare_tokenizers_and_estimate_processing_time():
    vocab_a = {0: b"<|endoftext|>", 1: b"a", 2: b"b", 3: b"ab"}
    merges_a = [(b"a", b"b")]
    tokenizer_a = Tokenizer(vocab_a, merges_a, special_tokens=["<|endoftext|>"])

    vocab_b = {0: b"<|endoftext|>", 1: b"a", 2: b"b"}
    merges_b: list[tuple[bytes, bytes]] = []
    tokenizer_b = Tokenizer(vocab_b, merges_b, special_tokens=["<|endoftext|>"])

    comparison = compare_tokenizers_on_documents(
        {"merged": tokenizer_a, "bytes": tokenizer_b},
        ["ab", "ab"],
    )
    assert comparison["merged"]["total_tokens"] < comparison["bytes"]["total_tokens"]

    estimate = estimate_processing_time(bytes_per_second=100.0, total_bytes=1000)
    assert estimate["seconds"] == 10.0
    assert estimate["hours"] == 10.0 / 3600.0
