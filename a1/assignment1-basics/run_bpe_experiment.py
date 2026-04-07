from __future__ import annotations

import argparse
import json
from pathlib import Path

from cs336_basics.bpe_experiments import (
    DEFAULT_EOT,
    compute_compression_stats,
    load_tokenizer,
    measure_encode_throughput,
    sample_documents,
    train_and_save_tokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer and write experiment results.")
    parser.add_argument("--input", required=True, help="Training corpus path")
    parser.add_argument("--vocab-size", required=True, type=int, help="Target vocabulary size")
    parser.add_argument("--name", required=True, help="Run name under BPEresult/")
    parser.add_argument("--special-token", action="append", default=None, help="Special token to preserve")
    parser.add_argument("--sample-docs", type=int, default=10, help="How many documents to sample for compression stats")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    special_tokens = args.special_token or [DEFAULT_EOT]
    output_dir = Path("BPEresult") / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = train_and_save_tokenizer(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        output_dir=output_dir,
    )
    tokenizer = load_tokenizer(output_dir)
    documents = sample_documents(args.input, num_documents=args.sample_docs, delimiter=special_tokens[0])
    compression_stats = compute_compression_stats(tokenizer, documents)
    throughput_stats = measure_encode_throughput(tokenizer, "".join(documents)) if documents else {
        "bytes": 0,
        "seconds": 0.0,
        "bytes_per_second": 0.0,
    }

    result = {
        "training": metadata,
        "compression_sample": compression_stats,
        "throughput_sample": throughput_stats,
        "sample_documents": len(documents),
    }

    with open(output_dir / "experiment.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
