from __future__ import annotations

import argparse
import json
from pathlib import Path

from cs336_basics.bpe_experiments import (
    compare_tokenizers_on_documents,
    estimate_processing_time,
    load_tokenizer,
    measure_encode_throughput,
    sample_documents,
)


PILE_BYTES = 825 * 10**9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze trained BPE tokenizers for assignment experiments.")
    parser.add_argument("--tiny-tokenizer-dir", required=True)
    parser.add_argument("--owt-tokenizer-dir", required=True)
    parser.add_argument("--tiny-sample-input", required=True)
    parser.add_argument("--owt-sample-input", required=True)
    parser.add_argument("--sample-docs", type=int, default=10)
    parser.add_argument("--output-dir", default="BPEresult/analysis")
    return parser.parse_args()


def _write_markdown(output_path: Path, analysis: dict) -> None:
    tiny_tiny = analysis["compression"]["tinystories_docs"]["tinystories_tokenizer"]["bytes_per_token"]
    owt_owt = analysis["compression"]["owt_docs"]["owt_tokenizer"]["bytes_per_token"]
    tiny_owt = analysis["compression"]["owt_docs"]["tinystories_tokenizer"]["bytes_per_token"]
    owt_tiny = analysis["compression"]["tinystories_docs"]["owt_tokenizer"]["bytes_per_token"]
    tiny_hours = analysis["throughput"]["tinystories_tokenizer"]["pile_estimate"]["hours"]
    owt_hours = analysis["throughput"]["owt_tokenizer"]["pile_estimate"]["hours"]

    lines = [
        "# BPE Experiment Summary",
        "",
        "## 2.5",
        f"- TinyStories 10k tokenizer summary: see `BPEresult/tinystories_10k/experiment.json`.",
        f"- OWT 32k tokenizer summary: see `BPEresult/owt_32k/experiment.json`.",
        "",
        "## 2.7",
        f"- TinyStories sample compressed to {tiny_tiny:.4f} bytes/token with the TinyStories tokenizer; the OWT sample compressed to {owt_owt:.4f} bytes/token with the OWT tokenizer.",
        f"- Using the TinyStories tokenizer on OWT degrades compression to {tiny_owt:.4f} bytes/token; using the OWT tokenizer on TinyStories gives {owt_tiny:.4f} bytes/token.",
        f"- Estimated 825GB tokenization time from sampled throughput: TinyStories tokenizer {tiny_hours:.2f} hours, OWT tokenizer {owt_hours:.2f} hours.",
        "- Encoded train/valid arrays are stored as `uint16` because both vocabularies are below 65536 entries.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiny_tokenizer = load_tokenizer(args.tiny_tokenizer_dir)
    owt_tokenizer = load_tokenizer(args.owt_tokenizer_dir)
    tiny_docs = sample_documents(args.tiny_sample_input, args.sample_docs)
    owt_docs = sample_documents(args.owt_sample_input, args.sample_docs)

    compression = {
        "tinystories_docs": compare_tokenizers_on_documents(
            {
                "tinystories_tokenizer": tiny_tokenizer,
                "owt_tokenizer": owt_tokenizer,
            },
            tiny_docs,
        ),
        "owt_docs": compare_tokenizers_on_documents(
            {
                "tinystories_tokenizer": tiny_tokenizer,
                "owt_tokenizer": owt_tokenizer,
            },
            owt_docs,
        ),
    }

    tiny_throughput = measure_encode_throughput(tiny_tokenizer, "".join(tiny_docs + owt_docs))
    owt_throughput = measure_encode_throughput(owt_tokenizer, "".join(tiny_docs + owt_docs))
    throughput = {
        "tinystories_tokenizer": {
            **tiny_throughput,
            "pile_estimate": estimate_processing_time(tiny_throughput["bytes_per_second"], PILE_BYTES),
        },
        "owt_tokenizer": {
            **owt_throughput,
            "pile_estimate": estimate_processing_time(owt_throughput["bytes_per_second"], PILE_BYTES),
        },
    }

    analysis = {
        "sample_documents": {
            "tinystories_docs": len(tiny_docs),
            "owt_docs": len(owt_docs),
        },
        "compression": compression,
        "throughput": throughput,
    }

    with open(output_dir / "analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    _write_markdown(output_dir / "analysis.md", analysis)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
