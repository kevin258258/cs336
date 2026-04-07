from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full BPE experiment pipeline required by the assignment.")
    parser.add_argument("--base-dir", default="BPEresult")
    parser.add_argument("--tiny-train", default="data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--tiny-valid", default="data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--owt-train", default="data/owt_train.txt")
    parser.add_argument("--owt-valid", default="data/owt_valid.txt")
    parser.add_argument("--sample-docs", type=int, default=10)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run_step(args: list[str], skip: bool = False) -> None:
    print(f"$ {' '.join(args)}")
    if skip:
        print("skipped")
        return
    subprocess.run(args, check=True)


def main() -> None:
    args = parse_args()
    python = sys.executable
    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    tiny_dir = base_dir / "tinystories_10k"
    owt_dir = base_dir / "owt_32k"

    run_step(
        [
            python,
            "run_bpe_experiment.py",
            "--input",
            args.tiny_train,
            "--vocab-size",
            "10000",
            "--name",
            tiny_dir.name,
            "--sample-docs",
            str(args.sample_docs),
        ],
        skip=args.skip_existing and (tiny_dir / "experiment.json").exists(),
    )
    run_step(
        [
            python,
            "run_bpe_experiment.py",
            "--input",
            args.owt_train,
            "--vocab-size",
            "32000",
            "--name",
            owt_dir.name,
            "--sample-docs",
            str(args.sample_docs),
        ],
        skip=args.skip_existing and (owt_dir / "experiment.json").exists(),
    )

    encode_jobs = [
        (tiny_dir, args.tiny_train, tiny_dir / "tokens_train.npy"),
        (tiny_dir, args.tiny_valid, tiny_dir / "tokens_valid.npy"),
        (owt_dir, args.owt_train, owt_dir / "tokens_train.npy"),
        (owt_dir, args.owt_valid, owt_dir / "tokens_valid.npy"),
    ]
    for tokenizer_dir, input_path, output_path in encode_jobs:
        run_step(
            [
                python,
                "prepare_bpe_dataset.py",
                "--tokenizer-dir",
                str(tokenizer_dir),
                "--input",
                input_path,
                "--output",
                str(output_path),
            ],
            skip=args.skip_existing and output_path.exists(),
        )

    run_step(
        [
            python,
            "analyze_bpe_experiments.py",
            "--tiny-tokenizer-dir",
            str(tiny_dir),
            "--owt-tokenizer-dir",
            str(owt_dir),
            "--tiny-sample-input",
            args.tiny_valid,
            "--owt-sample-input",
            args.owt_valid,
            "--sample-docs",
            str(args.sample_docs),
            "--output-dir",
            str(base_dir / "analysis"),
        ]
    )


if __name__ == "__main__":
    main()
