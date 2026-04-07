from __future__ import annotations

import argparse
import json
from pathlib import Path

from cs336_basics.bpe_experiments import export_token_ids, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode a corpus with a saved BPE tokenizer.")
    parser.add_argument("--tokenizer-dir", required=True, help="Tokenizer artifact directory under BPEresult/")
    parser.add_argument("--input", required=True, help="Input text file")
    parser.add_argument("--output", required=True, help="Output .npy file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer_dir)
    token_count = export_token_ids(tokenizer, args.input, args.output)
    summary = {
        "tokenizer_dir": str(Path(args.tokenizer_dir)),
        "input": str(Path(args.input)),
        "output": str(Path(args.output)),
        "token_count": token_count,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
