"""
简单的 BPE 训练脚本，用于 profiling 和验证。
用法：
    python train_bpe_demo.py                          # 默认跑 tinystories_sample.txt
    python train_bpe_demo.py --large                  # 跑 5M 版本
    python -m cProfile -s cumtime train_bpe_demo.py   # profiling
"""

import argparse
import time
from pathlib import Path

from cs336_basics.tokenizer import train_bpe, Tokenizer

FIXTURES = Path(__file__).parent / "tests" / "fixtures"

SMALL = FIXTURES / "tinystories_sample.txt"
LARGE = FIXTURES / "tinystories_sample_5M.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true", help="用 5M 数据集")
    parser.add_argument("--vocab-size", type=int, default=500, help="词表大小")
    args = parser.parse_args()

    input_path = LARGE if args.large else SMALL
    vocab_size = args.vocab_size
    special_tokens = ["<|endoftext|>"]

    print(f"输入文件: {input_path} ({input_path.stat().st_size / 1024:.1f} KB)")
    print(f"目标词表大小: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    print("-" * 50)

    t0 = time.perf_counter()
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    t1 = time.perf_counter()

    print(f"训练耗时: {t1 - t0:.3f}s")
    print(f"词表大小: {len(vocab)}")
    print(f"Merge 数量: {len(merges)}")
    print()

    print("前 20 条 merge:")
    for i, (a, b) in enumerate(merges[:20]):
        print(f"  {i:3d}. {a!r} + {b!r} -> {a + b!r}")
    print()

    tokenizer = Tokenizer(vocab, merges, special_tokens)

    test_texts = [
        "Once upon a time",
        "Hello, how are you?",
        "The little girl was so happy.",
        "<|endoftext|>",
        "She said <|endoftext|> goodbye.",
    ]

    print("编码/解码测试:")
    for text in test_texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        roundtrip = "OK" if decoded == text else "FAIL"
        print(f"  [{roundtrip}] \"{text}\"")
        print(f"        -> {len(ids)} tokens: {ids[:15]}{'...' if len(ids) > 15 else ''}")
        if decoded != text:
            print(f"        !! decoded: \"{decoded}\"")


if __name__ == "__main__":
    main()
