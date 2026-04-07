# 用 find_chunk_boundaries 按 <|endoftext|> 切分文件为 chunk
# 对每个 chunk：按 special token 分割文本（re.split + re.escape）
# 对每个片段：用 GPT-2 正则 re.finditer(PAT, segment) 提取 pre-token
# 将每个 pre-token 转为 UTF-8 bytes tuple："the" → (b't', b'h', b'e')
# 用 Counter 统计频次
# 用 multiprocessing.Pool 并行处理多个 chunk，合并 Counte

import os
from collections import Counter
from collections.abc import Iterable, Iterator
from multiprocessing import Pool

import regex as re  # 不是 re！pyproject.toml 里已经有了

from .pretokenization_example import find_chunk_boundaries

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pre_tokenize(input_path, start, end, special_tokens) -> Counter:
    with open(input_path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)
        chunk = raw.decode("utf-8", errors="ignore")
    if special_tokens:
        split_pattern = "|".join(re.escape(st) for st in special_tokens)
        segments = re.split(split_pattern, chunk)
    else:
        segments = [chunk]
    counts: Counter = Counter()
    for segment in segments:
        if not segment:
            continue
        else:
            for match in re.finditer(GPT2_PAT, segment):
                token_str = match.group()
                encode = token_str.encode("utf-8")
                byte_tuple = tuple(bytes([b]) for b in encode)
                counts[byte_tuple] += 1

    return counts


# 写个merge的辅助函数


def merge_words(word, pair) -> tuple:
    new_word = []
    a, b = pair[0], pair[1]
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            new_word.append(a + b)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def train_bpe(input_path, vocab_size, special_tokens):
    input_path = str(input_path)  # PathLike 转 str，方便传给子进程
    num_processes = os.cpu_count() or 4

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # 构造参数列表：每个 chunk 的 (path, start, end, special_tokens)
    args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with Pool(num_processes) as pool:
        results = pool.starmap(pre_tokenize, args)

    # 合并各进程的 Counter
    word_freqs: dict[tuple[bytes, ...], int] = {}
    for counter in results:
        for word, freq in counter.items():
            word_freqs[word] = word_freqs.get(word, 0) + freq

    # 初始化vocab
    vocab: dict[int, bytes] = {}
    next_id = 0
    for st in special_tokens:
        vocab[next_id] = st.encode("utf-8")
        next_id += 1
    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1

    # 根据之前的word初始化一下pair,然后维护一个反向的索引，方便之后更新pair的时候更改相关的words
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
            if pair not in pair_to_words:
                pair_to_words[pair] = set()
            pair_to_words[pair].add(word)
    # 初始化pair和反向hash之后可以开始merge了
    # 先挑出最大的pair,索引到对应的word开始更新
    # 然后修改pair_counts ,同时也要继续更新pair-to-words
    num_merges = vocab_size - len(vocab)
    merges: list[tuple[bytes, bytes]] = []

    for _ in range(num_merges):
        if not pair_counts:
            break
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        new_token = best[0] + best[1]
        vocab[next_id] = new_token
        next_id += 1
        merges.append(best)

        affected = pair_to_words.pop(best, set()).copy()
        del pair_counts[best]
        # 这里先把原本的词表和word——freq还有P2W更新了，然后根据这个更新pair-counts
        # 具体更新方法呢，就是先移除word的影响，在加上新word的，虽然感觉还有些可以优化的地方，但是懒了（

        for word in affected:
            freq = word_freqs.pop(word)
            new_word = merge_words(word, best)
            word_freqs[new_word] = word_freqs.get(new_word, 0) + freq
            for i in range(len(word) - 1):
                p = (word[i], word[i + 1])
                if p == best:
                    continue
                pair_counts[p] -= freq
                if pair_counts[p] <= 0:
                    del pair_counts[p]
                if p in pair_to_words:
                    pair_to_words[p].discard(word)
                    if not pair_to_words[p]:
                        del pair_to_words[p]
            for j in range(len(new_word) - 1):
                p = (new_word[j], new_word[j + 1])
                pair_counts[p] = pair_counts.get(p, 0) + freq
                if p not in pair_to_words:
                    pair_to_words[p] = set()
                pair_to_words[p].add(new_word)
    return vocab, merges


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)

        if self.special_tokens:
            for st in self.special_tokens:
                st_bytes = st.encode("utf-8")
                if st_bytes not in set(self.vocab.values()):
                    new_id = max(self.vocab.keys()) + 1
                    self.vocab[new_id] = st_bytes

        self.vocab_inv: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        if self.special_tokens:
            self._special_re = re.compile("(" + "|".join(re.escape(st) for st in self.special_tokens) + ")")
        else:
            self._special_re = None

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def _apply_merges(self, tokens: list[bytes]) -> list[bytes]:
        for a, b in self.merges:
            new_tokens: list[bytes] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(a + b)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        ids: list[int] = []
        parts = self._special_re.split(text) if self._special_re else [text]
        special_set = set(self.special_tokens)

        for part in parts:
            if not part:
                continue
            if part in special_set:
                ids.append(self.vocab_inv[part.encode("utf-8")])
                continue
            for match in re.finditer(GPT2_PAT, part):
                token_str = match.group()
                byte_list = [bytes([b]) for b in token_str.encode("utf-8")]
                merged = self._apply_merges(byte_list)
                for tok in merged:
                    ids.append(self.vocab_inv[tok])
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
