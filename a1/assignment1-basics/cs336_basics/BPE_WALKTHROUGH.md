# BPE 实现步骤引导

按这个顺序写，每写完一步就能跑对应的测试验证。

---

## Step 1：把 `find_chunk_boundaries` 搞进来

`pretokenization_example.py` 底部有裸代码，直接 import 会炸。
两种修法任选：

- **方案 A**：把 `find_chunk_boundaries` 函数复制到 `tokenizer.py` 里
- **方案 B**：把 `pretokenization_example.py` 底部的 `with open(...)` 那段包进 `if __name__ == "__main__":`

确认能正常 `from .pretokenization_example import find_chunk_boundaries`（或直接用复制的版本）。

---

## Step 2：写 `_pretokenize_chunk` 函数

这是个**模块顶层函数**（不是类方法），因为 multiprocessing 要能 pickle 它。

```
def _pretokenize_chunk(input_path, start, end, special_tokens) -> Counter:
```

做四件事：

1. `open(input_path, "rb")` → `seek(start)` → `read(end - start)` → `.decode("utf-8", errors="ignore")`
2. 按 special_tokens 切割（`regex.split`），避免 special token 被混进正常文本
3. 对每个片段用 `regex.finditer(GPT2_PAT, segment)` 提取 pre-token
4. 每个 pre-token → `token.encode("utf-8")` → 拆成单字节 tuple → 计入 Counter

**验证**：写个小脚本手动调用一下，传个小文本文件看 Counter 输出是否合理。

---

## Step 3：写 `train_bpe` 的前半段（预分词 + 初始化）

```
def train_bpe(input_path, vocab_size, special_tokens):
```

按顺序做：

1. **并行预分词**
  - `find_chunk_boundaries(f, num_processes, b"<|endoftext|>")`
  - `Pool(num_processes).starmap(_pretokenize_chunk, args)`
  - 合并所有 Counter → `word_freqs`
2. **初始化 vocab**
  - 先放 special tokens（按顺序分配 id 0, 1, ...）
  - 再放 256 个单字节 token（`bytes([0])` 到 `bytes([255])`）
3. **初始化 pair_counts**
  - 遍历 `word_freqs`，统计所有相邻 pair 的加权频次
4. **（可选）初始化 pair_to_words**
  - 记录每个 pair 出现在哪些 word 里，后面增量更新用

到这里先停，print 一下 `len(word_freqs)` 和 `pair_counts.most_common(5)` 看看数据是否合理。

---

## Step 4：写 merge 循环

这是性能关键。在 Step 3 后面续写：

```python
for _ in range(num_merges):
    # 1. 找频次最高的 pair（并列取字典序最大）
    best = max(pair_counts, key=lambda p: (pair_counts[p], p))

    # 2. 新 token = best[0] + best[1]，加入 vocab + merges

    # 3. 增量更新（只处理受影响的 word）：
    for word in pair_to_words[best].copy():
        freq = word_freqs.pop(word)
        new_word = _merge_word(word, best)
        word_freqs[new_word] = word_freqs.get(new_word, 0) + freq

        # 减：旧 word 中的旧 pair
        # 加：new_word 中的新 pair
        # 维护 pair_to_words
```

**关键细节**：

- 增量更新时要**先减旧再加新**，顺序不能反
- 旧 pair 中跳过 `best_pair` 本身（已经删了）
- `pair_counts` 降到 0 的要 `del` 掉，避免干扰 `max()`
- `word_freqs[new_word]` 可能已有值（两个不同的 word 合并后变成同一个），要累加

**验证**：

```bash
uv run pytest tests/test_train_bpe.py -k test_train_bpe_speed -v    # 必须 < 1.5s
uv run pytest tests/test_train_bpe.py -k test_train_bpe -v          # vocab + merges 精确匹配
uv run pytest tests/test_train_bpe.py -k test_train_bpe_special -v  # special token
```

三个都过了再往下走。

---

## Step 5：写 `Tokenizer.__init__`

```python
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
```

要做的事：

1. 存 vocab、merges
2. special_tokens 按**长度降序**排列（长的优先匹配）
3. 如果某个 special token 不在 vocab 里，追加进去
4. 建反向索引 `vocab_inv: dict[bytes, int]`
5. 编译 special token 的分割正则：`"(" + "|".join(re.escape(st)) + ")"`

---

## Step 6：写 `Tokenizer.decode`

最简单的方法，先写这个：

```python
def decode(self, ids: list[int]) -> str:
    # vocab[id] 查 bytes → 全部拼接 → .decode("utf-8", errors="replace")
```

就三行代码。

---

## Step 7：写 `Tokenizer._apply_merges`

```python
def _apply_merges(self, tokens: list[bytes]) -> list[bytes]:
```

按 `self.merges` 的顺序，逐个 merge 规则扫描 tokens 列表。
注意：不是找当前最优，是**固定顺序逐个应用**。

---

## Step 8：写 `Tokenizer.encode`

```python
def encode(self, text: str) -> list[int]:
```

流程：

1. 空字符串 → `[]`
2. 用 special_re 切分文本 → `[普通段, special, 普通段, ...]`
3. special 段 → 直接查 `vocab_inv`
4. 普通段 → `finditer(GPT2_PAT)` → 每个 pre-token 转单字节 list → `_apply_merges` → 查 `vocab_inv`

**验证**：

```bash
uv run pytest tests/test_tokenizer.py -k roundtrip -v       # encode→decode 往返
uv run pytest tests/test_tokenizer.py -k tiktoken -v         # 和 tiktoken 精确对齐
uv run pytest tests/test_tokenizer.py -k special -v          # special token 正确处理
```

---

## Step 9：写 `Tokenizer.encode_iterable`

```python
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
```

用 `yield from`，两行搞定。**不要**先收集到 list。

**验证**：

```bash
uv run pytest tests/test_tokenizer.py -k encode_iterable -v
uv run pytest tests/test_tokenizer.py -k memory -v          # 内存限制 1MB
```

---

## Step 10：填 adapters.py

打开 `tests/adapters.py`，把底部两个函数的 `raise NotImplementedError` 替换：

```python
# get_tokenizer:
from cs336_basics.tokenizer import Tokenizer
return Tokenizer(vocab, merges, special_tokens)

# run_train_bpe:
from cs336_basics.tokenizer import train_bpe
return train_bpe(input_path, vocab_size, special_tokens)
```

---

## 最终验证

```bash
uv run pytest tests/test_train_bpe.py -v      # 3 个全过
uv run pytest tests/test_tokenizer.py -v       # 全部过
```

Done! 🎉