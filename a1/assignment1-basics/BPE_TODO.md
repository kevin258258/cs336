# BPE 分词器实现 TODO

## Part A: BPE 训练 (`train_bpe`)

### A1. 预分词（Pre-tokenization）
- [ ] 用 `find_chunk_boundaries` 按 `<|endoftext|>` 切分文件为 chunk
- [ ] 对每个 chunk：按 special token 分割文本（`re.split` + `re.escape`）
- [ ] 对每个片段：用 GPT-2 正则 `re.finditer(PAT, segment)` 提取 pre-token
- [ ] 将每个 pre-token 转为 UTF-8 bytes tuple：`"the"` → `(b't', b'h', b'e')`
- [ ] 用 `Counter` 统计频次
- [ ] 用 `multiprocessing.Pool` 并行处理多个 chunk，合并 Counter

### A2. 初始化词表与 pair 计数
- [ ] 初始化 vocab：special tokens 分配 ID + 256 个单字节 token
- [ ] 遍历 `word_freqs`，统计所有相邻 pair 的频次 → `pair_counts`

### A3. 迭代 Merge
- [ ] 循环直到 `len(vocab) >= vocab_size`：
  - [ ] 找 `pair_counts` 中频率最高的 pair（并列选字典序最大）
  - [ ] 新 token = pair[0] + pair[1]（bytes 拼接）
  - [ ] 记录到 merges 列表
  - [ ] 加入 vocab
  - [ ] **增量更新**：遍历包含该 pair 的 word，合并 pair，更新 pair_counts
    - 减少：旧的相邻 pair 计数
    - 增加：新产生的相邻 pair 计数

### A4. 测试
- [ ] `uv run pytest tests/test_train_bpe.py -k test_train_bpe_speed -v`（< 1.5 秒）
- [ ] `uv run pytest tests/test_train_bpe.py -k test_train_bpe -v`（vocab + merges 精确匹配）
- [ ] `uv run pytest tests/test_train_bpe.py -k test_train_bpe_special_tokens -v`

---

## Part B: Tokenizer 类

### B1. `__init__`
- [ ] 存储 vocab (int→bytes)、merges (list)、special_tokens
- [ ] 构建反向查找 vocab_inv (bytes→int)
- [ ] 如果 special_tokens 不在 vocab 中，追加分配新 ID
- [ ] 编译 special token 的分割正则（按长度降序排列）

### B2. `encode(text) -> list[int]`
- [ ] 按 special tokens 分割文本（长的优先匹配）
- [ ] 对 special token 片段：直接查 vocab_inv 得 ID
- [ ] 对普通文本片段：
  - [ ] 用 GPT-2 正则预分词
  - [ ] 对每个 pre-token 转为 bytes tuple
  - [ ] 按 merges 顺序依次应用合并
  - [ ] 查 vocab_inv 得 ID
- [ ] 拼接所有 ID 返回

### B3. `decode(ids) -> str`
- [ ] 查 vocab 得每个 ID 的 bytes
- [ ] 拼接所有 bytes
- [ ] UTF-8 解码：`.decode("utf-8", errors="replace")`

### B4. `encode_iterable(iterable) -> Iterator[int]`
- [ ] 用 `yield` 实现 generator（不要收集到列表）
- [ ] 对 iterable 中每个 text 调用 encode，逐个 yield token ID

### B5. 测试
- [ ] `uv run pytest tests/test_tokenizer.py -k test_roundtrip -v`
- [ ] `uv run pytest tests/test_tokenizer.py -k test_matches_tiktoken -v`
- [ ] `uv run pytest tests/test_tokenizer.py -k test_special_token -v`
- [ ] `uv run pytest tests/test_tokenizer.py -k test_encode_iterable -v`
- [ ] `uv run pytest tests/test_tokenizer.py -v`（全部通过）

---

## Part C: 实验题（写进 writeup）

- [ ] 在 TinyStories 上训练 BPE（vocab_size=10000, special_token=`<|endoftext|>`）
- [ ] 记录训练时间、内存、最长 token
- [ ] profiling：找出最耗时部分
- [ ] 在 OpenWebText 上训练 BPE（vocab_size=32000）
- [ ] 比较两个 tokenizer 的词表差异
- [ ] 压缩率实验（bytes/token）
- [ ] tokenize TinyStories 和 OWT 的训练集/验证集，保存为 uint16 numpy 数组

---

## Adapter 对接备忘

`tests/adapters.py` 中需要填写：

```
run_train_bpe(input_path, vocab_size, special_tokens)
  → 调用你的训练函数，返回 (vocab, merges)

get_tokenizer(vocab, merges, special_tokens)
  → 实例化你的 Tokenizer 类并返回
  → 返回的对象需要有 .encode(), .decode(), .encode_iterable() 方法
```
