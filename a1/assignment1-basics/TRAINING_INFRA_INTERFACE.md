# 训练 Infra 接口与使用指南

本文档是当前 A1 训练栈的接口与使用参考。

适用场景：
- 日常训练使用（train/eval/checkpoint）。
- 未来替换底层实现（例如替换为 Flash Attention）时，保证外部行为兼容。

---

## 1. 模块总览

- `cs336_basics/model.py`
  - 模型结构与张量级算子实现。
- `cs336_basics/Function.py`
  - 训练工具函数：loss、batch、优化器、学习率调度、checkpoint。
- `train_lm.py`
  - 端到端训练入口（CLI 脚本）。
- `tests/adapters.py`
  - 测试使用的公共适配接口，改造后需保持行为兼容。

---

## 2. 模型接口（`cs336_basics/model.py`）

### 2.1 基础函数

#### `silu(x: torch.Tensor) -> torch.Tensor`
- 逐元素 SiLU 激活。

#### `softmax(x: torch.Tensor, dim: int) -> torch.Tensor`
- 在 `dim` 维做数值稳定的 softmax。

#### `scaled_dot_product_attention(q, k, v, mask=None) -> torch.Tensor`
- 输入形状：
  - `q`: `(..., q_len, d_k)`
  - `k`: `(..., k_len, d_k)`
  - `v`: `(..., k_len, d_v)`
  - `mask`（可选）：可广播到 `(..., q_len, k_len)`，`True=保留`，`False=屏蔽`。
- 输出形状：
  - `(..., q_len, d_v)`
- 契约：
  - 提供 `mask` 时，被屏蔽位置的注意力概率应为 0。
  - 是否因果由调用方决定，函数本身不强制因果。

### 2.2 模块

#### `Linear(in_features, out_features, device=None, dtype=None)`
- 参数：
  - `weight`: 形状 `(out_features, in_features)`
- 前向：
  - 输入 `(..., in_features)` -> 输出 `(..., out_features)`

#### `Embedding(num_embeddings, embedding_dim, device=None, dtype=None)`
- 参数：
  - `weight`: 形状 `(num_embeddings, embedding_dim)`
- 前向：
  - 输入 token id `(...)` -> 输出 `(..., embedding_dim)`

#### `RMSNorm(d_model, eps=1e-5, device=None, dtype=None)`
- 参数：
  - `weight`: 形状 `(d_model,)`
- 前向：
  - 输入 `(..., d_model)` -> 输出 `(..., d_model)`

#### `SwiGLU(d_model, d_ff=None, device=None, dtype=None)`
- 参数：
  - `w1.weight`: `(d_ff, d_model)`
  - `w2.weight`: `(d_model, d_ff)`
  - `w3.weight`: `(d_ff, d_model)`
- 前向：
  - 输入 `(..., d_model)` -> 输出 `(..., d_model)`

#### `RoPE(theta, d_k, max_seq_len, device=None)`
- 缓存（非持久化）：
  - `cos_cached`: `(max_seq_len, d_k/2)`
  - `sin_cached`: `(max_seq_len, d_k/2)`
- 前向：
  - `x`: `(..., seq_len, d_k)`
  - `token_positions`（可选）：`(..., seq_len)` 或 `(seq_len,)`
  - 输出：与 `x` 同形状
- 契约：
  - `d_k` 必须是偶数。
  - token 最大位置必须 `< max_seq_len`。

#### `MultiHeadSelfAttention(d_model, num_heads, max_seq_len=None, theta=None, ...)`
- 参数：
  - `q_proj.weight`、`k_proj.weight`、`v_proj.weight`、`output_proj.weight`：均为 `(d_model, d_model)`
- 前向：
  - `x`: `(..., seq_len, d_model)`
  - `token_positions` 可选（仅在启用 RoPE 时使用）
  - 输出：`(..., seq_len, d_model)`
- 内部契约：
  - 按头拆分：`d_head = d_model // num_heads`。
  - 内部使用因果 mask。
  - 若启用 RoPE，仅对 Q/K 应用旋转。

#### `TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta, ...)`
- 子模块：
  - `ln1`、`attn`、`ln2`、`ffn`
- 前向：
  - `x`: `(batch, seq_len, d_model)`
  - `token_positions` 可选
  - 输出：`(batch, seq_len, d_model)`
- pre-norm 残差顺序：
  - `x = x + attn(ln1(x))`
  - `x = x + ffn(ln2(x))`

#### `TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, ...)`
- 子模块：
  - `token_embeddings`、`layers`、`ln_final`、`lm_head`
- 前向：
  - `in_indices`: `(batch, seq_len)`，要求 `seq_len <= context_length`
  - 输出 logits：`(batch, seq_len, vocab_size)`

### 2.3 `state_dict` 键名契约

后续替换算子/后端时，以下键名必须保持不变：

- `token_embeddings.weight`
- `layers.{i}.attn.q_proj.weight`
- `layers.{i}.attn.k_proj.weight`
- `layers.{i}.attn.v_proj.weight`
- `layers.{i}.attn.output_proj.weight`
- `layers.{i}.ln1.weight`
- `layers.{i}.ffn.w1.weight`
- `layers.{i}.ffn.w2.weight`
- `layers.{i}.ffn.w3.weight`
- `layers.{i}.ln2.weight`
- `ln_final.weight`
- `lm_head.weight`

---

## 3. 训练工具接口（`cs336_basics/Function.py`）

### 3.1 Loss 与 Batch

#### `cross_entropy(logits, labels) -> torch.Tensor`
- `logits`: `(N, vocab_size)`
- `labels`: `(N,)`
- 返回标量平均交叉熵。

#### `get_batch(dataset, batch_size, context_length, device) -> tuple[x, y]`
- `dataset`: 一维 numpy 数组（`np.ndarray` 或 memmap）。
- 返回：
  - `x`: `(batch_size, context_length)`，位于 `device`
  - `y`: `(batch_size, context_length)`，位于 `device`
- 契约：
  - `y` 是 `x` 的右移一位目标序列。

### 3.2 优化器与学习率调度

#### `class AdamW(torch.optim.Optimizer)`
- 构造函数：
  - `(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)`
- 方法：
  - `step(closure=None)`
- 契约：
  - 使用 decoupled weight decay。
  - 参数状态包含：`step`、`exp_avg`、`exp_avg_sq`。

#### `gradient_clipping(parameters, max_l2_norm) -> None`
- 全局 L2 范数裁剪，原地修改梯度。

#### `get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters) -> float`
- 分段调度：
  - warmup
  - cosine 衰减
  - 低于最小学习率后封底

### 3.3 Checkpoint

#### `save_checkpoint(model, optimizer, iteration, out) -> None`
- 序列化字典键：
  - `"model"`
  - `"optimizer"`
  - `"iteration"`

#### `load_checkpoint(src, model, optimizer) -> int`
- 恢复模型和优化器状态。
- 返回保存时的 iteration。

---

## 4. 训练脚本 CLI（`train_lm.py`）

### 4.1 必填参数

- `--train-data`
- `--val-data`

### 4.2 常用运行参数

- 模型：
  - `--vocab-size`
  - `--context-length`
  - `--d-model`
  - `--num-layers`
  - `--num-heads`
  - `--d-ff`
  - `--rope-theta`
- 优化器/调度：
  - `--max-learning-rate`
  - `--min-learning-rate`
  - `--warmup-iters`
  - `--cosine-cycle-iters`
  - `--weight-decay`
  - `--beta1`
  - `--beta2`
  - `--eps`
  - `--max-grad-norm`
- 循环/IO：
  - `--steps`
  - `--batch-size`
  - `--eval-interval`
  - `--eval-batches`
  - `--log-interval`
  - `--save-interval`
  - `--out-dir`
  - `--resume-from`
  - `--device`
  - `--seed`

### 4.3 数据加载契约

- `--data-format` 支持：
  - `npy` -> 使用 `np.load(..., mmap_mode="r")`
  - `bin` -> 使用 `np.memmap(..., dtype=<data-dtype>, mode="r")`
  - `auto` -> 后缀为 `.npy` 时走 `npy`，否则走 `bin`

---

## 5. 使用示例

### 5.1 最小 CPU 训练

```bash
cd a1/assignment1-basics
./.venv/bin/python train_lm.py \
  --train-data /path/to/train.npy \
  --val-data /path/to/val.npy \
  --data-format npy \
  --vocab-size 10000 \
  --context-length 128 \
  --d-model 256 \
  --num-layers 2 \
  --num-heads 4 \
  --d-ff 768 \
  --steps 200 \
  --batch-size 8 \
  --device cpu \
  --out-dir checkpoints/dev
```

### 5.2 从 Checkpoint 恢复

```bash
./.venv/bin/python train_lm.py \
  --train-data /path/to/train.npy \
  --val-data /path/to/val.npy \
  --resume-from checkpoints/dev/latest.pt \
  --out-dir checkpoints/dev
```

### 5.3 可选 W&B 日志

```bash
./.venv/bin/python train_lm.py \
  --train-data /path/to/train.npy \
  --val-data /path/to/val.npy \
  --wandb \
  --wandb-project cs336-a1 \
  --wandb-run-name baseline-run
```

---

## 6. 后端替换兼容规则（如 Flash Attention）

如果只替换 attention 内核实现，以下外部契约必须保持：

1. 保持 `MultiHeadSelfAttention.forward(x, token_positions=None)` 签名不变。
2. 保持输出张量形状与 dtype 行为不变。
3. 保持因果 mask 语义不变。
4. 保持“只对 Q/K 应用 RoPE”的语义不变。
5. 保持全部 `state_dict` 键名不变。
6. 保持 adapter 侧输出与现有测试容差兼容。

推荐替换边界：
- 替换 `scaled_dot_product_attention` 内部实现，或 `MultiHeadSelfAttention.forward` 中的 attention 路径。
- 不要在迁移内核时改动 `TransformerBlock` / `TransformerLM` 的外部 API。

---

## 7. 测试与验证命令

### 7.1 快速正确性验证（排除长测）

```bash
cd a1/assignment1-basics
./.venv/bin/pytest -q -k "not memory_usage and not train_bpe_speed"
```

### 7.2 完整测试

```bash
./.venv/bin/pytest -q
```

---

## 8. 已知注意事项

- `test_encode_iterable_memory_usage` 约束严格，在部分环境会较慢。
- `test_train_bpe_speed` 对机器性能和调度比较敏感。
- `Function.py:get_batch` 当前会先拷贝 memmap 切片，再转 tensor，以避免 numpy 只读告警。

---

## 9. 扩展前快速检查清单

在合并 infra 变更前，建议至少确认：

1. API 签名与 `state_dict` 键名未变。
2. 先跑 adapter 和模型相关测试。
3. 至少跑一次 `train_lm.py` 的短程 smoke run。
4. 验证 checkpoint 保存/加载回环正确。
5. 最后再做速度/显存 benchmark 对比。
