# minimind-practice

在单机环境下复现 MiniMind 的「**预训练 → SFT 指令微调**」全流程，目的是跑通整条训练链路、验证工程细节，而不是训练一个可用的模型。

---

## 声明

本项目基于开源项目 [jingyaogong/minimind](https://github.com/jingyaogong/minimind)（Apache-2.0）进行复现练习。

- 模型架构、训练框架、数据集**均来自上游项目**，本仓库不包含上游源码、数据集与模型权重
- 本仓库只包含复现过程中编写的**辅助脚本、环境适配与实验记录**
- 原始代码与算法的著作权归上游作者所有，详见 [LICENSE](LICENSE)

如果你要训练自己的模型，请直接使用上游仓库，不要用本仓库。

---

## 做了什么

### 1. 数据切片（`_slice_data.py`）

上游完整的预训练/SFT 数据集有数 GB，单机跑不动。这个脚本按行数截取前 5%，生成 `pretrain_t2t_mini_sub.jsonl` 和 `sft_t2t_mini_sub.jsonl`，保持原 jsonl 格式不变。

```bash
python _slice_data.py
# dataset/pretrain_t2t_mini.jsonl: 行数 -> 5%，输出 dataset/pretrain_t2t_mini_sub.jsonl
```

> 注意：按行头截取会丢失数据分布的代表性（同一来源的样本集中在文件前部），这里纯粹是为了压缩耗时，不用于评估模型效果。

### 2. 预训练 + SFT

使用上游 `trainer/` 下的训练脚本，指向切片后的数据集，训练 1 个 epoch。

- 预训练产出：`out/pretrain_768.pth`（131 MB）
- SFT 产出：`out/full_sft_768.pth`（131 MB）

训练日志在终端输出，未落盘保存，因此本仓库不包含 loss 曲线。

### 3. 权重转换（`_convert_mine.py`）

把自训的 `.pth` 权重转换成 HuggingFace transformers 格式，供上游的 `web_demo` 和 `eval_llm.py` 加载。

模型配置按训练时实际使用的参数写死：

```python
MiniMindConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=8192, use_moe=False)
```

### 4. 自动评测（`_eval_auto.py`）

上游 `eval_llm.py` 启动时会用 `input()` 交互选择模式，在非交互终端下会因 stdin 编码问题卡住。这个脚本用 `runpy` 加载 `eval_llm.py`，并把 `input()` 劫持为固定返回 `0`（自动测试模式），从而实现无人值守评测。

```bash
python _eval_auto.py   # 等价于 eval_llm.py --load_from ./model --weight full_sft --device cuda，自动进入测试模式
```

---

## 环境

- Python 3.x + PyTorch（CUDA）
- 依赖见 [requirements.txt](requirements.txt)，**在上游基础上改了两个版本号**：

| 包 | 上游 | 本机 | 原因 |
|---|---|---|---|
| protobuf | 6.33.5 | 5.29.5 | 与 wandb / tensorboard 的依赖冲突 |
| ujson | 5.1.0 | 5.10.0 | 5.1.0 在当前 Python 版本下无预编译 wheel，需从源码编译且失败 |

`requirements.txt` 是上游完整依赖，仅供对照；单独安装本仓库的脚本只需标准库 + 上游环境。

---

## 复现步骤

```bash
# 1. 克隆上游项目
git clone https://github.com/jingyaogong/minimind.git
cd minimind
pip install -r requirements.txt

# 2. 放入本仓库的三个脚本
#    _slice_data.py / _convert_mine.py / _eval_auto.py

# 3. 切片数据
python _slice_data.py

# 4. 预训练（参数见上游 trainer/train_pretrain.py）
# 5. SFT（参数见上游 trainer/train_full_sft.py）

# 6. 转换为 transformers 格式
python _convert_mine.py

# 7. 评测
python _eval_auto.py
```

---

## 目录说明

```
minimind-practice/
├── _slice_data.py       # 数据集按比例切片
├── _convert_mine.py     # .pth -> transformers 格式
├── _eval_auto.py        # 无人值守评测
├── requirements.txt     # 上游依赖（含本机版本调整）
├── LICENSE              # 上游 Apache-2.0 协议
└── README.md
```

训练产物（不在本仓库）：

```
out/pretrain_768.pth          131 MB
out/full_sft_768.pth          131 MB
scripts/minimind-mine/        transformers 格式权重 + tokenizer
dataset/*_sub.jsonl           5% 切片后的语料
```

GitHub 单文件上限 100 MB，以上文件均超出限制，且数据集过大，故未上传。权重在本地生成，按上方步骤可复现。

---

## 已知限制

- **5% 子集 + 1 epoch 只能验证链路跑通，不能代表模型质量**。这样训练出来的模型回答质量很差，属于预期结果。
- 数据按行头截取，分布有偏。
- 未做分布式训练、断点续训、混合精度等工程优化，都是上游脚本的默认行为。
- 没有保存训练日志，无法复盘 loss 曲线。
