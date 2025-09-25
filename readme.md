# Seq2Seq与注意力机制实现库

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange.svg)](https://pytorch.org/)

一个全面实现深度学习中序列到序列(Seq2Seq)模型和多种注意力机制的Python库，适用于机器翻译、文本摘要、问答系统等自然语言处理任务。

## 📚 项目简介

本库提供了多种注意力机制及其在序列到序列模型中的应用实现，包括经典的加性注意力、缩放点积注意力和多头注意力等核心组件。所有实现基于PyTorch框架，设计简洁明了，便于学习和集成到实际项目中。

## 📁 项目结构

```
├── attention/                  # 注意力机制核心实现
│   ├── __init__.py             # 注意力模块导出定义
│   ├── additive_attention.py   # 加性注意力实现
│   ├── dotproduct_attention.py # 缩放点积注意力实现
│   └── multihead_attention.py  # 多头注意力实现
├── common/                     # 公共工具模块
│   ├── __init__.py             # 公共模块导出定义
│   ├── dataline.py             # 数据处理工具
│   ├── heatmaps.py             # 注意力热图可视化工具
│   └── mask_softmax.py         # 掩码softmax实现
├── decoder/                    # 编码器-解码器架构
│   ├── __init__.py             # 解码器模块导出定义
│   ├── bahdanau_enc_dec.py     # Bahdanau注意力编码器-解码器
│   └── seq2seq_enc_dec.py      # 基础序列到序列编码器-解码器
├── .gitignore                  # Git忽略配置
├── readme.md                   # 项目说明文档
├── test_additive_attention.py  # 加性注意力测试
├── test_bahdanau_enc_dec.py    # Bahdanau编码器-解码器测试
├── test_dotproduct_attention.py # 点积注意力测试
├── test_multihead_attention.py # 多头注意力测试
└── test_seq2seq_enc_dec.py     # 序列到序列模型测试
```

## 🔧 核心功能模块

### 注意力机制

- **加性注意力 (AdditiveAttention)**
  - 通过可学习的权重参数计算查询与键之间的相关性
  - 适用于查询和键维度不同的场景
  - 计算复杂度高于点积注意力

- **缩放点积注意力 (DotProductAttention)**
  - 通过点积操作高效计算注意力权重
  - 包含注意力掩码机制，支持可变长度序列处理
  - 对注意力分数进行缩放，缓解梯度消失问题

- **多头注意力 (MultiHeadAttention)**
  - 并行使用多个独立的注意力头，捕获不同子空间的特征
  - 结合注意力机制和线性变换，增强模型表达能力
  - 是Transformer架构的核心组件

### 编码器-解码器架构

- **基础序列到序列模型 (BasicEncoderDecoder)**
  - 传统的编码器-解码器框架，基于GRU实现
  - 编码器将输入序列转换为上下文向量
  - 解码器基于上下文向量生成目标序列

- **Bahdanau注意力编码器-解码器 (BahdanauEncoderDecoder)**
  - 结合Bahdanau注意力机制的编码器-解码器
  - 在解码过程中动态关注输入序列的不同部分
  - 解决了长距离依赖问题，提高长序列翻译质量

### 可视化工具

- **热图可视化 (show_heatmaps)**
  - 直观展示注意力权重分布
  - 支持自定义颜色映射和标注
  - 帮助理解模型关注的输入位置

## 🚀 快速开始

### 环境要求

- Python 3.7+
- PyTorch 1.9+
- NumPy
- Matplotlib
- d2l (用于部分测试函数)

### 安装方法

1. 克隆仓库
   ```bash
   git clone https://github.com/yourusername/seq2seq-attention.git
   cd seq2seq-attention
   ```

2. 安装依赖
   ```bash
   pip install torch numpy matplotlib
   ```

## 📝 使用示例

### 1. 使用缩放点积注意力

```python
from attention.dotproduct_attention import DotProductAttention
import torch

# 创建注意力模型
attention = DotProductAttention(dropout=0.5)

# 准备输入数据
batch_size = 2
num_queries = 4
num_kvpairs = 6
valid_lens = torch.tensor([3, 2])
queries = torch.ones((batch_size, num_queries, 10))
keys = torch.ones((batch_size, num_kvpairs, 10))
values = torch.ones((batch_size, num_kvpairs, 24))

# 计算注意力输出
output, attention_weights = attention(queries, keys, values, valid_lens)
print(f"输出形状: {output.shape}")  # (2, 4, 24)
print(f"注意力权重形状: {attention_weights.shape}")  # (2, 4, 6)
```

### 2. 使用多头注意力

```python
from attention.multihead_attention import MultiHeadAttention
import torch

# 创建多头注意力模型
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(
    key_size=num_hiddens, 
    query_size=num_hiddens, 
    value_size=num_hiddens, 
    num_hiddens=num_hiddens, 
    num_heads=num_heads, 
    dropout=0.5
)

# 准备输入数据
batch_size = 2
num_queries = 4
num_kvpairs = 6
valid_lens = torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))

# 计算多头注意力输出
output = attention(X, Y, Y, valid_lens)
print(f"输出形状: {output.shape}")  # (2, 4, 100)
```

### 3. 使用Bahdanau注意力编码器-解码器

```python
from decoder import BahdanauEncoderDecoder, BahdanauDecoder
from decoder import Seq2SeqEncoder
import torch

# 定义模型参数
vocab_size = 10000
embed_size = 256
num_hiddens = 512
num_layers = 2

# 创建编码器和解码器
encoder = Seq2SeqEncoder(
    src_vocab_size=vocab_size,
    embed_size=embed_size,
    num_hiddens=num_hiddens,
    num_layers=num_layers
)

decoder = BahdanauDecoder(
    vocab_size=vocab_size,
    embed_size=embed_size,
    num_hiddens=num_hiddens,
    num_layers=num_layers
)

# 创建Bahdanau注意力编码器-解码器模型
model = BahdanauEncoderDecoder(encoder, decoder)

# 准备输入数据
batch_size = 64
src_seq_len = 10
tgt_seq_len = 12

# 随机生成源序列和目标序列
src_tokens = torch.randint(0, vocab_size, (batch_size, src_seq_len))
tgt_tokens = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))
valid_lens = torch.randint(3, src_seq_len, (batch_size,))

# 前向传播
outputs, state = model(src_tokens, tgt_tokens, valid_lens)
print(f"输出形状: {outputs.shape}")
```

### 4. 可视化注意力权重

```python
from common.heatmaps import show_heatmaps
import torch

# 生成示例注意力权重
attention_weights = torch.rand((2, 3, 4, 4))  # 2个样本，3个注意力头，4x4注意力权重

# 可视化注意力热图
show_heatmaps(
    attention_weights, 
    xlabel='Keys', 
    ylabel='Queries',
    titles=['Head 1', 'Head 2', 'Head 3']
)
```

## 🧪 运行测试

本项目包含多个测试文件，用于验证各组件的功能正确性：

```bash
# 测试加性注意力
python test_additive_attention.py

# 测试点积注意力
python test_dotproduct_attention.py

# 测试多头注意力
python test_multihead_attention.py

# 测试序列到序列模型
python test_seq2seq_enc_dec.py

# 测试Bahdanau注意力编码器-解码器
python test_bahdanau_enc_dec.py
```

## 📚 理论背景

### 注意力机制原理

注意力机制允许模型在生成输出时动态地关注输入序列的不同部分，主要包括以下步骤：

1. **计算注意力分数**：通过查询(Query)和键(Key)计算相关性
2. **注意力分数归一化**：使用softmax将分数转换为权重
3. **加权求和**：使用权重对值(Value)进行加权求和

### 常见注意力机制对比

| 注意力类型 | 计算方式 | 优点 | 适用场景 |
|----------|---------|------|---------|
| 加性注意力 | 前馈网络 | 适用于Q和K维度不同 | 早期机器翻译模型 |
| 点积注意力 | 点积运算 | 计算效率高 | 序列长度较短场景 |
| 缩放点积 | 点积/√d_k | 缓解梯度消失 | Transformer模型 |
| 多头注意力 | 多组独立注意力 | 捕获多维度特征 | 复杂序列建模任务 |

## 🔍 模块导入指南

### 模块导入方式

1. **直接导入特定类或函数**
   ```python
   from attention.dotproduct_attention import DotProductAttention
   from decoder import BahdanauEncoderDecoder, BasicEncoderDecoder
   from common.heatmaps import show_heatmaps
   ```

2. **导入整个包**
   ```python
   import attention
   import decoder
   import common
   
   # 使用时
   attention_model = attention.MultiHeadAttention(...)
   decoder_model = decoder.BahdanauDecoder(...)
   ```

### 注意事项

- 确保项目目录在Python的搜索路径中
- 避免循环导入
- 推荐使用具体导入方式，提高代码可读性

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 鸣谢

本项目参考了以下资源：
- [d2l-ai](https://d2l.ai/) - 《动手学深度学习》
- [PyTorch官方文档](https://pytorch.org/docs/stable/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 论文