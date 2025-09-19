# 注意力机制与序列到序列模型实现

这是一个基于PyTorch实现的序列到序列(Seq2Seq)模型，用于机器翻译任务。该项目展示了基本的编码器-解码器架构，并为后续实现注意力机制提供了基础框架。

## 项目功能特点

- 实现了基本的编码器-解码器(Encoder-Decoder)架构
- 使用GRU(Gated Recurrent Unit)作为循环神经网络层
- 包含带遮蔽的Softmax交叉熵损失函数，处理变长序列
- 提供预测函数和BLEU评估指标
- 英语到法语的机器翻译示例
- 模块化设计，便于扩展实现注意力机制

## 目录结构

```
├── .gitignore          # Git忽略文件配置
├── readme.md           # 项目说明文档
└── seq2seq.py          # 主要实现代码
```

## 安装说明

### 依赖环境

- Python 3.6+
- PyTorch
- d2l库 (《动手学深度学习》配套库)

### 安装步骤

1. 克隆项目代码

```bash
git clone <项目地址>
cd attention
```

2. 安装必要的依赖包

```bash
pip install torch
pip install d2l
```

## 使用方法

### 训练模型

直接运行主程序将开始训练过程：

```bash
python seq2seq.py
```

训练过程将显示每个epoch的损失值和训练速度，并在训练结束后使用4个测试句子进行翻译测试和BLEU评分。

### 代码结构说明

1. **模型架构**
   - `Encoder`: 编码器基类
   - `Decoder`: 解码器基类
   - `EncoderDecoder`: 编码器-解码器整体架构
   - `Seq2SeqEncoder`: 具体的序列编码器实现
   - `Seq2SeqDecoder`: 具体的序列解码器实现

2. **训练与预测**
   - `train_seq2seq`: 训练函数，实现模型训练流程
   - `predict_seq2seq`: 预测函数，实现端到端的序列生成
   - `bleu`: BLEU评估指标计算函数
   - `MaskedSoftmaxCELoss`: 带遮蔽的交叉熵损失函数

3. **辅助函数**
   - `sequence_mask`: 生成序列掩膜，处理变长序列

## 模型参数

可在`main()`函数中调整以下超参数：

- `embed_size`: 嵌入层大小，默认为32
- `num_hiddens`: 隐藏层大小，默认为32
- `num_layers`: RNN层数，默认为2
- `dropout`: Dropout比率，默认为0.1
- `batch_size`: 批次大小，默认为64
- `num_steps`: 序列长度，默认为10
- `lr`: 学习率，默认为0.005
- `num_epochs`: 训练轮数，默认为300

## 扩展方向

这个基础实现可以扩展为包含注意力机制的更复杂模型：

1. 实现Bahdanau注意力或Luong注意力
2. 尝试不同的RNN变种(LSTM、双向RNN等)
3. 添加beam search提高翻译质量
4. 支持更多语言对的翻译
5. 实现Transformer架构

## 注意事项

- 训练过程可能需要较长时间，可根据实际硬件情况调整参数
- 该实现使用了d2l库的一些辅助函数，确保正确安装了该库
- 当前实现没有包含实际的注意力机制，仅提供了基础框架