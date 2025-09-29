import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

"""
尽管Transformer架构是为了序列到序列的学习而提出的
但Transformer编码器或Transformer解码器通常被单独用于不同的深度学习任务中。
"""

"""
基于位置的前馈网络:
"基于位置的前馈网络"(Position-wise Feed-Forward Network,简称 Position-wise FFN)是 Transformer 架构中的一个关键组件。
虽然名字里有"位置"，但它并不直接处理位置信息(位置信息通常由位置编码 Positional Encoding 提供),

*****而是指这个前馈网络对序列中每个位置(token)独立地、相同地应用"同一个"全连接网络.
*****这一步增强了模型的非线性表达能力,因为自注意力机制本身是线性的（只是加权求和）,需要 FFN 来引入非线性。

在原始 Transformer 论文(Vaswani et al., 2017)中,FFN 的隐藏层维度通常是输入维度的 4 倍,例如:Dg=512 → H=2048。
第一层：升维(如 512 → 2048)→ 增强模型容量，让网络在更高维空间中进行非线性变换。
第二层：降维(2048 → 512)→ 恢复原始维度，便于与残差连接相加。
"""
class PositionWiseFFN(nn.Module):
    # Dg:ffn_num_input
    # H: ffn_num_hiddens
    # Do:ffn_num_outputs
    # 一般来说:Dg = Do
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)

        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    # X(B,G,Dg)->(B,G,H)->(B,G,Do)
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


"""
残差连接 & 层归一化
self.dropout(Y) + X
残差连接(Residual Connection)是一种常用的神经网络结构,用于解决深度神经网络训练中的梯度消失问题。
它的基本思想是将输入直接添加到输出中,形成一个新的输出。
这在Transformer架构中被广泛使用,用于连接不同的层(如自注意力层、前馈网络层等)。

self.ln
层归一化(Layer Normalization)是一种常用的归一化技术,用于在神经网络中稳定训练过程。
它的基本思想是对每个样本的所有特征维度进行归一化,而不是对所有样本的所有特征维度进行归一化。
这在Transformer架构中被广泛使用,用于归一化残差连接的输出。
对于一个输入张量(比如形状为 [batch_size, seq_len, d_model]),
Layer Normalization 是对 每个样本的最后一个维度（即特征维度） 进行归一化。

在原始 Transformer 论文《Attention is All You Need》中,
采用的是 Post-LN 结构(先子层 → 再 Add & Norm),
但后来很多实现（如 BERT、大多数现代 Transformer)改用 Pre-LN(先 Norm → 再子层 → 再 Add)
你这段代码属于 Post-LN 风格。

残差连接(Add):
缓解深层网络的梯度消失问题。
允许信息直接跨层流动，提升训练稳定性。

Layer Normalization(Norm):
对每个样本的特征维度做归一化（不同于 BatchNorm)。
在 NLP 任务中特别有效,因为句子长度可变,BatchNorm 不稳定。

Dropout:
在残差连接前对子层输出做随机失活，增强泛化能力。
"""
class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    # X：通常是输入（比如前一层的输出）
    # Y：通常是某个子层（如多头注意力或前馈网络）的输出
    # 操作流程：
    # 1. 对子层输出 Y 应用 Dropout（用于正则化，防止过拟合）。
    # 2. 将 Dropout 后的 Y 与原始输入 X 相加 → 这就是 残差连接（skip connection）。
    # 3. 对相加结果进行 Layer Normalization。
    # 这样每一层都通过残差连接保留了原始信息，并通过 LayerNorm 稳定训练过程。
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

"""
Transformer编码器块
"""
class EncoderBlock(nn.Module):

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        
        # 多头自注意力
        self.attention = d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout,use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout) # 残差连接 & 规范化层
        
        # 基于位置的前馈网络
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout) # 残差连接 & 规范化层

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        Y = self.addnorm2(Y, self.ffn(Y))
        return Y

"""
Transformer编码器
"""
class TransformerEncoder(d2l.Encoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        
        # 位置编码
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        # 编码器块层
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。

        # 位置编码
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        # 编码器块层
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


"""
Transformer解码器块
"""
class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        # 多头自注意力层
        self.attention1 = d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        # 编码器－解码器多头自注意力层
        self.attention2 = d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        # 基于位置的前馈网络
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size,num_steps),
            # 其中每一行是[1,2,...,num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # 多头自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器－解码器多头自注意力
        # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        # 位置前馈网络
        return self.addnorm3(Z, self.ffn(Z)), state

"""
Transformer解码器
"""
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        # 位置编码
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        # 解码器块层
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        # 全连接层
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        # 位置编码
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        # 解码器块层
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        # 全连接层
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights