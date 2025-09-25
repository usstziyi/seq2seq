import torch
from torch import nn



"""
序列到序列模型的编码器-解码器架构：不包含注意力机制
"""

# 编码器接口
class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, inputs, *args):
        raise NotImplementedError

# 编码器接口
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, state, *args):
        raise NotImplementedError

    def forward(self, inputs, state):
        raise NotImplementedError

# 编码器-解码器
class EncoderDecoder(nn.Module):
    # 定义网络
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    
    # 定义前向传播
    def forward(self, src_inputs, tgt_inputs, *args):
        # 编码
        states, enc_state = self.encoder(src_inputs, *args)
        # 解码
        dec_state = self.decoder.init_state(enc_state, *args)
        return self.decoder(tgt_inputs, dec_state)

# Seq2Seq编码器：输入源句子
class Seq2SeqEncoder(Encoder):
    # 定义网络
    def __init__(self, src_vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层(D,E)
        self.embedding = nn.Embedding(
            num_embeddings=src_vocab_size, 
            embedding_dim=embed_size
        )
        # 循环神经网络层(E,H,L)
        self.rnn = nn.GRU(
            input_size=embed_size, 
            hidden_size=num_hiddens, 
            num_layers=num_layers, 
            dropout=dropout,
            batch_first=False
        )
        # 没有输出层
    
    # 定义前向传播
    # inputs(B,T)
    # states(T,B,H)
    # state(L,B,H)
    def forward(self, inputs, *args):
        # inputs(B,T)->(B,T,E)
        inputs = self.embedding(inputs)
        # inputs(T,B,E)
        inputs = inputs.permute(1, 0, 2)
        # states(T,B,H)
        # state(L,B,H)
        states, state = self.rnn(inputs)
        # states:包含每个时间步最后一层的信息
        # state:包含最后一个时间步每一层的信息
        return states, state
    
# Seq2Seq解码器：输入目标句子的词元，预测目标句子的下一个词元
class Seq2SeqDecoder(Decoder):
    # 定义网络
    def __init__(self, tgt_vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        # 嵌入层(D,E)
        self.embedding = nn.Embedding(
            num_embeddings=tgt_vocab_size, 
            embedding_dim=embed_size
        )
        # 循环神经网络层(E+H,H,L)
        self.rnn = nn.GRU(
            input_size=embed_size + num_hiddens, 
            hidden_size=num_hiddens, 
            num_layers=num_layers, 
            dropout=dropout,
            batch_first=False
        )
        # 输出层(H,D)
        self.dense = nn.Linear(
            in_features=num_hiddens, 
            out_features=tgt_vocab_size
        )


    # 初始化解码器状态
    def init_state(self, enc_state, *args):
        return enc_state # (L,B,H)

    # 定义前向传播
    # inputs(B,T)
    # output(B,T,D)
    # state(L,B,H)
    def forward(self, inputs, state):
        # inputs(B,T) -> (B,T,E)
        inputs = self.embedding(inputs)
        # inputs(T,B,E)
        inputs = inputs.permute(1, 0, 2)
        # inputs(T,B,E)
        # state(L,B,H),state[-1](B,H)
        # context(T,B,H)
        context = state[-1].repeat(inputs.shape[0], 1, 1)
        # inputs_and_context(T,B,E+H)
        inputs_and_context = torch.cat((inputs, context), 2)
        # states(T,B,H)
        # state(L,B,H)
        states, state = self.rnn(inputs_and_context, state)
        # 输出层，将隐藏状态映射到输出空间
        # output(T,B,H) -> (T,B,D)
        output = self.dense(states)
        # output(B,T,D)
        output = output.permute(1, 0, 2)
        return output, state
        # 在训练阶段，output 接下来送入 loss 损失函数更新梯度
        # 在预测阶段，output 接下来送入 argmax 函数获取预测结果

