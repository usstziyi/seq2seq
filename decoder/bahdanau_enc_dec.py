import torch
from torch import nn
from attention import additive_attention as at

# Bahdanau编码器(内嵌加性注意力)

"""
编码的时候:为了产生 key,编码器无论是训练还是预测，都需要输入固定长度(B,G)的序列,所以才能生成G个key
--- ##################
 |  ##################
 B  ##################
 |  ##################
--- |------T(G)------|

B:batch_size
G:sequence_length:group length

解码的时候:为了使用 key,解码器输入长度没有限制，训练时输入实际长度(B,Q)=(B,G),预测时输入实际长度(B,Q)=(1,1)
--- #########
 |  ##################
 B  ###############
 |  ######
--- |------T(Q)------|

B:batch_size
Q:real sequence_length:query length


"""



# 编码器接口
class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, inputs):
        raise NotImplementedError

# 编码器接口
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, states, last_state,valid_lens):
        raise NotImplementedError

    def forward(self, inputs, state):
        raise NotImplementedError

# Seq2Seq编码器网络模型：输入源句子，输出编码器的状态
class Seq2SeqEncoder(Encoder):
    # 定义网络
    def __init__(self, src_vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层(V,E)
        self.embedding = nn.Embedding(
            num_embeddings=src_vocab_size, 
            embedding_dim=embed_size
        )
        # 循环神经网络层(E,H)
        self.rnn = nn.GRU(
            input_size=embed_size, 
            hidden_size=num_hiddens, 
            num_layers=num_layers, 
            dropout=dropout,
            batch_first=False
        )
        # 没有输出层
    
    # 定义前向传播
    # inputs(B,G)
    # states(G,B,H)
    # last_state(L,B,H)
    def forward(self, inputs): # 编码的时候可以(B,G)按块进行
        # inputs(B,G)->(B,G,E)
        inputs = self.embedding(inputs)
        # inputs(G,B,E)
        inputs = inputs.permute(1, 0, 2)
        # states(G,B,H):包含每个时间步最后一层的信息
        # last_state(L,B,H):包含最后一个时间步每一层的信息
        states, last_state = self.rnn(inputs)

        return states, last_state

# 注意力机制解码器接口
class AttentionDecoder(Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError



# Bahdanau解码器(内嵌加性注意力)网络模型：输入原句子，输出目标句子
class BahdanauDecoder(AttentionDecoder):
    def __init__(self, tgt_vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(BahdanauDecoder, self).__init__(**kwargs)
        
        # 嵌入层(V,E):
        # 输入:当前时间步的输入(目标句子的单词索引)
        # 输出:当前时间步的嵌入向量(维度为E)
        self.embedding = nn.Embedding(
            num_embeddings=tgt_vocab_size,
            embedding_dim=embed_size
        )
        # 注意力层网络:查询编码器提供的信息，求的是注意力权重，
        # 注意力权重再与编码器的隐藏状态进行加权求和，得到上下文向量context
        self.attention = at.AdditiveAttention(
            key_size=num_hiddens,
            query_size=num_hiddens,
            num_hiddens=num_hiddens,
            dropout=dropout
        )

        # 循环层(E+H,H):
        # 输入:当前时间步的输入(嵌入层输出+上下文向量)
        # 隐藏状态:上一个时间步的隐藏状态
        # 输出:当前时间步的隐藏状态
        self.rnn = nn.GRU(
            input_size=embed_size + num_hiddens,
            hidden_size=num_hiddens,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=False
        )
        # 输出层(H,V):
        # 输入:当前时间步的隐藏状态
        # 输出:当前时间步的预测输出(维度为V)
        self.dense = nn.Linear(
            in_features=num_hiddens,
            out_features=tgt_vocab_size
        )

    
    # 注意：待翻译句子首先要经过编码器粗加工，得到半成品给解码器用
    # 初始化解码器状态:给编码器的states做维度转换，为了与注意力层的输入匹配
    # 把valid_lens也打包进来
    def init_state(self, states, last_state, valid_lens):
        # states(G,B,H)
        # last_state(L,B,H))
        # valid_lens(B):待翻译句子的有效长度，此时只是初加工，还没有翻译
        states = states.permute(1, 0, 2)
        # states(B,G,H)
        # last_state(L,B,H)
        # valid_lens(B)
        return states, last_state, valid_lens

    # inputs(B,Q):训练=(B,G),预测=(1,1)
    def forward(self, inputs, state): # 解码的时候最多按(B,1)处理
        # states(B,G,H)
        # last_state(L,B,H)
        # valid_lens(B)
        states, last_state, valid_lens = state

        outputs, self._attention_weights = [], []


        # states(B,G,H)->keys(B,K,Kd)=(B,G,H)
        keys = states # 编码器所有时间步的隐藏状态,作为注意力层的键
        # states(B,G,H)->values(B,K,V)=(B,G,H)
        values = states # 编码器所有时间步的隐藏状态,作为注意力层的值
 
        # 1.计算嵌入层
        # inputs(B,Q)->(B,Q,E)->(Q,B,E)
        inputs = self.embedding(inputs).permute(1, 0, 2)
        # 保留batch,分解group->input(B,E)
        for input in inputs: # 循环Q轮,进了这个for,Q这个维度就淡化了，主要处理B这个维度
            # 2.计算注意力层:在seq2seq中使用注意力机制
            # 编码器最新的隐藏状态是主动目标
            # 编码器所有时间步的隐藏状态就是被动目标，G个
            # 用最新的最后一步隐状态去查所有时间步的隐藏状态，求得分，求 weights,得到上下文向量
            # last_state(L,B,H)->(B,H)->query(B,1,H)
            query = last_state[-1].unsqueeze(1) # 编码器最后一个时间步的隐藏状态
            # query(B,Q,Qd)=(B,1,H)
            # keys(B,G,Gd)=(B,G,H)
            # values(B,G,V)=(B,G,H)
            # valid_lens(B)
            # context(B,Q,V)=(B,1,H):因为values(B,G,V)=(B,G,H),所以context(B,Q,V)=(B,1,H)
            context = self.attention(query, keys, values, valid_lens) # 调用注意力层(计算注意力)，计算上下文

            # 保存本轮的注意力权重(B,1,G)
            # attention_weights（B,Q,G)=(B,1,G)
            # list((B,1,G),(B,1,G)...(B,1,G)）一共G个
            self._attention_weights.append(self.attention.attention_weights.detach())

            # 3.计算循环层
            # input(B,E)->(B,1,E)
            input = input.unsqueeze(1)

            # input(B,1,E)
            # context(B,1,H)
            # combination(B,1,E+H)
            combination = torch.cat((input, context), dim=-1) # 拼接输入和上下文，在特征维度上拼接
            # combination(B,1,E+H)->(1,B,E+H)
            combination = combination.permute(1, 0, 2)
      
            # combination(1,B,E+H)
            # last_state(L,B,H)
            # out(1,B,H):当前时间步最后一层隐藏状态
            # last_state(L,B,H):当前时间步的隐状态
            out, last_state = self.rnn(combination, last_state)
            outputs.append(out) # (1,B,H),(1,B,H),(1,B,H)...



        # outputs(Q,B,H)
        outputs = torch.cat(outputs,dim=0)
        # outputs(Q,B,H)->(Q,B,V)->(B,Q,V)
        outputs = self.dense(outputs).permute(1, 0, 2)
        # outputs(B,Q,V)
        # states(B,G,H)
        # last_state(L,B,H)
        # valid_lens(B)
        return outputs, (states, last_state, valid_lens)

    @property
    def attention_weights(self):
        # 把列表中的注意力权重(B,1,G)拼接起来(B,Q,G)，生成张量
        # weight(B,1,G) 一共Q个
        # attention_weights(B,Q,G):表示B个句子的Q个时间步的注意力权重
        attention_weights = torch.cat(self._attention_weights, dim=1)
        return attention_weights


# 编码器-解码器
class EncoderDecoder(nn.Module):
    # 定义网络
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    
    # 定义前向传播
    # src_inputs(B,G)
    # tgt_inputs(B,G)
    # valid_lens(B)
    def forward(self, src_inputs, tgt_inputs, valid_lens):
        # 编码
        # src_inputs(B,G)
        # states(B,G,H)
        # last_state(L,B,H)
        states, last_state = self.encoder(src_inputs)
        # 初始化解码器 state
        # valid_lens(B)
        # state=(states, last_state, valid_lens)
        state = self.decoder.init_state(states,last_state,valid_lens)
        # 解码
        # tgt_inputs(B,G)
        # outputs(B,G,V)
        # states(B,G,H)
        # last_state(L,B,H)
        # valid_lens(B)
        # state=(states, last_state, valid_lens)
        outputs, state = self.decoder(tgt_inputs, state)
        # outputs(B,G,V)
        return outputs, state
