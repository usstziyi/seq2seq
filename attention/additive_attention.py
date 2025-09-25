import torch
from torch import nn
from common import heatmaps as hm
from common import masked_softmax


"""
注意力机制本质:
    0维(唐伯虎点秋香)
    首先要有被查询的对象，和对象指代的内容，
    然后要有人去查,去和每个对象比对,得到得分,再Mask+Softmax得到权重,
    最后根据权重，去加权求和对象指代的内容，得到最终的输出。

    1维(Q)
    被查对象:(G)由外部传入,固定长度
    对象内容:(G,V)
    原查对象:(Q)由外部传入,长度不固定
    权重分配:(Q,G)
    加权求和:(Q,G)*(G,V)->(Q,V)->Q拿到自己的V

    2维(B,Q)
    原查对象(B,Q)
    被查对象(B,G)
    对象内容(G,V)
    权重分配(B,Q,G)
    加权求和:(B,Q,G)bmm(B,G,V)->(B,Q,V)->(B,Q)拿到自己的V

"""


# 加性注意力网络
class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        # (Gd,H)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        # (Qd,H)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        # (H,1)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    # queries(B,Q,Qd)
    # keys(B,G,Gd)
    # values(B,G,V)
    # valid_lens(B) or (B,Q)
    def forward(self, queries, keys, values, valid_lens): # 可以按(B,Q)块处理
        # (B,Q,Qd)*(Qd,H)->(B,Q,H)
        queries = self.W_q(queries) # 一次处理Q步推理
        # (B,G,Gd)*(Gd,H)->(B,G,H)
        keys = self.W_k(keys)
        # queries(B,Q,H)->(B,Q,1,H)
        # keys(B,G,H)->(B,1,G,H)
        # features(B,Q,G,H)
        features = queries.unsqueeze(2) + keys.unsqueeze(1) # 加性
        # features(B,Q,G,H)->(B,Q,G,H)
        features = torch.tanh(features)
        # (B,Q,G,H)*(H,1)->(B,Q,G,1)
        scores = self.w_v(features) 
        # scores(B,Q,G,1)->(B,Q,G)
        scores = scores.squeeze(-1) # 关键点：去掉最后一个维度，变成(B,Q,G)
        
        # --------------------------------------

        # scores(B,Q,G)
        # valid_lens(B)
        # attention_weights(B,Q,G)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # attention_weights(B,Q,G)
        # dropped_weights(B,Q,G)
        dropped_weights = self.dropout(self.attention_weights)
        # dropped_weights(B,Q,G)
        # values(B,G,V)
        # outputs(B,Q,V)
        outputs = torch.bmm(dropped_weights, values)
        return outputs


