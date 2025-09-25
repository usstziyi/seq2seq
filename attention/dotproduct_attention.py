import math
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
    对象内容:(G,Dv)
    原查对象:(Q)由外部传入,长度不固定
    权重分配:(Q,G)
    加权求和:(Q,G)*(G,Dv)->(Q,Dv)->Q拿到自己的V

    2维(B,Q)
    原查对象(B,Q)
    被查对象(B,G)
    对象内容(G,Dv)
    权重分配(B,Q,G)
    加权求和:(B,Q,G)bmm(B,G,Dv)->(B,Q,Dv)->(B,Q)拿到自己的V

"""


# 缩放点积注意力网络,Dq=Gd才行
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries(B,Q,Dq)
    # keys(B,G,Dg)
    # values(B,G,Dv)
    # valid_lens(B) or (B,Q)
    def forward(self, queries, keys, values, valid_lens=None): # 可以按(B,Q)块处理
        # d = Dq
        d = queries.shape[-1]
        # queries(B,Q,Dq)
        # keys'T(B,Dg,G)  # 关键点:Dq=Dg，这里才可以bmm
        # score(B,Q,G)
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)  # 缩放点积

        # --------------------------------------

        # scores(B,Q,G)
        # valid_lens(B) or (B,Q)
        # attention_weights(B,Q,G)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # attention_weights(B,Q,G)
        # dropped_weights(B,Q,G)
        dropped_weights = self.dropout(self.attention_weights)
        # dropped_weights(B,Q,G)
        # values(B,G,Dv)
        # outputs(B,Q,Dv)
        outputs = torch.bmm(dropped_weights, values)
        return outputs




