import math
import torch
from torch import nn
from d2l import torch as d2l

"""
多头注意力
"""

# 多头注意力网络内嵌(缩放点积注意力网络)，线性顺序计算
class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
    
  
        self.attention = d2l.DotProductAttention(dropout)

        self.W_q_heads = []
        self.W_k_heads = []
        self.W_v_heads = []
        # 为每个头创建独立的线性层
        for head in range(num_heads):
            self.W_q_heads.append(
                nn.Linear(
                    in_features=query_size, # Q
                    out_features=num_hiddens, # H
                    bias=bias
                )
            )
            self.W_k_heads.append(
                nn.Linear(
                    in_features=key_size, # G
                    out_features=num_hiddens, # H
                    bias=bias
                )
            )
            self.W_v_heads.append(
                nn.Linear(
                    in_features=value_size, # V
                    out_features=num_hiddens, # H
                    bias=bias
                )
            )
        
        # 输出层
        self.W_o = nn.Linear(
            in_features=num_hiddens*num_heads,
            out_features=num_hiddens*num_heads,
            bias=bias
        )


    def forward(self, queries, keys, values, valid_lens):
        batch_size = queries.shape[0]
        num_queries = queries.shape[1]
        num_kvpairs = keys.shape[1]
        
        # 存储每个头的输出
        head_outputs = []
        num_heads = len(self.W_q_heads)
        for i in range(num_heads):
            # 每个头独立进行线性变换
            # queries_i 的形状: (batch_size, num_queries, head_dim)
            queries_i = self.W_q_heads[i](queries)
            # keys_i 的形状: (batch_size, num_kvpairs, head_dim)
            keys_i = self.W_k_heads[i](keys)
            # values_i 的形状: (batch_size, num_kvpairs, head_dim)
            values_i = self.W_v_heads[i](values)
            
            # 计算当前头的注意力
            # output_i 的形状: (batch_size, num_queries, head_dim)
            output_i = self.attention(queries_i, keys_i, values_i, valid_lens)
            head_outputs.append(output_i)
        
        # 拼接所有头的输出
        # output_concat 的形状: (batch_size, num_queries, num_hiddens)
        output_concat = torch.cat(head_outputs, dim=-1)
        print(f"每个头的输出shape: {head_outputs[0].shape}")
        print(f"拼接后的shape: {output_concat.shape}")
        
        return self.W_o(output_concat)


# num_hiddens, num_heads = 100, 5
# attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
# attention.eval()


# batch_size = 2
# num_queries = 4
# num_kvpairs =  6
# valid_lens = torch.tensor([3, 2])
# X = torch.ones((batch_size, num_queries, num_hiddens))
# Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
# attention(X, Y, Y, valid_lens).shape