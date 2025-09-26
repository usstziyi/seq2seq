import torch
from torch import nn
from attention import DotProductAttention

"""
多头注意力：顺序计算

画的是：侧面视图，维度(B,Dq)

queries(B,Q,Dq)
###########
###########
###########
###########

隐藏层变换

queries(B,Q,H)
######################
######################
######################
######################

单头注意里输出
output(B,Q,H)
######################
######################
######################
######################

拼接多头
outputs(B,Q,H*N)
##################################################################
##################################################################
##################################################################
##################################################################

输出层
outputs(B,Q,H*N)
##################################################################
##################################################################
##################################################################
##################################################################






"""

# 多头注意力网络内嵌(缩放点积注意力网络)，线性顺序计算
class MultiHeadAttention(nn.Module):
    # key_size(key的最后一个维度)
    # query_size(query的最后一个维度)
    # value_size(value的最后一个维度)
    # num_hiddens(每个头的隐藏维度)
    # num_heads(头的数量)
    # dropout(dropout概率)
    # bias(是否使用偏置项)
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
    
        self.num_heads = num_heads # N


        # 隐藏层(每个头的隐藏层)
        self.W_q_heads = []
        self.W_k_heads = []
        self.W_v_heads = []
        # 为每个头创建独立的线性层
        for head in range(num_heads):
            self.W_q_heads.append(
                nn.Linear(
                    in_features=query_size, # Dq
                    out_features=num_hiddens, # H
                    bias=bias
                )
            )
            self.W_k_heads.append(
                nn.Linear(
                    in_features=key_size, # Dg
                    out_features=num_hiddens, # H
                    bias=bias
                )
            )
            self.W_v_heads.append(
                nn.Linear(
                    in_features=value_size, # Dv
                    out_features=num_hiddens, # H
                    bias=bias
                )
            )
        

        # 注意力层
        self.attention = DotProductAttention(dropout)

        # 输出层(总的输出层)
        self.W_o = nn.Linear(
            in_features=num_hiddens * num_heads, # H * N
            out_features=num_hiddens * num_heads, # H * N
            bias=bias
        )


    def forward(self, queries, keys, values, valid_lens):
        
        # 存储每个头的输出
        head_outputs = []
        
        # 多头轮流计算注意力
        for i in range(self.num_heads):
            # queries(B,Q,Dq)
            # queries_i(B,Q,H)
            queries_i = self.W_q_heads[i](queries)
            # keys(B,G,Dg)
            # keys_i(B,G,H)
            keys_i = self.W_k_heads[i](keys)
            # values(B,G,Dv)
            # values_i(B,G,H)
            values_i = self.W_v_heads[i](values)
            
            # 计算当前头的注意力
            # queries_i(B,Q,H)
            # keys_i(B,G,H)
            # attention_i(B,Q,G)
            # values_i(B,G,H)
            # output_i(B,Q,H)
            output_i = self.attention(queries_i, keys_i, values_i, valid_lens) # 计算当前头的缩放点积注意力输出
            
            # 存储每个头的输出
            # head_outputs(B,Q,H),(B,Q,H)...
            head_outputs.append(output_i)
        
        # 拼接所有头的输出的最后一个维度
        # output_concat(B,Q,H*N)
        output_concat = torch.cat(head_outputs, dim=-1)
        
        # 输出层
        # output_concat(B,Q,H*N)
        # out(B,Q,H*N)
        output = self.W_o(output_concat)
        return output


