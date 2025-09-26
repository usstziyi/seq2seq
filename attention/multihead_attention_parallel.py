import torch
from torch import nn
from attention import DotProductAttention

"""
多头注意力：并行计算

画的是：侧面视图，维度(B,Dq)

queries(B,Q,Dq)
###########
###########
###########
###########

隐藏层变换

queries(B,Q,H)
##################################################################
##################################################################
##################################################################
##################################################################

并行变换

queries(B*N,Q,H/N)
######################
######################
######################
######################

######################
######################
######################
######################

######################
######################
######################
######################

多头注意力并行计算
outputs(B*N,Q,H/N)
######################
######################
######################
######################

######################
######################
######################
######################

######################
######################
######################
######################

恢复单头
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
class MultiHeadAttentionParallel(nn.Module):
    # key_size(键的最后一个维度)
    # query_size(查询的最后一个维度)
    # value_size(值的最后一个维度)
    # num_hiddens=单头隐藏层维度*头数，后面会拆分
    # num_heads(头数)
    # dropout(dropout概率)
    # bias(是否使用偏置项)
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttentionParallel, self).__init__(**kwargs)
    
        self.num_heads = num_heads # N


        # 多头并行大隐藏层
        self.W_q = nn.Linear(
            in_features=query_size, # Dq
            out_features=num_hiddens, # H
            bias=bias
        )
        self.W_k = nn.Linear(
            in_features=key_size, # Dg
            out_features=num_hiddens, # H
            bias=bias
        )
        self.W_v = nn.Linear(
            in_features=value_size, # Dv
            out_features=num_hiddens, # H
            bias=bias
        )

        # 注意力层
        self.attention = DotProductAttention(dropout)
        
        
        # 输出层(总的输出层)
        self.W_o = nn.Linear(
            in_features=num_hiddens, # H
            out_features=num_hiddens, # H
            bias=bias
        )
        


    def forward(self, queries, keys, values, valid_lens):
        
        # queries(B,Q,Dq)->(B,Q,H)
        queries = self.W_q(queries)
        # queries(B,Q,H)->(B,Q,N,H/N)
        queries = queries.reshape(queries.shape[0],queries.shape[1],self.num_heads,-1)
        # queries(B,Q,N,H/N)->(B,N,Q,H/N)
        queries = queries.permute(0,2,1,3)
        # queries(B,N,Q,H/N)->(B*N,Q,H/N)
        queries = queries.reshape(-1,queries.shape[2],queries.shape[3])
        
        # keys(B,G,Dg)->(B,G,H)
        keys = self.W_k(keys)
        keys = keys.reshape(keys.shape[0],keys.shape[1],self.num_heads,-1)
        keys = keys.permute(0,2,1,3)
        # keys(B*N,G,H/N)
        keys = keys.reshape(-1,keys.shape[2],keys.shape[3])
        
        # values(B,G,Dv)->(B,G,H)
        values = self.W_v(values)
        values = values.reshape(values.shape[0],values.shape[1],self.num_heads,-1)
        values = values.permute(0,2,1,3)
        # values(B*N,G,H/N)
        values = values.reshape(-1,values.shape[2],values.shape[3])

        if valid_lens is not None:
            # valid_lens(B)->(B*N)
            valid_lens = torch.repeat_interleave(
                input=valid_lens,
                repeats=self.num_heads,
                dim=0
            )

        # 多头并行计算缩放点积注意力输出
        # queries(B*N,Q,H/N)
        # keys(B*N,Q,H/N)
        # values(B*N,Q,H/N)
        # valid_lens(B*N)
        # output(B*N,Q,H/N)
        output = self.attention(queries, keys, values, valid_lens)  # 多头并行计算缩放点积注意力

        # 恢复单头
        # output(B*N,Q,H/N)->(B,N,Q,H/N)
        output = output.reshape(-1,self.num_heads,output.shape[1],output.shape[2])
        # output(B,N,Q,H/N)->(B,Q,N,H/N)
        output = output.permute(0,2,1,3)
        # output(B,Q,N,H/N)->(B,Q,H)
        output = output.reshape(output.shape[0],output.shape[1],-1)

        # 输出层
        # output(B,Q,H)->(B,Q,H)
        output = self.W_o(output)

        return output




