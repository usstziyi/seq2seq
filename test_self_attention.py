from attention import MultiHeadAttentionParallel
import torch

"""
1.三个矩阵投影(根据情况,可能不进行投影)

2. 投影后 shape 要求
Q 和 K 的最后一维必须相等(即dk),否则无法做QK'T
V 的最后一维可以不同(dv),它控制输出维度。
Q、K、V 的第一个维度(序列长度 n)通常相同，因为它们来自同一个输入序列。
但在某些变体(如 cross-attention)中,Q 和 K/V 的序列长度可以不同。

2. 举例(Transformer 中的常见设置)
在原始 Transformer 论文中：
d=512(输入维度)
dk = dv = 64(头维度)
多头注意力中，有 8 个头，每个头处理 64 维，拼接后回到 512 维。
此时 Q、K、V 的 shape 都是 nx64(单头),所以看起来一样，但这不是必须的。
"""


# 这里的测试用例是：用多头并行计算自注意力
def main():
    num_heads = 8
    single_num_hiddens = 64
    # 从多头注意力内部看
    # H=h*N
    # H表示隐藏层总维度
    # h表示每个头的维度

    # 从输入输出的角度看，要给个大的H,应为H进入后，会被平分到多个头中参与计算
    # 最后返回的H包含多个头的信息，每个头学到的信息包括比如：
    # 1. 不同位置的信息
    # 2. 不同特征的信息
    # 3. 不同时间步的信息
    # 4. 不同样本的信息
    # 5. 不同层的信息
    # ...

    # 输入：
    # queries(B,Q,Dq)=(B,Q,H)
    # keys(B,G,Dg)=(B,G,H)
    # values(B,G,Dv)=(B,G,H)
    # valid_lens(B)
    # 输出：
    # outputs(B,Q,H)=(B,Q,H)
    # 这里的H维度包含多个头学习到的信息
    num_hiddens = single_num_hiddens * num_heads # 512
    model = MultiHeadAttentionParallel(
        query_size=num_hiddens,
        key_size=num_hiddens,
        value_size=num_hiddens,
        num_hiddens=num_hiddens,
        num_heads=num_heads,
        dropout=0.5
    )
    model.eval()    

    batch_size = 2   # B

    # 在自注意力中Q=G
    num_queries = 6  # Q
    num_kvpairs =  6 # G

    # 输入：
    # queries(B,Q,Dq)=(B,Q,H)
    # keys(B,G,Dg)=(B,G,H)
    # values(B,G,Dg)=(B,G,H)
    # valid_lens(B)
    # queries = torch.ones((batch_size, num_queries, num_hiddens))
    # keys = torch.ones((batch_size, num_kvpairs, num_hiddens))
    # values = torch.ones((batch_size, num_kvpairs, num_hiddens))
    x = torch.ones((batch_size, num_queries, num_hiddens))
    valid_lens = torch.tensor([3, 2])

    # x(B,G,H)=(B,G,512)
    # outputs(B,G,H*single_num_heads)=(B,G,512)
    outputs = model(
        queries=x,
        keys=x,
        values=x,
        valid_lens=valid_lens
    )
    print(outputs.shape)

if __name__ == '__main__':
    main()