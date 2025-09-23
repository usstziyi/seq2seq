import math
import torch
from torch import nn
from d2l import torch as d2l
import heatmaps as hm

"""
K=G
"""

# score(B,Q,G)
# valid_lens(B):batch中不同样本的有效长度不同，每个查询共用一个key 查询库
# valid_lens(B,Q):batch中不同样本的有效长度不同，每个查询有自己独立的 key 查询库，所以需要加一个维度来存储不同的 key 查询库大小
def masked_softmax(score, valid_lens):
    shape = score.shape
    if valid_lens is None:
        return nn.functional.softmax(score, dim=-1)
    else:
        if valid_lens.dim() == 1:
            # valid_lens(B,) -> (B*Q)举个例子[1,1,1,2,2,2,3,3,3]->样本1,样本2,样本3
            valid_lens = torch.repeat_interleave(valid_lens, score.shape[1])
        else:
            # valid_lens(B,Q) -> (B*Q)
            valid_lens = valid_lens.reshape(-1) # 先内后外->样本1,样本2,样本3
        # score(B,Q,G) -> (B*Q,G)->样本1,样本2,样本3
        # valid_lens(B*Q,)->样本1,样本2,样本3
        # score(B*Q,G)->样本1,样本2,样本3
        score = d2l.sequence_mask(score.reshape(-1, score.shape[-1]), valid_lens, value=-1e6)
        # (B*Q,G)->(B,Q,G)
        score = score.reshape(shape)
        # return(B,Q,G)
        return nn.functional.softmax(score, dim=-1)

# 加性注意力网络
# T = Q
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
    # valid_lens(B)
    def forward(self, queries, keys, values, valid_lens):
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


# 缩放点积注意力网络
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries(B,Q,Qd)
    # keys(B,K,Kd)
    # values(B,K,V)
    # valid_lens(B,) or (B,Q)
    def forward(self, queries, keys, values, valid_lens=None):
        # d = Qd
        d = queries.shape[-1]
        # queries(B,Q,Qd)=(2,1,2)
        # keys'T(B,Kd,K)=(2,2,10)  # 关键点:keys转置
        # score(B,Q,K)=(2,1,10)
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)  # 缩放点积

        # --------------------------------------

        # scores(B,Q,K)=(2,1,10)
        # valid_lens(B,) or (B,Q)
        # attention_weights(B,Q,K)=(2,1,10)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # attention_weights(B,Q,K)=(2,1,10)
        # values(B,K,V)=(2,10,2)
        # outputs(B,Q,V)=(2,1,2)
        outputs = torch.bmm(self.dropout(self.attention_weights), values)
        return outputs



def test_additive_attention():
    """测试加性注意力"""
    # queries(B,Q,Qd)
    queries = torch.normal(mean=0, std=1, size=(2, 1, 20))
    # keys(B,K,Qd)
    keys = torch.ones(size=(2, 10, 2))
    # values(B,K,V)
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    # valid_lens(B)=(2,)
    valid_lens = torch.tensor([2, 6])

    # 创建模型
    model = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
    model.eval()
    # queries(B,Q,Qd)=(2,1,20)
    # keys(B,K,Kd)=(2,10,2)
    # values(B,K,V)=(2,10,4)
    # valid_lens(B)=(2,)
    # outputs(B,Q,V)=(2,1,4)
    with torch.no_grad():
        outputs = model(queries, keys, values, valid_lens)
    print(outputs)
    print(outputs.shape)
    return model.attention_weights

def test_dot_product_attention():
    """测试缩放点积注意力"""
    # queries(B,Q,Qd)=(2,1,2)
    queries = torch.normal(0, 1, (2, 1, 2))
    # keys(B,K,Kd)=(2,10,2)
    keys = torch.ones(size=(2, 10, 2))
    # values(B,K,V)
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    # valid_lens(B)=(2,)
    valid_lens = torch.tensor([2, 6])

    # 创建模型
    model = DotProductAttention(dropout=0.5)
    model.eval()
    # queries(B,Q,Qd)
    # keys(B,K,Kd)
    # values(B,K,V)
    # valid_lens(B,) or (B,Q)
    with torch.no_grad():
        outputs = model(queries, keys, values, valid_lens)
        print(outputs.shape)
        print(outputs)
    return model.attention_weights


def main():
    attention_weights = test_additive_attention()
    hm.show_heatmaps(
        # attention_weights(B,Q,K)=(2,1,10)->(1,1,2,10)
        matrices=attention_weights.reshape(1,1,attention_weights.shape[0],attention_weights.shape[2]),
        xlabel='Keys',
        ylabel='Queries',
        figsize=(9, 6),   # 宽一些，适合 2x3 布局
        cmap='Reds'
    )



    attention_weights = test_dot_product_attention()
    hm.show_heatmaps(
        # attention_weights(B,Q,K)=(2,1,10)->(1,1,2,10)
        matrices=attention_weights.reshape(1,1,attention_weights.shape[0],attention_weights.shape[2]),
        xlabel='Keys',
        ylabel='Queries',
        figsize=(9, 6),   # 宽一些，适合 2x3 布局
        cmap='Reds'
    )

if __name__ == '__main__':
    main()