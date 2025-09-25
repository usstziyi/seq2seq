import torch
from common import heatmaps as hm
from attention import DotProductAttention


def main():
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
    attention_weights = model.attention_weights

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