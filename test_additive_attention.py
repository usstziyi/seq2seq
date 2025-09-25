import torch
from common import heatmaps as hm
from attention import AdditiveAttention


def main():
    # queries(B,Q,Dq)
    queries = torch.normal(mean=0, std=1, size=(2, 1, 20))
    # keys(B,G,Dg)
    keys = torch.ones(size=(2, 10, 2))
    # values(B,G,Dv)
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    # valid_lens(B)=(2,)
    valid_lens = torch.tensor([2, 6])

    # 创建模型
    model = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
    model.eval()
    # queries(B,Q,Dq)=(2,1,20)
    # keys(B,G,Dg)=(2,10,2)
    # values(B,G,Dv)=(2,10,4)
    # valid_lens(B)=(2,)
    # outputs(B,Q,Dv)=(2,1,4)
    with torch.no_grad():
        outputs = model(queries, keys, values, valid_lens)
    print(outputs)
    print(outputs.shape)
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