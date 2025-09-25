from attention import MultiHeadAttentionParallel
import torch

def main():
    num_heads = 5
    # 并行隐藏层维度
    num_hiddens = 20 * num_heads
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
    num_queries = 4  # Q
    num_kvpairs =  6 # G
    # 输入：
    # queries(B,Q,Dq)=(B,Q,H)
    # keys(B,G,Dg)=(B,G,H)
    # values(B,G,Dg)=(B,G,H)
    # valid_lens(B)
    queries = torch.ones((batch_size, num_queries, num_hiddens))
    keys = torch.ones((batch_size, num_kvpairs, num_hiddens))
    values = torch.ones((batch_size, num_kvpairs, num_hiddens))
    valid_lens = torch.tensor([3, 2])

    # outputs(B,Q,H*num_heads)
    outputs = model(
        queries=queries,
        keys=keys,
        values=values,
        valid_lens=valid_lens
    )
    print(outputs.shape)

if __name__ == '__main__':
    main()