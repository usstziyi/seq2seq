import torch
import math
from torch.nn import functional as F
from torch import nn
from decoder import BahdanauDecoder
from common import heatmaps as hm
from d2l import torch as d2l
from decoder import BahdanauEncoderDecoder
from decoder import Seq2SeqEncoder
from common import dataline as dl
import collections



# 定义训练函数
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    # 初始化模型参数
    def xavier_init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # 初始化线性层的权重为均匀分布
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)  # 初始化循环神经网络层的权重为均匀分布
    # 应用初始化函数
    net.apply(xavier_init_weights)
    net.to(device)
    net.train()
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss(ignore_index=tgt_vocab['<pad>']) # 交叉熵损失函数，用于多分类问题

    # 初始化绘图
    fig, ax, line, x_list, y_list = dl.init_plot(lr, title='Training loss vs Epoch (Bahdanau Encoder-Decoder)')

    # 训练
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        # 定义累加器，用于记录每个epoch的损失总和和词元总数
        metric = d2l.Accumulator(2)
        # 按照批次训练，好处是梯度下降更稳定
        for batch in data_iter:
            # 拿取批次数据
            # src_inputs(B,G)
            # tgt_inputs(B,G)
            # tgt_valid_len(B)
            src_inputs, src_valid_len, tgt_inputs, tgt_valid_len = [x.to(device) for x in batch]

            # 制造教学输入
            # bos(B,1)
            bos = torch.tensor([tgt_vocab['<bos>']] * tgt_inputs.shape[0], device=device).reshape(-1, 1) # 目标输入在训练阶段需要添加<bos>标记
            # tgt_teach(B,1+G-1)->(B,G)
            tgt_teach = torch.cat([bos, tgt_inputs[:, :-1]], 1) # 目标输入在训练阶段需要去掉最后一个词元
            
            # 1.清零梯度
            optimizer.zero_grad()
            # 2.训练，执行强制教学
            # src_inputs(B,G)
            # tgt_teach(B,G)
            # src_valid_len(B)
            # outputs(B,G,V)
            outputs, _ = net(src_inputs, tgt_teach, src_valid_len) 
            # 3.计算损失
            # outputs(B,G,V)
            # tgt_inputs(B,G)
            # l(B,G)->(B)
            l = loss(outputs, tgt_inputs) # mean
            # l(B)->l(1)
            l = l.sum() # sum
            # 4.反向传播
            l.backward()
            # 5.梯度裁剪
            d2l.grad_clipping(net, 1)
            # 6.更新参数
            optimizer.step()

            # 累加批次的损失和词元数量
            with torch.no_grad():
                metric.add(l, tgt_valid_len.sum())
        train_loss = metric[0] / metric[1]
        train_speed = metric[1] / timer.stop()
        print(f'epoch {(epoch + 1):3d}/{num_epochs}, loss {train_loss:.3f}, {train_speed:.1f} 词元/秒 {str(device)}')

        # 更新绘图
        dl.update_plot(epoch+1, train_loss, x_list, y_list, line, ax)
    # 关闭绘图
    dl.close_plot()




# 带遮蔽的softmax交叉熵损失函数
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def __init__(self, ignore_index=0, **kwargs):  # 默认设为 0，适配你的 padding
        super(MaskedSoftmaxCELoss, self).__init__(
            ignore_index=ignore_index, # 忽略目标中值等于 ignore_index 的位置
            reduction='none',  # 保持逐元素计算
            **kwargs
        )

    def forward(self, input, target):
        # 计算逐位置损失，无效位置（ignore_index）损失为 0
        # input 需要 (B,V,G)，target 是 (B,G)
        # input(B,G,V)->(B,V,G)
        unweighted_loss = super().forward(input.permute(0, 2, 1), target)

        # 创建有效位置掩码：target 中不等于 ignore_index 的位置为 1
        # mask(B,G)
        mask = (target != self.ignore_index).float()

        # 对每个样本，计算有效损失总和
        # loss_sum(B)
        loss_sum = (unweighted_loss * mask).sum(dim=1)

        # 获取每个样本的有效 token 数量（避免除零）
        # valid_token_count(B)
        valid_token_count = mask.sum(dim=1).clamp(min=1)  # 至少为1，防止除零

        # 计算每个句子的平均损失
        # weighted_loss(B)
        weighted_loss = loss_sum / valid_token_count

        return weighted_loss

# 定义预测函数，端到端预测
# B=1
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    # 处理句子
    # 源句子小写化并分词
    source = src_sentence.lower().split(' ')
    # 源句子词元化
    src_tokens = src_vocab[source] + [src_vocab['<eos>']]
    # 源句子有效长度=Q
    src_valid_len = torch.tensor([len(src_tokens)], dtype=torch.long, device=device)
    # 源句子截断、填充,src_tokens=G,强制到G
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # src_inputs(B,G)=(1,G)
    src_inputs = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)

    pred_seq, attention_weight_seq = [], []
    
    # 编码
    # B=1
    # T=G
    net.eval()
    # src_inputs(B,G)=(1,G)
    # states(G,B,H)=(G,1,H)
    # last_state(L,B,H)=(L,1,H)
    states, last_state = net.encoder(src_inputs)

    # 初始化解码器
    # states(G,B,H)->(B,G,H)=(1,G,H)
    # last_state(L,B,H)=(L,1,H)
    # src_valid_len(1)
    # state=(states, last_state, src_valid_len)
    state = net.decoder.init_state(states, last_state, src_valid_len)

    # 解码
    # B=1
    # T=Q
    # 第一个输入是<bos>
    # next_input(B,Q)=(1,1)
    next_input = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)

    for _ in range(num_steps):
        # next_input(B,Q)=(1,1)
        # states(B,G,H)=(1,G,H)
        # last_state(L,B,H)=(L,1,H)
        # valid_lens(B)=(1)
        # state=(states, last_state, valid_lens)
        # outputs(B,Q,V)=(1,1,V)
        # state=(states, last_state, valid_lens)
        outputs, state = net.decoder(next_input, state)
        # tgt_hat(1,1,V)->(1,1)->next_input
        next_input = outputs.argmax(dim=2)
        # next_input(B,Q)=(1,1)->pred(1)
        pred = next_input.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重
        if save_attention_weights:
            # attention_weights(B,Q,G)=(1,1,G)
            # attention_weight_seq:(1,1,G),(1,1,G)
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列<eos>词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        pred_seq.append(pred) # 词元索引列表

    outputs = ' '.join(tgt_vocab.to_tokens(pred_seq)) # 词元列表转换为字符串
    # 把attention_weight_seq转成 tensor,在dim=1上拼接
    # attention_weight_seq:list(1,1,G),(1,1,G)
    # attention_weights:tensor(1,Q,G)
    attention_weights = torch.cat([weight for weight in attention_weight_seq], dim=1)
    return outputs, attention_weights

# 预测序列的评估-布鲁算法
def bleu(pred_src, label_src, k):
    # 预测序列和标签序列都小写化并分词
    pred_tokens, label_tokens = pred_src.lower().split(' '), label_src.lower().split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    # 计算惩罚分数
    score = math.exp(min(0, 1 - len_label / len_pred))
    # 计算n元语法的匹配率
    for n in range(1, k + 1):
        # 统计标签序列中n元语法的出现次数
        num_n_grams = len_label - n + 1 # 分母：标签序列中 n元语法的总数量
        label_subs = collections.defaultdict(int) # label_subs字典
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1 # label子序列出现的次数,因为 n元语法子序列有重复的，所以这里要归类累加
        
        # 统计pred子序列在label子序列中出现
        num_matches = 0 # 分子：pred序列与标签序列中匹配的 n元语法的数量
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0: # pred子序列在label子序列中出现
                num_matches += 1 # 匹配数加1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1 # 匹配的label子序列次数减1
        # 计算n元语法的匹配率
        p_n = num_matches / num_n_grams
        # 计算n元语法的BLEU分数
        score *= math.pow(p_n, math.pow(0.5, n))
    return score




def main():
    # 1.超参数
    # E=embed_size=32
    # H=num_hiddens=32
    # L=num_layers=2
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    # B=batch_size=64
    # G=num_steps=10
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

    # 2.加载数据集
    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
    
    # 3.初始化模型
    # V1 = len(src_vocab)
    # V2 = len(tgt_vocab)
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = BahdanauDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = BahdanauEncoderDecoder(encoder, decoder)

    # 4.训练模型
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    # 5.预测
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    print('----------------------------------------------------------------')
    attention_weights_list = []
    for eng, fra in zip(engs, fras):
        # attention_weights(B,Q,G)=(1,Q,G)
        translation, attention_weights = predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=True)
        # 因为预测阶段，每个句子的attention_weights的Q不同，所以这里attention_weights_list不做拼接，保留list
        # attention_weights_list:list(1,Q,G),(1,Q,G)...
        attention_weights_list.append(attention_weights)
        # 预测评估
        print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')



    # 6.可视化注意力权重
    # attention_weights(1,Q,G)=(1,1,10)
    attention_weights_padded_list = []
    # attention_weights(1,Q,G)
    for attention_weights in attention_weights_list:
        # 将 attention_weights 的第1维长度扩展到到 G，右补 0
        # attention_weights_padded(1,G,G)=(1,10,10)
        attention_weights_padded = F.pad(attention_weights, (0, 0, 0, 10 - attention_weights.shape[1],0,0), value=0)
        attention_weights_padded_list.append(attention_weights_padded.unsqueeze(1))
    
    # matrices(1,N,Q,Q)=(1,4,10,10)
    matrices = torch.cat(attention_weights_padded_list, dim=1).cpu()

    hm.show_heatmaps(
            matrices = matrices,
            xlabel='Key',
            ylabel='Query',
            figsize=(12,5),
            cmap='Reds'
            )       



if __name__ == '__main__':
    main()