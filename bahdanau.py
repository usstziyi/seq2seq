import torch
import math
import collections
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import attention as at
import heatmaps as hm
import torch.nn.functional as F

# 编码器接口
class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, inputs):
        raise NotImplementedError

# 编码器接口
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, states, last_state,valid_lens):
        raise NotImplementedError

    def forward(self, inputs, state):
        raise NotImplementedError

# Seq2Seq编码器网络模型：输入源句子，输出编码器的状态
class Seq2SeqEncoder(Encoder):
    # 定义网络
    def __init__(self, src_vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层(V,E)
        self.embedding = nn.Embedding(
            num_embeddings=src_vocab_size, 
            embedding_dim=embed_size
        )
        # 循环神经网络层(E,H,L)
        self.rnn = nn.GRU(
            input_size=embed_size, 
            hidden_size=num_hiddens, 
            num_layers=num_layers, 
            dropout=dropout,
            batch_first=False
        )
        # 没有输出层
    
    # 定义前向传播
    # inputs(B,Q)
    # states(Q,B,H)
    # state(L,B,H)
    def forward(self, inputs):
        # inputs(B,Q)->(B,Q,E)
        inputs = self.embedding(inputs)
        # inputs(Q,B,E)
        inputs = inputs.permute(1, 0, 2)
        # states(Q,B,H)
        # state(L,B,H)
        states, last_state = self.rnn(inputs)
        # states(Q,B,H):包含每个时间步最后一层的信息
        # last_state(L,B,H):包含最后一个时间步每一层的信息
        return states, last_state

# 注意力机制解码器接口
class AttentionDecoder(Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError



# Bahdanau解码器网络模型：输入原句子，输出目标句子
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        
        # 嵌入层(V,E)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size
        )
        # 注意力层网络
        self.attention = at.AdditiveAttention(
            key_size=num_hiddens,
            query_size=num_hiddens,
            num_hiddens=num_hiddens,
            dropout=dropout
        )

        # 循环层(E+H,H)
        self.rnn = nn.GRU(
            input_size=embed_size + num_hiddens,
            hidden_size=num_hiddens,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=False
        )
        # 输出层(H,V)
        self.dense = nn.Linear(
            in_features=num_hiddens,
            out_features=vocab_size
        )

    
    # 注意：待翻译句子首先要经过编码器粗加工，得到半成品给解码器用
    # states(Q,B,H)
    # state(L,B,H))
    # valid_lens(B,)：待翻译句子的有效长度，此时只是初加工，还没有翻译
    # 初始化解码器状态
    def init_state(self, states, last_state, valid_lens):
        # states(Q,B,H)->(B,Q,H)
        states = states.permute(1, 0, 2)
        return states, last_state, valid_lens

    # inputs(B,Q)
    def forward(self, inputs, state):
        # states(B,Q,H)
        # last_state(L,B,H)
        # valid_lens(B,)
        states, last_state, valid_lens = state

        outputs, self._attention_weights = [], []


        # states(B,Q,H)->keys(B,K,Kd)=(B,Q,H)
        keys = states # 编码器所有时间步的隐藏状态,作为注意力层的键
        # states(B,Q,H)->values(B,K,V)=(B,Q,H)
        values = states # 编码器所有时间步的隐藏状态,作为注意力层的值
 
        # 1.计算嵌入层
        # inputs(B,Q)->(B,Q,E)->(Q,B,E)
        inputs = self.embedding(inputs).permute(1, 0, 2)
        # 按时序步Q迭代 input(B,E)
        for input in inputs:

            # 2.计算注意力层
            # last_state(L,B,H)->(B,H)->query(B,1,H)
            query = last_state[-1].unsqueeze(1) # 编码器最后一个时间步的隐藏状态
            # query(B,Q,Qd)=(B,1,H)
            # keys(B,K,Kd)=(B,Q,H)
            # values(B,K,V)=(B,Q,H)
            # valid_lens(B,)
            # context(B,Q,V)=(B,1,H)
            context = self.attention(query, keys, values, valid_lens) # 调用注意力层，计算上下文
            # 解释：用最新的最后一步隐状态去查所有时间步的隐藏状态，求得分，求 weights,得到上下文向量
            # 被查对象：所有时间步的隐藏状态(B,Q,H),多少个Q个

            # attention_weights（B,Q,K)=(B,1,Q)
            # list((B,1,Q),(B,1,Q)...(B,1,Q)）一共Q个
            self._attention_weights.append(self.attention.attention_weights.detach())

            # 3.计算循环层
            # input(B,E)->(B,1,E)
            input = input.unsqueeze(1)
            # input(B,1,E)
            # context(B,1,H)
            # x(B,1,E+H)
            x = torch.cat((input, context), dim=-1) # 拼接上下文和输入，在特征维度上拼接
            # x(1,B,E+H)
            x = x.permute(1, 0, 2)
      
            # out(1,B,H)
            # last_state(L,B,H)
            out, last_state = self.rnn(x, last_state)
            outputs.append(out) # (1,B,H),(1,B,H),(1,B,H)...



        # outputs(Q,B,H)
        outputs = torch.cat(outputs,dim=0)
        # outputs(Q,B,H)->(Q,B,V)->(B,Q,V)
        outputs = self.dense(outputs).permute(1, 0, 2)
        return outputs, (states, last_state, valid_lens)

    @property
    def attention_weights(self):
        # weight(B,1,Q) 一共Q个
        # attention_weights(B,Q,Q)
        attention_weights = torch.cat([weight for weight in self._attention_weights], dim=1)
        return attention_weights


# 编码器-解码器
class EncoderDecoder(nn.Module):
    # 定义网络
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    
    # 定义前向传播
    # src_inputs(B,Q)
    # tgt_inputs(B,Q)
    # valid_lens(B,)
    def forward(self, src_inputs, tgt_inputs, valid_lens):
        # 编码
        states, last_state = self.encoder(src_inputs)
        # 初始化解码器 state
        state = self.decoder.init_state(states,last_state,valid_lens)
        # 解码
        # tgt_inputs(B,Q)
        # return outputs(B,Q,V), (states, last_state, valid_lens)
        return self.decoder(tgt_inputs, state)

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
    fig, ax, line, x_list, y_list = init_plot(lr)

    # 训练
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        # 定义累加器，用于记录每个epoch的损失总和和词元总数
        metric = d2l.Accumulator(2)
        # 按照批次训练，好处是梯度下降更稳定
        for batch in data_iter:
            # 拿取批次数据
            # src_inputs(B,Q)
            # tgt_inputs(B,Q)
            # tgt_valid_len(B)
            src_inputs, src_valid_len, tgt_inputs, tgt_valid_len = [x.to(device) for x in batch]

            # 制造教学输入
            # bos(B,1)
            bos = torch.tensor([tgt_vocab['<bos>']] * tgt_inputs.shape[0], device=device).reshape(-1, 1) # 目标输入在训练阶段需要添加<bos>标记
            # tgt_teach(B,1+Q-1)->(B,Q)
            tgt_teach = torch.cat([bos, tgt_inputs[:, :-1]], 1) # 目标输入在训练阶段需要去掉最后一个词元
            
            # 1.清零梯度
            optimizer.zero_grad()
            # 2.训练，执行强制教学
            # src_inputs(B,Q)
            # tgt_teach(B,Q)
            # src_valid_len(B,)
            # outputs(B,Q,V)
            outputs, _ = net(src_inputs, tgt_teach, src_valid_len) 
            # 3.计算损失
            # outputs(B,Q,V)
            # tgt_inputs(B,Q)
            # l(B,Q)->(B,)
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
        update_plot(epoch+1, train_loss, x_list, y_list, line, ax)
    # 关闭绘图
    close_plot()


# 初始化绘图
def init_plot(lr):
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots(figsize=(10, 6))
    x_list = [] # x轴数据
    y_list = [] # y轴数据
    line, = ax.plot(x_list, y_list, 'b-', linewidth=2, label='Perplexity')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('loss')
    ax.set_title(f'bahdanau attention Training loss vs Epoch (lr={lr})')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    return fig, ax, line, x_list, y_list
    # fig是画布
    # ax是坐标系区域(axis是带刻度的图表框)，一个fig 上可以有多个 ax
    # line是线对象
    # x_list和y_list是数据列表


# 更新绘图
def update_plot(x_item, y_item, x_list, y_list, line, ax):
    x_list.append(x_item)
    y_list.append(y_item)
    line.set_xdata(x_list)
    line.set_ydata(y_list)
    ax.set_xlim(0, x_item + 2)  # 确保x轴范围包含当前epoch，右边预留2个单位
    ax.set_ylim(0, max(y_list) * 1.1 if y_list else 1)  # 防止空列表报错，y轴预留10%
    plt.draw()
    plt.pause(0.01)

# 关闭绘图
def close_plot():
    plt.ioff()  # 关闭交互模式->恢复默认行为
    plt.show()  # 阻塞，保持窗口打开直到用户手动关闭


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
        # input 需要 (B,V,Q)，target 是 (B,Q)
        # input(B,Q,V)->(B,V,Q)
        unweighted_loss = super().forward(input.permute(0, 2, 1), target)  # (B, Q)

        # 创建有效位置掩码：target 中不等于 ignore_index 的位置为 1
        # mask(B,V)
        mask = (target != self.ignore_index).float()

        # 对每个样本，计算有效损失总和
        # loss_sum(B,)
        loss_sum = (unweighted_loss * mask).sum(dim=1)

        # 获取每个样本的有效 token 数量（避免除零）
        # valid_token_count(B,)
        valid_token_count = mask.sum(dim=1).clamp(min=1)  # 至少为1，防止除零

        # 计算每个句子的平均损失
        # weighted_loss(B,)
        weighted_loss = loss_sum / valid_token_count

        return weighted_loss

# 定义预测函数，端到端预测
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    # 处理句子
    # 源句子小写化并分词，source(Q,)
    source = src_sentence.lower().split(' ')
    # 源句子词元化,src_tokens(Q,)
    src_tokens = src_vocab[source] + [src_vocab['<eos>']]
    # 源句子有效长度
    src_valid_len = torch.tensor([len(src_tokens)], dtype=torch.long, device=device)
    # 源句子截断、填充,src_tokens(Q,)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # src_inputs（B,Q)=(1,Q)
    src_inputs = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)

    pred_seq, attention_weight_seq = [], []
    
    # 编码
    net.eval()
    # src_inputs(B,Q)=(1,Q)
    # states(Q,B,H)=(Q,1,H)
    # last_state(L,B,H)=(L,1,H)
    states, last_state = net.encoder(src_inputs)

    # 初始化解码器
    # states(Q,B,H)=(Q,1,H)
    # last_state(L,B,H)=(L,1,H)
    # src_valid_len(1,)
    # state=(states, last_state, src_valid_len)
    state = net.decoder.init_state(states, last_state, src_valid_len)

    # 解码
    # 第一个输入是<bos>，next_input(B,Q)=(1,1)
    next_input = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)

    for _ in range(num_steps):
        # next_input(B,Q)=(1,1)
        # tgt_hat(1,1,V)
        # state=(states, last_state, valid_lens)
        tgt_hat, state = net.decoder(next_input, state)
        # tgt_hat(1,1,V)->(1,1)->next_input
        next_input = tgt_hat.argmax(dim=2)
        # next_input(1,1)->pred(1,)
        pred = next_input.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重
        if save_attention_weights:
            # attention_weights(B,1,Q)=(1,1,Q)
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列<eos>词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        pred_seq.append(pred) # 词元索引列表

    outputs = ' '.join(tgt_vocab.to_tokens(pred_seq)) # 词元列表转换为字符串
    # 把attention_weight_seq转成 tensor,在dim=1上拼接
    # attention_weight_seq:list((B,1,Q)...)=((1,1,Q)...) 测试阶段，数量由实际查询次数决定
    # attention_weights(1,Q',Q)=(1,Q',10)
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
    # Q=num_steps=10
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

    # 2.加载数据集
    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
    
    # 3.初始化模型
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)

    # 4.训练模型
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    # 5.预测
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    print('----------------------------------------------------------------')
    attention_weights_list = []
    for eng, fra in zip(engs, fras):
        # attention_weights:tensor(1,Q',Q)=(1,Q',10)
        translation, attention_weights = predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=True)
        # 因为预测阶段，每个句子的attention_weights的Q'不同，所以这里attention_weights_list不做拼接，保留list
        attention_weights_list.append(attention_weights)
        # 预测评估
        print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')



    # 6.可视化注意力权重
    # attention_weights(1,Q',Q)=(1,Q',10)
    attention_weights_padded_list = []
    for attention_weights in attention_weights_list:
        # 将 attention_weights 的第1维长度扩展到到 10，右补 0
        # attention_weights_padded(1,Q,Q)=(1,10,10)
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