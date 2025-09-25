import torch
import math
import collections
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import attention as at
import heatmaps as hm
import torch.nn.functional as F
import dataline as dl

"""
编码的时候:为了产生 key,编码器无论是训练还是预测，都需要输入固定长度(B,G)的序列,所以才能生成G个key
--- ##################
 |  ##################
 B  ##################
 |  ##################
--- |------T(G)------|

B:batch_size
G:sequence_length:group length

解码的时候:为了使用 key,解码器输入长度没有限制，训练时输入实际长度(B,Q)=(B,G),预测时输入实际长度(B,Q)=(1,1)
--- #########
 |  ##################
 B  ###############
 |  ######
--- |------T(Q)------|

B:batch_size
Q:real sequence_length:query length


"""



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
        # 循环神经网络层(E,H)
        self.rnn = nn.GRU(
            input_size=embed_size, 
            hidden_size=num_hiddens, 
            num_layers=num_layers, 
            dropout=dropout,
            batch_first=False
        )
        # 没有输出层
    
    # 定义前向传播
    # inputs(B,G)
    # states(G,B,H)
    # last_state(L,B,H)
    def forward(self, inputs): # 编码的时候可以(B,G)按块进行
        # inputs(B,G)->(B,G,E)
        inputs = self.embedding(inputs)
        # inputs(G,B,E)
        inputs = inputs.permute(1, 0, 2)
        # states(G,B,H):包含每个时间步最后一层的信息
        # last_state(L,B,H):包含最后一个时间步每一层的信息
        states, last_state = self.rnn(inputs)

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
    def __init__(self, tgt_vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        
        # 嵌入层(V,E):
        # 输入:当前时间步的输入(目标句子的单词索引)
        # 输出:当前时间步的嵌入向量(维度为E)
        self.embedding = nn.Embedding(
            num_embeddings=tgt_vocab_size,
            embedding_dim=embed_size
        )
        # 注意力层网络:查询编码器提供的信息，求的是注意力权重，
        # 注意力权重再与编码器的隐藏状态进行加权求和，得到上下文向量context
        self.attention = at.AdditiveAttention(
            key_size=num_hiddens,
            query_size=num_hiddens,
            num_hiddens=num_hiddens,
            dropout=dropout
        )

        # 循环层(E+H,H):
        # 输入:当前时间步的输入(嵌入层输出+上下文向量)
        # 隐藏状态:上一个时间步的隐藏状态
        # 输出:当前时间步的隐藏状态
        self.rnn = nn.GRU(
            input_size=embed_size + num_hiddens,
            hidden_size=num_hiddens,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=False
        )
        # 输出层(H,V):
        # 输入:当前时间步的隐藏状态
        # 输出:当前时间步的预测输出(维度为V)
        self.dense = nn.Linear(
            in_features=num_hiddens,
            out_features=tgt_vocab_size
        )

    
    # 注意：待翻译句子首先要经过编码器粗加工，得到半成品给解码器用
    # 初始化解码器状态:给编码器的states做维度转换，为了与注意力层的输入匹配
    # 把valid_lens也打包进来
    def init_state(self, states, last_state, valid_lens):
        # states(G,B,H)
        # last_state(L,B,H))
        # valid_lens(B):待翻译句子的有效长度，此时只是初加工，还没有翻译
        states = states.permute(1, 0, 2)
        # states(B,G,H)
        # last_state(L,B,H)
        # valid_lens(B)
        return states, last_state, valid_lens

    # inputs(B,Q):训练=(B,G),预测=(1,1)
    def forward(self, inputs, state): # 解码的时候最多按(B,1)处理
        # states(B,G,H)
        # last_state(L,B,H)
        # valid_lens(B)
        states, last_state, valid_lens = state

        outputs, self._attention_weights = [], []


        # states(B,G,H)->keys(B,K,Kd)=(B,G,H)
        keys = states # 编码器所有时间步的隐藏状态,作为注意力层的键
        # states(B,G,H)->values(B,K,V)=(B,G,H)
        values = states # 编码器所有时间步的隐藏状态,作为注意力层的值
 
        # 1.计算嵌入层
        # inputs(B,Q)->(B,Q,E)->(Q,B,E)
        inputs = self.embedding(inputs).permute(1, 0, 2)
        # 保留batch,分解group->input(B,E)
        for input in inputs: # 循环Q轮,进了这个for,Q这个维度就淡化了，主要处理B这个维度
            # 2.计算注意力层:在seq2seq中使用注意力机制
            # 编码器最新的隐藏状态是主动目标
            # 编码器所有时间步的隐藏状态就是被动目标，G个
            # 用最新的最后一步隐状态去查所有时间步的隐藏状态，求得分，求 weights,得到上下文向量
            # last_state(L,B,H)->(B,H)->query(B,1,H)
            query = last_state[-1].unsqueeze(1) # 编码器最后一个时间步的隐藏状态
            # query(B,Q,Qd)=(B,1,H)
            # keys(B,G,Gd)=(B,G,H)
            # values(B,G,V)=(B,G,H)
            # valid_lens(B)
            # context(B,Q,V)=(B,1,H):因为values(B,G,V)=(B,G,H),所以context(B,Q,V)=(B,1,H)
            context = self.attention(query, keys, values, valid_lens) # 调用注意力层(计算注意力)，计算上下文

            # 保存本轮的注意力权重(B,1,G)
            # attention_weights（B,Q,G)=(B,1,G)
            # list((B,1,G),(B,1,G)...(B,1,G)）一共G个
            self._attention_weights.append(self.attention.attention_weights.detach())

            # 3.计算循环层
            # input(B,E)->(B,1,E)
            input = input.unsqueeze(1)

            # input(B,1,E)
            # context(B,1,H)
            # combination(B,1,E+H)
            combination = torch.cat((input, context), dim=-1) # 拼接输入和上下文，在特征维度上拼接
            # combination(B,1,E+H)->(1,B,E+H)
            combination = combination.permute(1, 0, 2)
      
            # combination(1,B,E+H)
            # last_state(L,B,H)
            # out(1,B,H):当前时间步最后一层隐藏状态
            # last_state(L,B,H):当前时间步的隐状态
            out, last_state = self.rnn(combination, last_state)
            outputs.append(out) # (1,B,H),(1,B,H),(1,B,H)...



        # outputs(Q,B,H)
        outputs = torch.cat(outputs,dim=0)
        # outputs(Q,B,H)->(Q,B,V)->(B,Q,V)
        outputs = self.dense(outputs).permute(1, 0, 2)
        # outputs(B,Q,V)
        # states(B,G,H)
        # last_state(L,B,H)
        # valid_lens(B)
        return outputs, (states, last_state, valid_lens)

    @property
    def attention_weights(self):
        # 把列表中的注意力权重(B,1,G)拼接起来(B,Q,G)，生成张量
        # weight(B,1,G) 一共Q个
        # attention_weights(B,Q,G):表示B个句子的Q个时间步的注意力权重
        attention_weights = torch.cat(self._attention_weights, dim=1)
        return attention_weights


# 编码器-解码器
class EncoderDecoder(nn.Module):
    # 定义网络
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    
    # 定义前向传播
    # src_inputs(B,G)
    # tgt_inputs(B,G)
    # valid_lens(B)
    def forward(self, src_inputs, tgt_inputs, valid_lens):
        # 编码
        # src_inputs(B,G)
        # states(B,G,H)
        # last_state(L,B,H)
        states, last_state = self.encoder(src_inputs)
        # 初始化解码器 state
        # valid_lens(B)
        # state=(states, last_state, valid_lens)
        state = self.decoder.init_state(states,last_state,valid_lens)
        # 解码
        # tgt_inputs(B,G)
        # outputs(B,G,V)
        # states(B,G,H)
        # last_state(L,B,H)
        # valid_lens(B)
        # state=(states, last_state, valid_lens)
        outputs, state = self.decoder(tgt_inputs, state)
        # outputs(B,G,V)
        return outputs, state

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
    fig, ax, line, x_list, y_list = dl.init_plot(lr)

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