import torch
from torch import nn
from d2l import torch as d2l




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
        # 这一步的目的：把 padding 的得分清零，但是保留查询对象位数共 G
        score = d2l.sequence_mask(score.reshape(-1, score.shape[-1]), valid_lens, value=-1e6)
        # (B*Q,G)->(B,Q,G),此时的G=(有效得分+剩下位置补0)
        score = score.reshape(shape)
        # weights(B,Q,G)
        weights = nn.functional.softmax(score, dim=-1)
        return weights
        """处理前
        score(B,Q,G)
        ########******
        ###***********
        ############**
        """


        """处理后
        weights(B,Q,G)
        #######0000000
        ###00000000000
        ############00
        """
