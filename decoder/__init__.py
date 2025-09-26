# 解码器包

"""解码器包，包含基础序列到序列模型和Bahdanau注意力编码器-解码器等实现。"""

# 使用别名导入以区分同名类
from .seq2seq_enc_dec import (
    Encoder,
    Decoder,
    Seq2SeqEncoder,
    Seq2SeqDecoder,
    EncoderDecoder as BasicEncoderDecoder  # 基础版编码器-解码器（不带注意力机制）
)
from .bahdanau_enc_dec import (
    BahdanauDecoder,
    EncoderDecoder as BahdanauEncoderDecoder  # Bahdanau注意力编码器-解码器
)
from .transformer_enc_dec import (
    TransformerEncoder,
    TransformerDecoder
)

__all__ = [
    'Encoder',
    'Decoder',
    'Seq2SeqEncoder',
    'Seq2SeqDecoder',
    'BahdanauDecoder',
    'BasicEncoderDecoder',  # 基础版编码器-解码器
    'BahdanauEncoderDecoder',  # Bahdanau编码器-解码器
    'TransformerEncoder',
    'TransformerDecoder'
]