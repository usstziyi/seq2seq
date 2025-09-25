# 注意力机制包

"""注意力机制实现包，包含加性注意力、缩放点积注意力和多头注意力等实现。"""

from .additive_attention import AdditiveAttention
from .dotproduct_attention import DotProductAttention
from .multihead_attention import MultiHeadAttention
from .multihead_attention_parallel import MultiHeadAttentionParallel

__all__ = [
    'AdditiveAttention',
    'DotProductAttention',
    'MultiHeadAttention',
    'MultiHeadAttentionParallel'
]