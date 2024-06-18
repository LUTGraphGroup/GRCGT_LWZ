import math
from torch import nn
import torch.nn.functional as F
from math import log
from typing import Optional, Tuple
import torch
# from torch import Tensor
# from torch.nn import Parameter
# from torch_sparse import SparseTensor, matmul
# from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.nn.conv.gcn_conv import gcn_norm
# from torch_geometric.typing import Adj, OptTensor
# from utils import glorot


# class GCN2Conv(MessagePassing):
#     _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
#     _cached_adj_t: Optional[SparseTensor]
#
#     def __init__(self, channels: int, alpha: float, theta: float = None,
#                  layer: int = None, shared_weights: bool = True,
#                  cached: bool = False, add_self_loops: bool = True,
#                  normalize: bool = True, **kwargs):
#
#         kwargs.setdefault('aggr', 'add')
#         super().__init__(**kwargs)
#
#         self.channels = channels
#         self.alpha = alpha
#         self.beta = 1.
#         if theta is not None or layer is not None:
#             assert theta is not None and layer is not None
#             self.beta = log(theta / layer + 1)
#         self.cached = cached
#         self.normalize = normalize
#         self.add_self_loops = add_self_loops
#
#         self._cached_edge_index = None
#         self._cached_adj_t = None
#
#         self.weight1 = Parameter(torch.Tensor(channels, channels))
#
#         if shared_weights:
#             self.register_parameter('weight2', None)
#         else:
#             self.weight2 = Parameter(torch.Tensor(channels, channels))
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         glorot(self.weight1)
#         glorot(self.weight2)
#         self._cached_edge_index = None
#         self._cached_adj_t = None
#
#     def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj,
#                 edge_weight: OptTensor = None) -> Tensor:
#         """"""
#
#         if self.normalize:
#             if isinstance(edge_index, Tensor):
#                 cache = self._cached_edge_index
#                 if cache is None:
#                     edge_index, edge_weight = gcn_norm(  # yapf: disable
#                         edge_index, edge_weight, x.size(self.node_dim), False,
#                         self.add_self_loops, self.flow, dtype=x.dtype)
#                     if self.cached:
#                         self._cached_edge_index = (edge_index, edge_weight)
#                 else:
#                     edge_index, edge_weight = cache[0], cache[1]
#
#             elif isinstance(edge_index, SparseTensor):
#                 cache = self._cached_adj_t
#                 if cache is None:
#                     edge_index = gcn_norm(  # yapf: disable
#                         edge_index, edge_weight, x.size(self.node_dim), False,
#                         self.add_self_loops, self.flow, dtype=x.dtype)
#                     if self.cached:
#                         self._cached_adj_t = edge_index
#                 else:
#                     edge_index = cache
#
#         # propagate_type: (x: Tensor, edge_weight: OptTensor)
#         x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
#
#         x.mul_(1 - self.alpha)
#         x_0 = self.alpha * x_0[:x.size(0)]
#
#         if self.weight2 is None:
#             out = x.add_(x_0)
#             out = torch.addmm(out, out, self.weight1, beta=1. - self.beta,
#                               alpha=self.beta)
#         else:
#             out = torch.addmm(x, x, self.weight1, beta=1. - self.beta,
#                               alpha=self.beta)
#             out = out + torch.addmm(x_0, x_0, self.weight2,
#                                     beta=1. - self.beta, alpha=self.beta)
#
#         return out
#
#     def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
#         return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
#
#     def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
#         return matmul(adj_t, x, reduce=self.aggr)
#
#     def __repr__(self) -> str:
#         return (f'{self.__class__.__name__}({self.channels}, '
#                 f'alpha={self.alpha}, beta={self.beta})')


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/modeling.py
    """
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5  # \frac{1}{\sqrt{d}}
        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)
        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]
        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)
        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)  # LN
        y = self.self_attention(y, y, y, attn_bias)  # MSA
        y = self.self_attention_dropout(y)
        x = x + y  # Equation（19）

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y  # Equation（20）
        return x


class InnerProductDecoder(nn.Module): # Equation（27）
    """
    decoder 解码器
    """
    def __init__(self, output_node_dim, dropout, num_dis):
        super(InnerProductDecoder, self).__init__()
        self.output_node_dim = output_node_dim
        self.dropout = dropout
        self.num_dis = num_dis
        self.weight = nn.Parameter(torch.empty(size=(self.output_node_dim, self.output_node_dim)))
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)

    def forward(self, inputs):
        inputs = F.dropout(inputs, self.dropout)
        Dis = inputs[0:self.num_dis, :]
        Meta = inputs[self.num_dis:, :]
        Meta = torch.mm(Meta, self.weight)
        Dis = torch.t(Dis)
        x = torch.mm(Meta, Dis)
        outputs = torch.sigmoid(x)
        return outputs
