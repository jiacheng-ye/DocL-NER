# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn.functional as F

from torch import nn
import math
from copy import deepcopy


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
                   torch.cumsum(mask, dim=1).type_as(mask) * mask
           ).long() + padding_idx


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        positions = make_positions(input, self.padding_idx)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(
            self,
            num_embeddings,
            embedding_dim,
            padding_idx,
    ):
        super(LearnedPositionalEmbedding).__init__(num_embeddings, embedding_dim, padding_idx)

    def forward(self, input):
        # positions: batch_size x max_len,
        positions = make_positions(input, self.padding_idx)
        return super(LearnedPositionalEmbedding).forward(positions)



class RelativeEmbedding(nn.Module):
    def forward(self, input):
        '''
        :param input: [bsz x seqlen]
        :return: [max_len*2, embed_size]
        '''
        bsz, seq_len = input.size()[:2]
        max_pos = self.padding_idx + seq_len
        if max_pos > self.origin_shift:
            # recompute/expand embeddings if needed
            weights = self.get_embedding(
                max_pos * 2,
                self.embedding_dim,
                self.padding_idx,
            )
            weights = weights.to(self._float_tensor)
            del self.weights
            self.origin_shift = weights.size(0) // 2
            self.register_buffer('weights', weights)

        positions = torch.arange(-seq_len, seq_len).to(input.device).long() + self.origin_shift  # 2*seq_len

        # mask = input.eq(self.padding_idx)
        # positions.masked_fill_(mask, self.padding_idx)

        embed = self.weights.index_select(0, positions.long()).detach()
        return embed


class RelativeSinusoidalPositionalEmbedding(RelativeEmbedding):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        """

        :param embedding_dim:
        :param padding_idx:
        :param init_size:
        """
        super(RelativeSinusoidalPositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        assert init_size % 2 == 0
        weights = self.get_embedding(
            init_size + 1,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('weights', weights)
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def get_embedding(self, num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        self.origin_shift = num_embeddings // 2 + 1
        return emb


class RelativePositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        super(RelativePositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        assert init_size % 2 == 0
        weights = self.get_embedding(
            init_size + 1,
            embedding_dim,
            padding_idx,
        )
        self.embed = nn.Parameter(weights)
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def get_embedding(self, num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        if hasattr(self, 'origin_shift'):
            raise RuntimeError("Cannot regenerate embedding")
        emb = nn.init.xavier_normal_(torch.randn(num_embeddings, embedding_dim))
        emb[padding_idx].fill_(0)
        self.origin_shift = num_embeddings // 2 + 1
        return emb

    def forward(self, input):
        bsz, seq_len = input.size()
        positions = torch.arange(-seq_len, seq_len).to(input.device).long() + self.origin_shift  # 2*seq_len

        embed = self.embed[positions.long()]
        return embed


class RelativeMultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_head, dropout, r_w_bias=None, r_r_bias=None, scale=True, rel_pos_embed='sin',
                 padding_idx=0):
        """

        :param int d_model:
        :param int n_head:
        :param dropout: dropout on attention map
        :param r_w_bias: n_head x head_dim or None,
        :param r_r_bias: n_head x head_dim or None,
        :param scale:
        :param rel_pos_embed:
        """
        super(RelativeMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout_layer = nn.Dropout(dropout)

        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.kv_linear = nn.Linear(d_model, d_model * 2, bias=False)
        self.r_linear = nn.Linear(self.head_dim, self.head_dim, bias=False)

        if rel_pos_embed == 'sin':
            self.pos_embed = RelativeSinusoidalPositionalEmbedding(d_model // n_head, padding_idx, 512)
        elif rel_pos_embed == 'fix':
            self.pos_embed = RelativePositionalEmbedding(d_model // n_head, padding_idx)
        else:
            raise

        if scale:
            self.scale = math.sqrt(d_model // n_head)
        else:
            self.scale = 1

        if r_r_bias is None or r_w_bias is None:  # Biases are not shared
            self.r_r_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, d_model // n_head)))
            self.r_w_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, d_model // n_head)))
        else:
            self.r_r_bias = r_r_bias  # r_r_bias = v
            self.r_w_bias = r_w_bias  # r_w_bias = u

    def forward(self, q, k, mask):
        """

        :param x: batch_size x max_len x d_model
        :param mask: batch_size x max_len x max_len
        :param weight: batch, max_len
        :return:
        """

        batch_size, max_len, d_model = q.size()
        pos_embed = self.pos_embed(mask)  # 2*max_len, d
        r = self.r_linear(pos_embed)  # 2*max_len, d

        q = self.q_linear(q)  # batch_size x max_len x d_model
        kv = self.kv_linear(k)
        k, v = torch.chunk(kv, chunks=2, dim=-1)

        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)  # b x n_head x max_len x d_model

        rw_head_q = q + self.r_r_bias[:, None]
        AC = torch.einsum('bnqd,bnkd->bnqk', rw_head_q, k)  # b x n_head x max_len x d_model, n = head

        D_ = torch.einsum('nd,ld->nl', self.r_w_bias, r)[None, :, None]  # head x 2max_len,
        B_ = torch.einsum('bnqd,ld->bnql', q, r)  # bsz x head  x max_len x 2max_len，
        BD = B_ + D_  # bsz x head x max_len x 2max_len
        BD = self._shift(BD)  # bsz x head x max_len x max_len
        attn = AC + BD

        attn = attn / self.scale  # batch, n_head, seq_len, seq_len

        attn = attn.masked_fill(mask[:, None, :, :].eq(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        v = torch.matmul(self.dropout_layer(attn), v).transpose(1, 2).reshape(batch_size, max_len, d_model)

        return v, attn

    def _shift(self, BD):
        """
        example:
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        to
        0   1  2
        -1  0  1
        -2 -1  0

        :param BD: batch_size x n_head x max_len x 2max_len
        :return: batch_size x n_head x max_len x max_len
        """
        bsz, n_head, max_len, _ = BD.size()
        zero_pad = BD.new_zeros(bsz, n_head, max_len, 1)
        BD = torch.cat([BD, zero_pad], dim=-1).view(bsz, n_head, -1, max_len)  # bsz x n_head x (2max_len+1) x max_len
        BD = BD[:, :, :-1].view(bsz, n_head, max_len, -1)  # bsz x n_head x 2max_len x max_len
        BD = BD[:, :, :, max_len:]
        return BD


class MultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, scale=False):
        """

        :param d_model:
        :param n_head:
        :param scale:
        """
        super(MultiHeadAttn, self).__init__()
        assert d_model % n_head == 0, "err"

        self.n_head = n_head
        self.qkv_linear = nn.Linear(d_model, 3 * d_model)
        self.dropout_layer = nn.Dropout(dropout)

        if scale:
            self.scale = math.sqrt(d_model // n_head)
        else:
            self.scale = 1

    def forward(self, x, k, mask):
        """

        :param x: bsz x max_len x d_model
        :param mask: bsz x max_len
        :return:
        """
        batch_size, max_len, d_model = x.size()
        x = self.qkv_linear(x)
        q, k, v = torch.chunk(x, 3, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        attn = torch.matmul(q, k)  # batch_size x n_head x max_len x max_len
        attn = attn / self.scale
        attn.masked_fill_(mask=mask[:, None, None].eq(0), value=float('-inf'))

        attn = F.softmax(attn, dim=-1)  # batch_size x n_head x max_len x max_len
        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v)  # batch_size x n_head x max_len x d_model//n_head
        v = v.transpose(1, 2).reshape(batch_size, max_len, -1)

        return v


class TransformerLayer(nn.Module):
    def __init__(self, d_model, self_attn, feedforward_dim, after_norm, dropout, last_layer=False):
        """
        :param int d_model:
        :param self_attn: self attention,
            input:batch_size x max_len x d_model, mask:batch_size x max_len
            output: batch_size x max_len x d_model
        :param int feedforward_dim: dimension in FFN
        :param bool after_norm: position of Layernorm
        :param float dropout:
        :param bool last_layer: whether this layer is the final layer, we only perform h->l attention on the last layer
        """
        super(TransformerLayer, self).__init__()

        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-12)

        self.self_attn_hh = deepcopy(self_attn)
        self.last_layer = last_layer
        if self.last_layer:
            self.self_attn_hl = deepcopy(self_attn)

        self.after_norm = after_norm

        self.ffn = nn.Sequential(nn.Linear(d_model, feedforward_dim),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(feedforward_dim, d_model),
                                 nn.Dropout(dropout),
                                 )

    def forward(self, h, l, mask):
        """

        :param x: batch_size x max_len x d_model
        :param mask: batch_size x max_len, position with zero value means the position is padded
        :return: batch_size x max_len x d_model
        """
        batch_size, max_len = mask.size()[:2]

        residual_h = h

        hh_mask = mask[:, None, :].expand(batch_size, max_len, max_len)
        attn_out_hh, _ = self.self_attn_hh(h, h, hh_mask)
        attn_out_hh = attn_out_hh.masked_fill(mask.unsqueeze(-1) == 0, 0)

        hh = attn_out_hh + residual_h
        hh = self.norm1(hh)

        # position-wise feed forward
        residual_hh = hh

        hh = self.ffn(hh)
        hh = residual_hh + hh
        hh = self.norm2(hh)

        if not self.last_layer:
            return hh, l, None

        attn_out_hl, attn = self.self_attn_hl(h, l, hh_mask)  # batch, seq_len, d_model
        hl = attn_out_hl.masked_fill(mask.unsqueeze(-1) == 0, 0)

        return hh, hl, attn

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_head, feedforward_dim, dropout, after_norm=True, attn_type='xl',
                 scale=True, dropout_attn=None, pos_embed=None, rel_pos_embed='sin', padding_idx=0):
        super(TransformerEncoder, self).__init__()
        if dropout_attn is None:
            dropout_attn = dropout

        if pos_embed is None:
            self.pos_embed = None
        elif pos_embed == 'sin':
            self.pos_embed = SinusoidalPositionalEmbedding(d_model, padding_idx, init_size=1024)
        elif pos_embed == 'fix':
            self.pos_embed = LearnedPositionalEmbedding(1024, d_model, padding_idx)

        if attn_type == 'naive':
            self_attn = MultiHeadAttn(d_model, n_head, dropout_attn, scale=scale)
        elif attn_type == 'xl':
            self_attn = RelativeMultiHeadAttn(d_model, n_head, dropout_attn, scale=scale, rel_pos_embed=rel_pos_embed,
                                              padding_idx=padding_idx)
        self.layers = nn.ModuleList(
            [TransformerLayer(d_model, self_attn, feedforward_dim, after_norm, dropout, False)
             for _ in range(num_layers - 1)])

        self.layers.append(TransformerLayer(d_model, self_attn, feedforward_dim, after_norm, dropout, True))

        self.h2dmodel = nn.Linear(d_model * 2, d_model)
        self.l2dmodel = nn.Linear(d_model * 2, d_model)

    def forward(self, h, l, mask, memory,  doc_idx, word_idx):
        """

        :param x: batch_size x max_len x d_model
        :param mask: batch_size x max_len
        :return:
        """
        if self.pos_embed is not None:
            h = h + self.pos_embed(mask)
            l = l + self.pos_embed(mask)

        hh, hl = h, l
        for i, layer in enumerate(self.layers):
            if memory is not None and i == len(self.layers) - 1:
                h_mem, l_mem = memory.get(hh,  doc_idx, word_idx)

                hh = self.h2dmodel(torch.cat([hh, h_mem], -1))
                hl = self.l2dmodel(torch.cat([hl, l_mem], -1))
            hh, hl, attn = layer(hh, hl, mask)
        return hh, hl, attn

