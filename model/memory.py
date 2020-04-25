# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Memory(nn.Module):
    def __init__(self, data):
        super(Memory, self).__init__()
        self.word_mat = data.word_mat
        self.h_dim = data.HP_hidden_dim
        self.l_dim = data.d_model
        self.default_h = nn.Parameter(torch.randn(data.d_model))
        self.default_l = nn.Parameter(torch.randn(data.d_model))
        self.attn = MultiHeadAttn(data.d_model, data.d_model, self.h_dim, self.l_dim, data.HP_memory_attn_nhead)
        self.mem_h = None
        self.mem_l = None

        self.max_read_memory = data.HP_max_read_memory

    def get(self, query_h, doc_idx, word_idx):
        batch_size, max_seq_len, hidden_dim = query_h.size()
        num = batch_size * max_seq_len
        idx = word_idx.new_tensor(self.word_mat[doc_idx[0]][word_idx.reshape(-1).cpu()]) # num * max_memory_size
        max_word_idx_len = (idx != 0).sum(-1).max().item()
        if max_word_idx_len==0:
            return self.default_h[None,None,...].expand_as(query_h), self.default_l[None,None,...].expand_as(query_h)

        idx = idx[...,:max_word_idx_len]
        mask = idx != 0
        h, l = self.attn(query_h.reshape((-1, 1, hidden_dim)),
                         self.mem_h[idx],
                         self.mem_l[idx],
                         mask.reshape((num, 1, max_word_idx_len))) # num, 1, d_model
        len_mask = mask.sum(-1) == 0
        h[len_mask] = self.default_h
        l[len_mask] = self.default_l

        h = h.reshape(batch_size, max_seq_len, hidden_dim)
        l = l.reshape(batch_size, max_seq_len, hidden_dim)
        return h, l

    def put(self, h, l, word_idx):
        max_idx = word_idx.max().item()
        self.mem_h = h.new_zeros(max_idx + 1, self.h_dim)
        self.mem_l = l.new_zeros(max_idx + 1, self.l_dim)
        self.mem_h.data[word_idx] = h.data
        self.mem_l.data[word_idx] = l.data


class MultiHeadAttn(nn.Module):
    def __init__(self, d_model, q_dim, k_dim, v_dim, n_head, dropout=0.1):
        super(MultiHeadAttn, self).__init__()
        self.n_head = n_head
        self.q_linear = nn.Linear(q_dim, d_model)
        self.k_linear = nn.Linear(k_dim, d_model)
        self.v_linear = nn.Linear(v_dim, d_model)
        self.dropout_layer = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model // n_head)


    def forward(self, q, k, v, mask=None):
        batch_size, q_len, d_model = q.size()
        batch_size, k_len, d_model = k.size()
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        q = q.view(batch_size, q_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, k_len, self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(batch_size, k_len, self.n_head, -1).transpose(1, 2)

        attn = torch.matmul(q, k)  # batch_size x n_head x q_len x k_len
        attn = attn / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask[:, None, :, :].eq(0), float('-inf'))
            attn = attn.masked_fill(mask[:, None, :, :].sum(-1).unsqueeze(-1) == 0, 0)  # 防止没有词可以attn的时候变成inf

        attn = F.softmax(attn, dim=-1)  # batch_size x n_head x q_len x k_len
        attn = self.dropout_layer(attn)

        v = torch.matmul(attn, v)  # batch_size x n_head x q_len x d_model//n_head
        v = v.transpose(1, 2).reshape(batch_size, q_len, -1)

        k = torch.matmul(attn, k.transpose(2, 3))
        k = k.transpose(1, 2).reshape(batch_size, q_len, -1)
        return k, v