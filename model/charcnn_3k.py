# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-04-26 13:21:40
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CharCNN(nn.Module):
    def __init__(self, alphabet_size, pretrain_char_embedding, embedding_dim, hidden_dim, dropout, gpu):
        super(CharCNN, self).__init__()
        print("build char sequence feature extractor: CNN ...")
        self.gpu = gpu
        self.hidden_dim = hidden_dim
        self.char_drop = nn.Dropout(dropout)
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim)
        if pretrain_char_embedding is not None:
            self.char_embeddings.weight.data.copy_(torch.from_numpy(pretrain_char_embedding))
        else:
            self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(alphabet_size, embedding_dim)))
        self.char_cnn2 = nn.Conv1d(embedding_dim, self.hidden_dim, kernel_size=2, padding=1)
        self.char_cnn3= nn.Conv1d(embedding_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.char_cnn4 = nn.Conv1d(embedding_dim, self.hidden_dim, kernel_size=4, padding=2)
        self.char_out_size = hidden_dim * 3

        if self.gpu:
            self.char_drop = self.char_drop.cuda()
            self.char_embeddings = self.char_embeddings.cuda()
            self.char_cnn2 = self.char_cnn2.cuda()
            self.char_cnn3 = self.char_cnn3.cuda()
            self.char_cnn4 = self.char_cnn4.cuda()


    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb


    def get_last_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(bsz*seq_len, word_length)
                seq_lengths: numpy array (bsz*seq_len,  1)
            output:
                Variable(bsz*seq_len, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input)) # bsz*seq_len, word_len, char_dim
        char_embeds = char_embeds.transpose(2,1).contiguous() # bsz*seq_len, char_dim, word_len
        char_cnn_out2 = self.char_cnn2(char_embeds)
        char_cnn_out3 = self.char_cnn3(char_embeds)
        char_cnn_out4 = self.char_cnn4(char_embeds)
        char_cnn_out = torch.cat([char_cnn_out2[:,:,1:], char_cnn_out3,char_cnn_out4[:,:,1:]], 1)
        char_cnn_out = F.max_pool1d(char_cnn_out, char_cnn_out.size(2)).view(batch_size, -1)
        return char_cnn_out
