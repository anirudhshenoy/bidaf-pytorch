# Taken from : https://github.com/galsang/BiDAF-pytorch
# All credits to @galsang 

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.nn import LSTM, Linear


class BiDAF(nn.Module):
    def __init__(self, 
                 char_vocab_size,
                 word_vocab_size,
                 pretrained,
                 word_dim = 100,
                 char_dim = 8,
                 char_channel_width = 5,
                 char_channel_size = 100,
                 dropout_rate = 0.2,
                 hidden_size = 100):
        
        super(BiDAF, self).__init__()

        self.word_dim = word_dim
        self.char_dim = char_dim
        self.char_channel_width = char_channel_width
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.char_vocab_size = char_vocab_size
        self.char_channel_size = char_channel_size
        self.word_vocab_size = word_vocab_size

        # 1. Character Embedding Layer
        self.char_emb = nn.Embedding(self.char_vocab_size, self.char_dim, padding_idx=1)
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)

        self.char_conv = nn.Conv2d(1, self.char_channel_size, (self.char_dim, self.char_channel_width))

        # 2. Word Embedding Layer
        # initialize word embedding with GloVe
        # Freeze layer to prevent gradient update
        self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=True)

        # highway network
        assert (self.hidden_size * 2) == (self.char_channel_size + self.word_dim)
        # Create 2 hidden layers 
        for i in range(2):
            setattr(self, 'highway_linear' + str(i),
                    nn.Sequential(Linear(self.hidden_size * 2, self.hidden_size * 2),
                                  nn.ReLU()))
            setattr(self, 'highway_gate' + str(i),
                    nn.Sequential(Linear(self.hidden_size * 2, self.hidden_size * 2),
                                  nn.Sigmoid()))

        # 3. Contextual Embedding Layer
        self.context_LSTM = LSTM(input_size=self.hidden_size * 2,
                                 hidden_size=self.hidden_size,
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=self.dropout_rate)

        # 4. Attention Flow Layer
        self.att_weight_c = Linear(self.hidden_size * 2, 1)
        self.att_weight_q = Linear(self.hidden_size * 2, 1)
        self.att_weight_cq = Linear(self.hidden_size * 2, 1)

        # 5. Modeling Layer
        self.modeling_LSTM1 = LSTM(input_size=self.hidden_size * 8,
                                   hidden_size=self.hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=self.dropout_rate)

        self.modeling_LSTM2 = LSTM(input_size=self.hidden_size * 2,
                                   hidden_size=self.hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=self.dropout_rate)

        # 6. Output Layer
        # No softmax applied here reason: https://stackoverflow.com/questions/57516027/does-pytorch-apply-softmax-automatically-in-nn-linear
        self.p1_weight_g = Linear(self.hidden_size * 8, 1, dropout=self.dropout_rate)
        self.p1_weight_m = Linear(self.hidden_size * 2, 1, dropout=self.dropout_rate)
        self.p2_weight_g = Linear(self.hidden_size * 8, 1, dropout=self.dropout_rate)
        self.p2_weight_m = Linear(self.hidden_size * 2, 1, dropout=self.dropout_rate)

        self.output_LSTM = LSTM(input_size=self.hidden_size * 2,
                                hidden_size=self.hidden_size,
                                bidirectional=True,
                                batch_first=True,
                                dropout=self.dropout_rate)

        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, batch):
        # TODO: More memory-efficient architecture
        def char_emb_layer(x):
            """
            :param x: (batch, seq_len, word_len)
            :return: (batch, seq_len, char_channel_size)
            """
            batch_size = x.size(0)
            # (batch, seq_len, word_len, char_dim)
            x = self.dropout(self.char_emb(x))
            # (batch * seq_len, 1, char_dim, word_len)
            x = x.view(-1, self.char_dim, x.size(2)).unsqueeze(1)
            # (batch * seq_len, char_channel_size, 1, conv_len) -> (batch * seq_len, char_channel_size, conv_len)
            x = self.char_conv(x).squeeze()
            # (batch * seq_len, char_channel_size, 1) -> (batch * seq_len, char_channel_size)
            x = F.max_pool1d(x, x.size(2)).squeeze()
            # (batch, seq_len, char_channel_size)
            x = x.view(batch_size, -1, self.char_channel_size)

            return x

        def highway_network(x1, x2):
            """
            :param x1: (batch, seq_len, char_channel_size)
            :param x2: (batch, seq_len, word_dim)
            :return: (batch, seq_len, hidden_size * 2)
            """
            # (batch, seq_len, char_channel_size + word_dim)
            x = torch.cat([x1, x2], dim=-1)
            for i in range(2):
                h = getattr(self, 'highway_linear' + str(i))(x)
                g = getattr(self, 'highway_gate' + str(i))(x)
                x = g * h + (1 - g) * x
            # (batch, seq_len, hidden_size * 2)
            return x

        def att_flow_layer(c, q):
            """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
            """
            c_len = c.size(1)
            q_len = q.size(1)

            # (batch, c_len, q_len, hidden_size * 2)
            #c_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            #q_tiled = q.unsqueeze(1).expand(-1, c_len, -1, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            #cq_tiled = c_tiled * q_tiled
            #cq_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1) * q.unsqueeze(1).expand(-1, c_len, -1, -1)
            
            # CALCULATE SIMILARITY MATRIX
            cq = []
            for i in range(q_len):
                #(batch, 1, hidden_size * 2)
                qi = q.select(1, i).unsqueeze(1)
                #(batch, c_len, 1)
                ci = self.att_weight_cq(c * qi).squeeze()
                cq.append(ci)
            # (batch, c_len, q_len)
            cq = torch.stack(cq, dim=-1)

            # (batch, c_len, q_len)
            s = self.att_weight_c(c).expand(-1, -1, q_len) + \
                self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                cq

            # (batch, c_len, q_len)
            a = F.softmax(s, dim=2)
            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
            c2q_att = torch.bmm(a, q)
            # (batch, 1, c_len)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
            q2c_att = torch.bmm(b, c).squeeze()
            # (batch, c_len, hidden_size * 2) (tiled)
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

            # (batch, c_len, hidden_size * 8)
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            return x

        def output_layer(g, m, l):
            """
            :param g: (batch, c_len, hidden_size * 8)
            :param m: (batch, c_len ,hidden_size * 2)
            :return: p1: (batch, c_len), p2: (batch, c_len)
            """
            # (batch, c_len)
            p1 = (self.p1_weight_g(g) + self.p1_weight_m(m)).squeeze()
            # (batch, c_len, hidden_size * 2)
            m2 = self.output_LSTM((m, l))[0]
            # (batch, c_len)
            p2 = (self.p2_weight_g(g) + self.p2_weight_m(m2)).squeeze()

            return p1, p2

        # 1. Character Embedding Layer
        c_char = char_emb_layer(batch.c_char)
        q_char = char_emb_layer(batch.q_char)
        # 2. Word Embedding Layer
        c_word = self.word_emb(batch.c_word[0])
        q_word = self.word_emb(batch.q_word[0])
        c_lens = batch.c_word[1]
        q_lens = batch.q_word[1]

        # Highway network
        c = highway_network(c_char, c_word)
        q = highway_network(q_char, q_word)
        # 3. Contextual Embedding Layer
        c = self.context_LSTM((c, c_lens))[0]
        q = self.context_LSTM((q, q_lens))[0]
        # 4. Attention Flow Layer
        g = att_flow_layer(c, q)
        # 5. Modeling Layer
        m = self.modeling_LSTM2((self.modeling_LSTM1((g, c_lens))[0], c_lens))[0]
        # 6. Output Layer
        p1, p2 = output_layer(g, m, c_lens)

        # (batch, c_len), (batch, c_len)
        return p1, p2
