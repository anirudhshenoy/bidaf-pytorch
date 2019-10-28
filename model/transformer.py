# Taken from : https://github.com/galsang/BiDAF-pytorch
# All credits to @galsang 

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.nn import LSTM, Linear


# https://github.com/andy840314/QANet-pytorch-/blob/master/models.py
def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return (x + signal.cuda())

def get_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales)-1))
    inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim = 1)
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal




# https://discuss.pytorch.org/t/depthwise-and-separable-convolutions-in-pytorch/7315
# https://stackoverflow.com/questions/56725660/how-does-the-groups-parameter-in-torch-nn-conv-influence-the-convolution-proces
class DepthwiseSepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.depth = nn.Conv1d(in_channels, out_channels, kernel_size, groups = in_channels)
        self.point = nn.Conv1d(in_channels, out_channels, 1, padding = 0)
        
    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        return F.relu(x)

class NormalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding = kernel_size // 2)
        
    def forward(self, x):
        x = self.conv(x)
        return F.relu(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, number_convs, hidden_size, kernel_size, attn_heads = 8, encoder_hidden_layer_size = 512,  is_depthwise = False):
        super().__init__()
        if is_depthwise:
            self.convs = nn.ModuleList([DepthwiseSepConv(hidden_size, hidden_size, kernel_size) for _ in range(number_convs)])
        else:
            self.convs = nn.ModuleList([NormalConv(hidden_size, hidden_size, kernel_size) for _ in range(number_convs)])
            
        self.conv_layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(number_convs)])
        
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead = attn_heads, dim_feedforward = encoder_hidden_layer_size)

    def forward(self, x):
        x = PosEncoder(x)
        for conv, layer_norm in zip(self.convs, self.conv_layer_norms):
            residual = x
            x = layer_norm(x)
            # Conv input is (batch_size, embed_dim, seq_len)
            x = conv(x.transpose(1,2)).transpose(1,2)
            
            # Residual Connection
            x = x + residual
            
        # Encoder input is (seq_len, batch_size, embed_dim)
        x = self.transformer_encoder(x.permute(1,0,2)).permute(1,0,2)
        return x
    
class ResizeConv(nn.Module):
    # To reduce word dim to hidden size of model
    def __init__(self, embed_dim, hidden_size):
        super().__init__()
        self.conv = nn.Conv1d(embed_dim, hidden_size, kernel_size = 1)
        
    def forward(self, x):
        x = self.conv(x.transpose(1,2)).transpose(1,2)
        return x
    
    
class BiDAF(nn.Module):
    def __init__(self, 
                 char_vocab_size,
                 word_vocab_size,
                 pretrained,
                 word_dim = 100,
                 char_dim = 8,
                 char_channel_width = 5,
                 char_channel_size = 100,
                 dropout_rate = 0.1,
                 hidden_size = 128,
                 encoder_hidden_layer_size = 512):
        
        super(BiDAF, self).__init__()

        self.word_dim = word_dim
        self.char_dim = char_dim
        self.char_channel_width = char_channel_width
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.char_vocab_size = char_vocab_size
        self.char_channel_size = char_channel_size
        self.word_vocab_size = word_vocab_size
        self.encoder_hidden_layer_size = encoder_hidden_layer_size

        # 1. Character Embedding Layer
        self.char_emb = nn.Embedding(self.char_vocab_size, self.char_dim, padding_idx=1)
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)

        self.char_conv = nn.Conv2d(1, self.char_channel_size, (self.char_dim, self.char_channel_width))

        # 2. Word Embedding Layer
        # initialize word embedding with GloVe
        # Freeze layer to prevent gradient update
        self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=True)
        
        self.emb_conv = ResizeConv(self.word_dim + self.char_channel_size , self.hidden_size)

        # highway network
        #assert (self.hidden_size * 2) == (self.char_channel_size + self.word_dim)
        # Create 2 hidden layers 
        for i in range(2):
            setattr(self, 'highway_linear' + str(i),
                    nn.Sequential(ResizeConv(self.hidden_size, self.hidden_size),
                                  nn.ReLU()))
            setattr(self, 'highway_gate' + str(i),
                    nn.Sequential(ResizeConv(self.hidden_size, self.hidden_size),
                                  nn.Sigmoid()))
            
        # Embedding Conv

        # Transformer
        # TO DO ADD RESIDUAL CONNECTION
        self.embedding_encoder_block = EncoderBlock(4, self.hidden_size, 7, encoder_hidden_layer_size = self.encoder_hidden_layer_size, is_depthwise = True)
        self.model_encoder_block = nn.ModuleList([EncoderBlock(2, self.hidden_size, 5, encoder_hidden_layer_size = self.encoder_hidden_layer_size, is_depthwise = True) for _ in range(7)])


        # 4. Attention Flow Layer
        self.att_weight_c = ResizeConv(self.hidden_size, 1)
        self.att_weight_q = ResizeConv(self.hidden_size, 1)
        self.att_weight_cq = ResizeConv(self.hidden_size, 1)
        
        # Modelling transformer
        self.resize_g_matrix = ResizeConv(self.hidden_size * 4 , self.hidden_size)

        
        # 6. Output Layer
        # No softmax applied here reason: https://stackoverflow.com/questions/57516027/does-pytorch-apply-softmax-automatically-in-nn-linear
        self.p1_weight_g = ResizeConv(self.hidden_size * 2, 1)
        self.p2_weight_g = ResizeConv(self.hidden_size * 2, 1)

        #self.transformer_output = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=200, nhead=4, dim_feedforward=512), num_layers=3)

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
            x = self.emb_conv(x)
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
            
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            x = self.resize_g_matrix(x)
            return x

        
        def output_layer(m0, m1, m2):
            """
            :param g: (batch, c_len, hidden_size * 8)
            :param m: (batch, c_len ,hidden_size * 2)
            :return: p1: (batch, c_len), p2: (batch, c_len)
            """
            # (batch, c_len)
            p1 = self.p1_weight_g(torch.cat([m0, m1], dim = -1)).squeeze()
            # (batch, c_len)
            p2 = self.p2_weight_g(torch.cat([m0, m2], dim = -1)).squeeze()

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
        
        # Transformer 
        
        #print("context size after highway: {}".format(c.size()))
        #print("question size after highway : {}".format(q.size()))
        
        c = self.embedding_encoder_block(c)
        q = self.embedding_encoder_block(q)
        #print("context size after encoder block: {}".format(c.size()))
        #print("question size after encoder block: {}".format(q.size()))

        #print("context size Embedding: {}".format(c.size()))
        #print("question size Embedding: {}".format(q.size()))
        
        # 4. Attention Flow Layer
        g = att_flow_layer(c, q)
        
        #print("G matrix size: {}".format(g.size()))
        
        enc_op_0 = g
        for model_enc in self.model_encoder_block:
            enc_op_0 = model_enc(enc_op_0)
        #print("model_op 0 size: {}".format(enc_op_0.size()))
        
        
        enc_op_1 = enc_op_0
        for model_enc in self.model_encoder_block:
            enc_op_1 = model_enc(enc_op_1)
        #print("model_op 1 size: {}".format(enc_op_1.size()))
        
        
        enc_op_2 = enc_op_1
        for model_enc in self.model_encoder_block:
            enc_op_2 = model_enc(enc_op_2)
        #print("model_op 2 size: {}".format(enc_op_2.size()))


        # 6. Output Layer
        p1, p2 = output_layer(enc_op_0, enc_op_1, enc_op_2)

        #print("p1 shape : {}".format(p1.size()))
        #print("p2 shape : {}".format(p2.size()))



        # (batch, c_len), (batch, c_len)
        return p1, p2
