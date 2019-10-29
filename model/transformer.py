# Taken from : https://github.com/galsang/BiDAF-pytorch
# All credits to @galsang 

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.nn import LSTM, Linear


# https://github.com/andy840314/QANet-pytorch-/blob/master/models.py
def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    #x = x.transpose(1,2)

    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return (x + signal.cuda())#.transpose(1,2)

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
        self.depth = nn.Conv1d(in_channels, in_channels, kernel_size, groups = in_channels, padding = kernel_size//2)
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
    def __init__(self, number_convs, hidden_size, kernel_size, dropout, attn_heads = 8, encoder_hidden_layer_size = 512,  is_depthwise = False):
        super().__init__()
        self.dropout_rate = dropout
        self.number_convs = number_convs
        if is_depthwise:
            self.convs = nn.ModuleList([DepthwiseSepConv(hidden_size, hidden_size, kernel_size) for _ in range(number_convs)])
        else:
            self.convs = nn.ModuleList([NormalConv(hidden_size, hidden_size, kernel_size) for _ in range(number_convs)])
            
        self.conv_layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(number_convs)])
        
        #self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead = attn_heads, dim_feedforward = encoder_hidden_layer_size)
        self.FFN_1 = ResizeConv(hidden_size, hidden_size, activation = True, bias = True)
        
        # FFN_2 activation required ? 
        self.FFN_2 = ResizeConv(hidden_size, hidden_size, activation = True, bias=True)
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.norm_2 = nn.LayerNorm(hidden_size)
        
        self.multihead_attn = nn.MultiheadAttention(hidden_size, attn_heads)

        self.dropout = nn.Dropout(p = dropout)
        self.spatial_dropout = nn.Dropout3d(p = dropout)



    def forward(self, x, mask):
        x = PosEncoder(x)
        i = 0
        for conv, layer_norm in zip(self.convs, self.conv_layer_norms):
            residual = x
            x = layer_norm(x)
            
            ## Dropout between every 2 layers? Probably not needed here. 
            if (i % 2) == 0 :
                x = self.dropout(x)
            # Conv input is (batch_size, embed_dim, seq_len)
            x = conv(x.transpose(1,2)).transpose(1,2)
            
            # Residual Connection
            x = self.spatial_dropout(x) + residual
            i += 1
            
        # Encoder input is (seq_len, batch_size, embed_dim)
        residual = x
        x = self.norm_1(x)
        x = self.dropout(x)
        x = x.permute(1,0,2)
        # Self Attention
        x = self.multihead_attn(x, x, x, key_padding_mask = mask)[0].permute(1,0,2)
        x = self.spatial_dropout(x) + residual
        

        residual = x
        x = self.norm_2(x)
        x = self.dropout(x)
        x = self.FFN_1(x)
        x = self.FFN_2(x)
        x = self.spatial_dropout(x) + residual
        
        return x
    
class ResizeConv(nn.Module):
    # To reduce word dim to hidden size of model
    def __init__(self, embed_dim, hidden_size, activation = False, bias = False):
        super().__init__()
        self.conv = nn.Conv1d(embed_dim, hidden_size, kernel_size = 1, bias = bias)
        
        if activation:   
            self.activation = True
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        else:
            self.activation = False
            nn.init.xavier_uniform_(self.conv.weight)

        
    def forward(self, x):
        x = self.conv(x.transpose(1,2)).transpose(1,2)
        
        if self.activation:
            x = F.relu(x)
        return x
    
    
class BiDAF(nn.Module):
    def __init__(self, 
                 char_vocab_size,
                 word_vocab_size,
                 pretrained,
                 word_dim = 100,
                 char_dim = 16,
                 char_channel_width = 5,
                 char_channel_size = 100,
                 dropout_rate = 0.1,
                 hidden_size = 128,
                 encoder_hidden_layer_size = 512,
                 attn_heads = 1):
        
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
        # Freeze layer to prevent gradient update
        self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=True)
        
        self.question_resize = ResizeConv(self.word_dim + self.char_channel_size , self.hidden_size)
        self.context_resize = ResizeConv(self.word_dim + self.char_channel_size , self.hidden_size)

        # highway network
        #assert (self.hidden_size * 2) == (self.char_channel_size + self.word_dim)
        # Create 2 hidden layers 
        for i in range(2):
            setattr(self, 'highway_linear' + str(i),
                    nn.Sequential(ResizeConv(self.hidden_size, self.hidden_size, bias = True),
                                  nn.ReLU()))
            setattr(self, 'highway_gate' + str(i),
                    nn.Sequential(ResizeConv(self.hidden_size, self.hidden_size, bias = True),
                                  nn.Sigmoid()))
            
        # Embedding Conv

        # Transformer
        self.embedding_encoder_block = EncoderBlock(4, self.hidden_size, 7, self.dropout_rate, attn_heads = attn_heads, encoder_hidden_layer_size = self.encoder_hidden_layer_size, is_depthwise = True)
        
        self.model_encoder_block = nn.ModuleList([EncoderBlock(2, self.hidden_size, 5, self.dropout_rate, attn_heads = attn_heads, encoder_hidden_layer_size = self.encoder_hidden_layer_size, is_depthwise = True) for _ in range(7)] )

        #self.embedding_encoder_block = EncoderBlockOLD(conv_num=4, ch_num=D, k=7)
        
        #self.model_encoder_block = nn.ModuleList([EncoderBlockOLD(conv_num=2, ch_num=D, k=5) for _ in range(7)] )

        

        # 4. Attention Flow Layer
        self.att_weight_c = torch.empty(self.hidden_size, 1)
        self.att_weight_q = torch.empty(self.hidden_size, 1)
        self.att_weight_cq = torch.empty(1, 1, self.hidden_size)
        nn.init.xavier_uniform_(self.att_weight_c )
        nn.init.xavier_uniform_(self.att_weight_q)
        nn.init.xavier_uniform_(self.att_weight_cq)
        self.att_weight_c  = nn.Parameter(self.att_weight_c )
        self.att_weight_q = nn.Parameter(self.att_weight_q)
        self.att_weight_cq = nn.Parameter(self.att_weight_cq)
        
        self.att_bias = torch.empty(1)
        nn.init.constant_(self.att_bias, 0)
        self.att_bias = nn.Parameter(self.att_bias)
        
        # Modelling transformer
        self.resize_g_matrix = ResizeConv(self.hidden_size * 4 , self.hidden_size)

        
        # 6. Output Layer
        # No softmax applied here reason: https://stackoverflow.com/questions/57516027/does-pytorch-apply-softmax-automatically-in-nn-linear
        self.p1_weight_g = ResizeConv(self.hidden_size *2, 1)
        self.p2_weight_g = ResizeConv(self.hidden_size *2, 1)


        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.dropout_char = nn.Dropout(p= 0.05)


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

            x = self.dropout_char(x)
            return x

        def highway_network(x):
            """
            :param x1: (batch, seq_len, char_channel_size)
            :param x2: (batch, seq_len, word_dim)
            :return: (batch, seq_len, hidden_size * 2)
            """
            # (batch, seq_len, char_channel_size + word_dim)
            
            for i in range(2):
                h = getattr(self, 'highway_linear' + str(i))(x)
                g = getattr(self, 'highway_gate' + str(i))(x)
                g = self.dropout(g)
                x = g * h + (1 - g) * x
            # (batch, seq_len, hidden_size * 2)
            return x

        def att_flow_layer(c, q, c_mask, q_mask):
            
            # https://github.com/andy840314/QANet-pytorch-/blob/master/models.py
            """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
            """
            
            # Add bias ? 
            c_len = c.size(1)
            q_len = q.size(1)
            
            batch_size = c.size(0)

            
            c = self.dropout(c)
            q = self.dropout(q)
            
            cq = torch.matmul(c * self.att_weight_cq, q.transpose(1,2))
            s_sub_1 = torch.matmul(c, self.att_weight_c).expand(-1, -1, q_len)
            s_sub_2 = torch.matmul(q, self.att_weight_q).permute(0, 2, 1).expand(-1, c_len, -1)


            # (batch, c_len, q_len)
            s = s_sub_1 + s_sub_2 + cq + self.att_bias

            # (batch, c_len, q_len)           
            c_mask = c_mask.view(batch_size, c_len, 1)
            q_mask = q_mask.view(batch_size, 1, q_len)

            
            S1 = F.softmax(mask_logits(s, q_mask), dim=2)
            S2 = F.softmax(mask_logits(s, c_mask), dim=1)

            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
            c2q_att = torch.bmm(S1, q)
            q2c_att = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), c)
            
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=2)
            x = self.resize_g_matrix(x)
            return x

        
        def output_layer(m0, m1, m2, mask):
            """
            :param g: (batch, c_len, hidden_size * 8)
            :param m: (batch, c_len ,hidden_size * 2)
            :return: p1: (batch, c_len), p2: (batch, c_len)
            """
            # (batch, c_len)

            p1 = self.p1_weight_g(torch.cat([m0, m1], dim = 2)).squeeze()
            # (batch, c_len)
            p2 = self.p2_weight_g(torch.cat([m0, m2], dim = 2)).squeeze()
            p1 = mask_logits(p1, mask)
            p2 = mask_logits(p2, mask)


            return p1, p2

        # 1. Character Embedding Layer
        c_char = char_emb_layer(batch.c_char)
        q_char = char_emb_layer(batch.q_char)
        
        
        # 2. Word Embedding Layer
        c_mask = torch.ones_like(batch.c_word[0]) == batch.c_word[0]
        q_mask = torch.ones_like(batch.q_word[0]) == batch.q_word[0]
        c_word = self.word_emb(batch.c_word[0])
        q_word = self.word_emb(batch.q_word[0])
        c_word = self.dropout(c_word)
        q_word = self.dropout(q_word)
        
        c_lens = batch.c_word[1]
        q_lens = batch.q_word[1]

        # Highway network
        c = self.context_resize(torch.cat([c_char, c_word], dim=-1))
        q = self.question_resize(torch.cat([q_char, q_word], dim=-1))
        
    
        c = highway_network(c)
        q = highway_network(q)
        
        # Transformer 
        
        #print("context size after highway: {}".format(c.size()))
        #print("question size after highway : {}".format(q.size()))
        
        c = self.embedding_encoder_block(c, c_mask)
        q = self.embedding_encoder_block(q, q_mask)
        #print("context size after encoder block: {}".format(c.size()))
        #print("question size after encoder block: {}".format(q.size()))
    
        # 4. Attention Flow Layer
        g = att_flow_layer(c, q, c_mask, q_mask)
        
        #print("G matrix size: {}".format(g.size()))
        
        M0 = self.dropout(g)
        for i, model_enc in enumerate(self.model_encoder_block):
            M0 = model_enc(M0, c_mask)
        #print("model_op 0 size: {}".format(enc_op_0.size()))
        
        M1 = M0
        for i, model_enc in enumerate(self.model_encoder_block):
            M0 = model_enc(M0, c_mask)
        #print("model_op 1 size: {}".format(enc_op_1.size()))
        
        M2 = self.dropout(M0)
        for i, model_enc in enumerate(self.model_encoder_block):
            M0 = model_enc(M0, c_mask)
        #print("model_op 2 size: {}".format(enc_op_2.size()))

        M3 = M0
        # 6. Output Layer
        p1, p2 = output_layer(M1, M2, M3, c_mask)

        # (batch, c_len), (batch, c_len)
        return p1, p2

def mask_logits(inputs, mask):
    mask = mask.type(torch.float32)
    return inputs + (-1e30) * (mask)



######### DELETE BELOW #########

"""

D = 96
Nh = 1
Dword = 100
Dchar = 16
batch_size = 3
dropout = 0.1
dropout_char = 0.05




class EncoderBlockOLD(nn.Module):
    def __init__(self, conv_num: int, ch_num: int, k: int):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSepConv(ch_num, ch_num, k) for _ in range(conv_num)])
        self.self_att = SelfAttention()
        self.FFN_1 = ResizeConv(ch_num, ch_num, activation=True, bias=True)
        self.FFN_2 = ResizeConv(ch_num, ch_num, bias=True)
        self.norm_C = nn.ModuleList([nn.LayerNorm(D) for _ in range(conv_num)])
        self.norm_1 = nn.LayerNorm(D)
        self.norm_2 = nn.LayerNorm(D)
        #self.transformer_encoder = nn.TransformerEncoderLayer(d_model=D, nhead = 1, dim_feedforward = 512)
        self.multihead_attn = nn.MultiheadAttention(96, 1)
        self.conv_num = conv_num
        self.spatial_dropout = nn.Dropout3d(p=0.2)
        
    def forward(self, x, mask, l, blks):
        total_layers = (self.conv_num+1)*blks
        out = PosEncoder(x)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out)
            if (i) % 2 == 0:
                out = F.dropout(out, p=dropout, training=self.training)
            out = conv(out.transpose(1,2)).transpose(1,2)
            #out = self.layer_dropout(out, res, dropout*float(l)/total_layers)
            out = self.spatial_dropout(out) + res
            l += 1
            
        #out = self.transformer_encoder(x.permute(1,0,2), src_key_padding_mask = mask).permute(1,0,2)
        res = out
        out = self.norm_1(out)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.multihead_attn(out.permute(1,0,2), out.permute(1,0,2), out.permute(1,0,2), key_padding_mask = mask)[0].permute(1,0,2)#.transpose(1,2)
        out = self.spatial_dropout(out) + res
        #out = self.layer_dropout(out, res, dropout*float(l)/total_layers)
        l += 1
        res = out
        out = self.norm_2(out)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.FFN_1(out)
        out = self.FFN_2(out)
        out = self.spatial_dropout(out) + res
        #out = self.layer_dropout(out, res, dropout*float(l)/total_layers)
        
        
        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0,1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual
            
"""