# flameMind 🔥, 1.0.0 license
"""Seq2Seq framework."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from models.EncoderDecoder import Encoder,Decoder, EncoderDecoder, AttentionDecoder
from models.attention import AdditiveAttention
from models.rnn_layer import get_rnn_layer

class Seq2SeqEncoder(Encoder):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, dims, num_hiddens, num_layers,get_rnn_layer,
                 selected_model, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        get_rnn_layer = get_rnn_layer(input_size=dims,
                                      num_hiddens=num_hiddens,
                                      num_layers=num_layers,
                                      dropout=dropout)
        self.rnn = get_rnn_layer.construct_rnn(selected_model=selected_model)
        # self.rnn = nn.GRU(input_size = dims,
        #               hidden_size = num_hiddens,
        #               num_layers = num_layers,
        #               dropout=dropout)
    def forward(self, X, *args):
        X = X.permute(1, 0, 2)
        Y, state = self.rnn(X)
        return Y, state

class Seq2SeqDecoder(Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, dims, embed_size, num_hiddens, num_layers, get_rnn_layer,
                 selected_model, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding =  nn.Embedding(dims, embed_size)
        get_rnn_layer = get_rnn_layer(input_size=embed_size + num_hiddens,
                                      num_hiddens=num_hiddens,
                                      num_layers=num_layers,
                                      dropout=dropout)
        self.rnn = get_rnn_layer.construct_rnn(selected_model=selected_model)
        # self.rnn = nn.GRU(input_size = embed_size + num_hiddens,
        #                   hidden_size = num_hiddens,
        #                   num_layers = num_layers,
        #                   dropout=dropout)
        self.dense1 = nn.Linear(num_hiddens, 32)
        self.dense2 = nn.Linear(32, 2)
    def init_state(self, enc_output, *args):
        return enc_output[1]
    def forward(self, X, state):
        X = self.embedding(X).permute(1,0,2)
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), -1)
        Y, state = self.rnn(X_and_context,state)

        output = self.dense2(F.relu(self.dense1(Y))).permute(1,0,2)
        return output, state

class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, dims, embed_size, num_hiddens, num_layers,get_rnn_layer,
                 selected_model,dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = AdditiveAttention(
            key_size=num_hiddens,
            query_size=num_hiddens,
            num_hiddens=num_hiddens,
            dropout=dropout)
        self.embedding = nn.Embedding(dims, embed_size)
        get_rnn_layer = get_rnn_layer(input_size=embed_size + num_hiddens,
                                      num_hiddens=num_hiddens,
                                      num_layers=num_layers,
                                      dropout=dropout)
        self.rnn = get_rnn_layer.construct_rnn(selected_model=selected_model)
        # self.rnn = nn.GRU(
        #     embed_size + num_hiddens, num_hiddens, num_layers,
        #     dropout=dropout)
        self.dense1 = nn.Linear(num_hiddens, 32)
        self.dense2 = nn.Linear(32, 2)

    def init_state(self, enc_outputs, *args):
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state)

    def forward(self, X, state):
        enc_outputs, hidden_state = state
        X = self.embedding(X).permute(1, 0, 2)  # (32, 10, 8) -> (10,32,8)

        outputs, self._attention_weights = [], []
        for x in X:
            query = torch.unsqueeze(hidden_state[-1], dim=1)  # (32, 1, 32)
            context = self.attention(
                queries=query,
                keys=enc_outputs,  # (32, 10, 32)
                values=enc_outputs)  # (32, 10, 32)
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        outputs = self.dense2(F.relu(self.dense1(torch.cat(outputs, dim=0))))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state]

    @property
    def attention_weights(self):
        return self._attention_weights

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-hiddens", type=int, default=32, help="number of hiddens")
    parser.add_argument("--num-layers", type=int, default=2, help="number of layers")
    parser.add_argument("--embed-size", type=int, default=8, help="embedding size")
    parser.add_argument("--dims", type=int, default=5, help="input size")
    parser.add_argument("--encoder-model", type=str, default="GRU", help="select rnn model, i.e. RNN, GRU, LSTM et.al")
    parser.add_argument("--decoder-model", type=str, default="GRU", help="select rnn model, i.e. RNN, GRU, LSTM et.al")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout")
    opt = parser.parse_args()
    return opt

def main(opt):
    # ===========测试==============#
    encoder = Seq2SeqEncoder(dims=opt.dims,
                             num_hiddens=opt.num_hiddens,
                             num_layers=opt.num_layers,
                             get_rnn_layer=get_rnn_layer,
                             selected_model=opt.encoder_model)
    encoder.eval()
    # decoder = Seq2SeqDecoder(dims=opt.dims,
    #                          embed_size=opt.embed_size,
    #                          num_hiddens=opt.num_hiddens,
    #                          num_layers=opt.num_layers,
    #                          get_rnn_layer = get_rnn_layer,
    #                          selected_model = opt.decoder_model)
    #
    decoder = Seq2SeqAttentionDecoder(dims=opt.dims,
                             embed_size=opt.embed_size,
                             num_hiddens=opt.num_hiddens,
                             num_layers=opt.num_layers,
                             get_rnn_layer=get_rnn_layer,
                             selected_model=opt.decoder_model)
    decoder.eval()
    net = EncoderDecoder(encoder, decoder)
    print(net)
if __name__ == '__main__':
   opt = parse_opt()
   main(opt)
