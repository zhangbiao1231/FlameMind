import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
from d2l import torch as d2l

#===========================================RNN=================================================#
def get_rnn_params(dims, num_hiddens, device):
    num_inputs = num_outputs = dims

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def rnn(inputs, state, params):
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形状：(批量大小，词表大小)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
#===========================================GRU=================================================#
# 初始化模型参数
def get_gru_params(dims, num_hiddens, device):
    num_inputs = num_outputs = dims  # 28

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal((num_inputs, num_hiddens)),  # dxh
                normal((num_hiddens, num_hiddens)),  # hxh
                torch.zeros(num_hiddens, device=device))  # 1xh

    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xh, W_hh, b_h = three()  # 候选状态参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))  # hxq
    b_q = torch.zeros(num_outputs, device=device)  # 1xq
    # 附加梯度
    params = [W_xr, W_hr, b_r,  # 重置门参数
              W_xz, W_hz, b_z,  # 更新门参数
              W_xh, W_hh, b_h,  # 候选状态参数
              W_hq, b_q]  # 输出层参数
    for param in params:
        param.requires_grad = True
    return params

def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device = device), ) # 返回(H,)
def gru(inputs, state, params):
    W_xr,W_hr,b_r,  W_xz,W_hz,b_z,  W_xh,W_hh,b_h,  W_hq,b_q = params
    H, = state
    outputs = []
    for X in inputs:
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        H_tilde = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilde
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
#===========================================LSTM=================================================#
# 初始化模型参数
def get_lstm_params(dims, num_hiddens, device):
    num_inputs = num_outputs = dims  # 28

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal((num_inputs, num_hiddens)),  # dxh
                normal((num_hiddens, num_hiddens)),  # hxh
                torch.zeros(num_hiddens, device=device))  # 1xh

    W_xf, W_hf, b_f = three()  # 遗忘门参数
    W_xi, W_hi, b_i = three()  # 输入门参数
    W_xc, W_hc, b_c = three()  # 候选记忆参数
    W_xo, W_ho, b_o = three()  # 输出门参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))  # hxq
    b_q = torch.zeros(num_outputs, device=device)  # 1xq
    # 附加梯度
    params = [W_xf, W_hf, b_f,  # 遗忘门参数
              W_xi, W_hi, b_i,  # 输入门参数
              W_xc, W_hc, b_c,  # 候选记忆参数
              W_xo, W_ho, b_o,  # 输出门参数
              W_hq, b_q]  # 输出层参数
    for param in params:
        param.requires_grad = True
    return params
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device = device),
            torch.zeros((batch_size, num_hiddens), device = device), ) # 返回(H,C)

# 定义长短时记忆网络模型，模型的架构与基本的循环神经网络单元时相同的，只是权重更新公式更加复杂。
def lstm(inputs, state, params):
    W_xf,W_hf,b_f,  W_xi,W_hi,b_i,  W_xc,W_hc,b_c, W_xo,W_ho,b_o,  W_hq,b_q = params
    (H, C) = state
    outputs = []
    # X的形状：（批量大小， 词表大小）
    for X in inputs:
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilde = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilde
        H = O * torch.tanh(C)
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)
#===========================================从零开始实现=================================================#
class RNNModelScratch: #@save
    """从零开始实现的循环神经网络模型"""
    def __init__(self, dims, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.dims, self.num_hiddens = dims, num_hiddens
        self.params = get_params(dims, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.dims).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)
#===========================================调用框架=================================================#
class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, rnn_layer, dims, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.dims = dims
        self.num_hiddens = self.rnn.hidden_size
        ## 如果RNN是双向的，num_directions=2，否侧是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, 2)  # W_hq，h_q
        else:
            self.directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, 2)

    def forward(self, inputs, state):
        X = inputs.permute(1, 0, -1).type(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # state 对于nn.GRU或nn.RNN作为张量
            # 非 nn.LSTM 模型（如 nn.RNN 或 nn.GRU）,函数返回一个三维张量。
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                               device=device)
        else:
            # state 对于nn.LSTM 是个元组
            # • 如果模型是 nn.LSTM，则返回一个包含两个张量的元组，分别表示 LSTM 的隐藏状态 (hidden state) 和单元状态 (cell state)。
            # • 每个张量的形状相同，均为 (num_directions * num_layers, batch_size, hidden_size)，其中：
            ##  • hidden state：储存每一层每个时间步的隐藏状态。
            ##  • cell state：储存每一层每个时间步的记忆状态，用于控制长期依赖信息。
            return (torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                                device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                                device=device))

class get_rnn_layer(nn.Module):
    def __init__(self, input_size, num_hiddens, num_layers, dropout,**kwargs):
        super(get_rnn_layer, self).__init__(**kwargs)
        self.input_size = input_size
        self.num_hiddens=num_hiddens
        self.num_layers=num_layers
        self.dropout = dropout
    def construct_rnn(self,selected_model):
        if selected_model == "RNN":
            return nn.RNN(input_size=self.input_size,
                          hidden_size=self.num_hiddens,
                          num_layers=self.num_layers,
                          dropout=self.dropout)
        elif selected_model == "GRU":
            return nn.GRU(input_size=self.input_size,
                          hidden_size=self.num_hiddens,
                          num_layers=self.num_layers,
                          dropout=self.dropout)
        else:
            return nn.LSTM(input_size=self.input_size,
                          hidden_size=self.num_hiddens,
                          num_layers=self.num_layers,
                          dropout=self.dropout)
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-hiddens", type=int, default=256, help="number of hiddens")
    parser.add_argument("--num-layers", type=int, default=2, help="number of layers")
    parser.add_argument("--dims", type=int, default=5, help="input size")
    parser.add_argument("--model-name", type=str, default="GRU", help="select rnn model, i.e. RNN, GRU, LSTM et.al")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout")
    opt = parser.parse_args()
    return opt
if __name__ == '__main__':
    opt = parse_opt()
    get_rnn_layer = get_rnn_layer(input_size=opt.dims,
                                  num_hiddens=opt.num_hiddens,
                                  num_layers=opt.num_layers,
                                  dropout=opt.dropout)
    rnn_layer = get_rnn_layer.construct_rnn(selected_model=opt.model_name)
    device = d2l.try_gpu()
    net = RNNModel(rnn_layer=rnn_layer,
                   dims=opt.dims)
    print(net)