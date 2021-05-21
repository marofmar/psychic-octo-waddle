import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor


class BaseRNN(nn.Module):
    """

    """
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN
    }

    def __init__(self,
                 input_size: int,
                 hidden_dim: int = 512,
                 num_layers: int = 1,
                 rnn_type: str = 'lstm',
                 dropout_p: float = 0.3,
                 bidirectional: bool = True,
                 device: torch.device = 'cuda'
                 ) -> None:
        super(BaseRNN, self).__init__()
        rnn_cell = self.supported_rnns[rnn_type]
        self.rnn = rnn_cell(input_size, hidden_dim, num_layrs, True, True, dropout_p, bidirectional)
        self.hidden_dim = hidden_dim
        self.device = device

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class BNReluRNN(baseRNN):
    """
    Batch Noramlization lyaer and Relu
    """
    def __init__(self,
                 input_size: int,
                 hidden_dim: int = 512,
                 rnn_type: str = 'gru',
                 bidirectional: bool = True,
                 dropout_p: float = 0.1,
                 device: torch.device = 'cuda'
                 ):
        super(BNReluRNN, self).__init__(input_size=input_size,
                                        hidden_dim=hidden_dim, num_layer=1,
                                        rnn_type=rnn_type, dropout_p=dropout_p,
                                        bidirectional=bidirectional, device=device)
        self.batch_norm = nn.BatchNorm1d(input_size)

    def forward(self, inputs: Tensor, input_length: Tensor):
        total_length = inputs.size(0)  # the number of inputs

        inputs = F.relu(self.batch_norm(inputs.transpose(1, 2)))  # (N,C,L) -> (N,L,C)
        inputs = inputs.transpose(1, 2)  # (N,L,C) -> (N,C,L)

        output = nn.utils.rnn.pack_padded_sequence(inputs, input_length)
        output, hidden = self.rnn(output)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, total_length=total_length)

        return output


class Linear(nn.Module):
    """
    torch.nn.linear wrapper
    weight initialization by xavier, bias initialization to zeros

    Originally,
    ~Linear.weight - initialized from U(-sqrt.k, sqrt.k) where k = 1/(size of in_features)
    ~Linear.bias - same as above
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bais=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor)-> Tensor:
        return self.linear(x)


class LayerNorm(nn.Module):
    """
    wrapper for torch.nn.LayerNorm
    근데 이건 꼭 래퍼 안 써도 그게 그거 같은데, 내가 아직 뭘 모르는거겠지..?
    분모 부분이 약간 다른것 같기도 하고 (~Originally, sqrt(var[x]+eps) -> 여기서는, std + eps)
    """
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, z:Tensor) -> Tensor:
        mean = z.mean(dim=-1, keepdim=True)
        std = z.std(dim=-1, keepdim=True)
        output = (z-mean)/(std+self.eps)
        output = self.gamma * output + self.beta

        return output


class View(nn.Moduel):
    """
    wrapper for torch.view() for sequential module!
    """
    def __init__(self, shape: tuple, contiguous: bool=False):
        super(View, self).__init__()
        self.shape = shape
        self.contiguous = contiguous

    def forward(self, inputs):
        if self.contiguous:
            inputs = inputs.contiguous()

        return inputs.view(*self.shape)


class Transpose(nn.Module):
    """
    wrapper for torch.transpose() for sequential module
    """
    def __init__(self, shape: tuple):
        super(Transpose,self).__init__()
        self.shape = shape

    def forward(selfself, inputs: Tensor):
        return inputs.transpose(*self.shape)


