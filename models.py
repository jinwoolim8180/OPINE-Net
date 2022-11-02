import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os

n_output = 1089
nrtrain = 88912   # number of training blocks
batch_size = 64


class MySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

MyBinarize = MySign.apply


class ConvGRUMod(nn.Module):
    def __init__(self, inp_dim, oup_dim):
        super().__init__()
        self.conv_ir = nn.Conv2d(inp_dim, oup_dim, 3, padding=1)
        self.conv_hr = nn.Conv2d(inp_dim, oup_dim, 3, padding=1)

        self.conv_iz = nn.Conv2d(inp_dim, oup_dim, 3, padding=1)
        self.conv_hz = nn.Conv2d(inp_dim, oup_dim, 3, padding=1)

        self.conv_in = nn.Conv2d(inp_dim, oup_dim, 3, padding=1)
        self.conv_hn = nn.Conv2d(inp_dim, oup_dim, 3, padding=1)

    def forward(self, x, h):

        if h is None:
            r = torch.sigmoid(self.conv_ir(x) * self.conv_hr(x))
            n = r * torch.tanh(self.conv_in(x))
            h = n
        else:
            r = torch.sigmoid(self.conv_ir(x) * self.conv_hr(h))
            z = torch.sigmoid(self.conv_iz(x) + self.conv_hz(h))
            n = torch.tanh(r * self.conv_in(x))
            h = (1 - z) * n + z * h

        return h, h


# Define OPINE-Net Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.gru1 = ConvGRUMod(32, 32)
        self.gru2 = ConvGRUMod(32, 32)

        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

        self.conv1_G = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_G = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

    def forward(self, x, PhiWeight, PhiTWeight, PhiTb, h1, h2):
        x = x - self.lambda_step * PhiTPhi_fun(x, PhiWeight, PhiTWeight)
        x = x + self.lambda_step * PhiTb
        x_input = x

        x_D = F.conv2d(x_input, self.conv_D, padding=1)
        x_D, h1 = self.gru1(x_D, h1)

        x = F.conv2d(x_D, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x_backward, h2 = self.gru2(x_backward, h2)

        x = F.conv2d(F.relu(x_backward), self.conv1_G, padding=1)
        x = F.conv2d(F.relu(x), self.conv2_G, padding=1)
        x_G = F.conv2d(x, self.conv3_G, padding=1)

        x_pred = x_input + x_G

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_D_est - x_D

        return [x_pred, symloss, h1, h2]


# Define OPINE-Net-plus
class OPINENetplus(torch.nn.Module):
    def __init__(self, LayerNo, n_input):
        super(OPINENetplus, self).__init__()
        self.n_input = n_input
        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, 1089)))
        self.Phi_scale = nn.Parameter(torch.Tensor([0.01]))

        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, x):

        # Sampling-subnet
        Phi_ = MyBinarize(self.Phi)
        Phi = self.Phi_scale * Phi_
        PhiWeight = Phi.contiguous().view(self.n_input, 1, 33, 33)
        Phix = F.conv2d(x, PhiWeight, padding=0, stride=33, bias=None)    # Get measurements

        # Initialization-subnet
        PhiTWeight = Phi.t().contiguous().view(n_output, self.n_input, 1, 1)
        PhiTb = F.conv2d(Phix, PhiTWeight, padding=0, bias=None)
        PhiTb = torch.nn.PixelShuffle(33)(PhiTb)
        x = PhiTb    # Conduct initialization

        # Recovery-subnet
        h1, h2 = None, None
        layers_sym = []   # for computing symmetric loss
        for i in range(self.LayerNo):
            [x, layer_sym, h1, h2] = self.fcs[i](x, PhiWeight, PhiTWeight, PhiTb, h1, h2)
            layers_sym.append(layer_sym)

        x_final = x

        return [x_final, layers_sym, Phi]


def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0,stride=33, bias=None)
    temp = F.conv2d(temp, PhiTW, padding=0, bias=None)
    return torch.nn.PixelShuffle(33)(temp)