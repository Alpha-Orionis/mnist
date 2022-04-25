import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function


class relu1(nn.Module):
    def __init__(self):
        super(relu1, self).__init__()
    def forward(self, input):
        x, y=input.shape
        for i in range(0, x):
            for j in range(0, y):
                if input[i, j] < 0:
                    input[i, j] = 0
        return input

#****************************************************************
class MyReLU(Function):

    @staticmethod
    def forward(ctx, input_):
        # 在forward中，需要定义MyReLU这个运算的forward计算过程
        # 同时可以保存任何在后向传播中需要使用的变量值
        ctx.save_for_backward(input_)         # 将输入保存起来，在backward时使用
        output = input_.clamp(min=0)               # relu就是截断负数，让所有负数等于0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 根据BP算法的推导（链式法则），dloss / dx = (dloss / doutput) * (doutput / dx)
        # dloss / doutput就是输入的参数grad_output、
        # 因此只需求relu的导数，在乘以grad_output
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_ < 0] = 0                # 上诉计算的结果就是左式。即ReLU在反向传播中可以看做一个通道选择函数，所有未达到阈值（激活值<0）的单元的梯度都为0
        return grad_input


class Relu666(nn.Module):
    def __init__(self):
        super(Relu666, self).__init__()
    def forward(self, x):
        return MyReLU.apply(x)

#********************************************************************



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            # The size of the picture is 28x28
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # The size of the picture is 14x14
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # The size of the picture is 7x7
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),

            torch.nn.Flatten(),
            torch.nn.Linear(in_features=7 * 7 * 64, out_features=128),
            Relu666(),
            torch.nn.Linear(in_features=128, out_features=10),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, input):
        output = self.model(input)
        return output





