import torch
from torch.autograd import Function
from torch import nn
import math

#**************************************************************************************
class power(Function):
    @staticmethod
    def forward(ctx, input, pow1):
        x, y = input.size()
        input1 = input.clamp(min=0)
        output = torch.zeros(x, y, dtype=torch.float)
        for i in range(x):
            for j in range(y):
                if input1[i][j]:
                    output[i][j] = pow(input1[i][j], pow1[i][j])
        ctx.save_for_backward(input, pow1, output, input1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, pow1, output, input1 = ctx.saved_tensors
        x, y = output.size()
        grad_input = torch.zeros(x, y, dtype=torch.float)
        grad_pow1 = torch.zeros(x, y, dtype=torch.float)
        if ctx.needs_input_grad[0]:
            for i in range(x):
                for j in range(y):
                    if input[i][j] <= 0:
                        grad_input[i][j] = 0
                    else:
                        grad_input[i][j] = pow1[i][j]*(pow(input[i][j], pow1[i][j]-1))*grad_output[i][j]

        if ctx.needs_input_grad[1]:
            for i in range(x):
                for j in range(y):
                    if input1[i][j]:
                        grad_pow1[i][j] = math.log(input[i][j])*output[i][j]*grad_output[i][j]

        return grad_input, grad_pow1

power1 = power.apply

class Pow(nn.Module):
    def __init__(self, in_x, in_y):
        super(Pow, self).__init__()
        self.in_x = in_x
        self.in_y = in_y
        self.pow = nn.Parameter(torch.randn(in_x, in_y, dtype=torch.float))
    def forward(self, input):
        x, y = input.size()
        return  power1(input, self.pow.reshape(x, y))

#**************************************************************************************
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            # The size of the picture is 28x28
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
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
            Pow(16, 128),
            torch.nn.Linear(in_features=128, out_features=10),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, input):
        output = self.model(input)
        return output





