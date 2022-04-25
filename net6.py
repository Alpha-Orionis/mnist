import torch
from torch.autograd import Function
from torch import nn
import math
import  random

#************************************************************************************
class powerL(Function):
    @staticmethod
    def forward(ctx, input, pow1, pow2, w1, w2, b):
        x, y = input.size()
        input1 = input.clamp(min=0)
        output1 = torch.zeros(x, y, dtype=torch.float)
        output2 = torch.zeros(x, y, dtype=torch.float)
        for i in range(x):
            for j in range(y):
                if input1[i][j]:
                    output1[i][j] = pow(input1[i][j], pow1[i][j])
                    output2[i][j] = pow(input1[i][j], pow2[i][j])
        ctx.save_for_backward(input, pow1, pow2, w1, w2, b, output1, output2, input1)
        output1 = output1.mm(w1.t())
        output2 = output2.mm(w2.t())
        return output1+output2+b

    @staticmethod
    def backward(ctx, grad_output):
        input, pow1, pow2, w1, w2, b, output1, output2, input1 = ctx.saved_tensors
        x, y = input.size()
        grad_input1 = torch.zeros(x, y, dtype=torch.float)
        grad_input2 = torch.zeros(x, y, dtype=torch.float)
        grad_pow1 = torch.zeros(x, y, dtype=torch.float)
        grad_pow2 = torch.zeros(x, y, dtype=torch.float)
        grad_w1 = grad_w2 = grad_b = None

        if ctx.needs_input_grad[3]:
            grad_w1 = grad_output.t().mm(output1)
            grad_w2 = grad_output.t().mm(output2)

        if ctx.needs_input_grad[0]:
            grad_ww1 = grad_output.mm(w1)
            grad_ww2 = grad_output.mm(w2)
            for i in range(x):
                for j in range(y):
                    if input[i][j] <= 0:
                        grad_input1[i][j] = 0
                        grad_input2[i][j] = 0
                    else:
                        grad_input1[i][j] = pow1[i][j]*(pow(input[i][j], pow1[i][j]-1)) * grad_ww1[i][j]
                        grad_input2[i][j] = pow2[i][j] * (pow(input[i][j], pow2[i][j] - 1))* grad_ww2[i][j]


        if ctx.needs_input_grad[1]:
            grad_pow1 = grad_output.mm(w1)
            grad_pow2 = grad_output.mm(w2)
            for i in range(x):
                for j in range(y):
                    if input1[i][j]:
                        grad_pow1[i][j] *= math.log(input[i][j]) * output1[i][j]
                        grad_pow2[i][j] *= math.log(input[i][j]) * output2[i][j]

        if ctx.needs_input_grad[5]:
            grad_b = grad_output

        return grad_input1+grad_input2, grad_pow1, grad_pow2, grad_w1, grad_w2, grad_b

power1 = powerL.apply

class PowL(nn.Module):
    def __init__(self, in_x, in_y, out_x):#in_x:层大小  in_y：通道数
        super(PowL, self).__init__()
        self.in_x = in_x
        self.in_y = in_y
        self.out_x = out_x
        self.pow1 = nn.Parameter(torch.ones(in_y, in_x, dtype=torch.float))
        self.pow2 = nn.Parameter(torch.ones(in_y, in_x, dtype=torch.float))
        self.w1 = nn.Parameter(torch.randn(out_x, in_x, dtype=torch.float))
        self.w2 = nn.Parameter(torch.randn(out_x, in_x, dtype=torch.float))
        self.b = nn.Parameter(torch.randn(in_y, out_x, dtype=torch.float))
    def forward(self, input):
        return  power1(input, self.pow1, self.pow2, self.w1, self.w2, self.b)

#**************************************************************************************

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            # The size of the picture is 28x28
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            # The size of the picture is 14x14
            torch.nn.Flatten(),
            #torch.nn.Linear(in_features=28 * 28 * 32, out_features=128),
            #nn.ReLU(),
            PowL(28 * 28 * 16, 48, 128),
            torch.nn.Linear(in_features=128, out_features=10),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, input):
        output = self.model(input)
        return output





