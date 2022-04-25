import torch
from torch.autograd import Function
from torch import nn
import math
import  random

#**************************************************************************************
class power(Function):
    @staticmethod
    def forward(ctx, input, pow1):
        input1 = input.clamp(min=0)
        output = torch.pow(input1, pow1)
        output = torch.where(input1 == 0, torch.full_like(output, 0), output)
        ctx.save_for_backward(input, pow1, output, input1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, pow1, output, input1 = ctx.saved_tensors
        grad_input = grad_pow1 = None
        if ctx.needs_input_grad[0]:
            grad_input = pow1 * torch.pow(input1, pow1-1) * grad_output
            grad_input = torch.where(input1 == 0, torch.full_like(grad_input, 0), grad_input)

        if ctx.needs_input_grad[1]:
            input1s = torch.where(input1 == 0, torch.full_like(input1, 1), input1)
            grad_pow1 = torch.log(input1s) * output * grad_output
            grad_pow1 = grad_pow1.mean(0)
        return grad_input, grad_pow1

power1 = power.apply

class Pow(nn.Module):
    def __init__(self, in_x):
        super(Pow, self).__init__()
        self.in_x = in_x
        self.pow = nn.Parameter(torch.rand(in_x, dtype=torch.float))
    def forward(self, input):
        return power1(input, self.pow)

#**************************************************************************************
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            # The size of the picture is 28x28
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            # The size of the picture is 14x14
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=28 * 28 * 32, out_features=128),
            Pow(128),
            torch.nn.Linear(in_features=128, out_features=10),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, input):
        output = self.model(input)
        return output





