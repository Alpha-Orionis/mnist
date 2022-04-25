import torch
from torch.autograd import Function
from torch import nn
import math
from multiprocessing.pool import ThreadPool as Pool
import  random

# ************************************************************************************
class powerL(Function):
    @staticmethod
    def forward(ctx, input, pow1, pow2, w1, w2, b):
        input1 = input.clamp(min=0)
        output1 = torch.pow(input1, pow1)
        output2 = torch.pow(input1, pow2)
        output1 = torch.where(torch.isinf(output1), torch.full_like(output1, 0), output1)
        output2 = torch.where(torch.isinf(output2), torch.full_like(output2, 0), output2)
        ctx.save_for_backward(pow1, pow2, w1, w2, output1, output2, input1)
        output1 = output1.mm(w1.t())
        output2 = output2.mm(w2.t())
        return output1 + output2 + b.unsqueeze(0).expand_as(output1)

    @staticmethod
    def backward(ctx, grad_output):
        pow1, pow2, w1, w2, output1, output2, input1 = ctx.saved_tensors
        grad_w1 = grad_w2 = grad_b = grad_input = None
        grad_pow1 = grad_output.mm(w1)
        grad_pow2 = grad_output.mm(w2)

        if ctx.needs_input_grad[3]:
            grad_w1 = grad_output.t().mm(output1)
            grad_w2 = grad_output.t().mm(output2)

        if ctx.needs_input_grad[0]:
            grad_input1 = torch.pow(input1, pow1 - 1)
            grad_input2 = torch.pow(input1, pow2 - 1)
            grad_input1 = torch.where(torch.isinf(grad_input1), torch.full_like(grad_input1, 0), grad_input1)
            grad_input2 = torch.where(torch.isinf(grad_input2), torch.full_like(grad_input2, 0), grad_input2)
            grad_input1 *= pow1 * grad_pow1
            grad_input2 *= pow2 * grad_pow2
            grad_input = grad_input1 + grad_input2

        if ctx.needs_input_grad[1]:
            input_log = torch.log(input1)
            grad_pow1s = grad_pow1 * input_log * output1
            grad_pow2s = grad_pow2 * input_log * output2
            grad_pow1 = torch.where(input1 == 0, grad_pow1, grad_pow1s).mean(0)
            grad_pow2 = torch.where(input1 == 0, grad_pow2, grad_pow2s).mean(0)

        if ctx.needs_input_grad[5]:
            grad_b = grad_output.mean(0)

        return grad_input, grad_pow1, grad_pow2, grad_w1, grad_w2, grad_b


power1 = powerL.apply


class PowL(nn.Module):
    def __init__(self, in_x, out_x):  # in_x:层大小  in_y：通道数
        super(PowL, self).__init__()
        self.in_x = in_x
        self.out_x = out_x
        self.pow1 = nn.Parameter(torch.full([in_x], 0.5, dtype=torch.float))
        self.pow2 = nn.Parameter(torch.full([in_x], 0.5, dtype=torch.float))
        self.w1 = nn.Parameter(torch.full([out_x, in_x], 1 ,dtype=torch.float))
        self.w2 = nn.Parameter(torch.full([out_x, in_x], 1 ,dtype=torch.float))
        self.b = nn.Parameter(torch.zeros(out_x, dtype=torch.float))

    def forward(self, input):
        return power1(input, self.pow1, self.pow2, self.w1, self.w2, self.b)


# **************************************************************************************

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            # The size of the picture is 28x28
            torch.nn.Conv2d(in_channels=3, out_channels=48, kernel_size=3, stride=1, padding=1),
            # The size of the picture is 14x14
            torch.nn.Flatten(),
            #torch.nn.Linear(in_features=28 * 28 * 32, out_features=128),
            #nn.ReLU(),
            PowL(28 * 28 * 48, 256),
            PowL(256, 128),
            torch.nn.Linear(in_features=128, out_features=10),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, input):
        output = self.model(input)
        return output





