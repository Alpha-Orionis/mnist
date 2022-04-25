import numpy
import torch
import matplotlib.pyplot as plt
from net5 import Net

device = "cuda:0" if torch.cuda.is_available() else "cpu"
net = Net()
model_path = r"./model5_2.pth"
stat_dict = torch.load(model_path, map_location=torch.device('cpu'))
net.load_state_dict(stat_dict)
net.to(device)
net.eval()
for name,parameters in net.named_parameters():
    #print(name, ':', parameters)

    if name=="model.3.pow":
        x = parameters
        print(x)

im = numpy.zeros((48, 128))
for i in range(48):
    for j in range(128):
        im[i][j] = x[i][j]
# 隐藏x轴和y轴
plt.axes().get_xaxis().set_visible(False)
plt.axes().get_yaxis().set_visible(False)
plt.imshow(im, cmap=  'gray')
