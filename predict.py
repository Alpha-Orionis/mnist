from net5 import Net
import torch
from PIL import Image as im
from PIL import Image
import numpy as np

nets = Net()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_path = r"./model5_4.pth"
stat_dict = torch.load(model_path, map_location=torch.device('cpu'))
nets.load_state_dict(stat_dict)
nets.to(device)
nets.eval()
img = im.open(r"./78.jpg")
img = img.resize((28, 28))
img = np.asarray(img)
img2 = img.transpose(2, 0, 1)
"""
对单通道图片的变换：
img2 = np.ones((3, 28, 28))
img2[0, :, :] = img
img2[1, :, :] = img
img2[2, :, :] = img
"""
img_in = torch.from_numpy(img2)
img_in = torch.unsqueeze(img_in, dim=0)
img_in = img_in.to(device)
img_in = img_in.float()
result_org = nets(img_in)
print(result_org)
