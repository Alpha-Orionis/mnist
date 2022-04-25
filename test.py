from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from net5 import Net

batch = 48
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# 加上transforms
transform = transforms.Compose([
    #transforms.Grayscale,
    transforms.Resize([28,28]),
    transforms.ToTensor(),  # 将图片转换为Tensor
])

dataset_train = ImageFolder(root='D:/Datas/Mnist/train2/', transform=transform)
datatrain = DataLoader(dataset_train, batch_size=batch, shuffle=True, drop_last=True)
dataset_test = ImageFolder(root='D:/Datas/Mnist/test/', transform=transform)
datatest = DataLoader(dataset_test, batch_size=batch, shuffle=True, drop_last=True)

#将模型转换到device中
net = Net()
model_path = r"./model5_5.pth"
stat_dict = torch.load(model_path, map_location=torch.device('cpu'))
net.load_state_dict(stat_dict)
net.to(device)
net.eval()
#构建迭代器与损失函数
lossF = torch.nn.CrossEntropyLoss()
"""
optimizer = torch.optim.SGD(params=net.parameters(),
                            lr=0.03,
                            momentum=0.9,
                            dampening=0.5,
                            weight_decay=0.01,
                            nesterov=False)
"""
#网络的训练模式
net.train(False)

EPOCHS = 1
# 存储训练过程
history = {'Test Loss': [], 'Test Accuracy': []}
for epoch in range(1, EPOCHS + 1):
    print(epoch)
    # 构建tqdm进度条
    processBar = tqdm(datatest, unit='step')
    correct, totalLoss = 0, 0
    for testImgs, labels in datatest:
        testImgs = testImgs.to(device)
        labels = labels.to(device)
        outputs = net(testImgs)
        loss = lossF(outputs, labels)
        predictions = torch.argmax(outputs, dim=1)
        totalLoss += loss
        correct += torch.sum(predictions == labels)
        testAccuracy = correct / (batch * len(datatest))
        testLoss = totalLoss / len(datatest)
        history['Test Loss'].append(testLoss)
        history['Test Accuracy'].append(testAccuracy)
        processBar.set_description("[%d/%d] Test Loss: %.4f, Test Acc: %.4f" %
                                    (epoch, EPOCHS, testLoss,
                                    testAccuracy))

    processBar.close()


