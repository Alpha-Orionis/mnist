from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from net5 import Net

batch = 128
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
#构建迭代器与损失函数
lossF = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())
"""
optimizer = torch.optim.SGD(params=net.parameters(),
                            lr=0.03,
                            momentum=0.9,
                            dampening=0.5,
                            weight_decay=0.01,
                            nesterov=False)
"""
#打开网络的训练模式
net.train(True)

EPOCHS = 30
# 存储训练过程
history = {'Test Loss': [], 'Test Accuracy': []}
for epoch in range(1, EPOCHS + 1):
    print(epoch)
    # 构建tqdm进度条
    processBar = tqdm(datatrain, unit='step')
    net.train(True)
    for step, (trainImgs, labels) in enumerate(processBar):
        trainImgs = trainImgs.to(device)
        labels = labels.to(device)

        net.zero_grad()
        outputs = net(trainImgs)
        loss = lossF(outputs, labels)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = torch.sum(predictions == labels) / labels.shape[0]
        loss.backward()

        optimizer.step()
        processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" %
                                   (epoch, EPOCHS, loss.item(), accuracy.item()))

    processBar.close()
    print("finish")
    torch.save(net.state_dict(), './model5_5.pth')
"""
        if step == len(processBar) - 1:
            correct, totalLoss = 0, 0
            net.train(False)
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
            history['Test Loss'].append(testLoss.item())
            history['Test Accuracy'].append(testAccuracy.item())
            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %
                                       (epoch, EPOCHS, loss.item(), accuracy.item(), testLoss.item(),
                                        testAccuracy.item()))
    processBar.close()
    """
# print("finish")
# torch.save(net.state_dict(), './model666.pth')




