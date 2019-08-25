#@Time   :2019/8/24 9:43
#@author : qtgavc
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

# hyper Parameters
EPOCH = 1           # train data n times
BATCH_SIZE = 50     # mini batch size
LR = 0.001          # learning rate
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)


#plot an example
print(train_data.train_data.size())
print(train_data.train_labels.size())
plt.imshow(train_data.train_data[0].numpy(), cmap = 'gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()


train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = torchvision.datasets.MNIST(root='./mnist', train=False)
test_x = Variable(torch.unsqueeze(test_data.data, dim=1)).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.targets[:2000]
'''
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1*28*28

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,      #黑白图片
                out_channels=16,
                kernel_size=5,      #5*5 filter
                stride=1,
                padding=2,          #padding=(kernel_size-1)/2
            ),
            # 16*28*28
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2       #2*2 filter
            ),
            # 16*14*14
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            # 32*14*14
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 32*7*7
        )
        self.out = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

cnn = CNN()
# print(cnn)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)

        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = float((pred_y.numpy() == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, 'train loss: %.4f' % loss.item(), 'test accuracy: %.2f' % accuracy)

#print 10 predictions from test data
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
'''