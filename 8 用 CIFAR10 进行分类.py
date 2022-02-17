import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.utils import data

# import matplotlib.pyplot as plt


# 定义数据的预处理器
transform = tv.transforms.Compose(
    [
        # 转化为 tensor, 同时把数值从 0-255 变换到 0-1 之间
        tv.transforms.ToTensor(),
        # 归一化
        tv.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        # 归一化的过程:
        # 对每个 image 做: image = (image - mean)/std
        # 原来 0-1 的最小值 0 变成 (0-0.5)/0.5=-1, 而最大值 1 则变成 (1-0.5)/0.5=1
    ]
)


# 加载数据集和数据加载器
# DataSet 对象是一个数据集, 可用下标访问, 返回形如 (data, label) 的数据
train_set = tv.datasets.CIFAR10(
    root="./data", train=True, transform=transform, download=True
)
# DataLoader 是一个可迭代对象, 每次迭代返回一个 batch 的数据
train_loader = data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)

test_set = tv.datasets.CIFAR10(
    root="./data", train=False, transform=transform, download=True
)
test_loader = data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=0)


# CIFAR10 数据集的 10 个类别
# CIFAR10 数据集中每张图片大小: 3×32×32
# classes = (
#     "plane",
#     "car",
#     "bird",
#     "cat",
#     "deer",
#     "dog",
#     "frog",
#     "horse",
#     "ship",
#     "truck",
# )


# 可视化数据集
# to_pil_image = tv.transforms.ToPILImage()  # 用于把 tensor 转换成 image
# example_data, example_label = train_set[0]
# print(classes[example_label])
# example_image = to_pil_image((example_data + 1) / 2).resize(size=(50, 50))
# plt.imshow(example_image)
# plt.show()

# train_iter = iter(train_loader)  # 转换成生成迭代器来加载部分数据
# example_datas, labels = train_iter.next()
# print(''.join('%11s' % classes[labels[i]] for i in range(4)))
# example_images = to_pil_image(tv.utils.make_grid(
#     (example_datas+1)/2)).resize(size=(200, 50))
# plt.imshow(example_images)
# plt.show()


# 构建网络
class Net(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)  # 将 x 展开, -1 表示自适应
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


# 训练
num_epochs = 2
torch.set_num_threads(8)  # 设置 CPU 线程数
torch.set_printoptions(precision=10)  # 设置 tensor 的打印精度
for epoch in range(num_epochs):
    running_loss = 0
    for i, train_data in enumerate(train_loader, start=0):
        inputs, labels = train_data
        optimizer.zero_grad()
        outputs = net(inputs)
        # 使用 MSELoss 时须将 labels 转化为 one-hot vector:
        # criterion = nn.MSELoss()
        # loss = criterion(outputs, F.one_hot(labels, num_classes=10))
        loss = criterion(outputs, labels)
        optimizer.step()
        running_loss += loss
        if i % 2000 == 1999:
            print("%d %5d loss: %.3f: " % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0


# 采用一些图片进行预测
random_data, labels = iter(test_loader).next()
outputs = net(random_data)
# torch.max 返回两个 tensor, 第一个 tensor 是每行的最大值, 第二个 tensor 是每行最大值的索引
_, preds = torch.max(outputs, dim=1)
print(100 * ((preds == labels).sum() / labels.size()[0]))


# 在测试集上进行检验
correct_count = 0
total_count = 0
with torch.no_grad():
    for test_data, labels in test_loader:
        outputs = net(test_data)
        _, preds = torch.max(outputs, dim=1)
        correct_count += (preds == labels).sum()
        total_count += labels.size()[0]
print(100 * correct_count / total_count)


# 使用 GPU 进行提速
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net.to(device)
# random_data = random_data.to(device)
# labels = labels.to(device)
# output = net(random_data)
# loss = criterion(output, labels)

# 在多 GPU 上并行计算
# 原理: 将输入的 batch 均分成多份, 分别送到对应的 GPU 进行计算, 然后将各个 GPU 得到的梯度累加
# 方法 ①: torch.nn.parallel.data_parallel(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None)
# 方法 ②: torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
#        其中, device_ids 指定在哪些GPU上进行优化, output_device 指定输出到哪个GPU上

# torch.nn.DataParallel 返回一个新的能够自动在多 GPU 上进行并行加速的 module
# new_net = nn.DataParallel(net, device_ids=[0, 1])
# output = new_net(input)

# torch.nn.parallel.data_parallel 直接利用多 GPU 并行计算得出结果
# output = nn.parallel.data_parallel(net, input, device_ids=[0, 1])
