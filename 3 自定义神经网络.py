import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss

################
## ⭐示例一⭐ ##
################

# 1) torch.nn 是构建于 Autograd 上可以用来定义和运行神经网络的接口.
# 2) 要想自定义一个网络, 则需要实现一个继承 nn.Module 的类.
# 3) 这里以构建卷积网络 LeNet 为例.


# ✅ 构建 LeNet
class LeNet(nn.Module):

    def __init__(self):
        # nn.Module 子类必须在构造函数中执行父类的构造函数
        super(LeNet, self).__init__()  # 等价于 nn.Module.__init__(self)

        # 1) 把网络中具有可学习参数的层放在构造函数中,
        #    不具有可学习参数的层直接在 forward 方法中通过 nn.functional 接口来添加.
        # 2) torch.nn.XXX: 先实例化后才能使用,
        #    在实例化时传入网络层的参数 (不包括 W 和 b, W 和 b 由内部自动管理)
        # 3) torch.nn.functional.XXX: 直接使用,
        #    在 input 后面添加网络层的参数 (包括 W 和 b, W 和 b 须手动进行管理)
        # 4) 注意: 每种网络层的参数都是针对单个样本的特征情况
        self.conv1 = nn.Conv2d(1, 6, 5)  # 输入图片的通道数, 卷积核数, 卷积核大小5×5
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # 在 forward 方法中实现网络的前向传播过程
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(x.fc1(x))
        x = F.relu(x.fc2(x))
        x = self.fc3(x)
        return x


# ✅ 实例化网络
net = LeNet()
net.zero_grad()  # 清除网络中可学习参数的梯度
net.eval()  # 将模型切换到 eval 模式, 该模式会禁用网络中的某些层 (比如 dropout)
net.train()  # 将模型切换到 train 模式, 该模式会恢复网络中被禁用的层

print(net.parameters())  # 获取网络的可学习参数
for name, param in net.named_parameters():  # 获取可学习参数及其名称
    print(name, param)

# ✅ 训练预准备
one_sample = torch.randn(1, 32, 32)
labels = torch.arange(0, 10).view(1, 10).float()
criterion = nn.MSELoss()  # 构建损失的计算函数
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)  # 构建优化器

# ✅ 单次训练过程
# torch.nn 只支持 batch 作为输入,
# 即 input 的形状必须为 (batch_size, featrue1, featrue2, ...).
# 如果要输入单样本, 则可用 .unsqueeze(0) 方法将 batch_size 设为 1.
input = one_sample.unsqueeze(0)  # shape=(1, 1, 32, 32)
output = net(input)  # 前向传播
loss = criterion(output, labels)  # 计算损失
optimizer.zero_grad()  # 效果同 net.zero_grad()
loss.backward()  # 反向传播
optimizer.step()  # 更新参数

# optimizer.step() 的手动实现如下:
# learning_rate = 0.01
# for p in net.parameters():
#     p.data.sub_(p.grad.data * learning_rate)

################
## ⭐示例二⭐ ##
################

# 1) nn.Module 会递归查找 nn.Parameter, 将其作为学习参数,
#    包括自身以及任意子层级的 nn.Module 定义的 nn.Parameter.
# 2) 这里以构建多层感知机网络为例.


# ✅ 构建 Linear
class Linear(nn.Module):
    # 在构造函数中定义需要学习的东西: 包括可学习参数和具有可学习参数的 layer
    def __init__(self, in_num, out_num):
        super(Linear, self).__init__()
        # 定义可学习参数要使用 nn.Parameter 进行封装,
        # 这是一种特殊的 tensor, 默认 requires_grad=True
        self.w = nn.Parameter(torch.randn(in_num, out_num))
        # 等价于 self.register_parameter('w', nn.Parameter(torch.randn(in_num, out_num)))
        self.b = nn.Parameter(torch.randn(out_num))

    # 在 forward 方法中实现前向传播过程.
    # 不具有可学习参数的任何方法应直接在此处调用, 不需要定义在构造函数中.
    # nn.Module 的 backward 方法会通过 AotoGrad 按照 forward 自动实现.
    def forward(self, x):
        x = x.mm(self.w)
        x += self.b.expand_as(x)
        return x


# ✅ 在已建立的 Linear 网络的基础上, 实现多层感知机网络
class Perceptron(nn.Module):

    def __init__(self, in_num, hidden_num, out_num):
        nn.Module(self).__init__()
        self.layer1 = Linear(in_num, hidden_num)
        self.layer2 = Linear(hidden_num, out_num)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.sigmoid(x)
        x = self.layer2(x)
        return x


#################
## ⭐一些结论⭐ ##
#################


# autograd.Function, nn.Module 和 nn.functional 的区别:
#   1)  autograd.Function 用于自定义处理数据的操作,
#       不具有参数, 且需要手动实现求导过程.
#   2)  nn.Module 定义的是 layers, 这些 layers 的参数由 nn.Module 自动进行管理.
#   3)  nn.functional 定义的也是 layers, 这些 layers 的参数由用户手动进行管理.
class MyLinear(nn.Module):

    def __init__(self):
        super(MyLinear, self).__init__()
        # 这些参数会由 nn.Module 自动管理
        self.weight = nn.Parameter(torch.randn(3, 4))
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, input):
        # nn.functional 的参数需要手动传入并手动进行管理
        return nn.functional.linear(input, self.weight, self.bias)


# nn.Module 中parameter的命名规范:
# 1) 对于 self.param_name = nn.Parameter(t.randn(3, 4)), 命名为 param_name;
# 2) 对于子 Module 中的 parameter, 会在名字前加上 Module 的名字,
#    如有 self.sub_module = SubModel(), 且 SubModel 中有个 parameter 的名字叫做 param_name,
#    那么该参数的命名就是 sub_module.param_name.

# 如何使用 nn.Module 和 nn.functional?
# nn.Module 中的大多数 layer 在 nn.functional 中都有与之相对应的函数.
# 二者在性能上没有太大差异, 具体的使用取决于个人的喜好, 但有下面两点建议:
# 1) 若有可学习参数, 则最好使用 nn.Module, nn.Module 可以自动管理可学习参数,
#    而 nn.functional 的参数需要手动进行管理.
# 3) 虽然 dropout 操作没有可学习操作, 但还是建议使用 nn.Dropout,
#    而不是 nn.functional.dropout, 因为 dropout 在训练和测试两个阶段的行为有所差别, 
#    使用 nn.Module 能够通过 .eval 和 .train 方法加以区分.
