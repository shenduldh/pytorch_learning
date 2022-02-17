from torch import optim
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.transforms as tf
import math
from PIL import Image
import matplotlib.pyplot as plt


###########################
## ⭐nn.Module layers⭐ ##
###########################

# ✅ 卷积层
# 举例: 利用 2d 卷积锐化图像
op_2tensor = tf.ToTensor()  # 用于将数据转化为 tensor 的操作
op_2image = tf.ToPILImage()  # 用于将数据转化为图像的操作
image_data = Image.open("assets/test.png")  # 读取图像数据
input = op_2tensor(image_data).unsqueeze(0)  # 将图像数据转化为 tensor, 并添加 batch 维度
# 构建卷积层
conv = nn.Conv2d(
    in_channels=1,  # 单个卷积核的通道数, 也是输入图像的通道数
    out_channels=1,  # 卷积核数量, 也是输出特征图的通道数
    kernel_size=(3, 3),  # 卷积核大小
    stride=1,  # 卷积核移动步长
    padding=0,  # 在输入矩阵的外围增加 0 圈的零填充
    bias=False,
)
# 手动设置卷积核的参数
print(conv.weight.data.size())
kernal = torch.ones(3, 3) / -9
kernal[1][1] = 1
conv.weight.data = kernal.view(1, 1, 3, 3)  # (out_channels, in_channels, kernel_size)
# 对图像进行卷积
output = conv(input)
# 可视化结果
image_data = op_2image(output.data.squeeze(0))  # 将 tensor 转化为图像数据
plt.imshow(image_data)
plt.show()

# ✅ 池化层
# 举例: 利用平均池化缩小图片
avg_pool = nn.AvgPool2d(2, 2)  # 构建池化层
output = avg_pool(input)  # 对图像进行池化
print(input.shape, output.shape)  # 缩小前后图像大小的对比
# 可视化结果
image_data = op_2image(output.data.squeeze(0))
plt.imshow(image_data)
plt.show()

# ✅ Linear layer
input = torch.randn(2, 3)
linear = nn.Linear(3, 4)
linear_out = linear(input)

# ✅ BatchNorm1d
bn = nn.BatchNorm1d(4)  # 4 表示输入的 1d 数据的长度
bn.weight.data = torch.ones(4) * 4  # 表示规范化后数据的标准差为 4
bn.bias.data = torch.zeros(4)  # 表示规范化后数据的均值为 0
bn_out = bn(linear_out)
# 方差是标准差的平方, 计算无偏方差分母会减1, 使用 unbiased=False 分母不减1
print(bn_out.mean(0))  # 接近 0
print(bn_out.var(0, unbiased=False))  # 接近 4^2=16

# ✅ Dropout
dropout = nn.Dropout(0.5)
output = dropout(bn_out)  # 每个元素以 0.5 的概率被舍弃
print(output)

# ✅ ReLU 激活函数
input = torch.randn(2, 3)
# ReLU 函数有个 inplace 参数, 如果设为 True,
# 它会把输出直接覆盖到输入中, 这样可以节省内存/显存.
# 之所以可以覆盖, 是因为在计算 ReLU 的反向传播时,
# 只需根据输出就可以推算出梯度, 不需要保存输入值.
# 但是只有少数的 autograd 操作支持 inplace (如 tensor.sigmoid_())
# 除非明确知道在做什么, 否则一般不要使用 inplace 操作
relu = nn.ReLU(inplace=True)
output = relu(input)
print(output)  # 小于 0 的元素都被截断为 0


#######################################
## ⭐nn.ModuleList, nn.Sequential⭐ ##
#######################################

# nn.ModuleList 和 nn.Sequential 都是 layer 容器,
# 利用 nn.ModuleList 和 nn.Sequential 可以简写前馈网络.

# ✅ nn.Sequential
# nn.Sequential 是一个顺序容器, 网络层将按照传入构造器的顺序依次被添加到计算图中执行
#   1) nn.Sequential 的第一种写法:
net1 = nn.Sequential()
net1.add_module("conv", nn.Conv2d(3, 3, 3))
net1.add_module("batch_norm", nn.BatchNorm2d(3))
net1.add_module("activator", nn.ReLU())
#   2) nn.Sequential 的第二种写法:
net2 = nn.Sequential(nn.Conv2d(3, 3, 3), nn.BatchNorm2d(3), nn.ReLU())
#   3) nn.Sequential 的第三种写法:
net3 = nn.Sequential(
    OrderedDict(
        [
            ("conv", nn.Conv2d(3, 3, 3)),
            ("batch_norm", nn.BatchNorm2d(3)),
            ("relu", nn.ReLU()),
        ]
    )
)

# ✅ nn.ModuleList 的写法和用法:
model_list = nn.ModuleList([nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)])
# nn.ModuleList 的执行需要手动进行
input = torch.randn(1, 3)
for model in model_list:
    input = model(input)
# nn.ModuleList 无法像 nn.Sequential 那样自动按序进行,
# 因此下面这行代码会报错:
# output = modelist(input)

# ✅ 注意事项
# 1) 不使用 Python 自带的 list 的原因:
#    nn.ModuleList 是 nn.Module 的子类,
#    当在 nn.Module 中使用它时, 就能自动识别为子 module.
# 2) 除 nn.ModuleList 之外还有 nn.ParameterList,
#    它是一个可以包含多个 parameter 的类 list 对象
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        # 由 list 包裹的 layers 无法被 MyModule 识别
        self.list = [nn.Linear(3, 4), nn.ReLU()]
        # 由 ModuleList 包裹的 layers 可以被 MyModule 识别
        self.module_list = nn.ModuleList([nn.Conv2d(3, 3, 3), nn.ReLU()])

    def forward(self):
        pass


##############
## ⭐RNN⭐ ##
##############

# pyTorch 实现了三种 RNN: RNN (vanilla RNN), LSTM 和 GRU,
# 以及相对应的三种 RNNCell.
# RNN 和 RNNCell 层的区别在于前者一次能够处理整个序列,
# 而后者一次只处理序列中一个时间点的数据.

# ✅ LSTM 示例
input = torch.randn(2, 3, 4)
lstm = nn.LSTM(4, 3, 1)  # 输入向量 4 维, LSTMCell 3 个, layer 1 层
# 初始状态: layer 1 层, 每批样本 3 个, LSTMCell 3 个
h0 = torch.randn(1, 3, 3)
c0 = torch.randn(1, 3, 3)
out, (hn, cn) = lstm(input, (h0, c0))

# ✅ LSTMCell 示例
input = torch.randn(2, 3, 4)
lstm = nn.LSTMCell(4, 3)  # 输入向量 4 维, LSTMCell 3 个
hx = torch.randn(3, 3)
cx = torch.randn(3, 3)
out = []
for i_ in input:
    hx, cx = lstm(i_, (hx, cx))
    out.append(hx)
out = torch.stack(out)


####################
## ⭐Embedding⭐ ##
####################

# 有 4 个词, 每个词用 5 维向量表示
embedding = nn.Embedding(4, 5)
# 可以用预训练好的词向量初始化 embedding
embedding.weight.data = torch.arange(0, 20).view(4, 5)
input = torch.arange(3, 0, -1).long()
output = embedding(input)


##################
## ⭐损失函数⭐ ##
##################

# batch_size=3, 计算对应每个类别的分数 (这里假设只有两个类别)
score = torch.randn(3, 2)
# 三个样本分别属于 1, 0, 1 类, label 必须是 LongTensor
label = torch.Tensor([1, 0, 1]).long()
# loss 与普通的 layer 无差异
criterion = nn.CrossEntropyLoss()
loss = criterion(score, label)


###################################
## ⭐优化器的使用和学习率的调控⭐ ##
###################################

# pyTorch 将深度学习中常用的优化方法全部封装在 torch.optim 中,
# 所有的优化方法都继承于基类 optim.Optimizer, 并实现了相应的优化步骤.

# 举例说明: 首先定义一个 LeNet 网络
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.classifier(x)
        return x


net = LeNet()  # 实例化网络
optimizer = optim.SGD(params=net.parameters(), lr=1)  # 定义优化器
optimizer.zero_grad()  # 梯度清零, 等价于 net.zero_grad()
input = torch.randn(1, 3, 32, 32)
output = net(input)
output.backward(output)  # 反向传播
optimizer.step()  # 执行优化

# 为不同子网络设置不同的学习率, 在 finetune 中经常用到.
# 如果对某个参数不指定学习率, 就使用最外层的默认学习率.
optimizer = optim.SGD(
    [
        {"params": net.features.parameters()},  # 学习率为 1e-5
        {"params": net.classifier.parameters(), "lr": 1e-2},
    ],
    lr=1e-5,
)

# 只为两个全连接层设置较大的学习率, 其余层的学习率较小
special_layers = nn.ModuleList([net.classifier[0], net.classifier[3]])
special_layers_params = list(map(id, special_layers.parameters()))
base_params = filter(lambda p: id(p) not in special_layers_params, net.parameters())
optimizer = torch.optim.SGD(
    [{"params": base_params}, {"params": special_layers.parameters(), "lr": 0.01}],
    lr=0.001,
)

# 对于调整学习率, 主要有两种做法:
# 一种是修改 optimizer.param_groups 中对应的学习率,
# 另一种是更为推荐的做法, 即新建优化器.
# 但后者对于使用动量的优化器 (如Adam), 会丢失动量等状态信息,
# 可能使得损失函数的收敛出现震荡等情况.
#   方法 ①: 通过手动 decay 来调整学习率, 会保存动量
for param_group in optimizer.param_groups:
    param_group["lr"] *= 0.1  # 学习率为之前的0.1倍
#   方法 ②: 通过新建 optimizer 来调整学习率, 不会保存动量
old_lr = 0.1
new_optimizer = optim.SGD(
    [
        {"params": net.features.parameters()},
        {"params": net.classifier.parameters(), "lr": old_lr * 0.1},
    ],
    lr=1e-5,
)


########################
## ⭐参数初始化策略⭐ ##
########################

# PyTorch 的 nn.init 模块专门用于初始化.
# 如果某种初始化策略 nn.init 不提供, 则可以自己手动进行初始化.
# 利用 nn.init 初始化, 等价于下面手动初始化的方式:
linear = nn.Linear(3, 4)
nn.init.xavier_normal_(linear.weight)
# 手动进行初始化, 也是采用 xavier_normal 的方式进行初始化
linear.weight.data.normal_(0, math.sqrt(2) / math.sqrt(7.0))


############################
## ⭐nn.Module 深入理解⭐ ##
############################

# ✅ nn.Module 基类的构造函数
def __init__(self):
    self._parameters = OrderedDict()
    self._modules = OrderedDict()
    self._buffers = OrderedDict()
    self._backward_hooks = OrderedDict()
    self._forward_hooks = OrderedDict()
    self.training = True
    # 每个属性的解释如下:
    # 1) _parameters: 保存用户设置的 parameter.
    #       比如 self.param = nn.Parameter(t.randn(3, 3)) 会被检测到,
    #       从而在字典中加入一个 key 为 'param', value 为对应 parameter 的 item.
    # 2) _modules: 通过 self.submodel = nn.Linear(3, 4) 指定的子 module 会保存于此.
    # 3) _buffers: 缓存.
    #       比如 batchnorm 使用 momentum 机制,
    #       每次前向传播需要用到上一次前向传播的结果.
    # 4) _backward_hooks 与 _forward_hooks: 钩子, 用来提取中间变量.
    # 5) training: 通过判断 training 值来决定前向传播时 BatchNorm 与 Dropout 层的策略
    # 6) _parameters, _modules 和 _buffers 这三个字典中的键值,
    #       都可以通过 self.key 方式获得.


# ✅ training 属性用于开启和关闭 batchnorm、dropout 和 instancenorm 等 layers
# 开启和关闭的方法:
#   方法 ①: 直接设置 module 的 training 属性
#   方法 ②: 调用 module.train() 和 module.eval() 函数


# ✅ nn.Module 的钩子函数
# 钩子不应修改输入和输出, 且在使用后应及时删除,
# 避免每次都运行钩子增加运行负载.
net = nn.Sequential()
net.add_module("linear", nn.Linear(3, 3))
net.add_module("activator", nn.ReLU())
features = torch.Tensor()


# 定义前向传播时的钩子函数
def forward_hook(module, input, output):
    # 把这层的输出拷贝到 features 中
    features.copy_(output.data)


# 定义反向传播时的钩子函数
def backward_hook(module, grad_input, grad_output):
    # 打印输出的梯度
    print(grad_output)


# 注册钩子函数
forward_hook_handler = net.linear.register_forward_hook(forward_hook)
backward_hook_handler = net.linear.register_backward_hook(backward_hook)
_ = net(torch.randn(1, 3))
# 用完 hook 后删除
forward_hook_handler.remove()
backward_hook_handler.remove()


# ✅ nn.Module 实例对象和普通对象在存取属性上的异同
# python 的默认行为:
obj = {"name": "Jack", "age": 20}
result = obj.name
# 当获取对象的属性时, 先调用 getattr(obj, 'name'),
# 如果没找到, 则会调用 obj.__getattr__('name')
obj.gender = "man"
# 当设定对象的属性时, 会调用 setattr(obj, 'name', value),
# 如果对象实现了 __setattr__ 方法, setattr 会直接调用 obj.__setattr__('name', value')

# nn.Module 对于 python 默认行为的修改:
# nn.Module 实现了自定义的 __setattr_ _函数,
# 当执行 module.name = value 时,
# 会在 __setattr__ 中判断 value 是否为 Parameter 或 nn.Module,
# 如果是则将这些对象加到 _parameters 或 _modules 两个字典中,
# 而如果是其它类型的对象, 如 Variable, List, Dict 等,
# 则调用默认操作, 将它们保存在 __dict__ 中.

# 因 _modules 和 _parameters 中的 item 未保存在 __dict__ 中,
# 所以默认的 getattr 方法无法获取它, 因而 nn.Module 实现了自定义的 __getattr__ 方法,
# 如果默认的 getattr 无法处理, 就调用自定义的 __getattr__ 方法,
# 尝试从 _modules, _parameters 和 _buffers 这三个字典中进行获取.
