import torch
from torch.utils import data
import torchvision
from torchvision import transforms
import PIL
import os
import random

###########################
## Dataset 和 DataLoader ##
###########################


# ✅ 使用自定义数据集来加载数据
# 实现自定义的数据集需要继承 Dataset 类, 并实现两个 Python 魔法方法:
#   1) __getitem__: 返回一条数据, 或一个样本. obj[index] 等价于 obj.__getitem__(index).
#   2) __len__: 返回样本的数量. len(obj) 等价于 obj.__len__().
# 假定数据集存放的方式为:
#   data/dogcat/
#   |-- cat.12484.jpg
#   |-- cat.12485.jpg
#   |-- cat.12486.jpg
#   |-- cat.12487.jpg
#   |-- dog.12496.jpg
#   |-- dog.12497.jpg
#   |-- dog.12498.jpg
#   `-- dog.12499.jpg
class DogCat(data.Dataset):

    def __init__(self, root, transforms=None):
        imgs = os.listdir(root)
        # 仅保存数据的路径, 当调用 __getitem__ 时才会真正读图片
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 0 if 'dog' in img_path.split('/')[-1] else 1
        data = PIL.Image.open(img_path)
        if self.transforms:  # 对数据进行规范化
            data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


# PyTorch 提供了 torchvision 视觉工具包,
# 其中 transforms 模块提供了对 PIL 的 Image 对象和 Tensor 对象的常用操作:
# ① 对 PIL 的 Image 对象的操作包括:
#  1) Scale：调整图片尺寸，长宽比保持不变
#  2) CenterCrop、RandomCrop、RandomResizedCrop： 裁剪图片
#  3) Pad：填充
#  4) ToTensor：将PIL Image对象转成Tensor，会自动将[0, 255]归一化至[0, 1]
# ② 对Tensor的操作包括:
#  1) Normalize：标准化，即减均值，除以标准差
#  2) ToPILImage：将Tensor转为PIL Image对象
# 如果要同时进行多个操作, 可通过 Compose 函数将这些操作拼接起来使用.
# 除了上述操作之外, transforms 还可以通过 Lambda 封装自定义的转换策略.
# 例如想对 Image 进行随机旋转, 则可写成 T.Lambda(lambda img: img.rotate(random()*360)).
transforms = transforms.Compose([
    transforms.Resize(224),  # 缩放图片, 最短边为 224 像素
    transforms.CenterCrop(224),  # 从图片中间切出 224*224 的图片
    transforms.ToTensor(),  # 将图片转成 Tensor
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至 [-1, 1]
])
dataset = DogCat('./data/dogcat/', transforms=transforms)
img, label = dataset[0]


# ✅ 使用 Pyotch 预先定义好的数据集来加载数据
def load_fashion_mnist(batch_size, resize=None):
    # 定义数据集的预处理器
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    # 加载数据集
    mnist_train = torchvision.datasets.FashionMNIST(root="./data",
                                                    train=True,
                                                    transform=trans,
                                                    download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data",
                                                   train=False,
                                                   transform=trans,
                                                   download=True)

    # 用 DataLoader 封装数据集, 使其成为可以返回 batch_samples 的加载器
    return (
        data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
        data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4),
    )


train_loader, test_loader = load_fashion_mnist(32, resize=(64, 64))
# 可通过 iter(DataLoader) 将 DataLoader 转化为生成器来加载数据
X_batch, Y_batch = next(iter(train_loader))

# ✅ 使用 ImageFolder 来加载数据
# ImageFolder(root, transform=None, target_transform=None, loader=default_loader)
#   1) root: 在 root 指定的路径下寻找图片
#   2) transform: 对 Image 进行的转换操作
#   3) target_transform: 对 label 的转换操作
#   4) loader: 给定路径后如何读取图片, 默认读取为 RGB 格式的 PIL Image 对象
# 假定数据集存放的方式为:
#   data/dogcat/
#   |-- cat
#   |   |-- cat.12484.jpg
#   |   |-- cat.12485.jpg
#   |   |-- cat.12486.jpg
#   |   `-- cat.12487.jpg
#   `-- dog
#       |-- dog.12496.jpg
#       |-- dog.12497.jpg
#       |-- dog.12498.jpg
#       `-- dog.12499.jpg
dataset = torchvision.datasets.ImageFolder(root='data/dogcat/')
# ImageFolder 假设所有的数据按文件夹来保存, 每个文件夹下存储同一种类别的图片.
# 如果 label 没有指定转换操作, 则默认用文件夹的索引来作为 label,
# 这里没有指定 target_transform, 因此 cat 文件夹下的数据的 label 为 0, dog 的为 1.
# 而且也没有指定任何的 transform, 因此返回的数据还是 PIL Image 对象.
# 第一维是第几张图, 第二维为 1 返回 label, 为 0 返回图片数据:
dataset[0][1]  # 0
dataset[0][0]  # cat.12484.jpg 的 PIL Image 对象
dataset[4][1]  # 1
dataset[4][0]  # dog.12496.jpg 的 PIL Image 对象

# ✅ 使用 DataLoader 来加载数据
# DataLoader 用于封装 Dataset, 使其可以批量返回数据.
# DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0,
#            collate_fn=default_collate, pin_memory=False, drop_last=False)
#  1) dataset：指定加载的数据集
#  2) batch_size：batch 的大小
#  3) shuffle: 是否将数据打乱
#  4) sampler: 样本抽样
#  5) num_workers: 使用多进程加载的进程数, 0 代表不使用多进程
#  6) collate_fn: 如何将多个样本数据拼接成一个 batch, 一般使用默认的拼接方式即可
#  7) pin_memory: 是否将数据保存在 pin memory 区, pin memory 中的数据转到 GPU 会快一些
#  8) drop_last: dataset 中的数据个数可能不是 batch_size 的整数倍,
#                drop_last 为 True 会将多出来不足一个 batch 的数据丢弃
dataloader = data.DataLoader(dataset, batch_size=3, shuffle=True)
# dataloader 是一个可迭代对象, 可用 for 循环来获取数据:
for batch_data, batch_label in dataloader:
    pass
# 也可以将 dataloader 转换成生成器来使用:
dataiter = iter(dataloader)
batch_data, batch_label = next(dataiter)

# DataLoader 封装了 Python 标准库 multiprocessing, 使其能够实现多进程加速.
# 但这样会出现一个问题, 在使用多进程时, 如果主程序异常终止, 相应的数据加载进程可能无法正常退出.
# 这时就需要手动强行杀掉进程, 建议使用如下命令: ps x | grep <cmdline> | awk '{print $1}' | xargs kill
#  1) ps x: 获取当前用户的所有进程
#  2) grep <cmdline>: 找到已停止的 PyTorch 程序的进程.
#                     例如通过 python train.py 启动的, 那就写 grep 'python train.py'
#  3) awk '{print $1}': 获取进程的pid
#  4) xargs kill: 杀掉进程, 根据需要可能要写成 xargs kill -9 强制杀掉进程
# 在执行这句命令之前, 建议先打印确认一下是否会误杀其它进程: ps x | grep <cmdline> | ps x

# PyTorch 提供了 sampler 模块, 用来对数据进行采样
#  1) 随机采样器 RandomSampler: 当 DataLoader 的 shuffle 参数为 True 时,
#                              会自动调用这个采样器实现打乱数据.
#  2) 顺序采样器 SequentialSampler: 按顺序一个一个进行采样, 是 DataLoader 的默认采样器.
#  3) 权重采样器 WeightedRandomSampler: 根据每个样本的权重选取数据,
#                                      在样本比例不均衡的问题中, 可用它来进行重采样.

# 构建 WeightedRandomSampler 时需提供三个参数:
#  1) weights: 每个样本的权重
#  2) num_samples: 共选取的样本总数
#  3) (可选参数) replacement: 用于指定是否可以重复选取某一个样本,
#                            默认为 True, 即允许在一个 epoch 中重复采样某一个数据.
#                            如果设为 False, 则当某一类的样本被全部选取完,
#                            但其样本数目仍未达到 num_samples 时,
#                            sampler 将不会再从该类中选择数据, 这就可能导致weights参数失效
dataset = DogCat('data/dogcat/', transforms=transforms)
# 设置权重, 使得狗的图片被取出的概率是猫的概率的两倍
weights = [2 if label == 1 else 1 for _, label in dataset]
sampler = data.sampler.WeightedRandomSampler(weights,
                                             num_samples=9,
                                             replacement=True)
dataloader = data.DataLoader(dataset, batch_size=3, sampler=sampler)

# 需要注意的是, 如果指定了 sampler, shuffle 将不再生效,
# 并且 num_samples 会覆盖 dataset 的实际大小,
# 即一个 epoch 返回的图片总数取决于 num_samples.


# ✅ 损坏数据的处理方式:
#  1) (推荐) 预先清洗数据, 将出错的样本剔除.
#  2) (次次推荐) 在 Dataset 的 __getitem__ 函数中对于出错的样本返回 None 对象,
#     然后在 Dataloader 中实现自定义的 collate_fn, 将空对象过滤掉.
#     但要注意, 在这种情况下 dataloader 返回的 batch 数目会少于 batch_size.
class NewDogCat(DogCat):  # 继承前面实现的DogCat数据集

    def __getitem__(self, index):
        try:
            return super(NewDogCat, self).__getitem__(index)
        except:
            return None, None


def my_collate_fn(batch):
    # 过滤为 None 的数据
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0: return torch.Tensor()
    # 用默认方式拼接过滤后的 batch 数据
    return data.DataLoader.default_collate(batch)


dataset = NewDogCat('./data/dogcat/', transforms=transforms)
dataloader = data.DataLoader(dataset, batch_size=2, collate_fn=my_collate_fn)


# 3) (次推荐) 随机取一个样本来代替异常样本, 可以保证每个 batch 的数目还是 batch_size.
class NewDogCat(DogCat):

    def __getitem__(self, index):
        try:
            return super(NewDogCat, self).__getitem__(index)
        except:
            new_index = random.randint(0, len(self) - 1)
            return self[new_index]


#################
## torchvision ##
#################

# ✅ torchvision 是 Pytorch 提供的视觉工具包, 主要包含三部分:
#  1) models: 提供各种经典网络的网络结构以及预训练好的模型, 包括 AlexNet, VGG, ResNet, Inception 等.
resnet34 = torchvision.models.squeezenet1_1(pretrained=True, num_classes=1000)
# 修改最后的全连接层为 10 分类问题 (默认是 ImageNet 上的 1000 分类)
resnet34.fc = torch.nn.Linear(512, 10)
#  2) datasets: 提供常用的数据集, 包括 MNIST, CIFAR10/100, ImageNet, COCO 等.
dataset = torchvision.datasets.MNIST('data/', download=True, train=False)
#  3) transforms: 提供常用的数据预处理操作, 主要包括对 Tensor 和 PIL Image 对象的操作.
#                 操作分为两步: ① 构建转换操作 transform = transforms.Normalize(mean=x, std=y)
#                              ② 执行转换操作 output = transform(input)
to_pilimage = torchvision.transforms.ToPILImage()
to_pilimage(torch.randn(3, 64, 64))

# ✅ torchvision 还提供了两个工具函数:
#   ① make_grid: 将多张图片拼接成一个网格中
#   ② save_img: 保存图片
dataiter = iter(dataloader)
# 拼成 4*4 的网格图片
img = torchvision.utils.make_grid(next(dataiter)[0], 4)
torchvision.utils.save_image(img, 'img.png')

###########################
## 使用 Visdom 进行可视化 ##
###########################

# Visdom 是 Facebook 专门为 PyTorch 开发的一款可视化工具.
# Visdom中有两个重要概念:
#  1) env: 环境. 不同环境的可视化结果相互隔离, 互不影响.
#          在使用时如果不指定env, 默认使用 main.
#  2) pane: 窗格. 窗格可用于可视化图像, 数值或打印文本等, 其可以拖动、缩放、保存和关闭.
#           一个程序中可使用同一个 env 中的不同 pane, 每个 pane 可视化或记录不同的信息.

# 安装 Visdom: pip install visdom
# 启动 Visdom 服务: python -m visdom.server
#                  nohup python -m visdom.server & (后台运行)
# Visdom 服务是一个 web 服务, 默认绑定 8097 端口.
# Visdom 的使用有两点需要注意的地方:
#  ① 需手动保存 env (可在 web 界面点击 save 按钮或在程序中调用 save 方法),
#    否则 Visdom 服务重启后, env 等信息将会丢失.
#  ② 客户端与服务器之间采用 tornado 异步框架进行非阻塞交互,
#    可视化操作不会阻塞当前程序, 网络异常也不会导致程序退出.

import visdom
# 新建一个 Visdom 客户端, 并指定使用名为 u'test1' 的 env
vis = visdom.Visdom(env=u'test1', host='localhost', port="3000")

# 常见的画图函数包括:
#  1) line: 绘制折线图
#  2) image: 可视化图片
#  3) text: 添加文字信息, 支持 html 格式
#  4) histgram: 可视化分布
#  5) scatter: 绘制散点图
#  6) bar: 绘制柱状图
#  7) pie: 绘制饼状图
# Visdom 同时支持 PyTorch 的 tensor 和 Numpy 的 ndarray 两种数据结构,
# 但不支持 Python 的 int, float 等类型, 因此每次传入时都需先将数据转成 ndarray 或 tensor.
# 上述画图函数的参数一般不同, 但有两个参数是绝大多数操作都具备的:
#  1) win: 用于指定 pane 的名字, 如果不指定, 则由 Visdom 自动分配一个新的 pane.
#          如果两次操作指定的 win 名字一样, 新的操作将覆盖当前 pane 的内容.
#          如果不想要覆盖之前的内容, 则可以通过设置 update 参数来实现.
#  2) opts: 接收一个字典, 用于设置pane的显示格式, 常见的选项包括 title, xlabel, ylabel, width 等.

# ✅ 可视化折线 y = x:
for i in range(0, 10):

    x = torch.Tensor([i])
    y = x
    vis.line(X=x,
             Y=y,
             win='line',
             name="I'm a old trace.",
             update='append' if i > 0 else None)

# 在同一个 pane 上新增一条折线:
x = torch.arange(0, 9, 0.1)
y = (x**2) / 9
vis.line(X=x, Y=y, win='line', name="I'm a new trace.", update='new')

# ✅ 可视化图片:
#  1) image 接收一个二维或三维向量, 前者是黑白图像, 后者是彩色图像.
#  2) images 接收一个四维向量, 第二维可以是 1 或 3, 分别代表黑白和彩色图像.
# 随机可视化一张黑白图片
vis.image(torch.randn(64, 64).numpy())
# 随机可视化一张彩色图片
vis.image(torch.randn(3, 64, 64).numpy(), win='img1')
# 可视化 36 张随机的彩色图片, 每一行显示 6 张图片
vis.images(torch.randn(36, 3, 64, 64).numpy(), nrow=6, win='img2')

# ✅ 可视化文字:
vis.text(u'<h1>Hello Visdom</h1>', win='text')

############################
## 使用 cuda 进行 GPU 加速 ##
############################

# PyTorch 的 Tensor 和 nn.Module 分为 CPU 和 GPU 两个版本,
# 它们都带有一个.cuda方法, 调用此方法即可将其转为对应的GPU对象.
# 注意, tensor.cuda 会返回一个新对象, 这个新对象的数据已转移至GPU,
# 而之前的 tensor 还在原来的设备上 (即 CPU).
# 而 module.cuda 则会将所有的数据都迁移至GPU, 并返回自己.
# 因此 module = module.cuda() 和 module.cuda() 所起的作用一致.
# nn.Module 在 GPU 与 CPU 之间的转换, 本质上还是利用了 Tensor 在 GPU 和 CPU 之间的转换.
# nn.Module 的 cuda 方法是将 nn.Module 下的所有 parameter 都转移至 GPU.

tensor = torch.Tensor(3, 4)
# 返回一个新的保存在第 1 块 GPU 上的 tensor, 但原来的 tensor 并没有改变
tensor.cuda(0)
tensor.is_cuda  # False
# 不指定所使用的 GPU 设备, 将默认使用第 1 块 GPU
tensor = tensor.cuda()
tensor.is_cuda  # True
# 将 module 迁移至 GPU
module = torch.nn.Linear(3, 4)
module.cuda(device=1)
module.weight.is_cuda  # True

# ✅ 关于使用 GPU 的一些建议:
#  1) GPU 对于很小的运算量来说, 并不能体现出它的优势
#  2) 数据在 CPU 和 GPU 之间的传递会比较耗, 应当尽量避免
#  3) 在进行低精度的计算时, 可以考虑 HalfTensor, 它相比于 FloatTensor 能节省一半的显存

# ✅ 大部分的损失函数也都属于 nn.Moudle,
# 但一般在使用 GPU 时, 不需要调用它的 .cuda 方法,
# 这在大多数情况下不会报错, 因为损失函数本身没有可学习的参数.
# 但在某些情况下会出现问题, 比如带带权重的交叉熵损失函数:
criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, 3]))
input = torch.randn(4, 2).cuda()
target = torch.Tensor([1, 0, 0, 1]).long().cuda()
# 下面这行会报错, 因为 criterion 的  weight 未被转移至 GPU
# loss = criterion(input, target)
# 对 criterion 调用 .cuda 方法后就不会报错了
criterion.cuda()
loss = criterion(input, target)

# ✅ 切换 GPU 的一些方法:
#  1) 可以使用 'with torch.cuda.device(1):' 来指定使用第二块 GPU
with torch.cuda.device(1):  # 指定默认使用 GPU 1
    # 在 with 包裹下创建的 GPU 数据都会在第二块 GPU 上
    tensor = torch.cuda.FloatTensor(2, 3)
#  2) 可以调用 torch.cuda.set_device(1) 指定使用第三块 GPU
torch.cuda.set_device(2)  # 后续创建的 GPU 数据都会在第三块 GPU 上
tensor = torch.cuda.FloatTensor(2, 3)
#  3) 可以使用 torch.set_default_tensor_type 使程序默认使用 GPU
torch.set_default_tensor_type('torch.cuda.FloatTensor')
#  4) 通过设置环境变量 CUDA_VISIBLE_DEVICES.
#     例如当export CUDA_VISIBLE_DEVICE=1 (下标是从0开始，1代表第二块GPU),
#     表示只使用第二块物理 GPU, 但在程序中这块 GPU 会被看成是第一块逻辑 GPU,
#     因此此时调用 tensor.cuda() 会将 Tensor 转移至第二块物理 GPU.
#     CUDA_VISIBLE_DEVICES 可以指定多个 GPU, 如 export CUDA_VISIBLE_DEVICES=0,2,3,
#     那么第一, 三, 四块物理 GPU 会被映射成第一, 二, 三块逻辑 GPU,
#     tensor.cuda(1) 会将 Tensor 转移到第三块物理 GPU 上.
#     设置 CUDA_VISIBLE_DEVICES 有以下方法:
#      ① 在命令行中 CUDA_VISIBLE_DEVICES=0,1 python main.py
#      ② 在程序中 import os;os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#      ③ 使用 Jupyter notebook, 可以使用 %env CUDA_VISIBLE_DEVICES=1,2 来设置环境变量

# ✅ 从 0.4 版本开始, Pytorch 新增了 tensor.to(device) 方法,
# 这个方法统一了 CPU 和 GPU 之间的切换方式 (原本 CPU->GPU 是调用 .cuda 方法,
# 而 GPU->CPU 是调用 .cpu 方法, 现在都是 .to 方法), 便于 CPU 和 GPU 之间的兼容.

# ✅ 快捷进行单机多卡并行: 直接调用 new_module = nn.DataParallel(module, device_ids)
#                     会默认把模型分布到单机的所有卡上
# 多卡并行的机制如下:
#  1) 将模型复制到每一张卡上;
#  2) 将形状为 (N,C,H,W) 的输入均等分为 n 份 (假设有n张卡), 每一份形状是 (N/n,C,H,W),
#     然后在每张卡上分别进行前向传播, 反向传播, 梯度求平均. 因此要求 batch_size 大于等于卡的个数 (N>=n).
# 如果想要获取原始的单卡模型, 需要通过 new_module.module 来访问.

###############
## 数据持久化 ##
###############

# 在 PyTorch 中, 以下对象都可以持久化到硬盘, 并能通过相应的方法加载到内存中:
#  1) Tensor
#  2) Variable
#  3) nn.Module
#  4) Optimizer
# 本质上上述这些信息最终都是保存成Tensor.
# Tensor 的保存和加载十分的简单, 使用 torch.save 和 torch.load 即可完成相应的功能.
# 在 save/load 时可指定使用的 pickle 模块, 在 load 时还可将 tensor 指定映射到 CPU 或 GPU 上.
# 对于 Module 和 Optimizer 对象, 这里建议保存对应的 state_dict,
# 而不是直接保存整个 Module/Optimizer 对象.

# ✅ Tensor 的保存与加载
a = torch.Tensor(3, 4)
if torch.cuda.is_available():
    a = a.cuda(1)
    torch.save(a, 'a.pth')
    # 加载为 b, 存储于 GPU1 上(因为保存时 tensor 就在 GPU1 上)
    b = torch.load('a.pth')
    # 加载为 c, 存储于 CPU
    c = torch.load('a.pth', map_location=lambda storage, loc: storage)
    # 加载为 d, 存储于 GPU0 上
    d = torch.load('a.pth', map_location={'cuda:1': 'cuda:0'})

# ✅ nn.Module 的保存与加载
model = torch.nn.Linear(1, 2)
torch.save(model.state_dict(), 'linear.pth')
model.load_state_dict(torch.load('linear.pth'))

# ✅ Optimizer 的保存与加载
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
torch.save(optimizer.state_dict(), 'optimizer.pth')
optimizer.load_state_dict(torch.load('optimizer.pth'))

# ✅ 同时保存 nn.Module 和 Optimizer 到一个文件
all_data = dict(optimizer=optimizer.state_dict(),
                model=model.state_dict(),
                info=u'模型和优化器的所有参数')
torch.save(all_data, 'all.pth')
all_data = torch.load('all.pth')
all_data.keys()  # dict_keys(['optimizer', 'model', 'info'])
