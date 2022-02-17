import torch
import numpy as np

# 查看 pytorch 的版本
torch.__version__

##############
## ⭐张量⭐ ##
##############

# ✅ 与 numpy 相互转换
# pytorch 与 numpy 兼容, 两者的基本数据单元（tensor 与 ndarray）和计算方法几乎相同
# 张量可以看成是被封装后的数组, 即在 python 数组的基础上增添了与运算相关的属性和方法
# tensor 和 ndarray 共享内存, 因此它们之间的转换很快, 几乎不会消耗什么资源
# 但当 numpy 的数据类型和 tensor 的类型不同时, 数据会被复制, 不会共享内存
numpy_data = np.arange(6)
torch_data = torch.from_numpy(numpy_data)  # ndarray -> tensor
# 不论输入的类型是什么, torch.tensor 都会进行数据拷贝, 不会共享内存
torch_data = torch.tensor(numpy_data)  # ndarray -> tensor
numpy_data = torch_data.numpy()  # tensor -> ndarray

# ✅ 张量的创建
torch.arange(6, dtype=torch.float32)  # 产生某一长度范围的 tensor
torch.zeros((2, 3, 4))  # 形状为 (2, 3, 4) 的全 0 张量
torch.ones((1, 2, 3))  # 全是 1 的 tensor
torch.eye(2, 2)  # 对角线为 1, 其余为 0 的 tensor
torch.randperm(5)  # 长度为 5 且随机排列的 tensor
torch.randn(1, 1)  # 由标准分布采样得到的 tensor
torch.rand(size=(2, 3))  # 由 [0,1] 均匀分布采样得到的张量
torch.normal(mean=0, std=1, size=(2, 3))  # 由指定的正态分布采样得到的 tensor
torch.tensor([[1, 2, 3], [4, 5, 6]])  # 将 python 数组转化成 tensor
torch.zeros_like(torch.tensor([[1, 2, 3]]))  # 产生与输入 tensor 形状一致的全 0 张量
torch.rand_like(torch.tensor([[1, 2, 3]]))  # 产生与输入 tensor 形状一致的随机张量
torch.cat((torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]])), dim=0)  # 拼接
torch.tensor([[1, 2, 3]]).clone()  # 克隆, 产生新的 tensor, 与原来的 tensor 不共享内存
torch.linspace(0, 10, steps=5)  # 从 0 到 10 的范围内均匀取出 5 个数作为 tensor

# ✅ 张量的属性
tensor = torch.ones((4, 5, 6))
tensor.shape, tensor.size()  # 获取 tensor 的形状
tensor.size()[1], tensor.size(1)  # 获取 tensor 在某一维的长度
tensor.numel, len(tensor)  # 获取 tensor 的元素总数

# ✅ 张量的数据类型
"""
数据类型 (dtype)                            |CPU tensor         |GPU tensor
32 位浮点型 (torch.float32 / torch.float)	|torch.FloatTensor	|torch.cuda.FloatTensor
64 位浮点型 (torch.float64 / torch.double)  |torch.DoubleTensor	|torch.cuda.DoubleTensor
16 位浮点型 (torch.float16 / torch.half)	|torch.HalfTensor	|torch.cuda.HalfTensor
8 位无符号整型 (torch.uint8)	             |torch.ByteTensor	 |torch.cuda.ByteTensor
8 位有符号整型 (torch.int8)                  |torch.CharTensor	 |torch.cuda.CharTensor
16 位有符号整型 (torch.int16 / torch.short)	 |torch.ShortTensor	 |torch.cuda.ShortTensor
32 位有符号整型 (torch.int32 / torch.int)	 |torch.IntTensor	 |torch.cuda.IntTensor
64 位有符号整型 (torch.int64 / torch.long)	 |torch.LongTensor	 |torch.cuda.LongTensor
"""
torch.set_default_tensor_type('torch.DoubleTensor')  # 设置默认的 tensor 数据类型
tensor.dtype  # 查看 tensor 的数据类型
tensor.new()  # 新建一个和 tensor 数据类型一致的 tensor
# 数据类型的转换
tensor.float()
tensor.type(torch.float32)
tensor.type_as(tensor)
tensor.cpu()
tensor.gpu()
tensor.to(
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

# ✅ 张量的索引操作
tensor[1:3]  # 取 1~2 行
tensor[1, 2] = 9  # 给第 1 行第 2 列的元素赋值
tensor[0:2, :] = 12  # 给第 0~1 行所有列的元素赋值
tensor[[0, 1, 2]]  # 根据 [0, 1, 2] 中指定的索引从 tensor 中取出相应的元素
tensor[[0, 1], [2], [3]]  # 取 tensor[0, 2, 3] 和 tensor[1, 2, 3]

# ✅ 张量的常用操作
torch.tensor(1).item(), float(
    torch.tensor(1))  # 将标量 tensor 转化为 python 的 number
tensor.reshape(2, 3)  # 改变 tensor 的形状
tensor.view(-1, 3)  # 改变 tensor 的形状, 必须保证改变前后的元素个数一致, -1表示自适应
tensor.resize_(3, 3)  # 改变 tensor 的形状, 不必保证改变前后的元素个数一致
tensor.expand(3, 3)  # 拓展 tensor 的形状, 返回新的 tensor
tensor.expand_as(torch.ones(3, 3))  # 拓展 tensor 的形状与输入 tensor 的一样, 返回新的 tensor
tensor.unsqueeze(dim=1), tensor[:, None]  # 在第 1 维上扩展一个长度为 1 的维度
tensor.squeeze(0)  # 压缩第 0 维, 前提是该维的长度等于 1
tensor.squeeze()  # 压缩所有长度为 1 的维度
# gather 表示根据 index 以 dim 指定的方向从 tensor 中取值
# 此处是取 tensor[0, [0, 1]] 和 tensor[1, [2, 3]]
tensor.gather(dim=1, index=torch.LongTensor([[0, 1], [2, 3]]))
# scatter_ 表示将 src 中的数据根据 index 以 dim 指定的方向填进 tensor
tensor.scatter_(dim=1,
                index=torch.LongTensor([[0, 1], [2, 3]]),
                src=torch.randn(2, 2))
# 从 tensor 中按照 dim 指定的维度和 index 指定的索引选取数据
tensor.index_select(dim=0, index=torch.LongTensor([1, 2, 3]))
# 从 tensor 中选取由 mask 指定的数据 (mask 的形状应和 tensor 一致)
tensor.masked_select(mask=(tensor > 0))  # 相当于 tensor[tensor > 0]
# 选取 tensor 中的非零数据
tensor.nonzero()

# ✅ 按元素运算
"""
abs/sqrt/div/exp/fmod/log/pow 绝对值/平方根/除法/指数/求余/求幂
cos/sin/asin/atan2/cosh       三角函数
ceil/round/floor/trunc	      上取整/四舍五入/下取整/只保留整数部分
clamp(input, min, max)	      截断超过 min 和 max 的部分
sigmod/tanh                   激活函数
"""
tensor + tensor
tensor - tensor
tensor * tensor
tensor / tensor
tensor**tensor
tensor == tensor
torch.exp(tensor)
torch.sin(tensor)
torch.nn.functional.relu(tensor)
torch.nn.functional.sigmoid(tensor)
torch.nn.functional.tanh(tensor)
torch.nn.functional.softplus(tensor)
torch.nn.functional.softmax(tensor)

# ✅ 运算的四种方式
torch.add(tensor, tensor)  # 第一种
tensor.add(tensor)  # 第二种
result = torch.tensor(size=(2, 3))  # 第三种
torch.add(tensor, out=result)
tensor.add_(tensor)  # 第四种
# 尾巴带下划线 _ 的函数会修改 tensor 自身, 称为 inplace 操作.
# 比如 x.add_(y) 和 x.t_() 会改变 x,
# 而 x.add(y) 和 x.t() 会返回新的 tensor, x 不变

# ✅ 归并运算
"""
mean/sum/median/mode 均值/和/中位数/众数
norm/dist	         范数/距离
std/var	             标准差/方差
cumsum/cumprod	     累加/累乘

假设输入的形状是 (m, n, k):
    如果指定 dim=0, 输出的形状就是 (1, n, k) 或者 (n, k)
    如果指定 dim=1, 输出的形状就是 (m, 1, k) 或者 (m, k)
    如果指定 dim=2, 输出的形状就是 (m, n, 1) 或者 (m, n)
size 中是否有 "1", 取决于参数 keepdim, keepdim=True 会保留维度 1.
"""
torch.mean(tensor)
tensor.sum(axis=1, keepdims=True)  # keepdims 表示保持轴数
tensor.cumsum(axis=0)  # 按某个轴累积求和
torch.max(tensor, dim=1)  # 返回元组 (每行的最大值, 每行最大值的索引)

# ✅ 比较运算
"""
gt/lt/ge/le/eq/ne 大于/小于/大于等于/小于等于/等于/不等
topk	          最大的 k 个数
sort	          排序
max/min	          比较两个 tensor 最大/最小值
"""

# ✅ 线性代数相关的运算
"""
trace	    对角线元素之和 (矩阵的迹)
diag	    对角线元素
triu/tril	矩阵的上三角/下三角，可指定偏移量
mm/bmm	    矩阵乘法，batch的矩阵乘法
addmm/addbmm/addmv/addr/badbmm	矩阵运算
t	        转置
dot/cross	内积/外积
inverse	    求逆矩阵
svd	        奇异值分解
"""
tensor.T, tensor.t()  # 矩阵转置
# 矩阵转置后会导致存储空间不连续, 可通过 .contiguous() 使其连续.
# 该方法会使数据复制一份, 不再与原来的数据共享内存.
torch.dot(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]))  # 向量内积
torch.mul(tensor, tensor)  # 形状相同的矩阵按位相乘
torch.mm(tensor, tensor.reshape(3, 2))  # 矩阵与矩阵相乘
torch.mv(tensor, torch.tensor([1, 2, 3], dtype=torch.float32))  # 矩阵与向量相乘
torch.norm(tensor)  # L2 范数: 元素平方和的平方根
torch.abs(tensor).sum()  # L1 范数: 元素的绝对值之和

# ✅ GPU 加速
# 将 tensor 转化为为 GPU 的 tensor, 从而享受加速.
# 但在不支持 CUDA 的机器下, 下一步还是在 CPU 上运行.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tensor = tensor.to(device)  # 通过 .to 迁移到 GPU 上
torch.tensor([1, 2, 3], device=device)  # 直接在 GPU 上在创建 tensor

# ✅ 广播机制
a = torch.arange(3).reshape((3, 1))
# tensor([[0],
#         [1],
#         [2]])
b = torch.arange(2).reshape((1, 2))
# tensor([[0, 1]])
a + b
# tensor([[0, 1],
#         [1, 2],
#         [2, 3]])

# ✅ 其他注意事项
# (1) 大多数 torch.function 都有一个参数out,
#     这时产生的结果将保存在 out 指定的 tensor 中
# (2) torch.set_num_threads 可以设置 PyTorch 进行 CPU 多线程并行计算时候所占用的线程数
# (3) torch.set_printoptions 可以用来设置打印 tensor 时的数值精度和格式

########################
## ⭐数据的保存和加载⭐ ##
#######################

# ✅ 张量的保存和加载
a = a.cuda(1)  # 把 a 转为 GPU1 上的 tensor
# 保存 tensor 到文件 a.pth
torch.save(a, "a.pth")  # 或 torch.save(a, 'a.pt')
# 加载为 b, 并存储在 GPU1 上 (因为保存时 tensor 就在 GPU1 上)
b = torch.load("a.pth")
# 加载为 c, 并存储在 CPU
c = torch.load("a.pth", map_location=torch.device("cpu"))
# 加载为 d, 并存储在 GPU0 上
d = torch.load("a.pth", map_location="cuda:0")

# ✅ 模型的保存和加载
net = torch.nn.Linear(2, 1)  # 创建网络
# 1. 仅保存参数
torch.save(net.state_dict(), "model.pth")  # 保存 net 的参数
net.load_state_dict(torch.load("model.pth"))  # 将保存的参数加载到 net
# 2. 保存完整的模型
torch.save(net, "model.pth")
loaded_net = torch.load("model.pth")

# ✅ 多种信息的保存和加载
# 保存信息 (比如优化器的参数, 损失函数, 迭代次数等)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()
epoch = 1
torch.save(
    {
        "epoch": epoch,
        "net_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "criterion": criterion,
    },
    "checkpoint.pth",
)
# 加载信息
checkpoint = torch.load("checkpoint.pth")
net.load_state_dict(checkpoint["net_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
criterion = checkpoint["criterion"]
epoch = checkpoint["epoch"]

###################
## ⭐AutoGrad⭐ ##
##################


# ✅ 手动求梯度
def get_grad_by_limit(f, x):
    h = 0.1
    result = 0
    for _ in range(10):
        # 随着增量 h 趋近于 0, 梯度越来越精确
        result = (f(x + h) - f(x)) / h
        h *= 0.1
    return result


# ✅ 自动求梯度
def get_grad_by_auto():
    input = torch.tensor(1, requires_grad=True, dtype=torch.float32)
    # 或 input.requires_grad = True
    # 或 input.requires_grad_(True)

    # 执行一些计算
    output = input.sigmoid()
    # 执行反向传播
    output.backward(gradient=torch.ones(output.size()),
                    retain_graph=False,
                    create_graph=False)
    # 获取梯度
    return tensor.grad


# ✅ 计算图
a = torch.ones(1)
b = torch.ones(1, requires_grad=True)
c = torch.ones(1)
y = a * c
z = a + b
g = z * b
"""
前向计算图:
(1) a → Mul → y
    c ↗

(2) a → Add → z → Mul → g
    b ↗       b ↗

反向计算图:
(1) da ← MulBackward ← dy
    dc ↙

(2) da ← AddBackward ← dz ← MulBackward ← dg
    db ↙               db ↙
"""
# 有两个计算图, 但实际上第一个是不存在的
# 因为 PyTorch 的计算图是根据 requires_grad 为 True 的节点来构建的
# 若一个节点的 requires_grad 为 False, 则它不会参与到计算图中
# 因此若要求某个节点的 grad, 则应该将其 requires_grad 设为 True

# 若某个节点的 requires_grad 为 True, 则依赖它的节点的 requires_grad 都会默认为 True
a.requires_grad  # False
b.requires_grad  # True
c.requires_grad  # False
y.requires_grad  # False
z.requires_grad  # True
g.requires_grad  # True

# 从前向计算图可以看出, a, b, c 均是叶子节点
# 但由于第一个计算图实际上不存在, 因此该图上的节点均默认是叶子节点
a.is_leaf  # True
b.is_leaf  # True
c.is_leaf  # True
y.is_leaf  # True
z.is_leaf  # False
g.is_leaf  # False

# 从反向计算图可以看出, g 的反向算子是 MulBackward
# z 的反向算子是 AddBackward, 其余节点都没有反向算子, 因此是 None
# 由于第一个计算图实际上不存在, 因此该图上的节点的反向算子都是 None
a.grad_fn  # None
b.grad_fn  # None
c.grad_fn  # None
y.grad_fn  # None
z.grad_fn  # <AddBackward0 object at 0x000001CD68FEBD60>
g.grad_fn  # <MulBackward0 object at 0x00000204119EB130>

# 从反向计算图可以看出, z 的反向算子的下一步算子是 da 和 db 的反向算子, 因此是 None
# 但由于 db 需要计算 grad, 它是累积的, 因此显示为 AccumulateGrad
# g 的反向算子的下一步算子是 dz 和 db 的反向算子, 因此是 AddBackward 和 None
# 但由于 db 需要计算 grad, 它是累积的, 因此显示为 AccumulateGrad
z.grad_fn.next_functions  # ((None, 0), (<AccumulateGrad object at 0x000001CD68FEBD90>, 0))
g.grad_fn.next_functions  # ((<AddBackward0 object at 0x000001CD68FEBD60>, 0), (<AccumulateGrad object at 0x000001CD68FEBDC0>, 0))

y.backward()  # 会报错, 因为它没有 grad_fn
g.backward(gradient=torch.ones((1, )), retain_graph=True,
           create_graph=False)  # 表示从 g 开始反向传播
# 参数解释:
# 1) gradient: 传入与调用对象形状相同的 tensor, 如果调用对象是标量, 则可以省略, 默认值为 1.
#             具体用法参考文章: https://blog.csdn.net/Konge4/article/details/114955821.
# 2) retain_graph: 是否保留计算图, 默认为 False.
#               作用: 在反向传播结束后会把计算图删掉, 若要重复进行 backward, 则应将计算图保留下来
# 3) create_graph: 对反向传播的过程也创建一个计算图,
#               然后通过 backward 的 backward 来实现高阶导数, 默认为 False

# 只有 requires_grad 为 True 的叶子节点的 grad 会被保留,
# 而其他节点的 grad 均会在反向传播结束后被清理掉 (置为 None),
# 即其他节点的梯度仅用于协助计算 requires_grad 为 True 的叶子节点的梯度.
g.grad  # None
z.grad  # None
a.grad  # None
b.grad  # tensor([3,])

# 多次反向传播会将每次计算的结果都累积起来
# 这也是 g.grad_fn.next_functions 中 AccumulateGrad 的含义
g.backward(retain_graph=True)
b.grad  # tensor([6,])

# 可以手动清除累积的梯度
b.grad.data.zero_()
g.backward(retain_graph=True)
b.grad  # tensor([3,])

# 可以关闭自动求导功能来节省内存/显存的开销
# torch.set_grad_enabled(False)
# 或者
with torch.no_grad():
    a = torch.ones(1)
    b = torch.ones(1, requires_grad=True)
    c = torch.ones(1)
    y = a * c
    z = a + b
    g = z * b

# 脱离计算图的方法
# 作用: 希望在计算某个 tensor 的指标 (如均值等) 时, 这些计算过程不会被计算图捕获
a = torch.ones((1, ), requires_grad=True)
b = torch.ones((1, )) * 2
z = a * b
z.backward()
a.grad  # tensor([2.])

# 1️⃣ 使用 .data
# a.data 和 a 共用数据, 即修改 a.data 也会修改 a
# 但 a.data 的 requires_grad 为 False, 对 a.data 的操作不会被计算图捕获
a.data.requires_grad  # False
z = a.data * b
z.backward()  # 会报错, 因为式子中没有需要求梯度的 tensor

# 2️⃣ 使用 .detach()
# a.detach() 会产生和 a.data 一样的 tensor
t = a.detach()
t.requires_grad  # False
z = t * b
z.backward()  # 会报错, 因为式子中没有需要求梯度的 tensor

# 3️⃣ 使用 .detach_()
# a.detach_() 会使得 a 本身的 requires_grad 变为 False
a.detach_()
a.requires_grad  # False
z = a * b
z.backward()  # 会报错, 因为式子中没有需要求梯度的 tensor
"""
✅ 开启和关闭 AutoGrad 功能
1) 关闭 AutoGrad 功能: torch.set_grad_enabled(False) 
2) 开启 AutoGrad 功能: torch.set_grad_enabled(True) 

✅ 可通过 hook 或 torch.autograd.grad 来获取中间节点的 grad:
1)  通过 hook
    hook_handler = middle_node.register_hook(lambda grad:print(grad)) # 注册 hook
    hook_handler.remove() # 移除 hook
2)  通过 torch.autograd.grad
    torch.autograd.grad(z, y) # 求 z 对 y 的梯度

✅ 注意事项
1)  版本变更:
    torch.Tensor 和 torch.autograd.Variable 在 v0.4 版本后是同一个类,
    torch.Tensor 能够和 Variable 一样追踪历史和反向传播,
    Variable 仍能够正常工作, 但返回的是 Tensor.
2)  梯度的计算是针对 tensor 中的每个元素（相互间独立）来进行的.
3)  grad 在反向传播的过程中是累加的, 这体现在: 
    (1) 多个样本（一个 batch）同时进行传播时, 
        每个样本的梯度都会叠加到一起, 因此最后要除以 batch_size 来求得梯度的期望.
        另一种解释: 当调用 .backward() 后, torch 会自动计算关于 tensor 的所有梯度,
        然后把这个 tensor 的所有梯度都累加到 .grad 属性上;
    (2) 每执行一次 backward, 梯度会直接累加到上一次 backward 计算的结果上,
        因此每次反向传播前都要将梯度清零, 即 tensor.grad.data.zero_().
4)  AutoGrad 构建的计算图是一个动态图, 即在运行时才开始构建.
5)  如果想要修改 tensor, 但又不想被 AutoGrad 记录, 
    则可以使用 tensor.data 或 tensor.detach().
6)  PyTorch 使用的是动态图, 它的计算图在每次前向传播时都是从头开始构建, 
    所以它能够根据需求使用 Python 控制语句 (如 for, if 等) 来创建计算图.
"""
