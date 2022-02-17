import torch
import random


# 生成数据
def generate_data(w, b, num):
    x = torch.normal(0, 1, size=(num, len(w)), dtype=torch.float32)
    noise = torch.normal(0, 0.01, size=(num, 1), dtype=torch.float32)
    y = torch.matmul(x, w)+b+noise
    return x, y


true_w = torch.tensor([[3], [4]], dtype=torch.float32)
true_b = torch.tensor(5, dtype=torch.float32)
train_x, train_y = generate_data(true_w, true_b, 1000)


# 构建 batch samples 迭代生成器
def get_batch(x, y, batch_size):
    x_count = len(x)
    indices = list(range(x_count))
    random.shuffle(indices)
    for i in range(0, x_count, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i+batch_size, x_count)])
        yield x[batch_indices], y[batch_indices]


# linear modal
def linear_regression(x, w, b):
    return torch.matmul(x, w)+b


# 计算 loss
def compute_loss(y_hat, y):
    # 使用平均 loss 会导致梯度下降的很慢, 需要大量的迭代次数
    # 使用下面的总和 loss 会训练的更快, 因为每次梯度更新的力度更大
    # ((y_hat-y.reshape(y_hat.shape))**2/2).sum()
    return ((y_hat-y.reshape(y_hat.shape))**2/2).mean()


# 更新参数
# @torch.no_grad()
def update_params(params, lr, batch_size):
    # torch.no_grad() 是一个上下文管理器, 被该语句 wrap 的部分将不会 track 梯度
    # 可以通过装饰器 @torch.no_grad() 的方式进行 wrap
    # 也可以通过 with torch.no_grad() 的方式进行 wrap
    with torch.no_grad():
        for param in params:
            # 一个 batch 关于 param 的梯度是所有 sample 关于 param 的梯度的累加和
            # 除以 batch_size 来得到平均梯度
            param -= lr*param.grad/batch_size
            param.grad.zero_()  # 清空梯度, 避免累积


# 训练
num_epochs = 500
batch_size = 100
w = torch.randn(size=(2, 1), dtype=torch.float32, requires_grad=True)
b = torch.zeros(size=(), dtype=torch.float32, requires_grad=True)

for epoch in range(num_epochs):
    print(w.data, b.data)
    for batch_x, batch_y in get_batch(train_x, train_y, batch_size):
        y_hat = linear_regression(batch_x, w, b)
        compute_loss(y_hat, batch_y).backward()
        update_params([w, b], lr=0.1, batch_size=batch_size)
    print(w.data, b.data)
