import torch
from torch.utils import data


# 生成数据
def generate_data(w, b, num):
    x = torch.normal(0, 1, size=(num, len(w)), dtype=torch.float32)
    noise = torch.normal(0, 0.01, size=(num, 1), dtype=torch.float32)
    y = torch.matmul(x, w)+b+noise
    return x, y


# 构建 batch samples 加载器
# 这是一个可迭代对象, 但不是一个生成器, 可通过 iter(DataLoader) 转化为生成器.
def load_batch_iteration(dataset, batch_size):
    dataset = data.TensorDataset(*dataset)  # 解构赋值
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


true_w = torch.tensor([[3], [4]], dtype=torch.float32)
true_b = torch.tensor(1, dtype=torch.float32)
train_x, train_y = generate_data(true_w, true_b, 1000)
get_batch = load_batch_iteration((train_x, train_y), 100)


# 构建网络
linear_regression = torch.nn.Sequential(torch.nn.Linear(2, 1))
linear_regression[0].weight.data.normal_(0, 0.01)
linear_regression[0].bias.data.fill_(0)
mse_loss = torch.nn.MSELoss()
trainer = torch.optim.SGD(linear_regression.parameters(), lr=0.1)


# 训练
num_epochs = 5

for epoch in range(num_epochs):
    for batch_x, batch_y in get_batch:
        trainer.zero_grad()  # 将梯度初始化为零, 避免累积
        y_hat = linear_regression(batch_x)
        mse_loss(y_hat, batch_y).backward()
        trainer.step()

    print(linear_regression[0].weight.data)
    print(linear_regression[0].bias.data)
