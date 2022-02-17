"""
链式分离求导法

神经网络可以看成是一个复杂的复合函数 f, 当求取某个变量 x 的梯度时, 
通常的做法是先通过链式求导法则写出整体的求导表达式, 
再用变量 x 简化并替换所有中间变量后, 代入变量 x 的值来求取最终的梯度. 

而链式分离求导法的做法是写出整体表达式后, 
分别单独计算每个子导数表达式的值 (在这个过程中每个中间变量保持原样, 
并通过直接代入自身的值来求出对应的导数值), 再把这些值相乘来得到最终的梯度. 
因此这个方法的过程为: 写出链式求导表达式 + 每个导数单独计算 + 将所有导数值相乘. 
pytorch 的 AutoGrad 对这个方法的实现: 
按照反向传播的方向 (即根据链式求导法则写出每个子导数表达式的顺序) 
以迭代的方式计算每个子导数表达式的值. 

举个例子: 假设整体的求导表达式为 (df/da * da/db * db/dx)
1.  首先按照求导规则计算第一个子表达式 df/da 的值, 然后将计算结果传递给下一个子表达式;
    1)  这个过程在 torch.autograd.Function 的 backward 方法中进行;
    2)  为了得到结果, 须代入变量 a 的值, 这个值会在前向传播 
        (即执行 torch.autograd.Function 的 forward 方法) 时,
        通过 save_for_backward 保存下来, 然后在 backward 中通过 saved_tensors 获取;
    3)  需要传递的计算结果通过 backward 方法的 return 返回, 
        返回后会被作为下一个子表达式的 backward 方法的 grad_output 参数;
    4)  第一个子表达式的 backward 方法的 grad_output 参数就是
        根节点 z 调用 backward 方法时传入的 gradient 参数.
2.  接着计算第二个子表达式 da/db 的值, 并乘上从前一个表达式 dz/da 传递下来的累积的梯度值, 
    然后将计算结果继续传递给下一个子表达式;
3.  后面以此类推, 直到所有子表达式都被计算完成, 
    最终计算的梯度结果会被赋值给目标变量 x 的 grad 属性上.
"""


import torch


#########################
## ⭐自定义 Function⭐ ##
#########################

# 1) 自定义 Function 须继承 torch.autograd.Function,
#    不需要构造函数, 且 forward 和 backward 方法都是静态的.
# 2) backward 方法的输出和 forward 函数的输入一一对应,
#    backward 函数的输入和 forward 函数的输出一一对应.
# 3) backward 方法的 grad_output 参数就是 tensor.backward 中的 gradient 参数.
# 4) 在 backward 方法中, 对于不需要求导的输入变量, 直接返回None.
# 5) 反向传播若需要利用前向传播的某些中间结果,
#    则须进行保存, 否则前向传播结束后这些对象就被释放了.


class WX_ADD_B(torch.autograd.Function):
    # 在 forward 中定义该 Function 需要执行的操作
    # w, x, b, not_requires_grad_param 均为输入变量
    @staticmethod
    def forward(self, w, x, b, not_requires_grad_param):
        self.save_for_backward(w, x)
        output = w * x + b
        return output

    # 在 backward 中定义该操作的求导过程, 并返回累积梯度
    @staticmethod
    def backward(self, grad_output):
        # grad_output 为前面所有求导表达式累积下来的梯度
        w, x = self.saved_tensors
        grad_w = grad_output * x
        grad_x = grad_output * w
        grad_b = grad_output * 1
        # 按 forward 输入变量的输入顺序返回各变量对应的梯度
        return grad_w, grad_x, grad_b, None


# WX_ADD_B 的示例 1:
x = torch.rand(1, requires_grad=False)
w = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)
z = WX_ADD_B.apply(w, x, b, None)  # 执行的是 WX_ADD_B 的 forward 方法
z.backward()
# x 的 requires_grad 属性为 False, 因此在反向传播后它的 grad 就被清除了,
# 即使在 WX_ADD_B 的 backward 方法中返回了 x 的 grad.
print(x.grad, w.grad, b.grad)

# WX_ADD_B 的示例 2:
x = torch.rand(1, requires_grad=False)
w = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)
z = WX_ADD_B.apply(w, x, b, None)
z.grad_fn.apply(torch.ones(1))
# 相当于执行了 WX_ADD_B 的 backward 方法
# 而且也仅仅是执行了 WX_ADD_B 的 backward 方法
# 并没有进行反向传播, 因此 x.grad, w.grad, b.grad 均没有值
print(x.grad, w.grad, b.grad)


#####################
## ⭐二阶求导举例⭐ ##
#####################

# 1) 要想实现高阶求导, 只需对反向传播的计算过程也建立一个计算图,
#    这可以通过设置参数 create_graph=True 来实现.
# 2) 反向传播和前向传播都是一样的, 都是一步一步地对变量进行计算,
#    高阶导数就是把反向传播也看成前向传播, 计算高阶导数就是对反向传播进行反向传播.
x = torch.tensor([5], requires_grad=True, dtype=torch.float)
y = x ** 2
grad_x = torch.autograd.grad(y, x, create_graph=True)
print(grad_x)  # grad_x = dy/dx = 2 * x
grad_grad_x = torch.autograd.grad(grad_x[0], x)
print(grad_grad_x)  # grad_grad_x = d(2x)/dx = 2


########################
## ⭐gradcheck 举例⭐ ##
########################

# 1) 使用 gradcheck 可以检测 backward 的实现是否正确.
# 2) gradcheck 通过数值逼近来计算梯度,
#    然后将计算结果与 backward 计算的结果进行比对.
# 3) gradcheck 通过控制参数 eps 的大小来控制可以容忍的误差.


class MySigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        output = 1 / (1 + torch.exp(-x))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        grad_x = output * (1 - output) * grad_output
        return grad_x


test_input = torch.randn(3, 4, requires_grad=True).double()
check_result = torch.autograd.gradcheck(MySigmoid.apply, (test_input,), eps=1e-3)
print("gradcheck 结果:", check_result)


########################
## ⭐举例: 实现 ReLU⭐ ##
########################

# 自定义实现 ReLU 激活函数的计算过程和求导过程
class MyReLU(torch.autograd.Function):
    def forward(self, input_):
        self.save_for_backward(input_)  # 将输入保存起来, 在 backward 时使用
        output = input_.clamp(min=0)
        return output

    def backward(self, grad_output):
        # 假设根据链式法则有 dz / dx = (dz / dReLU) * (dReLU / dx)
        # dz / dReLU 就是输入的参数 grad_output
        # 因此只需求 ReLU 的导数, 再乘以 grad_output
        (input_,) = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
