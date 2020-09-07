---
title: pytorch学习
date: 2020-09-06 16:46:45
tags: pytorch，深度学习框架
categories: 统计学习
mathjax: true
toc: true
---
在前面介绍深度学习的理论知识时，相信大家可以感受到，神经网络的实现主要有以下两个难题：
- 当网络结构复杂起来时，手写一个神经网络是非常困难(尤其是进行反向误差传播时)，也是十分费时的。
- 一个神经网络有着大量的参数，对计算机的计算能力要求非常高，而GPU是计算机中有着大量计算资源的部分，如何将这部分计算资源调动起来完成一个深度神经网络的训练是另一个难题。
<!--more-->

为了解决上述两个问题，深度学习框架便诞生了，目前主流的深度学习框架主要下图所示的这些:
![主流框架](https://raw.githubusercontent.com/xuejy19/xuejy19.github.io/source/Img/framework.jpeg)
这些框架比较如下图所示:
![深度学习框架](https://raw.githubusercontent.com/xuejy19/xuejy19.github.io/source/Img/framework1.png)
目前在科研机构中应用较多的是Tensorflow和pytorch,本文便以pytorch为例来进行讲解，本文内容主要参考官方教程：[pytorch官方教程](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py)。建议英文阅读能力较强的人还是去阅读原版教程，如果对于算法的理论理解已经到位的话，整个教程还是非常简洁明了的。本文按照以下结构组织:
- pytorch简介
- 张量和计算图 
- 低级API和高级API

### pytorch简介
pytorch 是一个开源的python 机器学习库，基于Torch，底层由C++实现，应用于人工智能领域，如自然语言处理(nlp)。它最初由Facebook的人工智能研究团队开发，并且被用于Uber的概率编程软件Pyro.
Pytorch 具有以下两大特征:
> - 提供类似于Numpy的张量计算，可使用gpu加速
> - 基于自动微分系统的深度神经网络

### 张量和计算图 
从caffee以来，基本上所有的深度学习框架都包含这两个概念，其实这两个关键概念也正是针对本文一开始提出的两个问题。 
#### 张量
> **张量：** 张量可以看作gpu下 numpy的替代品，在应用传统机器学习算法时，和我们打交道最多python库就是numpy，但遗憾的是numpy并不能够在gpu下进行运算。

输入:
```python
torch.randn(2,3)
```
输出:
```python
tensor([[ 1.9335,  0.8795, -0.6964],
        [ 0.1573,  1.5692, -0.2320]])
```
tensor的大部分操作与numpy是差不多的，具体属性和方法参考[torch.Tensor](https://pytorch.org/docs/stable/tensors.html?highlight=torch%20tensor#torch.Tensor)
#### 计算图
现代的神经网络架构可以拥有上百万个参数，从计算的角度看，训练神经网络包含两个阶段:
- 用于计算损失函数值的正向传递
- 向后传递以计算可学习参数的梯度

正向传递计算是非常简单的，前一层神经网络的输出是下一层神经网络的输入。后向传递有些复杂，因为这要求我们应用求导的链式法则来计算损失函数的梯度。

当一个神经网络的规模比较小(层数,各层神经元数)时，手算损失函数对各个参数的梯度是可行的，但随着神经网络规模的扩大，梯度的计算变得越来越困难，出于深度学习的这种现实需要，计算图便诞生了：
计算图属于有向无环图，计算图中的节点有两种，一种是基本数学运算符，另一种则是我们所定义的变量。比如下图就可以表示一个简单的神经网络:
$$
    \begin{aligned}
        b &= w_1 * a \\
        c &= w_2 * a \\
        d &= w_3 * b  + w_4 * c \\
        L & = 10 - d
    \end{aligned}
$$
在图中，变量$b,c$和$d$都是数学运算的结果，而变量$a ,w_1,w_2,w_3,w_4$ 由用户自己初始化，由于它们不是由任何数学运算符创建的，因此与其创建相对应的节点由其名称本身表示。

![计算图](https://raw.githubusercontent.com/xuejy19/xuejy19.github.io/source/Img/computation_graph.png)

下面来介绍如何使用计算图来进行梯度计算，首先我们计算在图中相连接的节点之间的梯度,将对应的梯度值写到边上，如下图所示:
![梯度计算图](https://raw.githubusercontent.com/xuejy19/xuejy19.github.io/source/Img/full_graph.png)
要计算损失$L$到某个变量$w$的梯度只需要按照以下步骤:
1. 找到从$L$到$w$所有可能的路径
2. 对于各个路径，将沿路径所有边相乘 
3. 将各个路径计算结果相加

 ### 低阶API和高阶API
我们在使用pytorch时主要接触到的pytorch的包的架构如下图所示:
![pytorch主要包](https://raw.githubusercontent.com/xuejy19/xuejy19.github.io/source/Img/Pytorch-package-hierarchy.jpg)
对于一个三层的全连结网络，我们可以仅仅使用tensor来实现，不过这种情况下需要手写梯度，下面我就一步步介绍如何用api来使你的代码变得简洁高效:
#### tensor 
```python
# -*- coding: utf-8 -*-

import torch


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```
#### autograd
这种情况下其实是没有显示的模型的概念的，只有自己定义的一个个参数。但当神经网络复杂之后，手算梯度变得不再现实，pytorch为解决这个问题提供了`autograd`这样一个好用的工具，如果我们将需要计算梯度的参数设置为`requires_grad = True`,那么在进行前向计算过程中，会定义一个计算图，同时会保存对应参数的梯度信息，也就是前面介绍的计算图中边上的梯度信息，通过`loss.backward()`操作，可以将损失对各个参数的梯度计算出来，由此可以重写上面的代码如下:
```python
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs.
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Tensors during the backward pass.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y using operations on Tensors; these
    # are exactly the same operations we used to compute the forward pass using
    # Tensors, but we do not need to keep references to intermediate values since
    # we are not implementing the backward pass by hand.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the scalar value held in the loss.
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call w1.grad and w2.grad will be Tensors holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    # An alternative way is to operate on weight.data and weight.grad.data.
    # Recall that tensor.data gives a tensor that shares the storage with
    # tensor, but doesn't track history.
    # You can also use torch.optim.SGD to achieve this.
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()
```
#### Extend autograd function
可以看到，有了`autograd`这个工具后，复杂网络的反向梯度计算也很容易解决。虽然`autograd.Function`中已经基本包含了我们所经常用到的运算函数(自动微分算子)，但若我们希望将一个其中没有的函数也加入到计算图中，则需要对`autograd.Function`进行扩展，比如我们可以重写`ReLu`函数，然后将其纳入自动微分:
```python
# -*- coding: utf-8 -*-
import torch


class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # To apply our Function, we use Function.apply method. We alias this as 'relu'.
    relu = MyReLU.apply

    # Forward pass: compute predicted y using operations; we compute
    # ReLU using our custom autograd operation.
    y_pred = relu(x.mm(w1)).mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # Use autograd to compute the backward pass.
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()
```
#### nn.Module
如果我们要构建一个深层神经网络，固然可以用这些基本的自动微分算子来构建，但这样做显得有些`low-level`，因为神经网络在进行理论推导时实际上就是`层的堆砌`，那么在进行编程实现时，可否沿用这个思想，将一些微分算子封装成神经网络层的形式，pytorch中实现这个功能是通过包`nn`，`nn`中除了定义了一系列模块`Modules`，同时还有大量的损失函数，下面我们就用`torch.nn.Sequential`来重新实现这样一个三层神经网络:
```python
import torch

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
```
可以看到，与不使用`nn`包相比，上面的代码实现一方面利用`nn.Module.Sequential`实现了对神经网络层的封装，另一方面应用`nn.MSELoss`来替代原始的手写损失函数，使得代码的可读性和简洁性进一步提升。同时需要注意的是，在进行反向传播之前通过`model.zero_grad()`将存储的梯度归0，在进行参数更新时必须要使用`with torch.no_grad()`上下文管理语句来避免在进行参数更新时仍记录梯度，提高计算速度。

#### optim
在实际进行优化时，我们不仅会用到`SGD`这种常规的优化算法，同时也会用到一些复杂的优化算法如`AdaGrad`,`RMSprop`, `Adam`等，`torch.optim`提供了大部分常用的优化算法，同时应用优化器后也不需要再使用`with torch.no_grad()`语句，可以直接使用`optimizer.step()`来实现参数更新，重写这个三层神经网络:

```python
# -*- coding: utf-8 -*-
import torch

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algorithms. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
```

#### Custom nn Modules
在前面定义这个神经网络是通过`torch.nn.Sequential`将各层函数封装成一个模型，有时可能需要定义一个更加复杂的模型，而这个模型中的有些模块需要自己手动实现，此时可以通过定义一个继承`torch.nn.Module`的子类来实现这个功能：
```python
# -*- coding: utf-8 -*-
import torch


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
可以看到，这样做其实与直接使用`torch.nn.Sequential`来定义模型在有些情况下是差不多的，但通过继承`nn.Module`一方面可以使得自己的模型更具扩展性，同时代码的可读性也进一步加强，因为你自己的神经网络模型是以一个类的形式存在的。

#### TensorDataset
在进行数据处理时，我们的数据基本都是`x_train,y_train,x_test,y_test`的形式，在进行迭代时可能必须要通过下标来进行迭代，而为了让迭代操作更加方便，`pytorch`引入了`torch.utils.data`来辅助进行数据处理，而其中`TensorDataset()`方法相当于是对`x,y`做了一个`wrap`操作，在进行迭代时可以一起迭代而不需要分别迭代：
>应用`TensorDataset()`方法:
>```python
>train_ds = TensorDataset(x_train, y_train)
>```
>   应用前迭代：
> ```python
> xb = x_train[start_i:end_i]
> yb = y_train[start_i:end_i]
> ```
> 应用后迭代：
> ```python
> xb,yb = train_ds[i*bs : i*bs+bs]
> ```

#### DataLoader
虽然使用`TensorDataset`使得迭代运算变得简单，但我们还是按照`batch_size`的大小来对迭代下标进行维护，为了能够更方便地应用数据集进行训练，`torch.utils.data`又为我们提供了另一个好用的工具`DataLoader`,它可以将`warp`之后的数据集按照`batch_size`大小进行分割，然后在迭代时每次取一个`batch`的数据。
> 应用`DataLoader`对数据集进行处理：
> ```python
> train_ds = TensorDataset(x_train, y_train)
> train_dl = DataLoader(train_ds, batch_size=bs)
> ```
> 对处理后的数据进行迭代：
> ```python
> for xb,yb in train_dl:
>    pred = model(xb)
> ```

#### MNIST数据集CNN代
有了应用上面的那些封装好的api，在MNIST数据集上实现一个CNN代码便不再困难，下面给出代码：
```python
from pathlib import Path
from matplotlib import pyplot as plt
from IPython.core.debugger import set_trace
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import requests
import math
import pickle
import gzip
import torch

'''
class Mnist_logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784,10)/math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))
    def forward(self, xb):
        return xb @ self.weights + self.bias
'''
'''
class Mnist_logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784,10)
    def forward(self, x):
        return self.linear(x)
'''
def get_model():
    model = Mnist_CNN().to(device)
    opt = optim.SGD(model.parameters(), lr = 0.5)
    return model, opt
def preprocess(x, y):
    return x.view(-1,1,28,28).to(device), y.to(device)
class WrappedDataLoader:
    def __init__(self, d1, func, batch_size):
        self.d1 = d1
        self.func = func 
        self.batch_size = 64
    def __len__(self):
        return len(self.d1)
    def __iter__(self):
        batches = iter(self.d1)
        for b in batches:
            yield (self.func(*b))
class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.pooling = nn.AdaptiveAvgPool2d(1)
    def forward(self, xb):
        xb = self.relu(self.conv1(xb))
        xb = self.relu(self.conv2(xb))
        xb = self.relu(self.conv3(xb))
        xb = self.pooling(xb)
        return xb.view(-1,xb.size(1))
def loss_batch(moedl, loss_func, xb, yb, opt = None):
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)

def fit(epoches, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epoches):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)
        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
            val_loss = np.sum(np.multiply(losses, nums)/np.sum(nums))
            if epoch % 10 == 0:
                train_aur, valid_aur = cal_acur(model, train_dl, valid_dl)    
                print('epoch:',epoch,'验证集损失:',np.str(val_loss),'训练集准确率:',np.str(train_aur),'验证集准确率:',np.str(valid_aur))

def get_data(train_ds, valid_ds, bs):
    train_dl = DataLoader(train_ds, batch_size = bs, shuffle = True)
    valid_dl = DataLoader(valid_ds, batch_size = bs, shuffle = True)
    return train_dl, valid_dl
def batch_acu(y_pred, y):
    batch_pred = torch.argmax(y_pred, dim = 1)
    return (batch_pred == y).sum()

def cal_acur(model, train_dl, valid_dl):
    train_pred_right = 0
    valid_pred_right = 0
    for xb, yb in train_dl:
        y_pred = model(xb) 
        train_pred_right += batch_acu(y_pred, yb)
    for xb, yb in valid_dl:
        y_pred = model(xb)
        valid_pred_right += batch_acu(y_pred, yb)
    train_acur = train_pred_right.float()/(len(train_dl)*train_dl.batch_size) 
    valid_acur = valid_pred_right.float()/(len(valid_dl)*valid_dl.batch_size)
    return train_acur.item(), valid_acur.item()
        

		

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    FILENAME = "data/mnist/mnist.pkl.gz"
    with gzip.open(FILENAME, "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
    x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
    print(torch.cuda.get_device_name(0))
    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)
    loss_func = F.cross_entropy
    epoches = 100
    # train and valid
    train_dl, valid_dl = get_data(train_ds, valid_ds, batch_size)
    train_dl = WrappedDataLoader(train_dl, preprocess, batch_size)
    valid_dl = WrappedDataLoader(valid_dl, preprocess, batch_size)
    model, opt = get_model()
    fit(epoches, model, loss_func, opt, train_dl, valid_dl)

```
### 总结
最后总结一下`pytorch`中常用的包：
- `torch.nn:`包含搭建神经网络层的模块(Modules)和一系列`loss`函数
- `torch.nn.functional:`提供常用的激活函数，这算是一个比较低级的api，大部分激活函数在`nn`下都有封装
- `torch.autograd:` 提供所有张量操作的自动求导方法
- `torch.optim:` 提供各种参数优化方法，如`SGD`、`Adam`、`AdaGrad`等
- `torch.utils.data:` 用于加载数据，比如上面介绍的`TensorDataset()`和`DataLoader()`方法
- `torchvision.datasets:`加载图像处理领域常用数据集，如`MNIST`、`coco`、`CIFAR10`、`Imagenet`等
- `torchvision.modules:` 图像处理领域常用模型，如`AlexNet`、`VGG`、`ResNet`、`DenseNet`等
- `torchvision.transforms:` 图片相关处理操作，如裁剪、尺寸缩放、归一化等

`pytorch`的学习其实并不是孤立的，其实是与其他知识相耦合的:
- `python`语法知识
- 算法理论知识

如果你本身`python`用的非常熟练，同时算法的理论基础也比较扎实，那么`pytorch`对你来说可能就是一个非常容易上手的工具，但如果本身这两方面的知识不够牢固，在学习`pytorch`的过程中，这两方面的能力也能得到很大的提高。
