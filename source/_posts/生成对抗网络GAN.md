---
title: 生成对抗网络GAN
date: 2020-09-09 23:34:49
tags: GAN
categories: 统计学习 
mathjax: true
#img: https://raw.githubusercontent.com/xuejy19/xuejy19.github.io/source/thumbnail/GAN.jpg
toc: true
---
这一部分介绍一种特殊的神经网路模型——生成对抗网络(GAN)，生成对抗网络由[Lan Goodfellow](https://en.wikipedia.org/wiki/Ian_Goodfellow)于2014年提出，该算法在形式上表现为两个神经网络的彼此对抗，对于生成对抗网络，我们可以从以下几个角度来对其进行限定：
- **本质：** 学习训练数据集的分布  
- **作用：** 产生新的样本，对于小样本任务可以起到数据增强的作用 
- **实现形式：** 两个神经网络彼此对抗 
 
本文按照以下结构进行组织:
- GAN算法思想
- GAN背后数学推导  
- GAN`pytorch`实现 
  
<!--more-->

### GAN 算法思想
![GAN](https://raw.githubusercontent.com/xuejy19/xuejy19.github.io/source/Img/GAN.jpg)
考虑这样一个场景，一个小镇里有一个造假钞的人，同时有一个警察，它们两个各自的目标分别是：
- **罪犯：** 不断提高自己的造假钞技术，使得自己的假钞足以以假乱真，让警察鉴别不出来
- **警察：** 不断提高自己的鉴别水平，能够准确地识别小偷制造的假钞

从博弈论的角度来看，这其实是一个零和博弈：
> 小偷的造假技术越高超，则警察鉴别起来就越困难;反之，警察的鉴别技术越高超，则小偷造出能欺骗过警察的假钞就更加困难，在对抗中，两个人都在变强。

接下来我们分析构想一个场景，我们有一组数据，但是数据的量不足，我们期望能够找到这样一个"造假数据"的人帮助我们产生一些新的图像，以达到数据增强的目的。但在这个"造假数据"的人造出一个假数据之后，我们需要判断一下这个数据到底合不合格，因此我们就需要另外一个鉴别"假数据"的"警察"。 

其实我们的本质目的是为了**造假数据**,但这个任务若无监督是很难进行下去的，因此我们需要为其提供一个"警察"对其进行监督，因此我们就需要有两个学习器，一个学习如何生成"假数据"，另一个则需要学习如何判别出这些"假数据"，在这两个学习器博弈的过程中我们最终得到了一个"造假数据"比较高超学习器。

### GAN 背后数学推导 
> **沃兹基**曾经说过：算法思想也就图一乐，真涨姿势还是要看数学推导

#### 从最大似然估计谈起
因为我们本质上是希望能够得到一个学习器，使得其产生的数据的能够与训练集数据同分布，那么数学推导就从分布入手，假设训练集数据概率分布为$p(x|\theta)$,其中$\theta$为该概率分布所依赖的参数，当我们得到一组数据$X = (x_1,x_2,\dots, x_N)$时，我们想要对参数进行估计，那么此时就要祭出已经老生常谈的**最大似然估计**，假设样本独立同分布，则写出似然函数如下:
$$
    L(X|\theta) = \prod_{i=1}^N p(x_i | \theta)
$$
而$\theta$的求解其实就是一个优化问题:
$$
    \begin{aligned}
        \theta^* &= argmax L(X|\theta) \\
        &= argmax \ln L(X|\theta) \\  
        &= argmax \frac{1}{N} L(X|\theta) \\
        &= argmax \frac{1}{N} \sum_{i=1}^N \ln p(x_i|\theta) \\
        &= argmax E_{x \sim p(x)} \ln p(x|\theta)  \\
        &= argmax \int_{x} p(x) \ln p(x|\theta) dx
    \end{aligned}
$$
这里需要特别说明一下这一步：
$$
    argmax \frac{1}{N} \sum_{i=1}^N \ln p(x_i|\theta) \Leftrightarrow argmax E_{x \sim p(x)} \ln p(x|\theta)
$$

当样本数量足够大时，空间中每个样本点都被覆盖，而具体空间中某一个点$x_i$会落入多少样本点则取决于数据分布$p(x)$在该点概率密度函数的大小，换句话说，如果从采样的角度，要对某点的概率密度函数进行估计，那么只需要原始数据进行采样，采样$N$个点，若最终有$n_{x_i}$个点落在了$x_i$处，那么该点的概率密度函数就可以估计为:
$$
    p(x_i) = \frac{n_{x_i}}{N}
$$
于是上面的等价性也是可以按照这种思想推导出来：
$$
    \begin{aligned}
        \lim_{N \rightarrow \infty} \frac{1}{N} \ln p(x_i|\theta) &= \sum_{x} \frac{n_{x}}{N} \ln p(x_i|\theta)  \\
        &= \int_x p(x) \ln p(x|\theta)  dx  \\
        &= E_{x \sim p(x)} \ln p(x|\theta)  
    \end{aligned}
$$
最大似然函数估计的目的是找到一组最合适的参数$\theta$使得分布$p(x|\theta)$更加符合数据分布，但是能够应用最大似然估计的场合一般是对数据分布有一个假设，这样才有参数可以优化，搜索空间限制在$\mathcal{p(x|\theta)}$,现在我们将这个约束去掉，在所有可能分布中找到一个分布$q(x)$,使得$q(x)$能够尽可能接近原始数据分布$p_{data}(x)$。 至此我们证明了：
$$
    \theta^* = argmax L(X|\theta)  \Leftrightarrow  argmax E_{x \sim p(x)} \ln q(x)
$$

#### 最大似然估计与KL散度
紧接上文，我们来看下$E_{x \sim p(x)} \ln q(x)$:
$$
    \begin{aligned}
    argmax_{q(x)}    E_{x \sim p(x)} \ln q(x) &= argmax_{q(x)}\int_x p(x) \ln q(x) dx  \\ 
    &= argmin_{q(x)} -\int_x p(x) \ln q(x) dx  \\
    &= argmin_{q(x)} -\int_x p(x) \ln q(x) dx + \int_x p(x) \ln p(x) dx \\
    &= argmin_{q(x)} D_{KL}(p(x)||q(x))
    \end{aligned}
$$
这说明我们要找到一个分布$q(x)$使得其对于原始分布$p(x)$产生数据的极大似然函数最大就等价于找到一个分布$q(x)$使得两个分布之间的KL散度最小(不知道啥是KL散度的，来[这里](https://xuejy19.github.io/2020/08/26/%E6%A6%82%E7%8E%87%E5%88%86%E5%B8%83%E4%B8%AD%E7%9A%84%E8%B7%9D%E7%A6%BB%E5%BA%A6%E9%87%8F/#more)补课)。同时中间结果则是也等价于最小化交叉熵，所以我们又得到了一个有用的结论:
> 最大化似然函数 $\Leftrightarrow$ 最小化交叉熵 $\Leftrightarrow$ 最小化KL散度

然后为了解决KL散度的不对称性，又引入了JS散度：
$$
    JS(p(x)||q(x)) = \frac{1}{2} D_{KL}(p(x)||\frac{p(x)+ q(x)}{2}) + \frac{1}{2} D_{KL} (q(x) || \frac{p(x) + q(x)}{2})
$$
本来想推导下优化KL散度等价于优化JS散度，但没推出来，只有一个不等关系:
$$
    JS(p||q) \leq \frac{1}{4} D_{KL}(p||q) + \frac{1}{4} D_{KL}(q||p)
$$
本身KL散度的取值范围是$[0, +\infty]$,而JS散度的取值范围是$[0,1]$($\log$以2为底)，但是KL散度与JS散度取最小值的条件是一样的，即两个分布完全相同，因此使用JS散度或者KL散度进行优化，应当是同样的优化效果。 因此可以继续将结论补充：
> 最大化似然函数 $\Leftrightarrow$ 最小化交叉熵 $\Leftrightarrow$ 最小化KL散度$\Leftrightarrow$ 最小化JS散度

#### 生成对抗网络
之前进行分布拟合，一般通过假设分布是一个高斯分布(或混合高斯分布)，然后对参数进行优化，使得对数据有一个较好的拟合效果，但高斯分布假设很多时候并不满足，因此需要一个更大的模型搜索空间。随着神经网络科学的发展，人们开始思考，能否使用神经网络来将一个高斯分布映射为我们期望的分布$p_G$，但由于$p_{data}$是未知的，我们并不知道应该如何衡量两个分布之间的差异，因此就需要有一个判别器。GAN的目标函数如下:
$$
    V(D, G) = E_{x \sim data} \log D(x) + E_{x \sim p_G} \log (1 - D(x))
$$
前一项中的$D(x)$表示判别器对来自训练集数据的评分，后一项中的$D(x)$表示判别器对来自生成器数据的评分，如果想提高判别器的性能，则要求判别器对来自训练数据集的数据评分较高，对来自生成器生成数据评分较低，因为$D(x) \in [0,1]$,因此$D(x)$也可以认为是一个样本是来自原始分布的概率。下面我们将$V(D,G)$写开:
$$
    \begin{aligned}
        V(D,G) &=  E_{x \sim data} \log D(x) + E_{x \sim p_G} \log (1 - D(x)) \\ 
        &= \int_x p_{data}(x) \log D(x) dx + \int_x p_G(x) \log(1-D(x)) dx \\
        &= \int_x [p_{data}(x) \log D(x) + p_G(x) \log(1 - D(x))] dx  
    \end{aligned}
$$
判别器希望最大化$V(D,G)$,即对于来自于真实样本分布的样本的打分尽可能多，对来自于生成器的样本的打分尽可能低，想要优化上式，即低于任意一个输入$x$，判别器应当:
$$
    \max_{D} v(D) = p_{data}(x) \log D(x) + p_G(x) \log(1 - D(x))
$$
该函数是一个凹函数，可以直接通过求导等于0求得$D(x)$最优解：
$$
    \frac{\partial v(D)}{\partial D(x)} = \frac{p_{data}(x)}{D(x)} - \frac{p_G(x)}{1 - D(x)} = 0
$$
可以求得： 
$$
    D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_{G}(x)}
$$
将求得的$D(x)$回代$V(D,G)$:
$$
    \begin{aligned}
         V(D^*,G) &= \int_x p_{data}(x) \log \frac{p_{data}(x)}{p_{data}(x) + p_{G}(x)} dx + \int_x p_G(x) \frac{p_{G}(x)}{p_{data}(x) + p_{G}(x)}  dx \\
         &= -2\log2 + 2JS(p_{data}(x) || p_G(x)) 
    \end{aligned}
$$
所以说我们接下来对生成器G的优化，其实就是:
$$
    G^* = \arg \min_G V(D^*,G) = \arg \min_{G} JS(p_{data}(x)||p_G(x))
$$
这说明我们对生成器进行优化，实质上就是期望能够找到一个分布，使得$p_G(x)$与原始数据分布$p_{data}(x)$的JS散度最小，由前面的等价性，其实就是找到一个分布使得对训练数据的最大似然函数最大。 因此整个优化问题可以写做:
$$
    \min_{G} \max_{D} V(D,G)
$$

#### GAN `Pytorch`实现 
在实现的时候需要关注的问题我觉得主要有以下几个：
- 损失函数选择 
- 网络结构设计 
- 优化顺序 
#### 损失函数选择 
首先讨论下损失函数选择，对于判别器而言，其面临的其实就是一个二分类问题，我们看下$V(D,G)$:
$$
    \begin{aligned}
        V(D,G) &= E_{x \sim data} \log D(x) + E_{x \sim p_G} \log (1 - D(x)) 
    \end{aligned}
$$
从采样的角度来看，$V(D,G)$可以写做:
$$
    \begin{aligned}
        V(D,G) = \sum_{i=1, x \sim p_{data}(x)}^{batch\_size} \log D(x) + 
        \sum_{i=1, x \sim p_{G}(x)}^{batch\_size} \log (1 - D(x))
    \end{aligned}
$$
若设置来自于真实数据集的数据的label为1，来自于生成器的数据的label为0，则上式其实可以看作两个二分类交叉熵的和，二分类交叉熵计算公式为：
$$
    BCE(x_n,y_n) = y_n \log x_n + (1-y_n) \log (1 - x_n) 
$$
对于label为1的样本，保留前一项；而对于label为0的样本，保留后一项，刚好就是$V(D,G)$。当然也可以通过直接就两个类别做one-hot，然后使用普通的交叉熵损失函数也可以。 

在对生成器进行优化的时候，因为我们期望造出的假样本能够尽可能真，因此只需要将假样本的label设置为1，然后进行反向传播优化生成器即可。
#### 网络结构设计 
生成对抗网络是出了名的难训练，因此合理的网络结构非常重要，对于新入门的人来说就只能参考一些现成的网络结构,同时对于不同的任务也应当选用不同的网络结构。目前生成对抗网络在图像处理领域较多，因此往往采用卷积神经网络。
![GAN](https://raw.githubusercontent.com/xuejy19/xuejy19.github.io/source/Img/generative-adversarial-network.png)

#### 优化顺序
生成对抗网络的优化是一个交替优化的过程，一般是先对判别器进行优化，接下来再对生成器进行优化，两者进行交替优化，互相博弈，下面简单地写下计算图：
> **判别器：** 
> 真实样本、虚假样本(生成器生成) $\rightarrow$ 判别器网络D $\rightarrow$  predict $\rightarrow$ Loss_D 
> **生成器：** 
> 虚假样本 $\rightarrow$ 生成器网络 $\rightarrow$ 虚假样本 $\rightarrow$ 判别器 $\rightarrow$ predict $\rightarrow$ Loss_G  

####   `pytorch` 代码实现
数据集采用的是[celeb-A faces Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html),代码实现如下:
```python
import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from tqdm import tqdm   
from IPython.display import HTML

import torch 
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vision_utils
def matplotlib_imshow(img , one_channel = False):
    if one_channel:
        img = img.mean(dim = 0)
    img = img/2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(img, cmap = 'Greys') 
    else:
        npimg  = np.transpose(npimg, (1,2,0)) 
        plt.imshow(npimg)
    plt.show()
def load_data(dataroot, image_size, batch_size, num_workers):
    transform = transforms.Compose([
                   transforms.Resize(image_size) ,
                   transforms.CenterCrop(image_size),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    dataset = datasets.ImageFolder(root = dataroot, transform = transform) 
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, 
                                        shuffle = True, num_workers = num_workers)
    # plot some training images 
    # real_batch, _ = iter(dataloader).next()
    
    #batch_image = vision_utils.make_grid(real_batch, padding = 2) 
    #matplotlib_imshow(batch_image)
    return dataloader
def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02) 
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__() 
        # nz = 100 H_out = (H_in - 1)*stride - 2*padding + kernel_size
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias = False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU())  
        # state size: (ngf*8)*4*4
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU())
        # state size: (ngf*4)*8*8 
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU())
        # state size: (ngf*2)*16*16 
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU()) 
        # state size: ngf*32*32 
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias = False),
            nn.Tanh())
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.conv5(x)
class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator,self).__init__()
        # input size: nc*64*64 
        self.conv1 = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias = False), 
            nn.LeakyReLU(0.2)) 
        # state size: ndf*32*32 
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, 2*ndf, 4, 2, 1, bias = False),
            nn.BatchNorm2d(2*ndf),
            nn.LeakyReLU(0.2))  
        # state size: (ndf*2)*16*16 
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2)) 
        # state size: (ndf*4)*8*8 
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2)) 
        # state size: (ndf*8)*4*4
        self.conv5 = nn.Sequential(
            nn.Conv2d(ndf*8,1, 4,1,0,bias = False),
            nn.Sigmoid()) 
        # state size: 1*1 [0,1] 
    def forward(self, x):
        x = self.conv1(x) 
        x = self.conv2(x) 
        x = self.conv3(x)
        x = self.conv4(x)
        return self.conv5(x)
def get_model():
    device = args.device
    netG = Generator(args.nz, args.ngf, args.channels).to(device)
    netG.apply(weights_init)
    netD = Discriminator(args.channels, args.ndf).to(device) 
    netD.apply(weights_init)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas = (args.beta1,0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas = (args.beta1,0.999))
    criterion = nn.BCELoss() 
    return netG, netD, optimizerD, optimizerG, criterion
def plot_figure(G_loss, D_loss, Dx, D_z1, D_z2, img_list):
    plt.figure(figsize = (10,5)) 
    plt.title('Generator and Discriminator Loss During Training') 
    plt.plot(G_loss, label = 'G') 
    plt.plot(D_loss, label = 'D') 
    plt.xlabel('iterations') 
    plt.ylabel('Loss') 
    plt.legend()
    plt.savefig('curve_folder/Loss.png')
    plt.figure(figsize = (10, 5)) 
    plt.title('Dx and Dz') 
    plt.plot(Dx, label = 'Dx') 
    plt.plot(D_z1, label = 'Dz1')
    plt.plot(D_z2, label = 'Dz2') 
    plt.xlabel('iterations')
    plt.ylabel('P') 
    plt.legend() 
    plt.savefig('curve_folder/D.png') 
    for i in range(len(img_list)):
        fig = plt.figure(figsize = (8,8)) 
        plt.axis('off') 
        plt.imshow(np.transpose(img_list[i], (1,2,0))) 
        plt.savefig('fake_image/fake_img' + str(i) + '.png')
def fit(epoches, dataloader, device, netG, netD, optimizerG, optimizerD):
    # Lists to keep track of progress 
    img_list = [] 
    G_losses = [] 
    D_losses = []
    Dx_list = [] 
    Dz1_list = []
    Dz2_list = []
    iters = 0
    real_label = 1.0
    fake_label = 0.0
    fixed_noise = torch.randn(64, args.nz, 1, 1, device = device)
    print('Starting Training Loop...') 

    for epoch in range(epoches):
        for i, data in enumerate(dataloader, 0):
            # Update D network: maximize log(D(x)) + log(1-D(G(z))) 
            ## Train with all real batch
            netD.zero_grad() 
            image_batch = data[0].to(device) 
            batch_size = image_batch.shape[0] 
            label = torch.full((batch_size,), real_label, dtype = torch.float,device = device) 
            #Forward pass real batch through D 
            output = netD(image_batch).view(-1) 
            errD_real = criterion(output, label)
            #Calculate gradients for D in backward pass 
            errD_real.backward() 
            D_x = output.mean().item() 

            ## Train with all-fake batch 
            # Generate batch of latent vectors  
            noise = torch.randn(batch_size,args.nz, 1, 1, device = device)  
            # Generate fake image batch with G 
            fake = netG(noise) 
            label.fill_(fake_label)
            # Classify all fake batch with D 
            # use detach() to make sure only change netD params 
            output = netD(fake.detach()).view(-1)  
            errD_fake = criterion(output, label) 
            errD_fake.backward() 
            D_G_z1 = output.mean().item() 
            errD = errD_real + errD_fake
            # update D 
            optimizerD.step() 

            #############################
            #(2) Update G network: maximize log(D(G(z))) 
            ############################# 
            netG.zero_grad() 
            label.fill_(real_label) # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all fake
            # batch through D
            output = netD(fake).view(-1) 
            # Calculate G's loss based on this output 
            errG = criterion(output, label) 
            # Calculate gradients for  G
            errG.backward() 
            D_G_z2 = output.mean().item() 
            # Update G 
            optimizerG.step() 

            # Output training stats 
            if i%50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\t Loss_G: %.4f\tD(x):%.4f\tD(G(z)): %.4f/%.4f' % (epoch, epoches, i,
                    len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                # Save Losses for plotting later
                G_losses.append(errG.item()) 
                D_losses.append(errD.item()) 
                Dx_list.append(D_x) 
                Dz1_list.append(D_G_z1)
                Dz2_list.append(D_G_z2)
                # Check how the generator is doing by saving G's output on
                # fixed noise
                if iters % 50 == 0:
                    with torch.no_grad():
                        fake = netG(fixed_noise).detach().cpu()
                    img_list.append(vision_utils.make_grid(fake[:64], padding=2,
                        normalize = True)) 
                iters += 1
                    
    plot_figure(G_losses, D_losses, Dx_list, Dz1_list, Dz2_list, img_list)

if __name__ == '__main__':
    manualSeed = 999
   # print('Random Seed: ', manualSeed) 
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    # argparse 
    parser = argparse.ArgumentParser(description = 'Train DCGAN') 
    parser.add_argument('--root', type = str, default = 'data', 
                        help = 'Root directory for dataset') 
    parser.add_argument('--device', type = str, default= 'cuda:0',
                        help = 'Choose the training device')
    parser.add_argument('--workers', type = int, default = 2, 
                        help = 'Number of workers for dataloader')
    parser.add_argument('--batch_size', type = int, default = 128, 
                        help = 'Batch size during training')
    parser.add_argument('--image_size', type = int, default = 64,
                        help = 'Spatial size of training images')
    parser.add_argument('--channels', type = int, default = 3,
                        help = 'Number of channels in the training images')
    parser.add_argument('--nz', type = int, default = 100,
                        help = 'Size of generator input')
    parser.add_argument('--ngf', type = int, default = 64,
                        help = 'Size of feature maps in generator') 
    parser.add_argument('--ndf', type = int, default = 64,
                        help = 'Size of feature maps in discriminator') 
    parser.add_argument('--epoches', type = int, default = 10,
                        help = 'Number of training epoches') 
    parser.add_argument('--lr', type = float, default = 2e-4,
                        help = 'Learning rate for optimizer') 
    parser.add_argument('--beta1', type = float, default = 0.5,
                        help = 'Beta1 hyperparam for Adam optimizer') 
    args = parser.parse_args() 
    dataloader = load_data(args.root, args.image_size, args.batch_size,
                           args.workers)
    netG, netD, optimizerD, optimizerG, criterion = get_model()
    
    fit(args.epoches, dataloader,args.device, netG, netD, optimizerG, optimizerD)
```
最后附上一些结果图：
**celeb-A faces Dataset**
![Loss](https://raw.githubusercontent.com/xuejy19/xuejy19.github.io/source/Img/Loss.png) 
![fake_image](https://raw.githubusercontent.com/xuejy19/xuejy19.github.io/source/Img/fake_image.png)

**MNIST** 
![Loss](https://raw.githubusercontent.com/xuejy19/xuejy19.github.io/source/Img/Loss_mn.png) 
![fake_msist](https://raw.githubusercontent.com/xuejy19/xuejy19.github.io/source/Img/mnist_gan.png)

网络结构比较简单，真正在使用时应当需要根据任务来进行调整，参数初始化以及部分超参数选择，**沃兹基**曾经说过：
> 深度神经网络理论学习是一门学问，代码实现又是一门学问，网络调参更是一门学问，想把神经网络搞掂，三种学问缺一不可。

GAN目前还面临很多问题，比如训练困难，在图像处理之外的领域效果并不理想，同时在理论支撑方面也不够扎实，如果想进一步学习GAN，可以阅读下面几篇文章：
- [这份攻略帮你「稳住」反复无常的 GAN](https://zhuanlan.zhihu.com/p/56943597)
- [海量案例！生成对抗网络（GAN）的18个绝妙应用](https://zhuanlan.zhihu.com/p/75789936)


