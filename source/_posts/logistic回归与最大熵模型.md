---
title: logistic回归与最大熵模型
date: 2020-08-23 15:16:31
tags: logistic回归, 最大熵模型, 优化学习
categories: 统计学习
toc: true
mathjax: true
---
在学习李航老师《统计学习》条件随机场章节时，对于学习算法感到有些陌生，后来发现在书中第六章“logistic回归与最大熵模型”有过一些介绍，因此本章节便总结一下相关知识，其中logistic回归模型做简要介绍，重点放在最大熵模型的学习算法上，本文按照如下结构组织：
- logistic回归
- 最大熵模型
<!--more-->
### logistic回归
#### logistic分布
首先介绍下logistic分布：
> **logistic分布**:设$X$是连续变量，$X$服从logistic分布是指$X$具有下列分布函数和密度函数：
> $$
    \begin{aligned}
        F(x) &= P(X \leq x) = \frac{1}{1 + e^{-\frac{x-\mu}{\gamma}}} \\
        f(x) &= F'(x) = \frac{e^{-\frac{x-\mu}{\gamma}}}{\gamma(1+e^{-\frac{x-\mu}{\gamma}})^2}
    \end{aligned}
> $$
> 式中，$\mu$为位置参数，$\gamma>0$为形状参数 

对于logistic分布，其概率密度函数和分布函数如下图所示：
![logistic](https://raw.githubusercontent.com/xuejy19/xuejy19.github.io/source/Img/logistic.jpeg)
分布函数的图形是一条$S$形曲线，该曲线以点$(\mu,\frac{1}{2})$为中心对称，即满足:
$$
    F(-x+\mu) - \frac{1}{2} = -F(x+\mu) +\frac{1}{2}
$$
曲线在中心增长速度较快，在两端增长速度慢，形状参数$\gamma$的值越小，曲线在中心附近增长得越快。
#### 二项logistic回归模型
二项logistic回归模型是一种二分类模型，由条件概率分布$P(Y|X)$表示，形式为参数化的logistic分布，随机变量$X$的取值为实数，随机变量$Y$取值为1或0，下面给出logistic回归模型定义：
> **logistic回归模型**：二项logistic回归模型是如下条件概率分布：
> $$
    \begin{aligned}
        P(Y=1|x) &= \frac{exp(w \cdot x + b)}{1 + exp(w \cdot x + b)} \\
        P(Y=0|x) &= \frac{1}{1 + exp(w \cdot x + b)} 
    \end{aligned}
> $$
> 其中，$x \in R^n$是输入，$Y \in \{0,1\}$是输出，$w \in R^n,b \in R$是参数。

若我们已经有了logistic模型，则判断样本点$x$属于哪一类只需要比较两个概率值的大小，下面给出对数几率的定义:
> **对数几率**：一个事件的几率是指该事件发生的概率与该事件不发生的概率的比值，如果一个事件发生的概率是$p$,那么该事件的几率是$\frac{p}{1-p}$,该事件的对数几率或logit函数是:
> $$
    logit(p) = log \frac{p}{1-p}
> $$

对于logistic回归而言，事件$(Y=1|X)$的对数几率为：
$$
    log \frac{P(Y=1|x)}{1 - P(Y=1|x)} = w \cdot x
$$
这说明，logistic回归模型可以看做是用线性回归模型去逼近真实标记的对数几率。因此logistic回归模型也被称为对数几率回归，可以看作是经过了如下的映射环节，将$R^n$空间中的$x$映射到了其标记的概率：
$$
    x \in R^n  \rightarrow w \cdot x \in R \rightarrow P(Y=1|x) \in [0,1]
$$

#### 模型参数估计
对于logistic回归模型，需要估计的参数是$w,b$,可以考虑将它们合并为一个$n+1$维向量:
$$
    w = [w;b]
$$
同时，给输入向量加一维，表示为:
$$
    x = [x;1]
$$
这样logistic回归模型就可以表示成更加紧凑的形式:
$$
    \begin{aligned}
        P(Y=1|x) &=  \frac{exp(w \cdot x) }{1+exp(w \cdot x)} \\
        P(Y=0|x) &=  \frac{1}{1 + exp(w \cdot x)}
    \end{aligned}
$$
logistic模型参数学习属于有监督的模型学习问题范畴，利用极大似然法就可以解决，设：
$$
    P(Y=1|x) = \pi(x), \quad P(Y=0|x) = 1- \pi(x)
$$
似然函数可以写成:
$$
    \prod_{i=1}^N [\pi(x_i)]^{y_i} [1 - \pi(x_i)]^{1-y_i}
$$
对数似然函数为:
$$
    \begin{aligned}
        L(w) &= \sum_{i=1}^N [y_i log \pi(x_i) + (1-y_i)log(1-\pi(x_i))] \\
            &= \sum_{i=1}^N [y_i log \frac{\pi(x_i)}{1 - \pi(x_i)} + log(1-\pi(x_i))] \\
            &= \sum_{i=1}^N [y_i(w \cdot x) - log(1 + exp(w \cdot x))]
    \end{aligned}
$$
对$L(w)$求极大值，得到$w$的估计值，常用梯度下降法及拟牛顿法求解(后面专开一章讲解)。
#### 多项logistic回归
上面介绍的是二项logistic模型，适用于二分类，可以将其推广为多项logistic分类模型，用于多类分类。假设离散型随机变量$Y$的取值集合是$\{1,2,\dots,K\}$,那么多项logistic回归的模型是:
$$
    \begin{aligned}
        P(Y=k|x) &= \frac{\exp(w_k \cdot x)}{1 + \sum_{k=1}^{K-1} \exp(w_k \cdot x)} \\
        P(Y=K|x) &= \frac{1}{1 + \sum_{k=1}^{K-1} \exp(w_k \cdot x)}
    \end{aligned}
$$
这里$x \in R^{n+1},w_k \in R^{n+1}$。二项logistic回归的参数估计法也可以推广到多项logistic回归

### 最大熵模型
最大熵模型，是由最大熵原理推导实现。本部分按照以下结构组织:
- 最大熵原理
- 最大熵模型推导
- 最大熵模型学习

#### 最大熵原理
最大熵原理是概率模型学习的一个准则，最大熵原理认为：
> 学习概率模型时，在所有可能的概率模型中，熵最大的模型是最好的模型。通常用约束条件来确定概率模型的集合，所以，最大熵原理也可以表述为在满足约束条件的模型集合中选取熵最大的模型。

我们首先来复习下信息论中关于熵的定义：
> 在信息论与概率统计中，熵是表示**随机变量不确定性**的度量，设$X$是一个取有限个值的离散随机变量，其概率分布为:
> $$
    P(X = x_i) = p_i, \quad i =1,2,\dots,n
> $$
则随机变量$X$的熵定义为:
$$
    H(X) = - \sum_{i=1}^n p_i \log p_i 
$$

若$p_i = 0$,则定义$0\log 0 =0$,式中的对数一般取以2为底或者以$e$为底，这时熵的单位被称作比特或纳特，由定义可知，熵只依赖于$X$的分布，而与$X$的取值无关，因此也可将$X$的熵记作$H(p)$,熵越大，随机变量的不确定性越大。因为$p_i \leq 0$,由$H(p)$函数性质(凹函数，应用拉格朗日乘子法可求最大值)可知:
$$
    0 \leq H(p) \leq \log n
$$
当且仅当$X$是均匀分布时右侧等号成立,也就是说，当$X$服从均匀分布时，熵最大，该随机变量的不确定性最大。下面给出条件熵的定义:
> 设有随机变量$(X,Y)$,其联合概率分布为：
$$
    P(X = x_i,Y = y_i) = p_{ij}\quad i = 1,2,\dots,n;\quad j = 1,2,\dots,m
$$
条件熵$H(Y|X)$表示在已知随机变量$X$条件下随机变量$Y$的不确定性。随机变量$X$给定的条件下随机变量$Y$的条件熵$H(Y|X)$,定义为$X$给定条件下$Y$的条件概率分布的熵对$X$的数学期望：
$$
    H(Y|X) = \sum_{i=1}^n p_i H(Y|X = x_i)
$$
当熵和条件熵中的概率由数据估计得到时，所对应的熵与条件熵分别称为经验熵和经验条件熵

直观的，最大熵原理认为要选择的概率模型首先必须满足已有的事实，即约束条件。在没有更多信息情况下，那些不确定的部分都是“等可能的”,最大熵原理通过熵的最大化来表示等可能性。下面通过一个简单的例子来介绍最大熵原理：
> 🌰：假设随机变量$X$有五个取值$\{A,B,C,D,E \}$,要估计取各个值的概率$P(A),P(B),P(C),P(D),P(E)$
> - 不提供任何额外约束，则只有这些概率的固有约束
> $$
    P(A) + P(B) + P(C) + P(D) + P(E) = 1
> $$
> 因为没有任何额外的信息，则根据最大熵原理，只能够认为该概率分布为均匀分布：
> $$
    P(A) = P(B) = P(C) = P(D) = P(E) = \frac{1}{5}
> $$
> - 从一些先验知识中得到了一些对概率值的约束条件，如$P(A) + P(B) = \frac{3}{10}$,满足约束的概率分布仍旧有无穷多个，在缺少其他信息的情况下，可以认为$A$和$B$是等概率的，$C,D,E$等概率。

实际上，最大熵原理就是为我们提供了一种选择最优概率模型的准则(保守)

#### 最大熵模型的定义
最大熵原理是统计学习的一般原理，将它应用到分类得到最大熵模型。
假设分类模型是一个条件概率分布$P(Y|X)$,$X \in \mathcal{X} \subseteq R^n$表示输入，$Y \in \mathcal{Y}$表示输出，$\mathcal{X}$和$\mathcal{Y}$分别是输入和输出的集合，这个模型表示的是对于给定的输入$X$，以条件概率$P(Y|X)$输出$Y$。
给定一个训练数据集:
$$
    T = \{  (x_1,y_1), (x_2,y_2), \dots, (x_N,y_N)\}
$$
学习的目标是应用最大熵模型选择最好的分类模型。首先考虑模型应该满足的条件，即该数据集为模型添加了哪些约束，给定训练数据集后，我们可以确定联合分布$P(X,Y)$的经验分布和边缘分布$P(X)$的经验分布，分别以$\tilde{P}(X,Y)$和$\tilde{P}(X)$表示:
$$
    \begin{aligned}
        \tilde{P}(X= x,Y=y) &= \frac{\nu(X = x,Y = y)}{N} \\
        \tilde{P}(X = x) &= \frac{\nu(X = x)}{N} 
    \end{aligned}
$$
式中$\nu(\cdot)$是频数统计函数，$N$表示训练样本容量。用特征函数$f(x,y)$描述输入$x$和输出$y$之间的某一个事实，其定义是:
$$
    f(x,y) = \begin{cases}
        1,  &x和y满足某一事实 \\
        0,  &else 
    \end{cases}
$$
特征函数$f(x,y)$关于经验分布$\tilde{P}(X,Y)$的期望值，用$E_{\tilde{P}}(f)$表示：
$$
    E_{\tilde{P}}(f) = \sum_{x,y} \tilde{P}(x,y) f(x,y)
$$
特征函数$f(x,y)$关于模型$P(Y|X)$i与经验分布$\tilde{P}(X)$的期望值，用$E_P(f)$表示：
$$
    E_P(f) = \sum_{x,y} \tilde{P}(x) P(y|x) f(x,y)
$$
如果模型能够获取训练数据中的信息，那么就可以假设这两个期望值相等，即：
$$
    E_P(f) = E_{\tilde{P}}(f)
$$
将这作为模型学习的约束条件，假如有$n$个特征函数$f_i(x,y),i=1,2,\dots,n$,那么就有$n$个约束条件，下面给出最大熵模型的定义:
> 最大熵模型:假设满足所有约束条件的模型集合为:
> $$
    C \equiv \{ P \in \mathcal{P} |E_P(f_i) = E_{\tilde{P}}(f_i), i = 1,2,\dots,n\} 
> $$
> 定义在条件概率分布$P(Y|X)$上的条件熵为：
> $$
    H(P) = -\sum_{x,y}  \tilde{P}(x)P(y|x) log P(y|x)
> $$
> 则模型集合$C$中条件熵$H(P)$最大的模型称为最大熵模型

从定义可以看出，最大熵模型是基于这样一种思想:
> 当我们需要对一个事件(变量)的概率分布进行预测时，最大熵原理告诉我们所有的预测应当满足全部已知的条件(约束)，而对未知的情况不要做任何主观假设，保留模型最大的不确定性。

#### 最大熵模型的学习
最大熵模型的学习过程就是求解最大熵模型的过程。最大熵模型学习可以形式化为约束最优化问题。
对于给定的训练数据集$T = \{ (x_1,y_1), (x_2,y_2),\dots, (x_N,y_N)\}$以及特征函数$f_i(x,y),i=1,2,\dots,n$,最大熵模型的学习等价于求解最优化问题:
$$
    \begin{aligned}
        &\max_{P \in C} \quad H(P) = -\sum_{x,y}  \tilde{P}(x)P(y|x) log P(y|x) \\
        &s.t. \quad E_P(f_i) = E_{\tilde{P}}(f_i),\quad i =1,2,\dots,n \\  
        & \quad \quad \sum_y P(y|x) =1
    \end{aligned}
$$
按照最优化问题的习惯，将求解最大值问题改为等价的求最小值问题:
$$
    \begin{aligned}
        &\min_{P \in C} \quad -H(P) = \sum_{x,y}  \tilde{P}(x)P(y|x) log P(y|x) \\
        &s.t. \quad E_P(f_i) - E_{\tilde{P}}(f_i) = 0,\quad i =1,2,\dots,n \\  
        & \quad \quad \sum_y P(y|x) =1
    \end{aligned}
$$
**该优化问题按照李航老师《统计学习》中的说法是满足强对偶性的，但书中只是一带而过，并没有进行详细说明，此处留坑，后面的推导就先默认该优化问题满足强对偶性，后面我搞懂了再来填坑**
***
**这个问题我又思考了一下，自变量是一个分布看似是一个很虚的东西，但它也不过是一个函数，若$X,Y$都是离散取值变量，那么这个条件概率分布只是一个有限长度的向量，而若$X,Y$取值连续，则该条件概率密度函数可以看作一个无限长的向量，这样来看原始待优化函数其实就是$\boldsymbol{x} \cdot \log \boldsymbol{x}$,这显然是一个凸函数，而约束条件也不过是待优化参数的线性组合，这显然是一个凸优化问题**
***
该优化问题共有$n+1$个等式约束，因此考虑引入拉格朗日乘子$w_0,w_1,\dots,w_n$,定义拉格朗日函数$L(P,w)$:
$$
    \begin{aligned}
        L(P,w) &= -H(P) + w_0 (1 - \sum_y P(y|x)) + \sum_{i=1}^n w_i (E_{\tilde{P}}(f_i) - E_P(f_i)) \\
        &= \sum_{x,y}  \tilde{P}(x)P(y|x) log P(y|x)  + w_0 (1 - \sum_y P(y|x)) \\
        &+ \sum_{i=1}^n w_i(\sum_{x,y} \tilde{P}(x,y) f(x,y) - \sum_{x,y} \tilde{P}(x) P(y|x) f(x,y))
    \end{aligned}
$$
原始优化问题的对偶问题写作：
$$
    \max_{w} \min_{P} L(P,w)
$$
首先求解对偶函数$\phi(w)$,因为$L(P,w)$关于$P$是凸函数，因此直接通过求导令导数为0即可：
$$
    \begin{aligned}
        \frac{ \partial L(P,w)}{\partial P(y|x)} &=  log P(y|x) +1 - w_0 - \sum_{i=1}^n w_i f_i (x,y) = 0 
    \end{aligned}
$$
在求导这一部分我不太认同李航老师书中的写法，因为$P(y|x)$本身可以看作是一个向量，因此求导相当于对该向量某个分量求导。由此可得$P(y|x)$的解析表达式:
$$
    P(y|x) = \exp(\sum_{i=1}^n w_i f_i(x,y) + w_0 -1) = \frac{\exp(\sum_{i=1}^n w_i f_i(x,y))}{\exp(1-w_0)}
$$
又因为有概率密度函数的固有约束$\sum_y P(y|x) = 1$,可得:
$$\exp(1-w_0) = \sum_y \exp(\sum_{i=1}^n w_i f_i(x,y))$$
因此，$P_w(y|x)$的解析表达式写作:
$$
    \begin{aligned}
        P_w(y|x) &= \frac{1}{Z_w(x)} \exp(\sum_{i=1}^n w_i f_i(x,y)) \\
        Z_w(x) &= \sum_y \exp(\sum_{i=1}^n w_i f_i(x,y))
    \end{aligned}
$$
至此，我们得到了对偶函数$\phi(w)$表达式：
$$
    \begin{aligned}
        \phi(w) &= \sum_{x,y} \tilde{P}(x,y) \sum_{i=1}^n w_i f_i(x,y) + \sum_{x,y} \tilde{P}(x)P_w(y|x)(\log P_w(y|x) - \sum_{i=1}^n w_i f_i(x,y)) \\
        &= \sum_{x,y} \tilde{P}(x,y) \sum_{i=1}^n w_i f_i(x,y) - \sum_{x,y} \tilde{P}(x)P_w(y|x) \log Z_w(x) \\
        &= \sum_{x,y} \tilde{P}(x,y) \sum_{i=1}^n w_i f_i(x,y) - \sum_x \tilde{P}(x) \log Z_w(x)
    \end{aligned}
$$
因此接下来便是求解对偶优化问题:
$$
    \max \phi(w)
$$
得到$w^\ast$,然后回代$P_w(y|x)$,得到最大熵模型表达式:
$$
    P_{w^\ast}(y|x)
$$
#### 极大似然估计
这一部分旨在证明一条性质:
> 最大熵模型求解过程中，对偶函数的极大化等价于最大熵模型的极大似然估计

首先写出对数似然函数的形式:
$$
    \begin{aligned}
         L_{\tilde{P}}(P_w) &= \log \prod_{x,y} P(y|x)^{\tilde{P}(x,y)} = \sum_{x,y} \tilde{P}(x,y) \log P(y|x) \\
         &= \sum_{x,y} \tilde{P}(x,y) \sum_{i=1}^n w_i f_i(x,)  - \sum_{x,y} \tilde{P}(x,y) \log Z_w(x) \\
         &= \sum_{x,y} \tilde{P}(x,y) \sum_{i=1}^n w_i f_i(x,y)  - \sum_{x} \tilde{P}(x) \log Z_w(x)
    \end{aligned}
$$
与对偶函数$\phi(w)$一致，这说明对对偶函数求最大等价于求最大熵模型的极大似然估计。
最大熵模型与logistic回归模型有类似的形式，它们又称为对数线性模型。模型学习就是对模型进行极大似然估计或者正则化的极大似然估计