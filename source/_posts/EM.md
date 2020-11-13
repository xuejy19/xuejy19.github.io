---
title: EM
date: 2020-07-30 16:01:11
tags: EM算法,隐变量,高斯混合模型
categories: 统计学习
toc: true
mathjax: true
---
在前面概率密度函数估计中，若概率模型的变量都是可观测变量，那么给定数据，便可以直接用极大似然估计法或者贝叶斯估计法来直接估计模型参数。但是当模型中含有隐变量时，便不能直接使用这些估计方法，**EM算法就是含有隐变量的概率模型参数的极大似然估计法**<!--more-->,本文便围绕EM算法展开，主要包含以下几部分：
- 算法引入
- 算法定义
- 算法导出
- 算法收敛性分析
- 算法应用-高斯混合模型(GMM)

### 算法引入
首先介绍一个使用EM算法的栗子，引出EM算法应用场景:
> 三硬币模型： 假设有三枚硬币A,B,C。这三枚硬币出现正面的概率分别为$\pi,p,q$。现进行如下掷币实验：首先掷出硬币A,根据其结果选出硬币B或硬币C，正面选硬币B,反面选硬币C；然后掷选出的硬币，掷硬币的结果，正面记做1，反面记做0,独立的重复$n$次实验，得到实验结果为$(x_1,\dots,x_n)$
> **问题**：根据实验结果估计三枚硬币参数(正面朝上概率)

这个问题的特殊性在于，我们只能够观察到最终实验结果，但是对于中间变量(硬币A的投掷结果)则没有记录，是隐变量，所以该问题是在有隐变量情况下的参数估计问题。下面首先尝试使用极大似然估计方法，做如下标记：
- 待估计参数$\theta = (\pi,p,q)$
- 观测变量$y$:最终掷硬币结果
- 隐含变量$z$:每次实验掷硬币A的结果

若直接采用最大似然估计，我们希望能够写出似然概率$P(y|\theta)$，但这个概率并不能够直接写出，需要引入隐含变量$z$，因此考虑将似然概率写成
$$
    \begin{aligned}
         P(y|\theta) &= \sum P(y,z|\theta) = \sum_{z} P(z|\theta) P(y|\theta,z) \\\\
         &= \pi p^y (1-p)^{1-y} + (1-\pi)q^y(1-q)^{1-y}
    \end{aligned}
$$
将观测数据记做$Y = (Y_1,\dots, Y_n)^T$,未观测数据记做$Z = (Z_1,\dots, Z_n)^T$,则似然函数可以写做：
$$
    L = P(Y|\theta) = \prod_{j=1}^n[\pi p^{y_j} (1-p)^{1-y_j} + (1-\pi)q^{y_j}(1-q)^{1-y_j}]
$$
对数似然函数则是:
$$
    logL = \sum_{j=1}^nlog[\pi p^{y_j} (1-p)^{1-y_j} + (1-\pi)q^{y_j}(1-q)^{1-y_j}]
$$
直接极大化对数似然函数并不能得到解析解，该问题只能通过EM算法迭代求解

### 算法推导
首先给出算法定义：
**EM算法**:
>   输入： 观测变量数据$Y$,隐变量数据$Z$，联合分布$P(Y,Z|\theta)$,条件分布$P(Z|Y,\theta$
>   输出： 模型参数$\theta$
1. 选择参数初始值$\theta^0$,开始迭代
2. E步：记$\theta^i$为第$i$次迭代参数$\theta$的估计值，在第$i+1$次迭代的E步，计算：

$$
    \begin{aligned}
        Q(\theta,\theta^i) &= E_Z[logP(Y,Z|\theta)|Y,\theta^i] \\\\
            &= \sum_Z log P(Y,Z|\theta) P(Z|Y,\theta^i)
    \end{aligned}
$$

3. M步： 求使得$Q(\theta,\theta^i)$极大化的$\theta$,确定第$i+1$次迭代的参数的估计值$\theta^{i+1}$ 
 $$ \theta^{i+1} = argmax_{\theta}Q(\theta,\theta^i)$$
4. 重复第2,3步，直到算法收敛

从形式上来看，$Q$函数与极大似然函数其实是一样的，不同之处在于$P(Z|Y,\theta^i)$中使用的是参数迭代估计值，用该迭代估计值代替真值，$Q$函数可以看作是：
> 完全数据的对数似然函数$logP(Y,Z|\theta)$关于在给定观测数据$Y$和当前参数$\theta^i$下对未观测数据$Z$的的条件概率分布$P(Z|Y,\theta^i)$的期望。

我们这样使用$Q$函数来近似优化最大似然函数，需要证明这样的迭代优化是有效的

### 算法导出
对数似然形式前面已经讨论过，形式为：
$$
    \begin{aligned}
        L(\theta) &= log P(Y|\theta) = log \sum_Z P(Y,Z|\theta) \\
        &= log(\sum_Z P(Y|\theta,Z) P(Z|\theta))
    \end{aligned}
$$
直接极大化对数似然函数困难之处在于两点：
- 含有隐变量$Z$
- 对数似然函数是和的对数的形式

假设在第$i$次迭代后参数$\theta$估计值为$\theta^i$,新的估计值是$\theta$，我们期望新的估计值$\theta$能够使得对数似然函数增大，因此，考虑计算两者的差：
$$
    \begin{aligned}
        L(\theta) - L(\theta^i) &= log(\sum_Z P(Y|\theta,Z) P(Z|\theta)) - logP(Y|\theta^i) 
        \\\\
        &= log(\sum_Z P(Z|Y,\theta^i) \frac{P(Y|Z,\theta)P(Z|\theta)}{P(Z|Y,\theta^i)}) - logP(Y|\theta^i) 
        \\\\
        &\geq \sum_Z P(Z|Y,\theta^i) log \frac{P(Y|Z,\theta)P(Z|\theta)}{P(Z|Y,\theta^i)} - logP(Y|\theta^i)  (Jensen不等式)  
        \\\\
        &= \sum_Z P(Z|Y,\theta^i) log \frac{P(Y|Z,\theta)P(Z|\theta)}{P(Z|Y,\theta^i)P(Y|\theta^i)}
    \end{aligned}
$$
令 
$$
    B(\theta,\theta^i) = L(\theta^i) + \sum_Z P(Z|Y,\theta^i) log \frac{P(Y|Z,\theta)P(Z|\theta)}{P(Z|Y,\theta^i)P(Y|\theta^i)}
$$
则有：
$$
    L(\theta) \geq B(\theta,\theta^i)
$$
至此，我们找到了对数似然函数的一个下界，同时在点$\theta^i$处，有：
$$
    L(\theta^i) = B(\theta^i,\theta^i)
$$
因此若我们可以找到$\theta^{i+1}$,使得$B(\theta^{i+1},\theta^i)>B(\theta^i,
\theta^i)$,则有：
$$
    L(\theta^{i+1}) \geq B(\theta^{i+1},\theta^i) > B(\theta^i,\theta^i) = L(\theta^i)
$$
因此我们可以考虑优化$B(\theta,\theta^i)$，从而变相使原始似然函数增大。接下来考虑$B(\theta,\theta^i)$中需要优化的具体是哪一项:
$$
    \begin{aligned}
        \theta^{i+1} &= argmax(L(\theta^i) +  \sum_Z P(Z|Y,\theta^i) log \frac{P(Y|Z,\theta)P(Z|\theta)}{P(Z|Y,\theta^i)P(Y|\theta^i)})  \\\\
        &= argmax  \sum_Z P(Z|Y,\theta^i) log \frac{P(Y|Z,\theta)P(Z|\theta)}{P(Z|Y,\theta^i)P(Y|\theta^i)} \\\\
        &= argmax \sum_Z P(Z|Y,\theta^i) log(P(Y|Z,\theta) P(Z|\theta)) \\\\
        &= argmax \sum_Z P(Z|Y,\theta^i) log P(Y,Z|\theta) \\\\
        &= argmax Q(\theta,\theta^i)
    \end{aligned}
$$
<div align=center>

![EM](https://raw.githubusercontent.com/xuejy19/Images/master/EM.png)
</div>

图中下方曲线为$B(\theta,\theta^i)$，上方曲线为$L(\theta)$,在$\theta^n$处，两者相等，当对$B(\theta,\theta^n)$进行优化时，似然函数也得到了优化，同样，这种迭代算法只能够保证每次迭代都会使得似然函数增加，但并不能保证为全局最优。
### 算法收敛性分析
该部分只给出一些结论，证明过程略过
> **定理1**:设$P(Y|\theta)$为观测数据的似然函数，$\theta^i$为EM算法得到的参数估计序列，$P(Y|\theta^i)$为对应的似然函数序列，则$P(Y|\theta_i)$是单调递增的，即：
    $$    
    P(Y|\theta^{i+1}) \geq P(Y|\theta^i) 
    $$
> **定理2**:设$L(\theta) = log P(Y|\theta)$为观测数据的对数似然函数，$\theta^i$为EM算法得到的参数估计序列，$L(\theta^i)$为对应的对数似然函数序列 
>1. 如果$P(Y|\theta)$有上界,则$L(\theta^i) = log P(Y|\theta^i)$ 收敛到某一值$L^*$
>2. 在函数$Q(\theta,\theta^{'})$与$L(\theta)$满足一定条件下，由EM算法得到的参数估计序列$
  \theta^i$的收敛点$\theta^*$是$L(\theta)$的稳定点

### 算法应用-高斯混合模型(GMM)
首先给出高斯混合模型定义:
> 高斯混合模型是指具有如下形式的概率分布模型：
$$
    P(y|\theta) = \sum_{k=1}^K \alpha_k \phi(y|\theta_k)
$$
其中,$\alpha_k$是系数，$\alpha_k \geq 0$,$\sum \alpha_k = 1$;$\phi(y|\theta_k)$是高斯概率密度函数，$\theta_k = (\mu_k,\sigma_k^2)$,称为第$k$个分模型

高斯模型是描述数据分布的一种常见模型，但在有时候单一的高斯模型并不足以对数据进行描述，因此便考虑该数据分布有没有可能用多个高斯模型来描述，即任意一个数据点可能来自于某一个分模型。假设观测数据$y_1,y_2,\dots,y_N$由高斯混合模型生成：
$$
      P(y|\theta) = \sum_{k=1}^K \alpha_k \phi(y|\theta_k)
$$
其中$\theta = (\alpha_1,\dots,\alpha_K;\theta_1,\dots,\theta_K)$,我们现在期望利用观测数据将这些参数估计出来。

#### 隐变量定义
首先来看对于这样一个模型，隐变量是什么，数据是这样产生的：首先依概率$\alpha_k$选择第$k$个高斯分布分模型，然后依第$k$个分模型的概率分布生成观测数据$y$。观测数据$y$是已知的，但是$y$究竟来自于哪个分模型是未知的，记该变量为隐变量$\gamma_{jk}$,其定义如下：
$$
    \begin{aligned}
        \gamma_{jk} &= \begin{cases}
            1,& 第j个观测来自第k个分模型 \\
            0,& other
        \end{cases}   
        \\\\
        j &= 1,\dots,N;k = 1,\dots,K
    \end{aligned}
$$
目前，我们的观测数据为$y_j$,未观测数据是$\gamma_{jk}$,完全数据是：
$$
    (y_j,\gamma_{j_1},\dots,\gamma_{jk}),j=1,\dots,N
$$
于是，可以写出完全数据的对数似然函数：

$$
    \begin{aligned}
         P(y,\gamma|\theta) &= \prod_{j=1}^N P(y_j,\gamma_{j1},\dots,\gamma_{jk}|\theta) \\\\
         &= \prod_{k=1}^K \prod_{j=1}^N [\alpha_k \phi(y_j|\theta_k)]^{\gamma_{jk} } \\\\
         &= \prod_{k=1}^K \alpha_k^{n_k} \prod_{j=1}^N [\alpha_k \phi(y_j|\theta_k)]^{\gamma_{jk} } \\\\
         &= \prod_{k=1}^K \alpha_k^{n_k} \prod_{j=1}^N [\frac{1}{\sqrt{2\pi}\sigma_k} exp(-\frac{(y_j-\mu_k)^2}{2\sigma_k^2})]^{\gamma_{jk}}
    \end{aligned}
$$

公式中$n_k = \sum_{j=1}^N \gamma_{jk},\sum_{k=1}^K n_k = N$,由此，便可以写出完全数据的对数似然函数：

$$
 logP(y,\gamma|\theta) = \sum_{k=1}^K \{ n_klog\alpha_k + \sum_{j=1}^N \gamma_{jk}[log(\frac{1}{\sqrt{2\pi}})-log\sigma_k - \frac{1}{2\sigma_k^2} (y_j-\mu_k)^2]\}
$$

**E步**：其实就是要在已知上次迭代参数估计值$\theta$和观测值$y$的情况下将$\gamma_{jk}$估计出来，然后代回完全数据的对数似然函数，下面就求解下$E(\gamma_{jk}|y,\theta)$:

$$
\begin{aligned}
    \hat{\gamma}_{jk} &= E(\gamma_{jk}|y,\theta) = P(\gamma_{jk}|y,\theta)\\\\
    &= \frac{P(\gamma_{jk = 1,y|\theta})}{\sum_{k=1}^K P(\gamma_{jk}=1,y_j|\theta)} \\\\
    &= \frac{P(y_j|\gamma_{jk}=1,\theta)P(\gamma_{jk}=1|\theta)}{\sum_{k=1}^K P(y_j|\gamma_{jk}=1,\theta)P(\gamma_{jk}=1|\theta)} \\\\
    &= \frac{\alpha_k \phi(y_j|\theta_k)}{\sum_{k=1}^K \alpha_k \phi(y_j|\theta_k)}
\end{aligned}
$$

将该估计值代回完全数据的对数似然函数，便得到了$Q$函数：

$$
    Q(\theta,\theta^i) =  logP(y,\gamma|\theta) = \sum_{k=1}^K \{ n_klog\alpha_k + \sum_{j=1}^N \hat{\gamma}_{jk}[log(\frac{1}{\sqrt{2\pi}})-log\sigma_k - \frac{1}{2\sigma_k^2} (y_j-\mu_k)^2]\}
$$

**M步**：迭代的M步是求函数$Q(\theta,\theta^i)$对$\theta$的极大值，即求新一轮迭代的模型参数:

$$
    \theta^{i+1} = \arg max_{\theta} Q(\theta,\theta^i)
$$

该步只需分别对$\alpha_K,\mu_k,\sigma_k$求偏导，令偏导等于0,注意在求解$\alpha_k$时需要利用$\alpha_k$天然的约束条件$\sum_{k=1}^K \alpha_k = 1$,最终求解结果为:

$$
\begin{aligned}
    \hat{\mu}_k &= \frac{\sum_{j=1}^N \hat{\gamma}_{jk}y_j}{\sum_{j=1}^N \hat{\gamma}_{jk}} \\\\
    \hat{\sigma}_k^2 &= \frac{\sum_{j=1}^N \hat{\gamma}_{jk}(y_j - \mu_k)^2}{\sum_{j=1}^N \hat{\gamma}_{jk}} \\\\
    \hat{\alpha}_k &= \frac{n_k}{N} = \frac{\sum_{j=1}^N \hat{\gamma}_{jk}}{N}
\end{aligned}
$$
#### 总结
该部分就高斯混合模型的EM算法进行总结
- 取参数的初始值开始迭代
- E步：根据当前模型参数，计算分模型$k$对观测数据$y_j$的响应度
- M步：计算新一轮迭代的模型参数 
- 重复直到算法收敛