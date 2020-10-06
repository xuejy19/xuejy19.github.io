---
title: SVM补充
date: 2020-10-05 20:14:57
tags: SVM 
categories: 统计学习 
toc: true 
mathjax: true 
---
之前章节关于[支持向量机](https://soundofwind.top/2020/08/12/svm/)的介绍已经比较详细了，最近发现有一些部分细节理解差一些意思，同时少了一部分内容，今天就在这里补充下，该部分章节按照以下结构组织:
- 优化问题等价 
- 从硬间隔$svm$到软间隔$svm$ 
- $svm$应用于多分类问题 
- 支持向量回归$svr$ 

### 优化问题等价 
在求解优化问题时，我们往往考虑会将优化问题进行等价转换，以使得优化问题的求解更加容易，在进行极大似然估计时我们转化为求解对数似然函数极大就是一个典型的例子。这部分就系统的总结一下如何进行优化问题的等价转换,这里首先给出等价优化问题的定义:
> **等价优化问题:** 对于两个优化问题，如果从一个优化问题的解可以很容易得到另一个优化问题的解，反之亦然，则称两个优化问题是等价的。 

一般优化问题的形式为:
$$
    \begin{aligned}
        &\min \quad f_0(x)  \\
        & \ s.t. \quad f_i(x) \leq 0,i=1,2,\dots,m \\
        &   \qquad \ \;  h_i(x)  = 0 , i =1,2,\dots, p
    \end{aligned}
$$
下面给出将其转化为一个等价优化问题的几种转化思路。 
#### 变量替换
假设存在函数$\phi: R^n \rightarrow R^n$, 并且是一个\textbf{双射}(one-map-one),同时该函数的值域可以覆盖原优化问题的定义域$\mathcal{D}$，即$\mathcal{D} \subseteq \phi(\rm{dom} \phi)$,此时定义函数$\tilde{f}_i, \tilde{h}_i$:
$$
\tilde{f}_{i}(z)=f_{i}(\phi(z)), \quad i=0, \ldots, m, \quad \tilde{h}_{i}(z)=h_{i}(\phi(z)), \quad i=1, \ldots, p
$$
则可以得到优化问题: 
$$
\begin{array}{ll}
\operatorname{minimize} & \tilde{f}_{0}(z) \\
\text { subject to } & \tilde{f}_{i}(z) \leq 0, \quad i=1, \ldots, m \\
& \tilde{h}_{i}(z)=0, \quad i=1, \ldots, p
\end{array}
$$
$x$与$z$之间的关系为$x = \phi(z)$, 假设$x$是原始优化问题的解，则$z = \phi^{-1}(x)$便是上述优化问题的解，同理，如果$z$是上述优化问题的解，则$x = \phi(z)$就是原始优化问题的解。
