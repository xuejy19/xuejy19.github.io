---
title: SVM
date: 2020-08-12 09:26:20
tags: 支持向量机
categories: 统计学习
toc: true
mathjax: true
---

这部分将介绍支持向量机(SVM)算法，该部分将按照以下几部分进行组织：
- 算法历史
- 线性可分支持向量机与硬间隔最大化
- 线性支持向量机与软间隔最大化
- 非线性支持向量机与核函数
- SVDD
<!--more-->

### 算法历史
支持向量机算法是在机器学习领域非常著名的算法，在神经网络算法得到普及之前，支持向量机算法可以说是统治了机器学习领域的半壁江山，该算法由Vapnik在贝尔实验室开发，根据Vapnik和Chervonekis提出的统计学习框架或VC理论，它提出了一种最可靠的预测方法（ 1974年）和瓦普尼克（Vapnik）（1982年，1995年）。给定一组训练示例，每个训练示例都标记为属于两个类别中的一个或另一个，则SVM训练算法会构建一个模型，该模型将新示例分配给一个类别或另一个类别，使其成为非概率 二进制 线性分类器（尽管方法例如Platt缩放存在以在概率分类设置中使用SVM）。SVM模型是将示例表示为空间中的点，并进行了映射，以使各个类别的示例被尽可能宽的明显间隙分开。然后，将新示例映射到相同的空间，并根据它们落入的间隙的侧面来预测属于一个类别。
除执行线性分类外，SVM还可以使用所谓的内核技巧有效地执行非线性分类，将其输入隐式映射到高维特征空间。

### 线性可分支持向量机与硬间隔最大化
对于线性可分数据，上一部分介绍的感知机算法便可以处理，但需要注意的是，感知机算法学习得到的超平面可能有无穷多个，那么这些超平面中哪个是最优的呢？支持向量机算法便回答了这一问题，该算法认为使两类样本点间隔最大的超平面是最优超平面。首先给出线性可分支持向量机的定义：
> {\ast}线性可分支持向量机{\ast}:给定线性可分训练数据集，通过间隔最大化或者等价地求解相应的凸二次规划问题学习得到的分离超平面为：
$$
    \omega^{\ast} \cdot x + b^{\ast} = 0
$$
以及相应的分类决策函数:
$$ f(x) = sign(\omega^{\ast} \cdot x + b^{\ast})$$

下面就结合下图来对支持向量机的算法思想进行阐述,图中红点与蓝点分别代表两个类别，我们现在期望找到一个超平面，使得两个类别的样本点都能够尽可能离该超平面距离远，也就是希望中间的margin尽可能大。
![SVM](https://raw.githubusercontent.com/xuejy19/xuejy19.github.io/source/Img/SVM.png)
这便是支持向量机的思想，下一步需要做的便是将这种想法翻译成一个数学问题，在这里首先引入两个间隔的概念：
> - 几何间隔：所谓几何间隔就是样本点$(x_i,y_i)$到超平面的实际距离，其计算公式为(此时样本点已经被超平面正确分类)：
$$ d = y_i \frac{\omega x_i + b}{||\omega||}$$
> - 函数间隔：不考虑几何间隔中$||\omega||$便得到了函数间隔：
$$ d = y_i(\omega x_i +b)$$

需要注意的是，函数间隔可以表示分类预测的正确性及确信度，在感知机学习中，我们便是使用函数间隔，这是由于数据线性可分，最终目标函数都会收敛到0，使用几何间隔与函数间隔最终效果并无不同(两种间隔符号一致)。但在支持向量机算法中，我们希望间隔最大，这时显然函数间隔是不够的，因为只要成比例的增大$\omega,b$，超平面并没有改变，但是函数间隔却便为两倍，因此在该问题中，我们应当使用几何间隔,至此支持向量机的思想可以写成如下优化问题的形式：
$$
    \begin{aligned}
        &max_{\omega,b} \quad \gamma  \\
        &s.t. \quad y_i \frac{\omega x_i +b}{||\omega||} \geq \gamma,\quad i = 1,2,\dots,N
    \end{aligned}
$$
上面是优化问题的原始形式，下面尝试对该优化问题进行等价转换以便于求解，首先根据几何间隔与函数间隔的关系，将优化问题转化为：
$$
    \begin{aligned}
         &max_{\omega,b} \quad \frac{\hat{\gamma}}{||\omega||}  \\
        &s.t. \quad y_i (\omega x_i +b) \geq \hat{\gamma},\quad i = 1,2,\dots,N
    \end{aligned}
$$
此时可以注意到函数间隔$\hat{\gamma}$并不影响该优化问题求解，因此考虑将$\hat{\gamma}$置为1，注意到最大化$\frac{1}{||\omega||}$等价于最小化$\frac{1}{2}||\omega||^2$,因此优化问题最终可以表述为:
$$
    \begin{aligned}
        &min_{\omega,b} \quad \frac{1}{2} ||\omega||^2 \\\\
        &s.t. \quad y_i (\omega x_i +b) \geq 1,\quad i = 1,2,\dots, N
    \end{aligned}
$$
这是一个典型的凸二次规划问题，运用凸优化知识可进行求解。

下面给出一个定理来说明最大间隔分离超平面唯一的定理：
> 定理：若训练数据集$T$线性可分，则可将训练数据集中的样本点完全分开的最大间隔分离超平面存在且唯一

#### 支持向量与间隔边界
首先给出支持向量的定义:
> 在线性可分情况下，训练数据集中样本点中与分离超平面距离最近的样本点的实例称为支持向量。

反映在优化问题中，支持向量满足约束条件的等号,支持向量在超平面:
$$
    \begin{aligned}
         H_1&: \omega \cdot x +b =1  \\
         H_2&: \omega \cdot x +b = -1
    \end{aligned}
$$
两个支撑超平面之间的距离称为间隔，该间隔距离为：
$$
    d = \frac{2}{||\omega||}
$$
在决定分离超平面时只有支持向量起作用，而其他实例点并不起作用。

#### 学习的对偶算法
在进行求解时，我们往往考虑不直接求解原问题，而是求解原问题的对偶问题, 又因为原问题是一个凸优化问题，同时满足Slater约束品性，因此考虑求解其对偶问题。首先写出Lagrange函数：
$$
    L(\omega,b,\alpha) = \frac{1}{2}||\omega^2|| - \sum_{i=1}^N \alpha_i y_i (\omega \cdot x_i +b) + \sum_{i=1}^N \alpha_i
$$
接下来求解对偶函数，对偶函数$g(\alpha)$表示为：
$$
    g(\alpha) = \min_{\omega,b} L(\omega,b,\alpha)
$$
因为$L(\omega,b,\alpha)$是凸函数，因此只需要对$\omega,b$求偏导，令偏导等于0即可：
$$
    \begin{aligned}
        \frac{\partial L(\omega,b,\alpha)}{\partial \omega} &= \omega - \sum_{i=1}^N \alpha_i y_i x_i = 0 \rightarrow \omega =  \sum_{i=1}^N \alpha_i y_i x_i  \\\\
        \frac{\partial L(\omega,b,\alpha)}{\partial b} &= -\sum_{i=1}^N \alpha_i y_i = 0 \rightarrow \sum_{i=1}^N \alpha_i y_i = 0
    \end{aligned}
$$
回代拉格朗日函数，可得对偶函数：
$$
    \begin{aligned}
          g(\alpha) &= \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_i x_i^T x_j - \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_i x_i^T x_j + \sum_{i=1}^N \alpha_i  \\\\
          &= -\frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_i x_i^T x_j + \sum_{i=1}^N \alpha_i
    \end{aligned}
$$
所以，对偶优化问题可以写做：
$$
    \begin{aligned}
        &\max_{\alpha} \quad -\frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_i x_i^T x_j + \sum_{i=1}^N \alpha_i \\\\
        &s.t. \quad \sum_{i=1}^N \alpha_i y_i =0;\alpha_i \geq 0,i=1,\dots,N
    \end{aligned}
$$
对于该问题的求解可以通过SMO(序列最小优化)算法进行求解，因为强对偶性成立,KKT条件满足：
$$
    \begin{aligned}
        &\nabla_{\omega} L(\omega^{\ast},b^{\ast},\alpha^{\ast}) = \omega^{\ast} - \sum_{i=1}^N \alpha_i^{\ast} y_i x_i = 0 \\
        &\nabla_{b} L(\omega^{\ast},b^{\ast},\alpha^{\ast}) = -\sum_{i=1}^N \alpha_i^{\ast} y_i = 0 \\
        &\alpha_i^{\ast} (y_i(\omega^{\ast}x_i+b^{\ast})-1) = 0 \\
       &y_i(\omega^{\ast} x_i +b^{\ast}) - 1 \geq 0 \\
        &\alpha_i^{\ast} \geq 0 
    \end{aligned}
$$
由此便可建立对偶问题最优解$\alpha^{\ast}$与原问题最优解$\omega^{\ast},b^{\ast}$之间的关系，首先至少有一个$\alpha_j^{\ast}>0$,这是因为若$\alpha^{\ast}$均为0，则$\omega^{\ast}$为0,这不是原始最优化问题的解，假设$\alpha_j^{\ast}>0$,则由互补松弛条件，此时有$y_j(\omega^{\ast} x_j +b^{\ast}) = 1$,而$\omega^{\ast}$表达式是知道的，由此可得$\omega^{\ast},b^{\ast}$表达式为：
$$
    \begin{aligned}
        \omega^{\ast} &= \sum_{i=1}^N \alpha_i^{\ast} y_i x_i \\
        b^{\ast} &=  y_j - \sum_{i=1}^N \alpha_i^{\ast} y_i(x_i\cdot x_j)
    \end{aligned}
$$
至此，我们便通过求解对偶问题得到了原始问题的最优解，得到了超平面方程以及判决方程：
$$
\begin{aligned}
    &\omega^{\ast} \cdot x + b^{\ast} = 0 \\
    &f(x) = sign(\omega^{\ast} \cdot x + b^{\ast})
\end{aligned}
$$
通过引入对偶问题，我们便可以从数学的角度导出支持向量:
> 训练数据集中对应于$\alpha_i^{\ast} >0$的样本点称为支持向量

### 线性支持向量机与软间隔最大化
#### 原始问题
前面介绍的算法是针对线性可分数据,那么对于大部分线性可分的数据集，支持向量机能否处理呢？答案是可以，但是需要对原始优化问题做适当变换,对于线性可分的数据，我们要求各个样本点到分类超平面的函数间隔都要大于1，对于不可分数据点，我们直观上的想法是考虑能否将该约束放松，因此我们考虑为每一个样本点引入一个松弛变量$\xi_i$,使得函数间隔加上松弛变量等于1。这样，约束条件变为:
$$
    y_i(\omega \cdot x_i +b) \geq 1 - \xi_i
$$
但需要注意的人，该松弛变量的引入是有代价的，需要在目标函数中进行惩罚，因此考虑在目标函数中加入惩罚项$\xi_i$，将目标函数改写为：
$$
    \frac{1}{2} ||\omega||^2  + C\sum_{i=1}^N \xi_i
$$
其中$C$是惩罚系数，$C$值大时对误分类惩罚增大。最小化该目标函数包含两层含义：使间隔尽量大；使误分类点尽量少。线性不可分支持向量机的学习问题变为如下凸二次规划问题：
$$
    \begin{aligned}
        &\min_{\omega,b,\xi} \quad \frac{1}{2} ||\omega||^2 + C \sum_{i=1}^N \xi_i \\\\
        &s.t. \quad y_i(\omega x_i +b) \geq 1- \xi_i,i=1,2,\dots,N \\\\
        &\qquad \xi_i \geq 0,i=1,2,\dots,N
    \end{aligned}
$$
对于该原始问题，可以证明$\omega$的解是唯一的，但$b$的解可能不唯一，而是存在一个区间。

#### 对偶问题
在该部分便不进行对偶问题推导了，步骤与上一部分一致，最终得到的对偶问题形式为：
$$
    \begin{aligned}
        &\max_{\alpha} \quad -\frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_i x_i^T x_j + \sum_{i=1}^N \alpha_i \\\\
        &s.t. \quad \sum_{i=1}^N \alpha_i y_i =0,i=1,\dots,N  \\\\
        & \qquad 0 \leq \alpha_i \leq C \quad i=1,2,\dots,N
    \end{aligned}
$$
可以看到引入松弛变量后对偶问题形式基本未发生变化，仅仅约束条件中$\alpha_i$范围变了。需要注意的是将原始问题转化为对偶问题的过程中，引入了两个Lagrange算子$\alpha_i,\mu_i$，但由于存在$
\alpha_i + \mu_i = C $以及$\alpha_i \geq 0, \mu_i \geq 0 $,因此将这些约束条件简化为$0 \leq \alpha_i \leq C$。
下面讨论如何根据对偶问题最优解求解原问题最优解，与硬间隔支持向量机不同，在该问题中，互补松弛条件有两个:
$$
    \begin{aligned}
        &\alpha_i^{\ast} (y_i(\omega^{\ast}x_i+b^{\ast})-1 + \xi_i) = 0  \\
        &\mu_i \xi_i = 0 
    \end{aligned}
$$
在求解$b^{\ast}$时，关键是要将支持向量确定出来，首先支持向量应当在支撑超平面上，这就要求$\alpha_i>0$,同时松弛间隔$\xi_i$应当为0，由第二个互补松弛条件可知$\mu_i$不能够为0，因此$\alpha_i$必须小于$C$,换句话说，$\alpha^{\ast} = C$所对应的点均在支撑超平面以内，若$\xi_i<1$,则仍能分类正确，若大于1，则就会分类错误。$\omega^{\ast}$和$b^{\ast}$表达式可以写做：
$$
    \begin{aligned}
        \omega^{\ast} &= \sum_{i=1}^N \alpha_i^{\ast} y_i x_i \\
        b^{\ast} &= y_j - \sum_{i=1}^N y_i \alpha_i^{\ast} (x_i \cdot x_j)
    \end{aligned}
$$

#### 非线性支持向量机与核函数
支持向量机是解决线性分类问题的一种有效方法，但并不能解决非线性分类问题，那么非线性的数据就不能用线性的方法解决么？ 直观上我们有这样一个方法，考虑降维度线性不可分的数据映射到高维度，使映射后的数据在高维度线性可分，由此便引出了本节的核函数。
#### 核函数
首先先举一个空间变换的栗子，设原空间为$\mathcal{X} \subset R^2,x = (x^1,x^2)^T$,新空间为$\mathcal{Z} \subset R^2,z = (z^1,z^2)^T$,定义从原空间到新空间的映射：
$$
    z = \phi(x) = ((x^1)^2,(x^2)^2)^T
$$
经过这样一个变换原空间中的椭圆：
$$
    w_1 (x^1)^2 + w_2 (x^2)^2 +b = 0
$$
变换为了新空间中的直线：
$$
    w_1 z^1 + w_2 z^2 + b = 0
$$
这样，原空间的非线性可分问题就变成了新空间的线性可分问题。从该问题出发，我们可以总结出用线性分类方法求解非线性分类问题的步骤：
- 首先使用一个变换将原空间中的数据映射到新空间
- 在新空间中使用线性分类学习方法从训练数据中学习分类模型

而在支持向量机中，达到这一目的的方法应用核技巧，下面首先给出核函数的定义：
> 核函数：设$\mathcal{X}$是输入空间，又设$\mathcal{H}$为特征空间(希尔伯特空间),如果存在一个从$\mathcal{X}$到$\mathcal{H}$的映射：
$$
    \phi(x):\mathcal{X} \rightarrow \mathcal{H}
$$
使得对所有$x,z \in \mathcal{X}$,函数$K(x,z)$满足条件:
$$
    K(x,z) = \phi(x) \cdot \phi(z)
$$
则称$K(x,z)$为核函数，$\phi(x)$为映射函数

核技巧的想法是，在学习与预测中只定义核函数$K(x,z)$,而不显式的定义映射函数$\phi$。同时，对于一个核$K(x,z)$,特征空间$\mathcal{H}$和映射函数$\phi$的取法并不唯一

#### 非线性支持向量机
在线性支持向量机的对偶问题中,无论是目标函数还是决策函数中均只包含实例的内积的形式，因此若用$K(x,y)$代替$x\cdot y$,实际上便是在对高维空间中的特征点设计线性分类器。那现在的关键问题便在于如何选择合适的核函数。

关于核函数这部分的理论我后面会另开一个章节进行介绍学习，在这一部分便直接给出一些常用的核函数：
- 多项式核函数:
$$ K(x,z) = (x\cdot z +1)^p$$
- 高斯核函数(径向基函数)
$$ K(x,z) = exp(- \frac{||x-z||^2}{2\sigma^2})$$

需要注意的是，在引入核函数后，又多引入了超参数，比如高斯核函数中的$\sigma$,而这些超参数的选择也会对分类效果产生影响。

### 一类分类器设计
在前面介绍的支持向量机算法中，针对的都是二分类情况，但当我们做故障检测时，往往就只有大量的正常样本点，现在我们需要根据这些正常样本点来设计一个分类器以对未出现的异常样本点进行检测。在这一部分介绍两种算法：
- OCSVM
- SVDD

#### OCSVM
在OCSVM中，将坐标原点作为唯一的异常点,最大化最优超平面到远点的距离，若记$\rho$为从原点到超平面的距离，$\xi_i$为松弛变量，则优化问题可以写做：
$$
    \begin{aligned}
        &\max_{\omega,\rho,\xi_i} \rho - C\sum_{i=1}^N \xi_i \\
        & s.t. \quad \omega \cdot x_i \geq \rho - \xi_i,\xi_i \geq 0,||\omega|| = 1
    \end{aligned}
$$

#### SVDD
SVDD的基本思想则是，对于$n$维空间中的数据点集，寻找最小超球面，使其成为所有点的边界，其对应优化问题的原问题为:
$$
    \begin{aligned}
         &\min_{a,R,\xi} \quad R^2 + C\sum_{i=1}^N \xi_i \\
         &s.t. \quad ||x_i -a||^2 \leq R^2 + \xi_i \\
         &\qquad \quad \xi_i \geq 0 
    \end{aligned}
$$