---
title: 隐马尔可夫模型
date: 2020-08-17 08:22:44
tags: 概率模型，HMM
categories: 统计学习
#img: https://raw.githubusercontent.com/xuejy19/xuejy19.github.io/source/thumbnail/HMM.png
toc: true
mathjax: true
---
在这一部分，我将对隐马尔可夫模型(HMM)做简要介绍,该章节分为以下几个部分进行组织:

- 基本概念
- 概率计算算法
- 学习算法
- 预测算法

<!--more-->

### 基本概念

首先给出隐马尔可夫模型的定义：

> **隐马尔可夫模型**:隐马尔可夫模型是关于时序的概率模型，描述由一个隐藏的马尔可夫链随机生成不可观测的随机状态序列,再由各个状态生成一个观测从而产生观测随机序列的过程。隐藏的马尔可夫链随机生成的状态序列称为状态序列；每个状态生成一个观测，而由此产生的观测的随机序列，称为观测序列，序列的每一个位置又可以看作是一个时刻。

隐马尔可夫模型的形式定义如下:设$Q$是所有可能的状态的集合,$V$是所有可能的观测的集合:

$$
Q = (q_1,q_1,\dots,q_N),V = (v_1,v_2,\dots,v_M)
$$

其中,$N$是可能的状态数，$M$是可能的观测数。$I$是长度为$T$的状态序列,$O$是对应的观测序列:

$$
I = (i_1,i_2,\dots,i_T),O = (o_1,o_2,\dots,o_T)
$$

$A$是状态转移概率矩阵:

$$
A = [a_{ij}]_{N \times N}
$$

其中:

$$
a_{ij} = P(i_{t+1} = q_j | i_t = q_i),i=1,\dots,N;j=1,\dots,N
$$

指代在时刻$t$处于状态$q_i$条件下,在时刻$t+1$转移到$q_j$的概率，$B$是观测概率矩阵:

$$
B = [b_j(k)]_{N\times M}
$$

其中:

$$
b_j(k) = P(o_t = v_k|i_t = q_j),\quad k=1,2,\dots,M;\quad,j=1,2,\dots,N
$$

指代在时刻$t$处于状态$q_j$条件下生成观测$v_k$的概率。$\pi$是初始是初始状态概率向量:

$$
\pi = (\pi_i)
$$

其中:

$$
\pi_i = P(i_1 = q_i),\quad i =1,2,\dots,N
$$

指代初始时刻处于状态$q_i$的概率。隐马尔可夫模型由初始状态概率向量$\pi$,状态转移概率矩阵$A$和观测概率矩阵$B$决定。$\pi$和$A$决定状态序列,$B$决定观测序列。因此，隐马尔可夫模型可以用三元符号表示:

$$
\lambda = (\pi,A,B)
$$

状态转移矩阵$A$与初始状态向量$\pi$确定了隐藏的马尔可夫链，生成不可观测的状态序列。观测概率矩阵$B$确定了如何从状态生成观测，与状态序列综合确定了如何产生观测序列。从定义来看，隐马尔可夫模型做了两个基本假设：

1. 齐次马尔可夫性假设，即假设隐藏的马尔可夫链在任意时刻$t$的状态只依赖于前一时刻的状态，与其他时刻的状态及观测无关，也与$t$无关：

$$
P(i_t| i_{t-1},o_{t-1},\dots,i_1,t_1) = P(i_t|i_{t-1}),\quad t = 1,2,\dots,T
$$

2. 观测独立性假设,即假设在任意时刻的观测只依赖于该时刻的马尔可夫链的状态，与其他观测及状态无关:

$$
P(o_t|i_t,i_{t-1},o_{t-1},\dots,i_1,t_1) = P(o_t|i_t)
$$

隐马尔可夫模型可用于标注,这时状态对应着标记。标注问题是给定观测的序列预测其对应的标记序列。可以假设标注问题的数据是由隐马尔可夫模型生成的。这样我们可以利用隐马尔可夫模型的学习与预测算法进行标注。

#### 观测序列的生成过程

根据隐马尔可夫模型定义,可以将一个长度为$T$的观测序列$O = (o_1,o_2,\dots,o_T)$的生成过程描述如下:

> 输入： 隐马尔可夫模型$\lambda = (A,B,\pi)$,观测序列长度$T$
> 输出： 观测序列$O = (o_1,o_2,\dots,o_T)$
>
> 1. 按照初始状态分布$\pi$产生状态$i_1$
> 2. 令$t=1$
> 3. 按照状态$i_t$的观测概率分布$b_{i_t}(k)$生成$o_t$
> 4. 按照状态$i_t$的状态转移概率分布$\{a_{i_t,i_{t+1}}\}$产生状态$i_{t+1}$
> 5. 令$t = t+1$,如果$t<T$,转步(3),否则，终止

#### 隐马尔可夫模型的3个基本问题

隐马尔可夫模型有3个基本问题:

- 概率计算问题,给定模型$\lambda = (A,B,\pi)$和观测序列$O = (o_1,o_2,\dots,o_T)$,计算在模型$\lambda$下观测序列$O$出现的概率$P(O|\lambda)$
- 学习问题，已知观测序列$O = (o_1,o_2,\dots,o_T)$,估计模型$\lambda = (A,B,\pi)$参数，使得在该模型下观测序列概率$P(O|\lambda)$。即用极大似然估计的方法估计参数。
- 预测问题，也称为解码问题。已知模型$\lambda = (A,B,\pi)$和观测序列$O = (o_1,o_2,\dots,o_T)$,求对给定观测序列条件概率$P(I|O)$最大的状态序列$I = (i_1,i_2,\dots,i_T)$。即给定观测序列，求最有可能的对应的状态序列。

### 概率计算算法

#### 直接计算法

首先介绍最直观的直接计算法，给定模型和观测序列后，计算观测序列出现的概率的直接方法便是按照概率公式直接计算。通过列举所有可能的长度为$T$的状态序列，求各个状态序列$I$与观测序列$O$的联合概率$P(I,O|\lambda)$,然后对所有可能的状态序列求和，得到$P(O|\lambda)$。

状态序列$I = (i_1,i_2,\dots,i_T)$的概率是:

$$
P(I|\lambda) = \pi_{i1}a_{i_1 i_2} a_{i_2 i_3} \dots a_{i_{T-1} i_T}
$$

对固定的状态序列$I = (i_1,i_2,\dots,i_T)$,观测序列$O = (o_1,o_2,\dots,o_T)$的概率是:

$$
P(O|I,\lambda) = b_{i_1}(o_1) b_{i_2}(o_2) \dots b_{i_T}(o_T)
$$

$O$和$I$同时出现的联合概率为:

$$
\begin{aligned}
         P(O,I|\lambda) &= P(O|I,\lambda) P(I|\lambda) \\\\
            &= \pi_{i_1}b_{i_1}(o_1) a_{i_1 i_2} b_{i_2}(o_2) \dots a_{i_{T-1}i_T} b_{i_T}(o_T)
    \end{aligned}
$$

然后，对所有可能的状态序列$I$求和，得到观测序列$O$的概率$P(O|\lambda)$，即:

$$
\begin{aligned}
         P(O,I|\lambda) &= \sum_{I} P(O|I,\lambda) P(I|\lambda) \\\\
            &= \sum_{i_1,i_2,\dots,i_T} \pi_{i_1}b_{i_1}(o_1) a_{i_1 i_2} b_{i_2}(o_2) \dots a_{i_{T-1}i_T} b_{i_T}(o_T)
    \end{aligned}
$$

这种直接计算的复杂度是$O(TN^T)$,在实际中并不可行。

#### 前向算法

首先定义前向概率：

> 前向概率:给定隐马尔可夫模型$\lambda$,定义到时刻$t$部分观测序列为$o_1,o_2,\dots,o_t$且状态为$q_i$的概率为前向概率，记做：
> $$
\alpha_t(i) = P(o_1,o_2,\dots,o_t,i_t = q_i|\lambda)
>$$
由此便可以递推求得前向概率$\alpha_t(i)$和观测序列概率$P(O|\lambda)$

> **观测序列概率的前向算法**
> 输入：隐马尔可夫模型$\lambda$,观测序列$O$
> 输出: 观测序列概率$P(O|\lambda)$
>
> - 初值：
> $$
> \alpha_1(i) = \pi_i b_i(o_1),\quad i = 1,2,\dots, N
> $$
> - 递推：对$t=1,2,\dots,T-1$
$$
\alpha_{t+1}(i) = [\sum_{j=1}^N \alpha_t(j) a_{ji}] b_i(o_{t+1}),\quad i= 1,2,\dots,N
$$
> - 终止:
$$
P(O|\lambda) = \sum_{i=1}^N \alpha_T(i)
$$

前向算法实际上是基于“状态序列的路径结构”递推计算$P(O|\lambda)$的算法。前向算法高效的关键在于其局部计算前向概率，然后利用路径结构将前向概率递推到全局,得到$P(O|\lambda)$。

#### 后向算法

首先给出后向概率的定义:

> **后向概率**：给定隐马尔可夫模型$\Lambda$,定义在时刻$t$状态为$q_i$的条件下，从$t+1$到$T$的部分观测序列为$o_{t+1},o_{t+2},\dots,o_T$的概率为后向概率，记做：
> $$
\beta_t(i) = P(o_{t+1},o_{t+2},\dots,o_T|i_t = q_i,\lambda)
> $$

与前向概率计算类似,后向概率的计算也可以采用递推算法:

> **观测序列概率的后向算法**
> 输入：隐马尔可夫模型$\lambda$,观测序列$O$
> 输出：观测序列概率$P(O|\lambda)$
>
> - 初始化:
$$
\beta_T(i) = 1,\quad i =1,2,\dots,N
$$
> - 对$t = T-1,T-2,\dots,1$
> $$
\beta_t(i) = \sum_{j=1}^N a_{ij} b_j(o_{t+1})\beta_{t+1}(j),
\quad i=1,2,\dots,N
> $$
> - 终止：
$$
P(O|\lambda) = \sum_{i=1}^N \pi_i b_i(o_1)\beta_1(i)
$$

利用前向概率与后向概率的定义可以将观测序列概率统一写成:

$$
P(O|\lambda) =\sum_{i=1}^N \alpha_t(i) \beta_t(i) =\sum_{i=1}^N \sum_{j=1}^N \alpha_t(i)
    a_{ij} b_j(o_{t+1}) \beta_{t+1}(j)
$$

该联合概率公式告诉我们，若我们知道了某一个时刻所有状态下的前向概率，同时知道下一个时刻所有状态下的后向概率，则可以直接计算出观测序列概率。前面介绍的三个计算观测序列概率的公式实际上为我们提供了三种递推思路:

- 前向概率递推，即从前向后递推
- 后向概率递推，即从后向前递推
- 前向后向概率递推，即从两个方向开始递推

这三种方法的计算复杂度均为$O(TN^2)$级别，对比直接计算的$O(TN^T)$级别，计算复杂度大大降低。

#### 一些概率与期望值的计算

利用前向概率和后向概率，可以得到关于单个状态和两个状态概率的计算公式。
1. 给定模型$\lambda$和观测$O$,在时刻$t$处于状态$q_i$的概率,记：
$$
\gamma_t(i) = P(i_t = q_i|O,\lambda)
$$
可以通过前向后向概率计算,由贝叶斯公式可得:
$$
\gamma_t(i) = P(i_t = q_i|O,\lambda) = \frac{P(i_t = q_i,O|\lambda)}{P(O|\lambda)}
$$
由此可以得到：
$$
\gamma_t(i) = \frac{\alpha_t(i) \beta_t(i)}{\sum_{j=1}^N \alpha_t(j)\beta_t(j)}
$$
2. 给定模型$\lambda$和观测$O$,在时刻$t$处于状态$q_i$且在时刻$t+1$处于状态$q_j$的概率为：
$$
\xi_t(i,j) = P(i_t = q_i, i_{t+1} = q_j | O,\lambda)
$$
由贝叶斯公式可得：
$$
\begin{aligned}
     \xi_t(i,j)  &= \frac{P(i_t = q_i,i_{t+1} = q_j, O|\lambda)}{P(O|\lambda)} \\\\
     &= \frac{P(i_t = q_i,i_{t+1} = q_j, O|\lambda)}{\sum_{i=1}^N \sum_{j=1}^N P(i_t = q_i,i_{t+1} = q_j, O|\lambda)}
\end{aligned}
$$
公式中$P(i_t = q_i,i_{t+1} = q_j, O|\lambda)$计算公式为:
$$
P(i_t = q_i,i_{t+1} = q_j, O|\lambda) = \alpha_t(i) a_{ij} b_j (o_{t+1}) \beta_{t+1}(j)
$$
3. 将$\gamma_t(i)$和$\xi_t(i,j)$对各个时刻$t$求和，可以得到一些期望值：
- 在观测$O$下状态$i$出现的期望值：
$$
\sum_{t=1}^T \gamma_t(i)
$$
- 在观测$O$下由状态$i$转移的期望值:
$$
\sum_{t=1}^{T-1} \gamma_t(i)
$$
- 在观测$O$下由状态$i$转移到状态$j$的期望值：
$$
\sum_{t=1}^{T-1} \xi_t(i,j)
$$

### 学习算法

隐马尔可夫的学习算法可以根据数据类型分为两种：

- 训练数据包含观测序列和状态序列, 此时可采用监督学习算法
- 训练数据仅有观测序列，此时需采用无监督学习算法

#### 监督学习方法
假设已给训练数据包含$s$个长度相同的观测序列和对应的状态序列$\{ (O_1,I_1),\dots, 
(O_S,I_S)\}$,那么可以直接采用极大似然法来估计马尔可夫参数,具体方法如下:
1. 转移概率$a_{ij}$的估计：设样本中时刻$t$处于状态$i$,而时刻$t+1$处于状态$j$出现
   的频数为$A_{ij}$,那么状态转移概率$a_{ij}$的估计为：
$$
\hat{a}_{ij} = \frac{A_{ij} }{\sum_{j=1}^N A_{ij} }
$$
这其实就是一个伯努利分布的最大似然估计问题，计算比较简单。
1. 观测概率$b_j(k)$的估计,设样本中状态为$j$并且观测为$k$的频数是$B_{jk}$,那么状态为$j$观测为$k$的估计是：
$$
\hat{b}_j(k) = \frac{B_{jk}}{\sum_{k=1}^M B_{jk}}
$$
3. 初始状态概率$\pi_i$的极大似然估计为$S$个样本中初始状态为$q_i$的频率

### 无监督学习方法-BW算法

假设训练数据只包含$S$个长度为$T$的观测序列$\{ O_1,O_2,\dots, O_S\}$
而没有相应的状态序列，学习的目标是学习隐马尔可夫模型的参数。我们将观测序列
看做观测数据$O$,状态序列数据看做不可观测的隐数据$I$,那么隐马尔可夫实际上是
一个含有隐变量概率模型：

$$
P(O|\lambda) = \sum_{I} P(O|I,\lambda)P(I|\lambda)
$$

这是在EM算法解决框架下，可以按照EM算法步骤进行计算

- 确定完全数据的对数似然函数，所有的观测数据写成$O = (o_1,o_2,\dots,o_T)$,
  所有隐数据写成$I = (i_1,i_2,\dots,i_T)$。完全数据的对数似然函数是
  $log P(O,I|\lambda)$
- EM算法的E步:求$Q$函数$Q(\lambda,\bar{\lambda})$
$$
\begin{aligned}
        Q(\lambda,\bar{\lambda}) &= E_I[logP(O,I|\lambda)P(O,I|\bar{\lambda})] \\\\
        &= \sum_I log P(O,I|\lambda) P(I|O,\bar{\lambda}) \\\\
        &= \sum_I log P(O,I|\lambda) \frac{P(I,O|\bar{\lambda})}{P(O|\bar{\lambda})}
\end{aligned}
$$
又因为$P(O|\bar{\lambda})$与$\lambda$无关，而在接下来M步我们要做的是找到合适的
$\lambda$极大化$Q$函数，因此可以考虑将该项略去，将$Q$函数写做:
$$
Q(\lambda,\bar{\lambda}) = \sum_I log P(O,I|\lambda) P(I,O|\bar{\lambda})
$$
其中,$\bar{\lambda}$是隐马尔可夫模型的当前参数估计值,$\lambda$是要极大化的隐马尔
可夫模型参数。
$$
P(O,I|\lambda) = \pi_{i_1} b_{i_1}(o_1) a_{i_1,i_2} b_{i_2}(o_2)
    \dots a_{i_{T-1}i_T} b_{i_T}(o_T)
$$
于是$Q$函数可以写做:
$$
\begin{aligned}
        Q(\lambda,\bar{\lambda}) &= \sum_I log \pi_{i_1} P(O,I|\bar{\lambda}) + \sum_I (\sum_{t=1}^{T-1} log a_{i_t i_{t+1}})
        P(O,I|\bar{\lambda})\\ &+ \sum_{I}(\sum_{t=1}^T log b_{i_t}(o_t))P(O,I|\bar{\lambda})
    \end{aligned}
$$
- EM算法M步：极大化$Q$函数$Q(\lambda,\bar{\lambda})$求模型参数$A,B,\pi$
由最终$Q$函数的写法可以看出，要极大化参数单独出现在三项中，因此只需对各项
分别极大化
1. 求$\pi_{i_1}$估计值，第一项可以重写为：
$$
\sum_I log \pi_{i_1} P(O,I|\bar{\lambda}) = \sum_{i=1}^N log \pi_{i} P(O,i_1 = i|\bar{\lambda})
$$
同时需要注意到关于$\pi_i$的天然约束$\sum_{i=1}^N \pi_i = 1$,该优化问题可以通过
Lagrange乘子法进行求解，最终求解值为:
$$
\pi_i = \frac{P(O,i_1 = i|\bar{\lambda})}{P(O|\bar{\lambda})}
    = \gamma_1(i)
$$
2. 求$a_{ij}$估计值，第二项可以重写为：
$$
\sum_I (\sum_{t=1}^{T-1} log a_{i_t i_{t+1}})P(O,I|\bar{\lambda}) = 
    \sum_{i=1}^N \sum_{j=1}^N \sum_{t=1}^{T-1} log a_{i j} P(O,i_t = i,i_{t+1} = j|\bar{\lambda})
$$
注意存在约束条件$\sum_{j=1}^N a_{ij} = 1$,应用Lagrange乘子法可得:
$$
a_{ij} = \frac{\sum_{t=1}^{T-1} P(O,i_t = i,i_{t+1} = j|\bar{\lambda})}{\sum_{t=1}^{T-1} P(O,i_t = i|\bar{\lambda})}
     = \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}
$$
3. 求$b_j(k)$的估计值，首先将第三项重写为:
$$
\sum_{I}(\sum_{t=1}^T log b_{i_t}(o_t))P(O,I|\bar{\lambda}) = 
    \sum_{j=1}^N \sum_{t=1}^{T} log b_j(o_t) P(O,i_t = j|\bar{\lambda})
$$
约束条件是$\sum_{k=1}^M b_j(k) = 1$。只有在$o_t = v_k$时,$b_j(o_t)$对$b_j(k)$的
偏导数才不为0,以$I(o_t = v_k)$表示,求得:
$$
b_j(k) = \frac{\sum_{t=1}^T P(O,i_t =j|\bar{\lambda})I(o_t = v_k)}{\sum_{t=1}^T P(O,i_t = j|\bar{\lambda})}
    = \frac{\sum_{t=1,o_t = v_k}^T \gamma_t(j)}{\sum_{t=1}^T \gamma_t(j)}
$$
> Baum-Welch算法：
> 输入：观测数据$O = (o_1,o_2,\dots,o_T)$
> 输出：隐马尔可夫模型参数
> - 初始化:对$n=0$,选取$a_{ij}^{(0)}, b_j(k)^{(0)},\pi_i^{(0)}$，得到模型
>   $\lambda^{(0)} = (A^{(0)},B^{(0)},\pi^{(0)})$
> - 根据上面给出的公式进行递推
> - 重复E步和M步。直到算法收敛

### 预测算法

下面介绍两种隐马尔可夫模型的预测算法：近似算法与维特比算法。

#### 近似算法

近似算法的想法是，在每个时刻$t$选择在该时刻最有可能出现的状态$i_t^*$,从而
得到一个状态序列$I^\ast = (i_1^\ast,i_2^\ast,\dots,i_T^\ast)$,将该序列作为预测的结果。
给定隐马尔可夫模型$\lambda$和观测序列$O$,在时刻$t$处于状态$q_i$的概率
$\gamma_t(i)$是：

$$
\gamma_t(i) = \frac{\alpha_t(i)\beta_t(i)}{P(O|\lambda)} = 
 \frac{\alpha_t(i)\beta_t(i)}{\sum_{j=1}^N \alpha_t(j)\beta_t(j)}
$$

在每一时刻$t$最有可能的状态$i_t^*$是:

$$
i_t^* = argmax_{1 \leq i \leq N} [\gamma_t(i)],\quad t = 1,2,\dots,T
$$

进而得到状态序列

$$
I^* = (i_1^*,i_2^*,\dots,i_T^*)
$$

近似算法优点是计算简单，其缺点是不能够保证预测的序列整体是最有可能的状态序列，这是因为预测的状态序列可能有实际不发生的部分。

#### 维特比算法

维特比算法实际上是用动态规划来解隐马尔可夫模型的预测问题，即用动态规划来求概率最大路径(最优路径)。这时一条路径对应着一个最优序列。
首先导入两个变量$\delta,\varPhi$,定义在时刻$t$状态为$i$的所有单个路径$(i_1,i_2,\dots,i_t)$中概率最大值为:
$$
    \delta_t(i) = \max_{i_1,i_2,\dots,i_{t-1}} P(i_t = i,i_{t-1},\dots,i_1,o_t,\dots,o_1|\lambda)
$$
由定义可知变量$\delta$的递推公式：
$$
    \begin{aligned}
     \delta_{t+1}(i) &= \max_{i_1,i_2,\dots,i_{t}} P(i_{t+1} = i,i_{t},\dots,i_1,o_{t+1},\dots,o_1|\lambda) \\
    &= max_{1 \leq j \leq N} [\delta_t(j) a_{ji}]b_i(o_{t+1})
    \end{aligned}
$$
定义在时刻$t$状态为$i$的所有单个路径$(i_1,i_2,\dots,i_{t-1},i)$中概率最大的路径的第
$t-1$个结点为：
$$
    \varPhi_t(i) = argmax_{1 \leq j \leq N}[\delta_{t-1}(j) a_{ji}],\quad i=1,\dots,N 
$$
下面给出维特比算法的流程:
> **维特比算法：**
> - 输入： 模型$\lambda = (A,B,\pi)$和观测$O = (o_1,o_2,\dots,o_T)$
> - 输出： 最优路径$I^\ast -= (i_1^\ast,i_2^\ast,\dots,i_T^\ast)$
> 1. 初始化
> $$
>   \delta_1(i) = \pi_i b_i(o_1),\varPhi_1(i) = 0
> $$
> 2. 递推：对$t = 2,3,\dots,T$
> $$
> \begin{aligned}
>   \delta_{t+1}(i) &= \max_{1 \leq j \leq N} [\delta_t(j) a_{ji}]b_i(o_{t+1}) \\
> \varPhi_t(i) &= argmax_{1 \leq j \leq N}[\delta_{t-1}(j) a_{ji}]
> \end{aligned}
> $$
> 3. 终止
> $$
>   P^* = \max_{1 \leq i \leq N} \delta_T(i), i_T^* = argmax_{1 \leq i \leq N}
> [\delta_T(i)]
> $$
> 4. 最优路径回溯。对$t = T-1,T-2,\dots,1$
> $$
>   i_t^* = \varPhi_{t+1}(i_{t+1}^*)
> $$
> 求得最优路径$I^{\ast} = (i_1^\ast,i_2^\ast,\dots,i_T^\ast)$
