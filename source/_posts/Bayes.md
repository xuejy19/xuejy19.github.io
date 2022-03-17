---
title: 贝叶斯理论
date: 2020-07-23 10:32:46
tags: 
categories: 统计学习
toc: true 
mathjax: true
---
-----
该部分按照以下层次进行组织：
> * 从贝叶斯公式谈起
> * 贝叶斯决策 
<!--more-->
-----
## 贝叶斯公式
**概念介绍**(From 维基百科)：
> * **条件概率**：条件概率就是事件A在事件B发生条件下的概率，记做$P(A|B)$.
> * **先验概率**：先验概率是指在考虑“观测数据”之前，对某一不确定量的估计,如$P(A)$
> * **后验概率**：在贝叶斯统计中，一个随机事件或者一个不确定事件的后验概率是在考虑和给出相关证据或数据后所得到的条件概率，“后验”指代考虑了被测试事件的相关证据，如$P(B|A)$可以看作事件B的后验概率

贝叶斯公式形式如下：
$$ P(A|B) = \frac{P(A)P(A|B)}{P(B)} $$
再根据全概率公式：
$$P(B) = \sum_j P(B|A_j)P(A_j) $$
可以得到贝叶斯公式另一种形式：
$$P(A_i|B) = \frac{P(A_i)P(B|A_i)}{\sum_j P(B|A_j)P(A_j) }$$
> 公式分析：在现实世界中，我们往往期望通过一些观察到的事件来对一些不可以直接观测的事件进行推断，公式中的$B$变量指代可以直接观测到的事件，而我们的目的则是希望通过$B$事件与$A$事件之间的关联(因果性,相关性)来间接对A事件进行观测，贝叶斯公式便为我们搭起了这样一做桥梁。

从公式角度出发，可以看出这样进行推理是有代价的，或者是必须拥有某些知识：

> - 先验概率$P(A)$
> - 类条件概率$P(B|A)$
---

## 贝叶斯决策
这部分主要介绍三种基于贝叶斯理论的决策策略：
> - 基于最小错误率的贝叶斯决策
> - 基于最小风险的贝叶斯决策
> - 在限定一类错误率条件下使另一类错误率为最小的两类别决策
---
### 最小错误率贝叶斯决策
从贝叶斯公式出发，通过比较后验概率大小，选择后验概率大的类别作为label。以而分类问题为例，$\omega_i(i=1,2)$是类别标签，$x$为特征相量，由贝叶斯公式，可得$P(\omega_i|x)$：
$$ P(\omega_i|x) = \frac{P(x|\omega_i)P(\omega_i)}{\sum_{j=1}^2 P(x|\omega_j)P(\omega_j)}$$
最小错误率下等价的决策规则有：
> - 若$P(\omega_i|x) = max_{j=1,2}P(\omega_j|x)$,则$x \in \omega_i$
> - 若$P(x|\omega_i)P(\omega_i) = max_{j=1,2}P(x|\omega_j)P(\omega_j)$,则$x \in \omega_i$
> - 对于二分类问题，若$l(x) = \frac{P(x|\omega_1)}{P(x|\omega_2)} > \frac{P(\omega_2)}{P(\omega_1)}$,则$x \in \omega_1$

下面证明该种决策策略为最小错误率，首先给出平均错误率公式：
$$ P(e) = \int_{-\infty}^{\infty}P(e,x)dx = \int_{-\infty}^{\infty} P(e|x)p(x)dx$$
由上面给出的决策规则可知：
$$ P(e|x) = \begin{cases} P(\omega_1|x),if P(\omega_2|x) > P(\omega_1|x) \\ 
P(\omega_2|x),if P(\omega_1|x) > P(\omega_2|x) \end{cases}$$
若$t$为两类的决策分界面，在该分界面上$P(\omega_1|x) = P(\omega_2|x)$，$t$将整个决策空间分为了两部分$\mathcal{R_1}$和$\mathcal{R_2}$,分别对应将$x$归为$\omega_1$和$\omega_2$,据此平均错误率可以写为：
$$
\begin{aligned}
P(e) &= P(x \in \mathcal{R_1},\omega_2) + P(x \in \mathcal{R_2},\omega_1) \\
&= P(x\in \mathcal{R_1}|\omega_2) P(\omega_2) + P(x \in \mathcal{R_2}|\omega_1) P(\omega_1) \\
&= P(\omega_2) \int_{\mathcal{R_1}} p(x|\omega_2)dx + P(\omega_1)\int_{\mathcal{R_2}}p(x|\omega_1)dx \\
&= P(\omega_2) P_2(e) + P(\omega_1) P_1(e)
\end{aligned}
$$
![最小错误率](https://raw.githubusercontent.com/xuejy19/Images/master/bayes2.png)

图中斜线部分代表$P(e)$的第一项，纹线部分代表第二项，而从最小错误率公式来看，若要使平均错误率$P(e)$最小，只需要对于任意的$x$，保证$P(e|x)$最小，而这恰恰是贝叶斯最小错误率决策规则。

### 最小风险贝叶斯决策
最小风险贝叶斯决策，或者称为最小损失贝叶斯决策，这种决策思想也是非常朴素：
> 同样是决策错误，但不同的错误带来的损失并不相同，有时甚至天差地别，比如去医院看病，对于早起癌症，漏警要比虚警严重的多，可能会让患者失去早期治疗的时机。

最小风险错误率准则正是考虑各种错误造成损失不同而提出的一种决策规则，若将决策空间记做$\mathcal{A}$,而每个决策都将带来一定的损失，它通常是决策和自然状态的函数，比如$\lambda(a_i,\omega_j)$则指代在$x$处于状态$\omega_j$而做出$a_i$决策时的损失，下面给出数学符号定义：
- 观测向量$x$为$d$维随机向量
$$ x = [x_1, x_2 , \dots, x_d]^T$$
- 状态空间$\Omega$由$c$个自然状态(c类)组成
$$ \Omega = \{ \omega_1, \dots, \omega_c \}$$
- 决策空间由$n$个决策组成
$$ \mathcal{A} = \{ a_1, \dots, a_n \} $$
- 决策表，决策表中第$i$行$j$列为相应损失值$\lambda(a_i,\omega_j)$

由于引入了损失的概念，在进行决策时便不能够只根据后验概率大小来做决策，必须考虑所采取的决策是否使得损失最小。对于给定的$x$，如果我们采取决策$a_i$,其对应$c$种可能的损失$\lambda(a_i,\omega_j)$,每种损失出现的概率为$P(\omega_i|x)$,由此可以计算条件期望损失$R(a_i|x)$:
$$
    R(a_i|x) = E(\lambda(a_i,\omega_j)) = \sum_{j=1}^c \lambda(a_i,\omega_j) P(\omega_j|x), i=1,\dots,n
$$
在决策论种$R(a_i|x)$被称作条件风险，由于$x$是随机向量的观察值，因此对于不同的$x$，采取决策$a_i$时，其条件风险是不同的，所以最终采取何种决策应当与$x$有关，在这种意义下，可以将决策$a$看作是$x$的函数，记做$a(x)$,可以定义期望风险$R$为：
$$
    R = \int R(a(x)|x) p(x)dx
$$
若想使得期望风险$R$最小，只需要保证对于任意的$x$，$R(a(x)|x)$最小，因此最小损失贝叶斯决策规则为：
$$
    if R(a_k|x) = \min \limits_{i=1,\dots,n}R(a_i|x) \Rightarrow a = a_k
$$
### 在限定一类错误率条件下，使得另一类错误率最小
在两类别决策问题中，有两种错误分类的可能性,分别对应两个错误率$P_1(e)$和$P_2(e)$,而在实际应用中，我们往往期望某一个错误率不大于某个常数，而使另一类错误率尽可能小。比如在癌症诊断中，我们认识到将异常判断为正常非常严重，我们期望这类错误率为一个很小的值$P_2(e) = \epsilon_0$,在满足该条件下，期望另一类错误率$P_1(e)$尽可能低，这其实是一个带有等式约束的优化问题，可以用拉格朗日乘子法进行求解，拉格朗日函数$L$为:
$$
    L = P_1(e) + \lambda(P_2(e) - \epsilon_0)
$$
其中$P_1(e)$为：
$$
    P_1(e) = \int_{\mathcal{R_2}} p(x|\omega_1)dx 
$$
$P_2(e)$类似，因此拉格朗日函数可以写做：
$$
\begin{aligned}
    L &= \int_{\mathcal{R_2}} p(x|\omega_1) + \lambda(\int_{\mathcal{R_1}} p(x|\omega_2)dx - \epsilon_0)  \\
    &= 1-\lambda \epsilon_0 + \int_{\mathcal{R_1}} [\lambda  p(x|\omega_2) - p(x|\omega_1)dx]
\end{aligned}
$$
$L$分别对分界点$t$和$\lambda$求导，可得：
$$
 \begin{cases}
    \frac{\partial L}{\partial t} = 0 \Rightarrow \lambda = \frac{p(t|\omega_1)}{p(t|\omega_2)} \\
    \frac{\partial L}{\partial \lambda} = 0 \Rightarrow P_2(e) = \epsilon_0
 \end{cases} 
$$
由此便可得到Neyman-Pearson决策准则：
$$
    if \frac{p(x|\omega_1)}{p(x|omega_2)} > \lambda \Rightarrow x \in \omega_1 
$$
