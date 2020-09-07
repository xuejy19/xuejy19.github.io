---
title: pytorch学习
date: 2020-09-06 16:46:45
tags: pytorch
categories: 统计学习
mathjax: true
toc: true
---
在前面介绍深度学习的理论知识时，相信大家可以感受到，神经网络的实现主要有以下两个难题：
- 当网络结构复杂起来时，手写一个神经网络是非常困难，也是十分费时的。
- 一个神经网络有着大量的参数，对计算机的计算能力要求非常高，而GPU是计算机中有着大量计算资源的部分，如何将这部分计算资源调动起来完成一个深度神经网络的训练是另一个难题。
<!--more-->
为了解决上述两个问题，深度学习框架便诞生了，目前主流的深度学习框架主要下图所示的这些:
![主流框架](https://raw.githubusercontent.com/xuejy19/xuejy19.github.io/source/Img/framework.jpeg)
这些框架比较如下图所示:
![深度学习框架](https://raw.githubusercontent.com/xuejy19/xuejy19.github.io/source/Img/framework1.png)
