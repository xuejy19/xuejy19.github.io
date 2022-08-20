---
title: transformer_self-attention机制学习
date: 2020-11-13 10:11:02
tags: transformer, self-attention机制 
categories: 统计学习  
toc: true 
mathjax: true 
---
基于RNN的网络结构(LSTM,gru等)在nlp领域有着广泛应用，但RNN这样的网络结构使其固有长程梯度消失的问题，对于较长的句子，我们很难寄希望于将输入的序列转化为定长的向量(embedding)而保存所有有效的信息。 为了解决这一由长序列到定长向量转化而造成信息损失的瓶颈，attention机制诞生了。 
<!--more-->
![self-attention](https://raw.githubusercontent.com/xuejy19/xuejy19.github.io/source/Img/transformer.gif) 

上面的`gif`给出了self-attention机制的直观演示，对于传统的RNN网络，网络结构本身就导致经过Encoder之后的向量中更多包含后面样本的信息，而前面样本的信息被稀释了，而self-attention机制可以无差别的注意到序列中的任意一个单位。 
#### Transformer 详解
> **关于transformer结构的详解请参考这篇博文:**
> [图解transformer](http://jalammar.github.io/illustrated-transformer/)

#### 实验部分 
后面会补一个实验，先挖坑在这
