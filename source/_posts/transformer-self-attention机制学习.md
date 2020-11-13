---
title: transformer_self-attention机制学习
date: 2020-11-13 10:11:02
tags: transformer, self-attention机制 
categories: 统计学习  
toc: true 
mathjax: true 
---
基于RNN的网络结构(LSTM,gru等)在nlp领域有着广泛应用，但RNN这样的网络结构使其固有长程梯度消失的问题，对于较长的句子，我们很难寄希望于将输入的序列转化为定长的向量(embedding)而保存所有有效的信息。 为了解决这一由长序列到定长向量转化而造成信息损失的瓶颈，attention机制诞生了。 

![self-attention](https://vdn1.vzuu.com/SD/feb4e6b8-2397-11eb-9748-0eb20c5883f6.mp4?disable_local_cache=1&bu=pico&expiration=1605237178&auth_key=1605237178-0-0-4d7df87f75ff00202b3fe71c765b814c&f=mp4&v=hw)