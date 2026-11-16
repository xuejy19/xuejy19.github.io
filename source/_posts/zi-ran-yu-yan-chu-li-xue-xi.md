---
title: 自然语言处理学习
date: '2022-10-06 12:00:00'
updated: '2022-10-14 12:00:00'
permalink: 2022/10/06/zi-ran-yu-yan-chu-li-xue-xi/
tags:
- nlp
categories:
- 环境配置
- 灵魂雕琢
math: true
legacy_html: true
img: /medias/featureimages/5.jpg
summary: '最近打算入门一下nlp,选择的课程是CS224n: Natural Language Processing with Deep Learning，打算以该课程的课件为脉络来了解nlp领域，本篇博客用来进行学习记录。
  Lecture1: Introduction and word vectors传统的'
---
{% raw %}
<p>最近打算入门一下nlp,选择的课程是<a target="_blank" rel="noopener" href="https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/">CS224n: Natural Language Processing with Deep Learning</a>，打算以该课程的课件为脉络来了解nlp领域，本篇博客用来进行学习记录。<span id="more"></span> </p>
<h4 id="Lecture1-Introduction-and-word-vectors"><a href="#Lecture1-Introduction-and-word-vectors" class="headerlink" title="Lecture1: Introduction and word vectors"></a>Lecture1: Introduction and word vectors</h4><h4 id="传统的词表示"><a href="#%E4%BC%A0%E7%BB%9F%E7%9A%84%E8%AF%8D%E8%A1%A8%E7%A4%BA" class="headerlink" title="传统的词表示"></a>传统的词表示</h4><p>如何让计算机理解自然语言的含义？ </p>
<ul>
<li><code>Wordnet</code>: 包含同义词集和上位词列表的同义词库，通过维护一个分类清晰的庞大词表使计算机能够理解某个单词在语言框架中所处的位置(<code>词性</code>、<code>含义</code>、<code>上位词</code>)。但这样做存在以下问题: <ul>
<li>无法区分词之间的细微差别 </li>
<li>不够完整，难以保持up-to-date </li>
<li>带有主观性 </li>
<li>依赖于人力构建 </li>
<li>无法计算词之间的相似度 </li>
</ul>
</li>
<li>在传统的NLP，我们将单次看作离散的符号，通过one-hot编码的形式来表示每个单词<br><img src="/.io//onehot.jpg" alt><ul>
<li>两个不同的词的one-hot表征是彼此正交的，没有办法衡量两个词之间的相似度 <ul>
<li>通过<code>Wordnet</code>等词表来获取单词之间的相似度，但仍具有上述问题 </li>
<li><strong>在向量空间对相似度进行编码</strong></li>
</ul>
</li>
</ul>
</li>
</ul>
<h4 id="Word-vectors"><a href="#Word-vectors" class="headerlink" title="Word vectors"></a>Word vectors</h4><p>针对如何在向量空间对单词相似度进行编码，后续又有了一系列的工作: </p>
<ul>
<li><strong>分布语义</strong>：一个单词的意义是由经常出现在其旁边的单词给出的 </li>
<li>考虑通过一个单词的上下文来构建该单词的表示<br><img src="/.io//context.jpg" alt></li>
</ul>
<blockquote>
<p><strong>Word vectors/embeddings/representations</strong>: 为每个单词找到一个稠密向量表示，使得出现在相似上下文的单词具有相似的向量表示 </p>
</blockquote>
<ul>
<li><p>Word2Vec: Overview</p>
<ul>
<li>有大量单词的文本语料库 </li>
<li>词汇表中的每个单词都用一个向量(one-hot)表示 </li>
<li>对于文本中的每个位置，都有一个中心单词$c$和上下文单词集合$\mathbf{o}$</li>
<li>考虑使用中心单词$c$的单词向量来预测上下文单词$\mathbf{o}$(反之亦然) </li>
<li>调整单词向量使得对上下文/中心向量的预测尽可能准确，最大化$p(\mathbf{o}|t)$<br><img src="/.io//wordvector.jpg" alt></li>
</ul>
</li>
<li><p>推导 </p>
<ul>
<li>似然函数 <script type="math/tex; mode=display">
L(\theta)=\prod_{\substack{t=1}} \prod_{\substack{m \leq j \leq m \\ j \neq 0}} P\left(w_{t+j} \mid w_t ; \theta\right)</script></li>
<li>一般进行优化都是最小化负对数似然: <script type="math/tex; mode=display">
J(\theta)=-\frac{1}{T} \sum_{t=1}^T \sum_{\substack{m \leq j \leq m \\ j \neq 0}} \log P\left(w_{t+j} \mid w_t ; \theta\right)</script></li>
<li>条件概率$P(w_{t+j}|w_t;\theta)$：softmax function <script type="math/tex; mode=display">
P(o \mid c)=\frac{\exp \left(u_o^T v_c\right)}{\sum_{w \in V} \exp \left(u_w^T v_c\right)}</script></li>
</ul>
</li>
<li>两种模型<ul>
<li>Skip-gram: 用中间单词来预测上下文</li>
<li>Continuous Bag of Words(CBOW): 用上下文单词预测中间单词 </li>
</ul>
</li>
</ul>
<h5 id="CBOW"><a href="#CBOW" class="headerlink" title="CBOW"></a>CBOW</h5><p>基本思想是用上下文单词来预测中间单词，流程如下： </p>
<blockquote>
<ul>
<li>假设词库容量为$V$, 窗口宽度为$m$, embedding维度为$N$, 单个句子$x$, 中心单词位置$c$</li>
<li>task: $(x<em>{c-m}, x</em>{c-m+1}, … , x<em>{c-1}, x</em>{c+1}, … , x_{c+m}) \rightarrow x_c$</li>
</ul>
</blockquote>
<p><img src="/.io//CBOW.jpg" alt> </p>
<ul>
<li>2个矩阵<ul>
<li>Embedding mat $W_e \in \mathbb{R}^{V \times N}$</li>
<li>Predict mat $W_p \in \mathbb{R}^{N \times V}$ </li>
</ul>
</li>
<li>流程 <ul>
<li>通过$W_e$得到各个单词的embedding $v_i = W_e^Tx_i \in \mathbb{R}^{N \times 1}$(本质上就是抽取了$W_e$的某一行) </li>
<li>取平均： $\hat{v} = \frac{v<em>{c-2m} + … + v</em>{c+2m}}{2m}$  </li>
<li>通过$W_p$得到预测结果： $z = W_p^T \hat{v} \in \mathbb{R}^{V \times 1}$ </li>
<li>将该预测结果通过softmax函数转换成概率形式: $\hat{y} = softmax(z)$ </li>
</ul>
</li>
</ul>
<h5 id="Skip-Gram-Model"><a href="#Skip-Gram-Model" class="headerlink" title="Skip-Gram Model"></a>Skip-Gram Model</h5><p>基本思想是使用中间单词来预测其上下文单词 </p>
<blockquote>
<ul>
<li>假设词库容量为$V$, 窗口宽度为$m$, embedding维度为$N$, 单个句子$x$, 中心单词位置$c$</li>
<li>task: $x<em>c \rightarrow (x</em>{c-m}, x<em>{c-m+1}, … , x</em>{c-1}, x<em>{c+1}, … , x</em>{c+m})$</li>
</ul>
</blockquote>
<ul>
<li>流程: 基本就是CBOW的逆过程 </li>
</ul>
<h5 id="负采样-Negative-sampling"><a href="#%E8%B4%9F%E9%87%87%E6%A0%B7-Negative-sampling" class="headerlink" title="负采样(Negative sampling)"></a>负采样(Negative sampling)</h5><p>如果不进行优化，那么两种wordvec方法在计算过程中都需要更新词汇表中所有单词用于训练，而对于输出层，正样本其实只对应着极少数的样本，而其余均是大量的负样本(输出结果的one-hot编码不匹配)，负采样的直观想法便是随机选择一小部分negative words来更新权重，可以大大提高计算效率 </p>
<h5 id="Hierachical-softmax"><a href="#Hierachical-softmax" class="headerlink" title="Hierachical softmax"></a>Hierachical softmax</h5><p>通过分层的形式将一个$V$分类问题变成了$logV$个二分类问题，</p>
{% endraw %}
