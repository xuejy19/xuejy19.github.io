---
title: Domain generalization
date: '2022-10-28 12:00:00'
updated: '2022-10-31 12:00:00'
permalink: 2022/10/28/domain-generalization/
tags:
- 前沿方向
categories:
- 论文阅读
- 技术学习
math: true
legacy_html: true
img: /medias/featureimages/0.jpg
summary: '本篇文章是对文章Generalizing to Unseen Domains: A Survey onDomain Generalization的研读介绍，旨在对域泛化(domain
  generalization)领域研究现状有粗略了解，考虑将该领域的一些研究思想迁移到故障检测/溯源/软测量等任务上'
---
{% raw %}
<p>本篇文章是对文章<a target="_blank" rel="noopener" href="https://arxiv.org/pdf/2103.03097.pdf">Generalizing to Unseen Domains: A Survey on<br>Domain Generalization</a>的研读介绍，旨在对域泛化(domain generalization)领域研究现状有粗略了解，考虑将该领域的一些研究思想迁移到故障检测/溯源/软测量等任务上去。<span id="more"></span> </p>
<h3 id="出发点"><a href="#%E5%87%BA%E5%8F%91%E7%82%B9" class="headerlink" title="出发点"></a>出发点</h3><ul>
<li><p>现有算法面临的挑战<br>传统的机器学习算法的泛化能力依赖于训练域与测试域的独立同分布(i.i.d)假设，而在现实数据中，这样的假设往往无法得到满足。</p>
</li>
<li><p>目的<br>如果我们已经拥有了一个/多个域的数据，希望让训练得到的模型在未知域上仍能有良好的泛化能力。 </p>
</li>
</ul>
<h3 id="问题数学描述"><a href="#%E9%97%AE%E9%A2%98%E6%95%B0%E5%AD%A6%E6%8F%8F%E8%BF%B0" class="headerlink" title="问题数学描述"></a>问题数学描述</h3><blockquote>
<ul>
<li><p>domain: 记$\mathcal{X}$为非空的输入空间，$\mathcal{Y}$为输出空间，输入输出之间的关系通过联合概率分布$P<em>{XY}$来描述，域则由从该分布中采样得到的样本组成$S = {(x_i,y_i)}</em>{i=1}^n \sim P_{XY}$。</p>
</li>
<li><p>domain generalization: 在域泛化任务中，我们有$M$个训练/源域的数据</p>
<script type="math/tex; mode=display">
  S_{train} = {S_i|i=1,...,M}, where \quad S^i = \{(x_j^i, y_j^i)\}_{j=1}^{n_i} \quad denotes \quad i_{th} \quad domain</script><p>不同域的数据具有不同的联合概率分布，即$P<em>{XY}^i \neq P</em>{XY}^j $；域泛化的目标是学习到一个<strong>鲁棒且可泛化的预测方程</strong>$h: \mathcal{X} \rightarrow \mathcal{Y}$, 该模型在未知域$S_{test}$上的预测误差要尽可能小: </p>
<script type="math/tex; mode=display">
  \min_{h} \mathbb{E}_{(x,y) \in S_{test}} [\ell(h(x),y)]</script></li>
</ul>
</blockquote>
<h3 id="相关研究领域介绍"><a href="#%E7%9B%B8%E5%85%B3%E7%A0%94%E7%A9%B6%E9%A2%86%E5%9F%9F%E4%BB%8B%E7%BB%8D" class="headerlink" title="相关研究领域介绍"></a>相关研究领域介绍</h3><ul>
<li><strong>多任务学习(Multi-task learning)</strong>: 在多个任务上联合优化模型，目的是为了提高模型在原始任务上的泛化能力，多域学习(multi-domain learning)可以看作多任务学习的一种特例。</li>
<li><strong>迁移学习(Transfer learning)</strong>: 在源任务/域上训练得到模型，目的在于提高模型在目标任务/域上的性能，迁移学习最广泛的应用就是<strong>预训练模型+微调</strong>的范式，与域泛化的区别在于迁移学习中目标域的数据是可获取的。</li>
<li><strong>域适应(domain adaptation)</strong>: 该领域可以看作是迁移学习的一个分支，也是需要能够获取目标域的数据。 </li>
<li><strong>元学习(Meta-learning)</strong>: 学习学习算法，即learning to learn, 元学习是一种通用的学习策略可以用于实现域泛化。 </li>
<li><strong>终身学习/持续学习</strong>: 随着任务/数据域的不断更新，模型需要不断利用新的知识来更新模型参数。 </li>
<li><strong>零样本学习(Zero-shot learning)</strong>: 利用已有类别的数据学习模型，对新类别的样本仍能分辨出来。 </li>
</ul>
<h3 id="方法分类"><a href="#%E6%96%B9%E6%B3%95%E5%88%86%E7%B1%BB" class="headerlink" title="方法分类"></a>方法分类</h3><p><img src="/.io//%E6%96%B9%E6%B3%95%E5%88%86%E7%B1%BB.png" alt></p>
<p>目前域泛化的方法可以分为三类： </p>
<ul>
<li><strong>数据操纵</strong>: 基本思路是提高训练数据的涵盖范围，常用的思路有两个: <ul>
<li>数据扩充: 数据扩展/随机变换 </li>
<li>数据生成: 生成更具多样性的样本 </li>
</ul>
</li>
<li><strong>表征学习</strong>: 基本思想是学习到一些稳定的特征，可细分为两个方向: <ul>
<li>域不变表征学习 </li>
<li>特征解偶</li>
</ul>
</li>
<li><strong>学习策略</strong>: 探索一般性的学习策略以提升模型泛化能力，常用思路有: <ul>
<li>集成学习</li>
<li>元学习 </li>
<li>梯度操作 </li>
<li>分布鲁棒优化 </li>
<li>自监督学习 </li>
</ul>
</li>
</ul>
<h4 id="数据操纵-data-manipulation"><a href="#%E6%95%B0%E6%8D%AE%E6%93%8D%E7%BA%B5-data-manipulation" class="headerlink" title="数据操纵(data manipulation)"></a>数据操纵(data manipulation)</h4><p>提高训练数据的多样性，数据增强的有效性在于增强过后的数据未来有可能出现在测试域中，<strong>能否考虑应用一些机理知识来进行数据增强，类似于反事实的概念</strong>。 </p>
<h5 id="基于数据扩充的域泛化"><a href="#%E5%9F%BA%E4%BA%8E%E6%95%B0%E6%8D%AE%E6%89%A9%E5%85%85%E7%9A%84%E5%9F%9F%E6%B3%9B%E5%8C%96" class="headerlink" title="基于数据扩充的域泛化"></a>基于数据扩充的域泛化</h5><p>常用的数据扩充操作有翻转、旋转、裁剪、添加噪声等</p>
<ul>
<li>域随机化: 改变目标大小/位置/光线/随机噪声 </li>
<li>对抗数据增广: 实现有效数据增广，同标签，沿差异大的方向扩充样本 </li>
</ul>
<h5 id="基于数据生成的域泛化"><a href="#%E5%9F%BA%E4%BA%8E%E6%95%B0%E6%8D%AE%E7%94%9F%E6%88%90%E7%9A%84%E5%9F%9F%E6%B3%9B%E5%8C%96" class="headerlink" title="基于数据生成的域泛化"></a>基于数据生成的域泛化</h5><p>一般是通过VAE或者GAN来实现数据增强</p>
<h4 id="表征学习"><a href="#%E8%A1%A8%E5%BE%81%E5%AD%A6%E4%B9%A0" class="headerlink" title="表征学习"></a>表征学习</h4><p>将预测方程$h$拆分未$h = f \cdot g$, 表征学习的目标为: </p>
<script type="math/tex; mode=display">
\min _{f, g} \mathbb{E}_{\mathbf{x}, y} \ell(f(g(\mathbf{x})), y)+\lambda \ell_{\mathrm{reg}}</script><h5 id="基于域不变性表征学习的域泛化"><a href="#%E5%9F%BA%E4%BA%8E%E5%9F%9F%E4%B8%8D%E5%8F%98%E6%80%A7%E8%A1%A8%E5%BE%81%E5%AD%A6%E4%B9%A0%E7%9A%84%E5%9F%9F%E6%B3%9B%E5%8C%96" class="headerlink" title="基于域不变性表征学习的域泛化"></a>基于域不变性表征学习的域泛化</h5><p>理论保证: 如果不同数据域的表征相同，那么说明该表征具有一般性并且可以迁移到其他域，在该表征函数下，源域的数据和目标域的数据更接近了，使得泛化误差上界更紧。</p>
<ul>
<li>基于核的方法(Kernel-based methods): 将原始数据映射到高维空间，通过核函数可以在不知道样本高维坐标情况下计算两样本内积，将不同域样本映射到高维空间，分布一致。</li>
<li>域对抗学习(Domain adversarial learning): 判别器来识别数据属于不同域，生成器用来学习不变表征让判别器分辨不出来。</li>
<li>显式特征对齐(Explicit feature alignment): 在学习的过程显式让特征对齐 </li>
<li>不变风险最小化(Invariant risk minimization): </li>
</ul>
<h5 id="基于特征解耦的域泛化"><a href="#%E5%9F%BA%E4%BA%8E%E7%89%B9%E5%BE%81%E8%A7%A3%E8%80%A6%E7%9A%84%E5%9F%9F%E6%B3%9B%E5%8C%96" class="headerlink" title="基于特征解耦的域泛化"></a>基于特征解耦的域泛化</h5><p>将原始数据投影到特征空间，将特征分为domain-shared feature 和 domain-specific feature, 目标函数一般为如下形式: </p>
<script type="math/tex; mode=display">
\min _{g_c, g_s, f} \mathbb{E}_{\mathbf{x}, y} \ell\left(f\left(g_c(\mathbf{x})\right), y\right)+\lambda \ell_{\mathrm{reg}}+\mu \ell_{\mathrm{recon}}\left(\left[g_c(\mathbf{x}), g_s(\mathbf{x})\right], \mathbf{x}\right)</script><ul>
<li>多成分分析: 用不同的特征提取器来提取数据中的不同成分 </li>
<li>生成模型: Domain-invariant VAE </li>
<li>因果启发的方法: </li>
</ul>
<h4 id="学习策略"><a href="#%E5%AD%A6%E4%B9%A0%E7%AD%96%E7%95%A5" class="headerlink" title="学习策略"></a>学习策略</h4><h5 id="基于集成学习的域泛化"><a href="#%E5%9F%BA%E4%BA%8E%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0%E7%9A%84%E5%9F%9F%E6%B3%9B%E5%8C%96" class="headerlink" title="基于集成学习的域泛化"></a>基于集成学习的域泛化</h5><p>每个域都有一个模型，结合输出结果 </p>
<h5 id="基于元学习的域泛化"><a href="#%E5%9F%BA%E4%BA%8E%E5%85%83%E5%AD%A6%E4%B9%A0%E7%9A%84%E5%9F%9F%E6%B3%9B%E5%8C%96" class="headerlink" title="基于元学习的域泛化"></a>基于元学习的域泛化</h5><p>元学习的关键思想是从多个任务中学习一个通用的模型</p>
<h5 id="基于梯度操作的域泛化"><a href="#%E5%9F%BA%E4%BA%8E%E6%A2%AF%E5%BA%A6%E6%93%8D%E4%BD%9C%E7%9A%84%E5%9F%9F%E6%B3%9B%E5%8C%96" class="headerlink" title="基于梯度操作的域泛化"></a>基于梯度操作的域泛化</h5><p>self-challenging训练算法 </p>
<h5 id="基于分布鲁棒优化的方法"><a href="#%E5%9F%BA%E4%BA%8E%E5%88%86%E5%B8%83%E9%B2%81%E6%A3%92%E4%BC%98%E5%8C%96%E7%9A%84%E6%96%B9%E6%B3%95" class="headerlink" title="基于分布鲁棒优化的方法"></a>基于分布鲁棒优化的方法</h5><h5 id="基于自监督学习的域泛化"><a href="#%E5%9F%BA%E4%BA%8E%E8%87%AA%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0%E7%9A%84%E5%9F%9F%E6%B3%9B%E5%8C%96" class="headerlink" title="基于自监督学习的域泛化"></a>基于自监督学习的域泛化</h5><h3 id="其他的研究方向"><a href="#%E5%85%B6%E4%BB%96%E7%9A%84%E7%A0%94%E7%A9%B6%E6%96%B9%E5%90%91" class="headerlink" title="其他的研究方向"></a>其他的研究方向</h3><ul>
<li>单源域泛化</li>
<li>半监督域泛化 </li>
<li>域泛化的联邦学习</li>
</ul>
<h3 id="域泛化的应用"><a href="#%E5%9F%9F%E6%B3%9B%E5%8C%96%E7%9A%84%E5%BA%94%E7%94%A8" class="headerlink" title="域泛化的应用"></a>域泛化的应用</h3><p><img src="/.io//%E5%BA%94%E7%94%A8.png" alt></p>
<h3 id="我的想法"><a href="#%E6%88%91%E7%9A%84%E6%83%B3%E6%B3%95" class="headerlink" title="我的想法"></a>我的想法</h3><p>通过域对抗训练的方式使得模型能够提取出与模态解耦的中介变量<br><img src="/.io//%E5%9F%9F%E5%AF%B9%E6%8A%97%E8%AE%AD%E7%BB%83.png" alt></p>
<p>现有研究进展:</p>
<ul>
<li>源域数据+标签均可见，目标域仅有数据 </li>
<li>局限在视觉领域 </li>
<li>视觉领域是否有已将该工作迁移到未知域上,已经有了<a target="_blank" rel="noopener" href="https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Domain_Generalization_With_CVPR_2018_paper.pdf">2019cvpr</a></li>
<li>衡量分布差异: 最大平均偏差(MMD)，将特征映射到高维空间，然后用在高维空间的一阶矩来度量两个分布的差异。 </li>
</ul>
<p>可以有的创新方向: </p>
<ul>
<li>应用创新，在故障诊断领域还没这么搞得，同时可以结合之前的因果框架，提取出的中介变量更具说服力。 </li>
<li>将有标签数据与无标签数据混用，<strong>有模态标签，但无故障标签</strong>，在实际中也是常见的场景</li>
<li>结合时序数据的一些特点 </li>
</ul>
<h4 id="论文复现"><a href="#%E8%AE%BA%E6%96%87%E5%A4%8D%E7%8E%B0" class="headerlink" title="论文复现"></a>论文复现</h4><p>将<a target="_blank" rel="noopener" href="https://www.jmlr.org/papers/volume17/15-239/15-239.pdf">domain-adversarial training of neural network</a>论文结果在TEP上运行了一下，模型训练不收敛。</p>
{% endraw %}
