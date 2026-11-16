---
title: 数据结构与算法
date: '2022-09-29 12:00:00'
updated: '2022-10-17 12:00:00'
permalink: 2022/09/29/shu-ju-jie-gou-yu-suan-fa/
tags:
- 数据结构
categories:
- 算法岗八股文
- 灵魂雕琢
math: true
legacy_html: true
img: /medias/featureimages/3.png
summary: '数据结构与算法算是必考内容，如果能在笔试/面试的过程中快速找到算法题的解题思路，无疑会给面试官留下很好的印象，这类题目主要考察两个方面: 数据结构的掌握程度，能否熟练掌握各类数据结构的特点及操作
  常见算法，比如贪心算法、动态规划、BFS/DFS等 面试官的考察一般是循序渐进的： 一般首先会考察一些数'
---
{% raw %}
<p>数据结构与算法算是必考内容，如果能在笔试/面试的过程中快速找到算法题的解题思路，无疑会给面试官留下很好的印象，这类题目主要考察两个方面: </p>
<ul>
<li>数据结构的掌握程度，能否熟练掌握各类数据结构的特点及操作</li>
<li>常见算法，比如贪心算法、动态规划、BFS/DFS等 <span id="more"></span>
面试官的考察一般是循序渐进的：</li>
<li>一般首先会考察一些数据结构的知识来看被面试者的数据结构基础 </li>
<li>接下来会抛出一些常见算法，询问算法的实现细节、优缺点 </li>
<li>最后从一道具体的算法题目出发，考察被面试者分析问题，对算法举一反三的运用能力，一定要多注意思考有没有更好的解法</li>
<li>上机编程，考察被面试者的编程实现功底</li>
</ul>
<p>从上面的分析可以看出，前两项是后两项的理论基础，只有有了扎实的理论基础，后续在leetcode/牛客网刷题的时候才能事倍功半。 </p>
<h3 id="基础知识"><a href="#%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86" class="headerlink" title="基础知识"></a>基础知识</h3><h4 id="编程语言"><a href="#%E7%BC%96%E7%A8%8B%E8%AF%AD%E8%A8%80" class="headerlink" title="编程语言"></a>编程语言</h4><p>从大的层面来看，编程语言分为两类: </p>
<ul>
<li>高级语言: 编译型语言、解释型语言 </li>
<li>低级语言: 机器语言、汇编语言 </li>
</ul>
<h5 id="编译型语言"><a href="#%E7%BC%96%E8%AF%91%E5%9E%8B%E8%AF%AD%E8%A8%80" class="headerlink" title="编译型语言"></a>编译型语言</h5><blockquote>
<p><strong>编译型语言:</strong> C/C++、Pascal/Object Pascal、Golang<br><strong>定义:</strong> 程序在执行之前需要一个专门的编译过程，把程序编译成为机器语言的文件，运行时则不需要重新编译，直接运行编译后的结果就可以。</p>
<ul>
<li>优点: 程序执行效率高 </li>
<li>缺点: 依赖于编译器，跨平台性差 </li>
</ul>
</blockquote>
<h5 id="解释型语言"><a href="#%E8%A7%A3%E9%87%8A%E5%9E%8B%E8%AF%AD%E8%A8%80" class="headerlink" title="解释型语言"></a>解释型语言</h5><blockquote>
<p><strong>解释型语言:</strong> Python、Matlab、Javascript、Java、Ruby、C#、PHP等<br><strong>定义:</strong> 程序不需要编译，每次运行，都需要一边运行，一边翻译</p>
<ul>
<li>优点: 跨平台性能好 </li>
<li>缺点: 效率低 </li>
</ul>
</blockquote>
<h4 id="时间复杂度"><a href="#%E6%97%B6%E9%97%B4%E5%A4%8D%E6%9D%82%E5%BA%A6" class="headerlink" title="时间复杂度"></a>时间复杂度</h4><blockquote>
<p><strong>算法复杂度：</strong> 输入数据为$N$时，算法运行所需花费的时间 </p>
<ul>
<li>统计的是算法的<strong>计算操作数量</strong>，而不是算法的<strong>运行绝对时间</strong>，计算操作数量与算法运行绝对时间成正相关，但并不相等，还受到<strong>编程语言、计算机处理器速度、运行环境</strong>等多种因素的影响。 </li>
<li>体现的是计算操作随数据大小$N$变化时的变化情况，按照从小到大，常见的时间复杂度有: <script type="math/tex; mode=display">O(1) < O(logN) < O(N) < O(NlogN) < O(N^2) < O(2^N) < O(N!)</script></li>
</ul>
</blockquote>
<h4 id="空间复杂度"><a href="#%E7%A9%BA%E9%97%B4%E5%A4%8D%E6%9D%82%E5%BA%A6" class="headerlink" title="空间复杂度"></a>空间复杂度</h4><p>空间复杂度涉及的空间类型有: </p>
<ul>
<li>输入空间: 存储输入数据所需的空间大小 </li>
<li>暂存空间: 算法运行过程中，存储所有中间变量和对象等数据所需的空间大小 </li>
<li>输出空间: 算法运行返回时，存储输出数据所需的空间 </li>
</ul>
<p>空间复杂度的计算一般只需要考虑 <strong>暂存空间</strong> + <strong>输出空间</strong><br>根据不同来源，算法使用的内存空间分为三类： </p>
<ul>
<li>指令空间: 编译后，程序指令所使用的内存空间 </li>
<li>数据空间: 算法中的各项变量使用的空间</li>
<li>栈帧空间: 一般在递归时会需要考虑 </li>
</ul>
<p>按照从小到大排列，常见的算法空间复杂度有: </p>
<script type="math/tex; mode=display">O(1) < O(logN) < O(N) < O(N^2) < O(2^N)</script><h3 id="数据结构"><a href="#%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84" class="headerlink" title="数据结构"></a>数据结构</h3><p>在这一部分，我将回顾各种数据结构的概念以及优劣势</p>
<h4 id="数组"><a href="#%E6%95%B0%E7%BB%84" class="headerlink" title="数组"></a>数组</h4><p>数组一般用来存储相同类型的数据 </p>
<ul>
<li>特点<ul>
<li>长度固定，一般不可以动态扩展 </li>
<li>占据连续的内存空间 </li>
<li>数据的访问方式为随机访问 </li>
</ul>
</li>
<li>优点: 查改效率高，通过数组名和下标进行访问，¬$O(1)$复杂度 </li>
<li>缺点: 增删效率低，需要移动增删元素之后的所有元素 </li>
</ul>
<p><img src="/.io//%E6%95%B0%E7%BB%84.jpeg" alt="数组"></p>
<h4 id="链表"><a href="#%E9%93%BE%E8%A1%A8" class="headerlink" title="链表"></a>链表</h4><p>链表相较于数组，除了数据域，还增加了指针域用于存储下一个节点的地址，链表中每一个元素都包含此节点的数据和指向下一节点地址的指针。</p>
<ul>
<li>特点<ul>
<li>长度可动态变化</li>
<li>非连续的内存空间</li>
<li>数据的访问方式为顺序访问 </li>
</ul>
</li>
<li>优点: 增删效率后，$O(1)$的时间复杂度 </li>
<li>缺点: 查改效率低，只能通过遍历节点依次查询，$O(N)$的时间复杂度  </li>
</ul>
<p><img src="/.io//%E9%93%BE%E8%A1%A8.jpeg" alt="链表"></p>
<h4 id="栈"><a href="#%E6%A0%88" class="headerlink" title="栈"></a>栈</h4><p>栈本身是一个线性表，但在这个表中只有一个口子允许数据的进出，其特点在于栈中元素的<strong>先进后出(FILO)</strong>，常用的操作有<code>push()、pop()、top()</code></p>
<p><img src="/.io//%E6%A0%88.gif" alt="栈"></p>
<h5 id="单调栈"><a href="#%E5%8D%95%E8%B0%83%E6%A0%88" class="headerlink" title="单调栈"></a>单调栈</h5><p>单调栈是一种特殊的栈，栈中的元素是单调递增/递减: </p>
<ul>
<li>单调递增栈: 可以找到元素向左遍历第一个比它小的元素 </li>
<li>单调递减栈: 可以找到元素向左遍历第一个比它大的元素 </li>
</ul>
<p>借助单调栈可以在线性时间内完成左侧第一个小于/大于元素的搜索，下面介绍几个典型题目 </p>
<ul>
<li>接雨水<br><img src="/.io//%E6%8E%A5%E9%9B%A8%E6%B0%B4.png" alt><ul>
<li><blockquote>
<p>题目描述: 给定一个列表表示高度图，计算下雨后能接多少雨水 </p>
</blockquote>
</li>
<li><blockquote>
<p>解题思路: 维护一个单调递减栈，一旦发现当前元素大于栈顶元素，当前元素便是右边界，栈顶元素出栈，此时单调栈的栈顶元素便成为了左侧边界，取两者最小值作为高度，乘以左右距离便得到了装水量。</p>
</blockquote>
</li>
</ul>
</li>
<li>求直方图中最大矩阵面积<br><img src="/.io//%E6%9C%80%E5%A4%A7%E7%9F%A9%E9%98%B5%E9%9D%A2%E7%A7%AF.png" alt><ul>
<li><blockquote>
<p>题目描述: 给定一个列表表示直方图，求该直方图中最大矩阵面积</p>
</blockquote>
</li>
<li><blockquote>
<p>解题思路: 维护一个单调递增栈，当遇到大的元素时直接进栈，一旦发现当前元素小于栈顶元素，则当前元素便是栈顶元素的右边界，而栈顶元素出栈后，下一个栈顶元素便是左边界，由此便可计算矩阵面积，遍历一次找到最大的矩阵面积即可。 </p>
</blockquote>
</li>
</ul>
</li>
<li>最大矩形<br><img src="/.io//%E6%9C%80%E5%A4%A7%E7%9F%A9%E5%BD%A2.png" alt><ul>
<li><blockquote>
<p>题目描述: 给定一个仅包含 0 和 1 、大小为 rows x cols 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。</p>
</blockquote>
</li>
<li><blockquote>
<p>解题思路: 按行遍历，然后每次都堆叠列得到一个柱状图，然后通过单调栈算法，求最大矩形面积</p>
</blockquote>
</li>
</ul>
</li>
</ul>
<p>单调栈算法总结: </p>
<blockquote>
<p>单调栈是一个原理简单，但不太好想到的一种思路，在leetcode往往是以hard题目形式出现，线性时间复杂度是该数据结构最大的优势，对于此类题目往往会先考虑暴力解法，如果需要寻找各个元素的左边界/右边界，则考虑能否应用单调栈降低时间复杂度。 </p>
</blockquote>
<h4 id="队列"><a href="#%E9%98%9F%E5%88%97" class="headerlink" title="队列"></a>队列</h4><p>队列是一种线性数据结构，与栈的后进先出相对应，队列是<strong>先进先出(FIFO)</strong><br><img src="/.io//%E9%98%9F%E5%88%97.gif" alt="队列"></p>
<pre class="line-numbers language-lang-python"><code class="language-lang-python">import queue 
q = queue.Queue()
q.put(1)   #向队列中加入元素 
q.get()    #取出队列中最先进入的元素 
q.empty()  #判断队列是否为空 
q.qsize()  #获取队列长度
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span><span></span><span></span></span></code></pre>
<h4 id="树"><a href="#%E6%A0%91" class="headerlink" title="树"></a>树</h4><p>树作为一种树状的数据机构，其将有限个节点根据不同的层次关系进行排列，从而形成数据之间的父子关系，从某种角度来说，可以将树看作链表的高配版，对普通链表的指针域进行了扩展。<br>下面介绍一些常见的树的类型: </p>
<ul>
<li>完全二叉树: 除了最后一层节点，其他层的节点数都达到了最大值，同时最后一层的节点都是按照从左到右依次排布</li>
<li>满二叉树: 除了最后一层，其他层的节点都有两个子节点 </li>
<li>二叉排序树((二叉搜索树): 左子树上所有结点的值均小于根结点的值，右子树上所有结点的值均大于根结点，左右子树也分别时二叉排序树</li>
<li>平衡二叉树: 平衡二叉树又被称为AVL树，它是一个二叉排序树，且左右子树的高度差距不超过1，左右子树也都是二叉平衡树 <ul>
<li>优点: 平衡二叉树为了追求高度平衡，需要通过平衡处理使得左右子树的高度差必须小于等于1，高度平衡带来了更高的搜索效</li>
<li>缺点： 缺点则是对树中结点进行插入删除时需要经过多次旋转实现复衡，这导致AVL的插入和删除效率并不高。</li>
</ul>
</li>
<li>红黑树: <ul>
<li>每个结点要么是红的，要么是黑的</li>
<li>根结点是黑的 </li>
<li>每个叶结点(None)都是黑的 </li>
<li>如果一个结点是红的，那么它的两个儿子都是黑的</li>
<li>对于任意结点而言，其到叶结点树尾端None指针的每条路径都包含相同数目的黑结点<br><img src="/.io//%E7%BA%A2%E9%BB%91%E6%A0%91.jpeg" alt="红黑树"></li>
</ul>
</li>
<li>红黑树与平衡二叉树对比<br><img src="/.io//%E7%BA%A2%E9%BB%91vs%E5%B9%B3%E8%A1%A1.png" alt="AVLvs红黑树"></li>
</ul>
<p>树的遍历方式有四种: </p>
<ul>
<li>前序遍历: 按照 根结点-&gt;左子结点-&gt;右子结点的顺序进行遍历 </li>
<li>中序遍历: 按照 左子结点-&gt;根结点-&gt;右子结点的顺序进行遍历 </li>
<li>后序遍历: 按照 左子结点-&gt;右子结点-&gt;根结点的顺序进行遍历 </li>
<li>层次遍历: 从根结点开始，按结点所处层级进行遍历 <h4 id="堆"><a href="#%E5%A0%86" class="headerlink" title="堆"></a>堆</h4>堆通常可以看做一棵树的数组对象，堆的实现一般不通过指针域，而是通过构建一个一维数组与二叉树的父子结点进行对应，因此堆总是一棵完全二叉树。</li>
<li>对于任意一个父结点的序号$n$来说，它的子结点的序号一定是$2n+1, 2n+2$，因此可以直接用一个数组表示一个堆</li>
<li>堆中某个结点的值总是不大于或者不小于其父节点的值: 根结点最大的堆称为<strong>最大堆</strong>;根结点最小的堆称为<strong>最小堆</strong></li>
<li>堆常用来实现优先队列，比如堆排序、top$K$问题等 </li>
</ul>
<p><img src="/.io//%E5%A0%86.jpeg" alt></p>
<pre class="line-numbers language-lang-python"><code class="language-lang-python">import heapq  
pq = [] 
value = 1
heapq.heappush(pq, value)    #向小根堆中加入元素 
heapq.heapreplace(pq, value) #替换堆顶元素 
heapq.pop(pq)                #弹出堆顶元素
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span><span></span><span></span></span></code></pre>
<h4 id="哈希表"><a href="#%E5%93%88%E5%B8%8C%E8%A1%A8" class="headerlink" title="哈希表"></a>哈希表</h4><p>哈希表也叫散列表，是一种通过键值堆直接访问数据的结构，借助于哈希表可以实现$O(1)$的数据访问效率。<br><img src="/.io//%E5%93%88%E5%B8%8C%E8%A1%A8.jpeg" alt></p>
<p>哈希表实现的最关键就是哈希函数的定义和选择，一般常用的有以下几种： </p>
<ul>
<li>直接寻址法: 取关键字或者关键字的某个线性函数作为哈希地址 </li>
<li>数字分析法: 通过数据的特点，发现数据中冲突较少的部分，构造哈希地址。 </li>
<li>平方取中法:</li>
<li>取随机数法: 通常适用于关键字长度不同的场合</li>
<li>除留取余法: 取关键字被某个不大于散列表的表长$n$的数$m$除后所得的余数$p$作为散列地址，一般取素数或者直接用$n$。 </li>
</ul>
<p><strong>哈希冲突</strong>: 当不同的key值访问到同一个地址时，就发生了哈希冲突。<br>哈希冲突主要有以下解决方案： </p>
<ul>
<li>链地址法: 在冲突的地方再做一个链表</li>
<li>开放寻址法: 探测后续未被占用的地址 </li>
<li>再哈希: 使用关键字的其他部分继续计算地址，更换哈希函数 </li>
</ul>
<p><img src="/.io//%E9%93%BE%E5%9C%B0%E5%9D%80%E6%B3%95.jpeg" alt> </p>
<h4 id="图"><a href="#%E5%9B%BE" class="headerlink" title="图"></a>图</h4><p>图一般包含顶点和边，可粗略分为有向图和无向图。</p>
<ul>
<li>图的存储一般通过<strong>邻接矩阵</strong>实现<br><img src="/.io//%E9%82%BB%E6%8E%A5%E7%9F%A9%E9%98%B51.jpeg" alt></li>
<li>邻接矩阵往往是稀疏的，存储大量的0导致浪费了大量的空间，，一种只存储相连顶点关系的邻接表应运而生。<ul>
<li>在邻接表中，图的每一个顶点都是一个链表的头结点，其后连接着该节点能够到达的相邻节点。</li>
<li>邻接表只包含个个结点的出度信息</li>
</ul>
</li>
<li>逆邻接表: 包含结点的入度信息</li>
<li>十字链表 </li>
</ul>
<h3 id="常见算法"><a href="#%E5%B8%B8%E8%A7%81%E7%AE%97%E6%B3%95" class="headerlink" title="常见算法"></a>常见算法</h3><h4 id="动态规划"><a href="#%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92" class="headerlink" title="动态规划"></a>动态规划</h4><p>动态规划是算法与数据结构的重难点，其中包含了「分治思想」、「空间换时间」、「最优解」等多种算法思想，这部分将从以下三个方面对动态规划算法进行介绍: </p>
<ul>
<li>动态规划问题特点，<strong>动态规划</strong>和<strong>分治算法</strong>的联系与区别 </li>
<li>介绍<strong>重叠子问题</strong>与<strong>最优子结构</strong>分别是什么，以及动态规划是如何解决的 </li>
<li>动态规划的解题框架总结 </li>
</ul>
<h5 id="动态规划与分治"><a href="#%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E4%B8%8E%E5%88%86%E6%B2%BB" class="headerlink" title="动态规划与分治"></a>动态规划与分治</h5><ol>
<li><strong>分治算法</strong>: 「分治」是算法中的一种基本思想，通过将<strong>原问题分解为子问题</strong>，不断递归地将子问题分解为更小的子问题，并通过<strong>组合子问题的解</strong>来得到原问题的解， e.g. 归并排序 </li>
<li><strong>动态规划</strong>: 与「分治」的相同点在于「动态规划」也是通过组合子问题的解来得到原问题的解，不同的是，能用「动态规划」解决的问题具有「<strong>重叠子问题</strong>」和「<strong>最优子结构</strong>」两大特性。</li>
</ol>
<h5 id="动态规划问题特性"><a href="#%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E9%97%AE%E9%A2%98%E7%89%B9%E6%80%A7" class="headerlink" title="动态规划问题特性"></a>动态规划问题特性</h5><ul>
<li><strong>重叠子问题</strong>: 动态规划的子问题是有重叠的，即各个子问题中包含<strong>重复的更小子问题</strong>，若使用暴力法穷举，求解这些相同子问题会产生大量重复计算，效率低下。<br><img src="/.io//%E6%96%90%E6%B3%A2%E9%82%A3%E5%A5%91%E6%95%B0%E5%88%97.png" alt><pre class="line-numbers language-lang-python"><code class="language-lang-python"># 求第 n 个斐波那契数
def fibonacci(n):
  if n == 0: return 0       # 若求 f(0) 则直接返回 0
  a, b = 0, 1               # 初始化 f(0), f(1)
  for i in range(2, n + 1): # 状态转移求取 f(2), f(3), ..., f(n) 
      a, b = b, a + b
  return b                  # 返回 f(n)
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span><span></span><span></span><span></span></span></code></pre>
</li>
<li><strong>最优子结构</strong>: 如果一个问题的最优解可以由其子问题的最优解组合而成，并且这些子问题可以独立求解，那么称此问题具有最优子结构。<br><img src="/.io//%E8%9B%8B%E7%B3%95%E6%9C%80%E9%AB%98%E5%94%AE%E4%BB%B7.png" alt><pre class="line-numbers language-lang-python"><code class="language-lang-python"># 输入蛋糕价格列表 price_list ，求重量为 n 蛋糕的最高售价
def max_cake_price(n, price_list):
  if n &lt;= 1: return price_list[n] # 蛋糕重量 &lt;= 1 时直接返回
  dp = [0] * (n + 1)              # 初始化 dp 列表
  for j in range(1, n + 1):       # 按顺序计算 f(1), f(2), ..., f(n)
      for i in range(j):          # 从 j 种组合种选择最高售价的组合作为 f(j)
          dp[j] = max(dp[j], dp[i] + price_list[j - i])
  return dp[n]
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></span></code></pre>
</li>
</ul>
<h5 id="动态规划解题框架"><a href="#%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E8%A7%A3%E9%A2%98%E6%A1%86%E6%9E%B6" class="headerlink" title="动态规划解题框架"></a>动态规划解题框架</h5><p>首先分析问题是否具有「<strong>重叠子问题</strong>」与「<strong>最优子结构</strong>」两个特性，若有则可以考虑用动态规划按照以下步骤进行求解: </p>
<ol>
<li><strong>状态定义</strong>: 构建问题最优解模型，包括问题<strong>问题最优解的定义</strong>、<strong>有哪些计算解的自变量</strong></li>
<li><strong>初始状态</strong>: 确定基础子问题的解，原问题和子问题的解都是以基础子问题的解为起始点，在迭代计算中得到 </li>
<li><strong>转移方程</strong>: 确定原问题与子问题之间的关系，以及使用何种选择规则从子问题最优解中组合出原问题的最优解 </li>
<li><strong>返回值</strong>: 确定应返回的问题的解是什么，即动态规划在何处停止迭代</li>
</ol>
<p><strong>示例: 斐波那契数列</strong></p>
<ul>
<li><strong>状态定义</strong>: 一维$dp$列表，设第$i$个斐波那契数为$dp[i]$ </li>
<li><strong>初始状态</strong>: 已知第$0,1$个斐波那契数为$dp[0] = 0, dp[1] = 1$</li>
<li><strong>转移方程</strong>: 后一个数字等于前两个数字之和<script type="math/tex; mode=display">dp[i] = dp[i-1] + dp[i-2]</script></li>
<li><strong>返回值</strong>: 第$n$个斐波那契数$dp[n]$</li>
</ul>
<h4 id="排序算法"><a href="#%E6%8E%92%E5%BA%8F%E7%AE%97%E6%B3%95" class="headerlink" title="排序算法"></a>排序算法</h4><h5 id="概述"><a href="#%E6%A6%82%E8%BF%B0" class="headerlink" title="概述"></a>概述</h5><p>排序算法用作实现列表的排序，列表元素可以是整数，也可以是浮点数、字符串等其他数据类型，生活中有许多需要用到排序算法的场景: </p>
<ul>
<li><strong>数字排序</strong>: 对于一个数组，我们希望将所有数字从小到大/从大到小排序 </li>
<li><strong>字符串排序</strong>: 对于一个字符串列表，我们希望将所有单词按照字符先后排序 </li>
<li><strong>自定义排序</strong>: 对于任意一个<strong>已定义比较规则</strong>的集合，我们希望将其按规则排序 </li>
</ul>
<p><img src="/.io//%E6%8E%92%E5%BA%8F%E6%A6%82%E8%BF%B0.png" alt></p>
<h5 id="常见排序算法"><a href="#%E5%B8%B8%E8%A7%81%E6%8E%92%E5%BA%8F%E7%AE%97%E6%B3%95" class="headerlink" title="常见排序算法"></a>常见排序算法</h5><p>常见排序算法包括「冒泡排序」、「插入排序」、「选择排序」、「快速排序」、「归并排序」、「堆排序」、「基数排序」、「桶排序」。如下图所示，为各排序算法的核心特性与时空复杂度总结<br><img src="/.io//%E6%8E%92%E5%BA%8F%E7%AE%97%E6%B3%95%E6%97%B6%E7%A9%BA%E5%A4%8D%E6%9D%82%E5%BA%A6.png" alt><br>如下图所示，为在 「随机乱序」、「接近有序」、「完全倒序」、「少数独特」四类输入数据下，各常见排序算法的排序过程。</p>
<p><img src="/.io//%E5%86%92%E6%B3%A1%E6%8E%92%E5%BA%8F%E5%8A%A8%E7%94%BB.gif" alt><br><img src="/.io//%E6%8F%92%E5%85%A5%E6%8E%92%E5%BA%8F%E5%8A%A8%E7%94%BB.gif" alt><br><img src="/.io//%E9%80%89%E6%8B%A9%E6%8E%92%E5%BA%8F%E5%8A%A8%E7%94%BB.gif" alt><br><img src="/.io//%E5%BF%AB%E9%80%9F%E6%8E%92%E5%BA%8F%E5%8A%A8%E7%94%BB.gif" alt><br><img src="/.io//%E5%BD%92%E5%B9%B6%E6%8E%92%E5%BA%8F%E5%8A%A8%E7%94%BB.gif" alt><br><img src="/.io//%E5%A0%86%E6%8E%92%E5%BA%8F%E5%8A%A8%E7%94%BB.gif" alt></p>
<h5 id="排序算法主要可根据稳定性、就地性、自适应性分类，理想的排序算法具有以下特征"><a href="#%E6%8E%92%E5%BA%8F%E7%AE%97%E6%B3%95%E4%B8%BB%E8%A6%81%E5%8F%AF%E6%A0%B9%E6%8D%AE%E7%A8%B3%E5%AE%9A%E6%80%A7%E3%80%81%E5%B0%B1%E5%9C%B0%E6%80%A7%E3%80%81%E8%87%AA%E9%80%82%E5%BA%94%E6%80%A7%E5%88%86%E7%B1%BB%EF%BC%8C%E7%90%86%E6%83%B3%E7%9A%84%E6%8E%92%E5%BA%8F%E7%AE%97%E6%B3%95%E5%85%B7%E6%9C%89%E4%BB%A5%E4%B8%8B%E7%89%B9%E5%BE%81" class="headerlink" title="排序算法主要可根据稳定性、就地性、自适应性分类，理想的排序算法具有以下特征:"></a>排序算法主要可根据<strong>稳定性、就地性、自适应性</strong>分类，理想的排序算法具有以下特征:</h5><ul>
<li><strong>稳定性</strong>: 相等元素的相对位置不变化 </li>
<li><strong>就地性</strong>: 不使用额外的辅助空间 </li>
<li><strong>自适应性</strong>: 时间复杂度受元素分布影响 </li>
</ul>
<p>按照稳定性来分类: </p>
<ul>
<li>稳定排序: 冒泡排序、插入排序、归并排序、基数排序、桶排序 </li>
<li>非稳定排序: 选择排序、快速排序、堆排序 </li>
</ul>
<p>按照就地性进行分类: </p>
<ul>
<li>原地排序: 不使用额外辅助数组，例如: 冒泡排序、插入排序、选择排序、快速排序、堆排序 </li>
<li>非原地排序: 使用额外辅助数组，例如: 基数排序、桶排序等 </li>
</ul>
<h5 id="时空复杂度"><a href="#%E6%97%B6%E7%A9%BA%E5%A4%8D%E6%9D%82%E5%BA%A6" class="headerlink" title="时空复杂度"></a>时空复杂度</h5><p>常见排序算法时空复杂度见下图</p>
<p><img src="/.io//%E6%8E%92%E5%BA%8F%E5%A4%8D%E6%9D%82%E5%BA%A6.png" alt></p>
<ul>
<li>普通「冒泡排序」的最佳时间复杂度为$O(N^2)$，通过增加标志位实现提前返回 ，可以将最佳时间复杂度降低至$O(N)$</li>
<li>在输入列表完全倒序下，普通「快速排序」的空间复杂度劣化至$O(N)$通过代码优化 Tail Call Optimization 保持算法递归较短子数组，可以将最差递归深度降低至 $\log N$ </li>
<li>普通「快速排序」总以最左或最右元素为基准数，因此在输入列表有序或倒序下，时间复杂度劣化至$O(N^2)$; 通过随机选择基准数, 可极大减少此类最差情况发生, 尽可能地保持$O(NlogN)$的时间复杂度。</li>
<li>若输入列表是数组，则归并排序的空间复杂度为$O(N)$; 而若排序链表, 则「归并排序」不需要借助额外辅助空间，空间复杂度可以降低至$O(1)$</li>
</ul>
<h5 id="快速排序"><a href="#%E5%BF%AB%E9%80%9F%E6%8E%92%E5%BA%8F" class="headerlink" title="快速排序"></a>快速排序</h5><p>快速排序分为两步: </p>
<ul>
<li><strong>哨兵划分</strong>: 以数组某个元素为<strong>基准数</strong>，将所有小于基准数的元素移至其左边，大于基准数的元素移动至其右边</li>
<li><strong>递归</strong>: 对左子数组和右子数组分别执行<strong>哨兵划分</strong>，直至子数组长度为1终止递归 </li>
</ul>
<p><img src="/.io//%E5%BF%AB%E9%80%9F%E6%8E%92%E5%BA%8F.png" alt></p>
<pre class="line-numbers language-lang-python"><code class="language-lang-python">def quick_sort(nums, l, r):
    # 子数组长度为 1 时终止递归
    if l &gt;= r: return
    # 哨兵划分操作
    i = partition(nums, l, r)
    # 递归左（右）子数组执行哨兵划分
    quick_sort(nums, l, i - 1)
    quick_sort(nums, i + 1, r)

def partition(nums, l, r):
    # 以 nums[l] 作为基准数
    i, j = l, r
    while i &lt; j:
        while i &lt; j and nums[j] &gt;= nums[l]: j -= 1
        while i &lt; j and nums[i] &lt;= nums[l]: i += 1
        nums[i], nums[j] = nums[j], nums[i]
    nums[l], nums[i] = nums[i], nums[l]
    return i

# 调用
nums = [3, 4, 1, 5, 2]
quick_sort(nums, 0, len(nums) - 1)
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></span></code></pre>
<h5 id="归并排序"><a href="#%E5%BD%92%E5%B9%B6%E6%8E%92%E5%BA%8F" class="headerlink" title="归并排序"></a>归并排序</h5><p>归并排序体现了<strong>分治</strong>的算法思想，具体为: </p>
<ul>
<li><strong>分</strong>: 不断将数组从中点位置划分开，将原数组的排序问题转化为子数组的排序问题</li>
<li><strong>治</strong>: 划分到子数组长度为1时，开始向上合并，不断将<strong>左右两个较短排序数组</strong>合并为一个<strong>较长排序数组</strong>，直至合并至原数组完成排序</li>
</ul>
<p><img src="/.io//%E5%BD%92%E5%B9%B6%E6%8E%92%E5%BA%8F.png" alt></p>
<p>代码实现</p>
<pre class="line-numbers language-lang-python"><code class="language-lang-python">def merge_sort(nums, l, r):
    # 终止条件
    if l &gt;= r: return
    # 递归划分数组
    m = (l + r) // 2
    merge_sort(nums, l, m)
    merge_sort(nums, m + 1, r)
    # 合并子数组
    tmp = nums[l:r + 1]       # 暂存需合并区间元素
    i, j = 0, m - l + 1       # 两指针分别指向左/右子数组的首个元素
    for k in range(l, r + 1): # 遍历合并左/右子数组
        if i == m - l + 1:
            nums[k] = tmp[j]
            j += 1
        elif j == r - l + 1 or tmp[i] &lt;= tmp[j]:
            nums[k] = tmp[i]
            i += 1
        else:
            nums[k] = tmp[j]
            j += 1

# 调用
nums = [3, 4, 1, 5, 2, 1]
merge_sort(0, len(nums) - 1)
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></span></code></pre>
<h4 id="搜索与回溯算法"><a href="#%E6%90%9C%E7%B4%A2%E4%B8%8E%E5%9B%9E%E6%BA%AF%E7%AE%97%E6%B3%95" class="headerlink" title="搜索与回溯算法"></a>搜索与回溯算法</h4><h5 id="概述-1"><a href="#%E6%A6%82%E8%BF%B0-1" class="headerlink" title="概述"></a>概述</h5><ul>
<li><strong>搜索</strong>是计算机解题中的常用算法，其本质是通过某种规则实现对解空间的遍历</li>
<li><strong>回溯</strong>搜索算法中的一种控制策略，它的基本思想是: 为了求得问题的解，先选择某一种可能的情况向前探索，在探索过程中，一旦发现上一步选择是错的，就退回一步重新选择，继续向前探索，直至得到解或者证明无解。 </li>
</ul>
<p>这类算法大致可以分为两类: </p>
<ul>
<li>暴力搜索(无信息搜索算法): 广度优先搜索(BFS)、广度优先搜索(DFS)</li>
<li>启发式搜索(有信息搜索算法): 贪心搜索、A*搜索 </li>
</ul>
<h5 id="广度优先搜索算法"><a href="#%E5%B9%BF%E5%BA%A6%E4%BC%98%E5%85%88%E6%90%9C%E7%B4%A2%E7%AE%97%E6%B3%95" class="headerlink" title="广度优先搜索算法"></a>广度优先搜索算法</h5><p>BFS，类似地毯式搜索，以距离为层数逐层搜索，直到找到答案或者确认无解<br><img src="/.io//BFS.gif" alt><br>广度优先搜索一般通过队列实现: </p>
<ul>
<li>起始结点入队列</li>
<li>遍历与启示结点有连通的结点，入队列 </li>
<li>起始结点出队列</li>
<li>重复以上步骤，直至找到目标点或者队列为空(无解)</li>
</ul>
<h5 id="深度优先搜索算法"><a href="#%E6%B7%B1%E5%BA%A6%E4%BC%98%E5%85%88%E6%90%9C%E7%B4%A2%E7%AE%97%E6%B3%95" class="headerlink" title="深度优先搜索算法"></a>深度优先搜索算法</h5><p>深度优先搜索(DFS)会从第一个指定的顶点开始遍历图，沿着这条路径直到最后一个顶点被访问了，再原路折回(回溯)探索下一条路径。<br><img src="/.io//DFS.gif" alt></p>
<p>深度优先搜索一般是通过递归回溯实现 </p>
<h5 id="贪心搜索"><a href="#%E8%B4%AA%E5%BF%83%E6%90%9C%E7%B4%A2" class="headerlink" title="贪心搜索"></a>贪心搜索</h5><p>当边上有权值时，BFS便不适用了，此时需要加入松弛变量来表示结点与顶点之间的距离，这就是dijkstra算法，对于无权图/树，dijkstra算法就退化成了BFS</p>
<p>动态规划是贪心的泛化，贪心是动态规划的特例(贪婪解也恰好是Bellman最优时)</p>
<h5 id="A-搜索"><a href="#A-%E6%90%9C%E7%B4%A2" class="headerlink" title="A*搜索"></a>A*搜索</h5><p>做到题目了再来补</p>
<h4 id="位运算"><a href="#%E4%BD%8D%E8%BF%90%E7%AE%97" class="headerlink" title="位运算"></a>位运算</h4><h5 id="概念"><a href="#%E6%A6%82%E5%BF%B5" class="headerlink" title="概念"></a>概念</h5><p>程序中的所有数在计算机内存中都是以二进制的形式存储的，位运算就是直接对整数在内存中的二进制位进行运算，常见的运算操作符有: </p>
<ul>
<li>与(&amp;) </li>
<li>或(|) </li>
<li>异或(^) </li>
<li>取反(~) </li>
<li>左移(&lt;&lt;) </li>
<li>右移(&gt;&gt;) </li>
</ul>
<h5 id="与运算"><a href="#%E4%B8%8E%E8%BF%90%E7%AE%97" class="headerlink" title="与运算"></a>与运算</h5><p>两位同时为1，结果为1，否则为0<br>与运算的特殊用途: </p>
<ul>
<li>清零: 任何数字与0相与都是0 </li>
<li>取指定位: 与一个特定位为1，其他位为0的数字进行与运算即可取原数字中该特定位</li>
</ul>
<p><img src="/.io//%E4%B8%8E%E8%BF%90%E7%AE%97.png" alt></p>
<h5 id="或运算"><a href="#%E6%88%96%E8%BF%90%E7%AE%97" class="headerlink" title="或运算"></a>或运算</h5><p>二进制对应位两两进行或运算(有1则1，无1则0)<br>或运算的特殊用途: </p>
<ul>
<li>用来将数字的某些位置1 </li>
</ul>
<p><img src="/.io//%E6%88%96%E8%BF%90%E7%AE%97.png" alt></p>
<h5 id="异或运算"><a href="#%E5%BC%82%E6%88%96%E8%BF%90%E7%AE%97" class="headerlink" title="异或运算"></a>异或运算</h5><p>二进制对应位两两进行异或运算(相同为1，不同为0)<br><img src="/.io//%E5%BC%82%E6%88%96%E8%BF%90%E7%AE%97.png" alt></p>
<p>异或运算的特殊用途: </p>
<ul>
<li>将低位1置0(n&amp;(n-1)将低位1置0) </li>
<li>使某些特定的位翻转(与指定位为1的数字进行异或运算) </li>
<li>判断两个值是否相等(相等则异或值为0)</li>
<li>实现两个值的交换(a = a^b, b = b^a, a = a^b) </li>
</ul>
<h5 id="按位取反"><a href="#%E6%8C%89%E4%BD%8D%E5%8F%96%E5%8F%8D" class="headerlink" title="按位取反"></a>按位取反</h5><p>二进制的0变成1，1变成0<br><img src="/.io//%E5%8F%96%E5%8F%8D%E8%BF%90%E7%AE%97.png" alt> </p>
<h5 id="移位运算符"><a href="#%E7%A7%BB%E4%BD%8D%E8%BF%90%E7%AE%97%E7%AC%A6" class="headerlink" title="移位运算符"></a>移位运算符</h5><ul>
<li>左移运算符$&lt;&lt;$: 左移后右侧补0 </li>
<li>右移运算符$&gt;&gt;$: 右移后左边补原左位值</li>
</ul>
<p>说明: </p>
<ul>
<li>左移一位相当于整个数乘以二 </li>
<li>右移一位相当于整数除以二</li>
</ul>
<p><img src="/.io//%E7%A7%BB%E4%BD%8D%E8%BF%90%E7%AE%97%E7%AC%A6.png" alt> </p>
<h5 id="位运算小技巧"><a href="#%E4%BD%8D%E8%BF%90%E7%AE%97%E5%B0%8F%E6%8A%80%E5%B7%A7" class="headerlink" title="位运算小技巧"></a>位运算小技巧</h5><ul>
<li>判断奇偶数<br>```python</li>
</ul>
<p>if(n &amp; 1 == 1){<br>// n 是个奇数。<br>} </p>
<pre><code>- 交换两个数
```python
a=a^b; #a=a^b
b=a^b; #b=(a^b)^b=a^0=a
a=a^b; #a=(a^b)^(a^b^b)=0^b=0
</code></pre><h5 id="经典问题"><a href="#%E7%BB%8F%E5%85%B8%E9%97%AE%E9%A2%98" class="headerlink" title="经典问题"></a>经典问题</h5><ul>
<li>不用加减乘除做加法</li>
<li>二进制中1的个数</li>
<li>只出现一次的数字 </li>
</ul>
<p>….未完待续，后序刷题碰到未掌握的知识点后回来补充</p>
{% endraw %}
