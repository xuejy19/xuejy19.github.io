---
title: 剑指offer题目总结
date: '2022-09-29 12:00:00'
updated: '2022-10-09 12:00:00'
permalink: 2022/09/29/suan-fa-ti-zong-jie/
tags:
- 数据结构
categories:
- 算法岗八股文
- 游戏
math: false
legacy_html: true
img: /medias/featureimages/4.png
summary: 剑指offer题目总结链表题目 从尾到头打印链表 反转链表 合并两个排序的链表(递归/遍历) 两个链表的第一个公共节点(双指针/哈希表) 链表中环的入口节点(快慢指针/字典)
  链表中倒数最后k个节点(快慢指针) 复杂链表的复制(1. 复制 2. 随机指针复制 3.打断两个链表) -手写可能出错 删除
---
{% raw %}
<h3 id="剑指offer题目总结"><a href="#%E5%89%91%E6%8C%87offer%E9%A2%98%E7%9B%AE%E6%80%BB%E7%BB%93" class="headerlink" title="剑指offer题目总结"></a>剑指offer题目总结</h3><h4 id="链表题目"><a href="#%E9%93%BE%E8%A1%A8%E9%A2%98%E7%9B%AE" class="headerlink" title="链表题目"></a>链表题目</h4><ul>
<li>从尾到头打印链表  </li>
<li>反转链表 </li>
<li>合并两个排序的链表(递归/遍历)</li>
<li>两个链表的第一个公共节点(双指针/哈希表) </li>
<li>链表中环的入口节点(快慢指针/字典) </li>
<li>链表中倒数最后k个节点(快慢指针) </li>
<li>复杂链表的复制(1. 复制 2. 随机指针复制 3.打断两个链表) -手写可能出错 </li>
<li>删除链表中的节点(简单遍历)<span id="more"></span>
<blockquote>
<p>总结：简单遍历+快慢指针+哈希 </p>
</blockquote>
</li>
</ul>
<h4 id="树题目"><a href="#%E6%A0%91%E9%A2%98%E7%9B%AE" class="headerlink" title="树题目"></a>树题目</h4><ul>
<li>二叉树的深度(简单递归/队列辅助层次遍历)</li>
<li>按之字顺序打印二叉树(层次遍历 + flag判断奇偶层)</li>
<li>二叉搜索树的第k个节点(深度优先遍历，结合二叉搜索树的特点)</li>
<li>重建二叉树(递归 + 结合前序中序遍历的特点) </li>
<li>树的子结构(递归，注意三种匹配情况) </li>
<li>二叉树的镜像(递归，左右子树镜像，交换)</li>
<li>从上往下打印二叉树(简单的层次遍历) </li>
<li>二叉搜索树的后序遍历序列(递归，后序遍历，根节点) </li>
<li>二叉树中和为某一值的路径1(递归，不需保存路径)</li>
<li>二叉树中和为某一值的路径2(递归，保存路径, dfs)</li>
<li>二叉搜索树与双向链表(中序遍历，维护一个pre_node)</li>
<li>判断是不是平衡二叉树(递归判断，专门写一个判断树深度的函数即可)</li>
<li>二叉树的下一个节点(获取中序遍历序列/分情况讨论)</li>
<li>对称的二叉树(递归，输入两棵树，左右子树是否match， 因为单一递归，同时只能到一部分，需要两部分同步递归)</li>
<li>把二叉树打印成多行(用队列实现树的BFS)</li>
<li>序列化二叉树(确定规则，指针遍历，节点值后面加符号来分割树节点) </li>
<li>二叉树中和为某一值的路径3(层次遍历 + dfs/借助哈希表，一次遍历)</li>
<li>在二叉树中找到两个节点的最近公共祖先(dfs+最后一个共同节点/递归)</li>
<li>二叉搜索树的最近公共祖先(简单递归，利用二叉搜索树的特点)</li>
</ul>
<blockquote>
<p>总结: 树的题目基本都是围绕递归遍历展开 </p>
<ul>
<li>几种遍历方式: 前序、中序、后序、层次 </li>
<li>合理考虑递归边界，特殊情况 </li>
</ul>
</blockquote>
<h4 id="队列-amp-栈"><a href="#%E9%98%9F%E5%88%97-amp-%E6%A0%88" class="headerlink" title="队列&amp;栈"></a>队列&amp;栈</h4><ul>
<li>用两个栈实现队列(进行两次压栈操作即可) </li>
<li>包含min函数的栈(普通栈+最小栈) </li>
<li>栈的压入弹出序列(辅助栈模拟) </li>
<li>翻转单词序列(两次翻转/直接split+join)</li>
<li>滑动窗口的最大值(双向队列，维护前向最大值)</li>
</ul>
<blockquote>
<p>总结：数据结构特点的基础应用 </p>
</blockquote>
<h4 id="搜索算法"><a href="#%E6%90%9C%E7%B4%A2%E7%AE%97%E6%B3%95" class="headerlink" title="搜索算法"></a>搜索算法</h4><ul>
<li>数字在升序数组中出现的次数(二分查找 target-0.5 target+0.5) </li>
<li>二维数组中的查找(有序数组，从角落开始搜索)</li>
<li>旋转数组中的最小数字(二分查找，特殊情况考虑，注意等号) </li>
<li>字符串的排列(遍历) </li>
<li>数字序列中的某一位数字(模拟, 几位数字位数关系) </li>
</ul>
<blockquote>
<p>总结：二分搜索 or 模拟遍历  </p>
</blockquote>
<h4 id="动态规划"><a href="#%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92" class="headerlink" title="动态规划"></a>动态规划</h4><ul>
<li>连续子数组的最大和(一维dp, 可以不另外开辟数组) </li>
<li>连续子数组最大和(二)(一维dp,维护起始位置，可以不另外开辟数组) </li>
<li>跳台阶(斐波那契数列) </li>
<li>斐波那契数列 </li>
<li>正则表达式匹配(二维dp) </li>
<li>跳台阶扩展问题($2^{n-1}$) </li>
<li>矩阵覆盖(斐波那契数列)</li>
<li>买股票的最好时机(一维dp) </li>
<li>礼物的最大价值(二维dp, 可以不另外开辟数组) </li>
<li>最长不含重复字符的子字符串(借助字典，1维dp)</li>
<li>把数字翻译成字符串(一维dp) </li>
</ul>
<blockquote>
<p>总结： 这类问题的关键在于</p>
<ul>
<li>状态的定义 </li>
<li>状态转移方程的形式 </li>
</ul>
</blockquote>
<h4 id="回溯"><a href="#%E5%9B%9E%E6%BA%AF" class="headerlink" title="回溯"></a>回溯</h4><ul>
<li>矩阵中的路径(dfs + 标志矩阵)</li>
<li>机器人的运动范围(dfs + 标志矩阵)</li>
</ul>
<blockquote>
<p>dfs + 标志矩阵  </p>
</blockquote>
<h4 id="排序"><a href="#%E6%8E%92%E5%BA%8F" class="headerlink" title="排序"></a>排序</h4><ul>
<li>数组中重复的数字(哈希/位置重排) </li>
<li>数组中的逆序对(归并排序) </li>
<li>最小的k个数 </li>
<li>数据流中的中位数 </li>
</ul>
<blockquote>
<p>总结：熟练掌握归并排序和快速排序 </p>
</blockquote>
<h4 id="位运算"><a href="#%E4%BD%8D%E8%BF%90%E7%AE%97" class="headerlink" title="位运算"></a>位运算</h4><ul>
<li>不用加减乘除做加法(异或运算获取非进位和，与运算加左移算子获取进位信息) </li>
<li>二进制中1的个数(n&amp;n-1, 将低位1变为0，注意处理负数) </li>
<li>数值的整数次方(借助二进制运算实现快速幂) </li>
<li>数组中只出现一次的两个数字(异或运算的特点) </li>
<li>求1+2+…+n(通过与运算终止递归) </li>
</ul>
<h4 id="模拟"><a href="#%E6%A8%A1%E6%8B%9F" class="headerlink" title="模拟"></a>模拟</h4><ul>
<li>顺时针打印矩阵(简单模拟，通过边界控制循环结束)</li>
<li>扑克牌顺子(借助hash)</li>
<li>把字符串转换成整数(去空格，找符号，转换，取模)</li>
<li>表示数值的字符串 </li>
</ul>
<h4 id="其他算法"><a href="#%E5%85%B6%E4%BB%96%E7%AE%97%E6%B3%95" class="headerlink" title="其他算法"></a>其他算法</h4><ul>
<li>构建乘积数组 </li>
<li>第一个只出现一次的字符 </li>
<li>替换空格 </li>
<li>调整数组顺序使奇数位于偶数前面 </li>
<li>数组中出现超过一半次数的数字 </li>
<li>整数中1出现的次数 </li>
<li>把数组排成最小的数 </li>
<li>丑数 </li>
<li>和为S的连续正数序列 </li>
<li>和为S的两个数字 </li>
<li>左旋转字符串 </li>
<li>孩子们的游戏(约瑟夫环)</li>
<li>字符流中第一个不重复地字符 </li>
<li>剪绳子 </li>
<li>调整数组顺序使奇数位于偶数前面</li>
<li>剪绳子(进阶版)</li>
<li>打印从1到最大的n位数<h4 id="python中常用数据结构模块"><a href="#python%E4%B8%AD%E5%B8%B8%E7%94%A8%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84%E6%A8%A1%E5%9D%97" class="headerlink" title="python中常用数据结构模块"></a>python中常用数据结构模块</h4></li>
</ul>
<ol>
<li>队列 <code>queue.Queue()</code> </li>
</ol>
<ul>
<li><code>Queue().qsize()</code>: 返回队列的大小 </li>
<li><code>Queue().empty()</code>: 如果队列为空返回<code>True</code>，反之返回<code>False</code> </li>
<li><code>Queue().full()</code>: 如果队列满了，返回<code>True</code>,反之返回<code>False</code> </li>
<li><code>Queue().get()</code>: 获取队列的第一个值，出队列 </li>
<li><code>Queue().put()</code>: 写入队列 </li>
<li>题目: 从上往下打印二叉树 <pre class="line-numbers language-lang-python"><code class="language-lang-python">import queue 
class Solution:
  def PrintFromTopToBottom(self , root: TreeNode) -&gt; List[int]:
      # write code here 
      res = [] 
      if not root:
          return res 
      q = queue.Queue() 
      q.put(root) 
      while not q.empty(): 
          cur_node = q.get() 
          res.append(cur_node.val) 
          if cur_node.left: 
              q.put(cur_node.left) 
          if cur_node.right: 
              q.put(cur_node.right) 
      return res
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></span></code></pre>
</li>
</ul>
<ol>
<li>双向队列<code>collections.deque()</code> </li>
</ol>
<ul>
<li><code>deque()</code>: 创建 </li>
<li><code>deque().append()</code>: 向队列中添加元素 </li>
<li><code>deque().pop()</code>: 抛出最后一个元素 </li>
<li><code>deque().leftpop()</code>: 抛出第一个元素 </li>
<li><code>deque().extend()</code>: 添加list项到原<code>deque()</code>中 </li>
<li><code>deque().extendleft()</code>: 添加list项到原<code>deque()</code>头部 </li>
<li><p>题目: 滑动窗口的最大值 </p>
<pre class="line-numbers language-lang-python"><code class="language-lang-python">from collections import deque 
class Solution:
  def maxInWindows(self , num: List[int], size: int) -&gt; List[int]:
      # write code here 
      res = [] 
      dq = deque()
      if size &gt; len(num) or not size: 
          return [] 

      for i in range(size): 
          while len(dq) &gt; 0 and num[dq[-1]] &lt; num[i]: 
              dq.pop() 
          dq.append(i) 

      for i in range(size, len(num)): 
          res.append(num[dq[0]]) 
          while len(dq) &gt; 0 and dq[0] &lt; i - size + 1: 
              dq.popleft() 
          while len(dq) &gt; 0 and num[dq[-1]] &lt; num[i]:
              dq.pop() 
          dq.append(i) 
      res.append(num[dq[0]]) 
      return res
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></span></code></pre>
</li>
</ul>
<ol>
<li>小根堆<code>heapq</code></li>
</ol>
<ul>
<li><code>heapq.heappush(pq, value)</code> 入堆 </li>
<li><code>heapq.heapreplace(pq,value)</code> 替换堆中元素 </li>
<li><code>heapq.pop(pq)</code> 出堆 </li>
</ul>
<ol>
<li><code>collections.defaultdict()</code><br><code>dict</code>的一个子类，可以使创建的字典具有默认值，可以预先指定默认值类型：<pre class="line-numbers language-lang-python"><code class="language-lang-python">import collections 
mp = collections.defaultdict(list) # 默认值为[]
mp = collections.defaultdict(int)  # 默认值为0
mp = collections.defaultdict(lambda:999) # 默认值为999
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span></span></code></pre>
在python中，可哈希的数据类型有：</li>
</ol>
<ul>
<li>数字类型(<code>int</code>、<code>float</code>、<code>bool</code>), 字符串<code>str</code>、元组<code>tuple</code></li>
</ul>
<p>不可哈希的数据类型有:</p>
<ul>
<li>字典<code>dict</code>、列表<code>list</code>、集合<code>set</code><h4 id="算法题常用解题思路"><a href="#%E7%AE%97%E6%B3%95%E9%A2%98%E5%B8%B8%E7%94%A8%E8%A7%A3%E9%A2%98%E6%80%9D%E8%B7%AF" class="headerlink" title="算法题常用解题思路"></a>算法题常用解题思路</h4><h5 id="递归"><a href="#%E9%80%92%E5%BD%92" class="headerlink" title="递归"></a>递归</h5></li>
<li>什么样的问题可以用递归解决? <ul>
<li>主问题可以拆分为若干子问题 </li>
<li>主问题和子问题的解题思路一致</li>
<li>存在终止条件 </li>
</ul>
</li>
<li>递归问题如何解决？ <ul>
<li>找到递推公式 </li>
<li>找到终止条件 </li>
<li>翻译成代码 </li>
</ul>
</li>
</ul>
{% endraw %}
