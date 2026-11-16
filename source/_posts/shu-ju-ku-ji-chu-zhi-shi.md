---
title: 数据库基础知识
date: '2022-10-31 12:00:00'
updated: '2022-11-01 12:00:00'
permalink: 2022/10/31/shu-ju-ku-ji-chu-zhi-shi/
tags:
- 数据库
categories:
- 技术学习
- 论文阅读
math: false
legacy_html: true
img: /medias/featureimages/4.png
summary: 定义数据库是结构化信息或数据的有序集合，一般以电子形式存储在计算系统中，通常由数据库管理系统(DBMS)来控制，在现实中，数据、DBMS 及关联应用一起被称为数据库系统，通常简称为数据库。为了提高数据处理和查询效率，当今最常见的数据库通常以行和列的形式将数据存储在一系列的表中，支持用户便捷地访问、管
---
{% raw %}
<h4 id="定义"><a href="#%E5%AE%9A%E4%B9%89" class="headerlink" title="定义"></a>定义</h4><p>数据库是<strong>结构化信息或数据的有序集合</strong>，一般以电子形式存储在计算系统中，通常由数据库管理系统(DBMS)来控制，在现实中，数据、DBMS 及关联应用一起被称为数据库系统，通常简称为数据库。<br>为了提高数据处理和查询效率，当今最常见的数据库通常以行和列的形式将数据存储在一系列的表中，支持用户便捷地访问、管理、修改、更新、控制和组织数据。另外，大多数数据库都使用结构化查询语言 (SQL) 来编写和查询数据。<span id="more"></span></p>
<h4 id="结构化查询语言SQL"><a href="#%E7%BB%93%E6%9E%84%E5%8C%96%E6%9F%A5%E8%AF%A2%E8%AF%AD%E8%A8%80SQL" class="headerlink" title="结构化查询语言SQL"></a>结构化查询语言SQL</h4><p>目前几乎所有的关系数据库都使用 SQL 编程语言来查询、操作和定义数据，进行数据访问控制。SQL 最初于 20 世纪 70 年代由 IBM 开发，当时 Oracle 是一个主要的贡献者，这推动了 SQL ANSI 标准的实施，而 SQL 的兴起也刺激了 IBM、Oracle 和 Microsoft 等公司开始全面扩张。时至今日，虽然 SQL 仍被广泛使用，但是新的编程语言也已经崭露头角。</p>
<h4 id="数据库的发展历程"><a href="#%E6%95%B0%E6%8D%AE%E5%BA%93%E7%9A%84%E5%8F%91%E5%B1%95%E5%8E%86%E7%A8%8B" class="headerlink" title="数据库的发展历程"></a>数据库的发展历程</h4><p>自 20 世纪 60 年代初诞生至今，数据库已经发生了翻天覆地的变化。最初，人们使用<strong>分层数据库</strong>（树形模型，仅支持一对多关系）和<strong>网络数据库</strong>（更加灵活，支持多种关系）这样的导航数据库来存储和操作数据。这些早期系统虽然简单，但缺乏灵活性。20 世纪 80 年代，<strong>关系数据库</strong>开始兴起；20 世纪 90 年代，<strong>面向对象的数据库</strong>开始成为主流。最近，随着互联网的快速发展，为了更快速地处理非结构化数据，NoSQL 数据库应运而生。现在，云数据库和自治驾驶数据库在数据收集、存储、管理和利用方面正不断取得新的突破。</p>
<h4 id="常用SQL语言"><a href="#%E5%B8%B8%E7%94%A8SQL%E8%AF%AD%E8%A8%80" class="headerlink" title="常用SQL语言"></a>常用SQL语言</h4><ul>
<li>查询表中所有数据 <pre class="line-numbers language-lang-sql"><code class="language-lang-sql">SELECT * FROM 表名
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
</li>
<li><p>查找表中特定内容</p>
<pre class="line-numbers language-lang-sql"><code class="language-lang-sql">SELECT selection_list /*要查询的列名称*/
FROM table_list /*要查询的表名称*/
WHERE condition /*行条件*/
GROUP BY grouping_columns /*对结果分组*/
HAVING condition /*分组后的行条件*/
ORDER BY sorting_columns /*对结果分组*/
LIMIT offset_start, row_count /*结果限定*/
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span><span></span><span></span><span></span></span></code></pre>
</li>
<li><p>聚合函数: 做纵向运算</p>
<pre class="line-numbers language-lang-sql"><code class="language-lang-sql">COUNT()：统计指定列不为NULL的记录行数；
MAX()：计算指定列的最大值，如果指定列是字符串类型，那么使用字符串排序运算；
MIN()：计算指定列的最小值，如果指定列是字符串类型，那么使用字符串排序运算；
SUM()：计算指定列的数值和，如果指定列类型不是数值类型，那么计算结果为0；
AVG()：计算指定列的平均值，如果指定列类型不是数值类型，那么计算结果为0；
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span><span></span></span></code></pre>
</li>
</ul>
{% endraw %}
