---
title: conda介绍及常用命令
date: '2022-10-17 12:00:00'
updated: '2022-10-18 12:00:00'
permalink: 2022/10/17/conda-jie-shao-ji-chang-yong-ming-ling/
tags:
- 命令行
categories:
- 环境配置
- linux
math: false
legacy_html: true
img: /medias/featureimages/0.jpg
summary: 随着机器学习、深度学习的爆火，大量的python第三方库也如雨后春笋一般冒了出来，这给开发者带来了极大便利的同时，也使得管理这些库成为了一件复杂度极高的事情，相信很多人都有过为了处理库冲突而抓狂的时刻，为了简化python包管理的工作，Anaconda应运而生。
  AnacondaAnaconda可以
---
{% raw %}
<p>随着机器学习、深度学习的爆火，大量的python第三方库也如雨后春笋一般冒了出来，这给开发者带来了极大便利的同时，也使得管理这些库成为了一件复杂度极高的事情，相信很多人都有过为了处理库冲突而抓狂的时刻，为了简化python包管理的工作，Anaconda应运而生。<span id="more"></span></p>
<h5 id="Anaconda"><a href="#Anaconda" class="headerlink" title="Anaconda"></a>Anaconda</h5><p>Anaconda可以看作是包含python、conda、numpy、notebook等常用包的一个集合，具有如下特点： </p>
<ul>
<li>安装过程简单 </li>
<li>高性能使用python和R语言 </li>
<li>免费的社区支持</li>
</ul>
<p>对于大多数用户来说，下载Anaconda的目的主要有两个: </p>
<ul>
<li>高效简单的配置python开发环境(python+一些基础常用包) </li>
<li><strong>后续通过<code>conda</code>这个神器来进行包管理和虚拟环境管理</strong></li>
</ul>
<h5 id="Conda"><a href="#Conda" class="headerlink" title="Conda"></a>Conda</h5><p><code>conda</code>是包及其依赖项和环境的管理工具: </p>
<ul>
<li>适用语言: Python, R, Ruby, Lua, Scala, Java, JavaScript, C/C++, FORTRAN </li>
<li>适用平台: Windows, Macos, Linux </li>
<li>优点 <ul>
<li>快速安装、运行和升级包及其依赖项</li>
<li>在计算机中便捷地创建、保存、加载和切换环境 <blockquote>
<p>如果你需要的包要求不同版本的Python，你无需切换到不同的环境，因为conda同样是一个环境管理器。仅需要几条命令，你可以创建一个完全独立的环境来运行不同的Python版本，同时继续在你常规的环境中使用你常用的Python版本。——Conda官方网站</p>
</blockquote>
</li>
</ul>
</li>
</ul>
<p><strong>conda常用命令</strong>: </p>
<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">conda activate xxxx #开启xxxx环境
conda deactivate #关闭环境
​
conda env list #显示所有的虚拟环境
conda info --envs #显示所有的虚拟环境
conda info -e #显示所有已经创建的环境
​
conda update -n base conda #update最新版本的conda
​
conda create -n xxxx python=3.6 #创建python3.6的xxxx虚拟环境
​
conda remove --name xxxx  --all #彻底删除旧环境
​
conda remove -n tensorflow --all  #彻底删除旧环境
​
#Conda是没有重命名环境的功能, 要实现这个基本需求, 可以通过克隆-删除的过程。
#切记不要直接mv移动环境的文件夹来重命名, 会导致一系列无法想象的错误的发生!
conda create --name newname --clone oldname //克隆环境
conda remove --name oldname --all //彻底删除旧环境
​
conda list          #查看已经安装的文件包
conda list -n xxx   #指定查看xxx虚拟环境下安装的package
conda update xxx    #更新xxx文件包
conda uninstall xxx #卸载xxx文件包
​
#pip 安装本地包
pip install   ～/Downloads/a.whl
#conda 安装本地包
conda install --use-local  ~/Downloads/a.tar.bz2
​
​
#显示目前conda的数据源有哪些
conda config --show channels
​
#添加数据源：例如, 添加清华anaconda镜像：
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
​
#删除数据源
conda config --remove channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
​
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></span></code></pre>
<h5 id="conda配置国内镜像源"><a href="#conda%E9%85%8D%E7%BD%AE%E5%9B%BD%E5%86%85%E9%95%9C%E5%83%8F%E6%BA%90" class="headerlink" title="conda配置国内镜像源"></a><code>conda</code>配置国内镜像源</h5><ul>
<li><p>方法1: 用<code>vim</code>打开文件，添加以下镜像源</p>
<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">  vim ~/.condarc
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
<p>  修改文件内容: </p>
<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">      #清华源
  channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  ssl_verify: true
  #中科大源
  channels:
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
  - https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
  ssl_verify: true
  #上交大源
  channels:
  - https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/
  - https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/free/
  - https://mirrors.sjtug.sjtu.edu.cn/anaconda/cloud/conda-forge/
  ssl_verify: true
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></span></code></pre>
</li>
<li>方法2: 直接通过终端命令修改 <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">  ​# 中科大镜像源
  conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
  conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
  conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
  conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/msys2/
  conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/bioconda/
  conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/menpo/
  conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/
  ​
  # 阿里镜像源
  conda config --add channels https://mirrors.aliyun.com/pypi/simple/
  ​
  # 豆瓣的python的源
  ​
  conda config --add channels http://pypi.douban.com/simple/ 
  ​
  # 显示检索路径，每次安装包时会将包源路径显示出来
  ​
  conda config --set show_channel_urls yes
  ​
  conda config --set always_yes True
  ​
  # 显示所有镜像通道路径命令
  ​
  conda config --show channels
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></span></code></pre>
</li>
</ul>
{% endraw %}
