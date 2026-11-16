---
title: 常用Linux命令整理
date: '2022-10-18 12:00:00'
updated: '2022-10-20 12:00:00'
permalink: 2022/10/18/chang-yong-linux-ming-ling-zheng-li/
tags:
- 命令行
categories:
- linux
- 技术学习
- 环境配置
math: false
legacy_html: true
img: /medias/featureimages/3.png
summary: '在服务器上训练深度学习模型时不可避免地需要进行大量的命令行操作，熟练使用命令行已经成为了一项程序员必备的技能，本文对常用的linux命令进行整理，以加深印象和方便后续查询。
  文件管理 cat: 用于连接文件并打印到标准输出设备上 使用权限: 所有使用者 语法格式: cat [-AbeEnstTuv]'
---
{% raw %}
<p>在服务器上训练深度学习模型时不可避免地需要进行大量的命令行操作，熟练使用命令行已经成为了一项程序员必备的技能，本文对常用的linux命令进行整理，以加深印象和方便后续查询。<br><span id="more"></span> </p>
<h4 id="文件管理"><a href="#%E6%96%87%E4%BB%B6%E7%AE%A1%E7%90%86" class="headerlink" title="文件管理"></a>文件管理</h4><ul>
<li><code>cat</code>: 用于连接文件并打印到标准输出设备上 <ul>
<li>使用权限: 所有使用者 </li>
<li>语法格式: <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">cat [-AbeEnstTuv] [--help] [--version] fileName
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
</li>
<li>参数说明: <ul>
<li>-n/—number: 由1开始对所有输出的行号编号 </li>
<li>-b: 与-n类似，只是对空白行不编号 <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">cat file1 &gt; file2 #将file1的内容拷贝到file2中(覆盖)
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
</li>
</ul>
</li>
</ul>
</li>
<li><code>chmod</code>: 控制用户对文件的权限 <ul>
<li>使用权限: 只有文件所有者和超级用户可以修改文件或者目录的权限 </li>
<li>说明: Linux/unix的文件调用权限分为三级: 文件所有者、用户组、其他用户<br>用户创建一个新文件时，默认权限是<code>-rw-r--r--</code><br><img src="/.io//%E6%96%87%E4%BB%B6%E6%9D%83%E9%99%90.png" alt><br>可以通过使用绝对模式(八进制数字模式)指定文件的权限<br><img src="/.io//%E7%94%A8%E6%88%B7%E7%BB%84.png" alt></li>
<li>语法: <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">chmod [-cfvR] [--help] [--version] mode file...
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
</li>
<li>参数说明: <ul>
<li>mode: 权限设定字串<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">  符号模式: [ugoa...][[+-=][rwxX]...][,...]
  八进制模式: 777
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span></span></code></pre>
</li>
</ul>
</li>
<li>使用示例: <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">chmod a+rwx file #符号模式向三类用户开放读写执行权限
chmod 777 file   #八进制数字模式，功能与上面等价
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span></span></code></pre>
</li>
</ul>
</li>
<li><code>chown</code>: change owner, 将文件的拥有者改为指定的用户或组<ul>
<li>使用权限: 超级用户 </li>
<li>语法: <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">chown [-cfhvR] [--help] [--version] user[:group] file...
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
</li>
<li>参数说明: <ul>
<li>user: 新的文件拥有者或用户id </li>
<li>group: 新的文件拥有者的使用者组 </li>
<li>-c: 显示更改的部分的信息 </li>
<li>-R: 处理指定目录以及其子目录下的所有文件 </li>
</ul>
</li>
<li>使用示例: <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">chown -R root dir #将文件夹dir及其子目录的所有者设定为root
chown -R xuejianye dir #将文件夹dir及其子目录的所有者设定为xuejianye
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span></span></code></pre>
</li>
</ul>
</li>
<li><code>diff</code>: 比较文本文件的异同，若相同则没有输出，有差异则会输出差异位置及内容<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">  diff file1 file2
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
</li>
<li><code>find</code>: 在指定目录下查找文件<ul>
<li>语法: <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">  find   path   -option   [   -print ]   [ -exec   -ok   command ]   {} \;
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
</li>
<li>使用示例: <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">  find . -name "*.txt" #在当前目录下找到所有txt文件
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
</li>
</ul>
</li>
<li><code>mv</code>: 将文件或目录移入其他位置/修改文件或目录的名称 <ul>
<li>语法: <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">mv [options] source dest
mv [options] source... directory
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span></span></code></pre>
</li>
<li>参数: <ul>
<li>-b: 当目标文件或目录存在时，在执行覆盖前，会为其创建一个备份 </li>
<li>-i: 如果指定移动的源目录或文件与目标的目录或文件同名，则会先询问是否覆盖旧文件，输入 y 表示直接覆盖，输入 n 表示取消该操作。</li>
<li>-f: 如果指定移动的源目录或文件与目标的目录或文件同名，不会询问，直接覆盖旧文件。</li>
<li>-n: 不要覆盖任何已存在的文件或目录。 </li>
</ul>
</li>
<li>使用示例: <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">mv source_file(文件) dest_file(文件) #将源文件名 source_file 改为目标文件名 dest_file
mv source_file(文件) dest_directory(目录) #将文件 source_file 移动到目标目录 dest_directory 中
mv source_directory(目录) dest_directory(目录) #目录名 dest_directory 已存在，将 source_directory 移动到目录名 dest_directory 中；目录名 dest_directory 不存在则 source_directory 改名为目录名 dest_directory
mv aaa bbb #将文件aaa改名为bbb
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span></span></code></pre>
</li>
</ul>
</li>
<li><code>paste</code>: 合并文件的列<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">  icarus@DESKTOP-7Q4C11S  ~  cat test.txt copy.txt
  123
  456
  123
  456
  icarus@DESKTOP-7Q4C11S  ~  cat test.txt copy.txt &gt; new.txt
  icarus@DESKTOP-7Q4C11S  ~  cat new.txt
  123
  456
  123
  456
  icarus@DESKTOP-7Q4C11S  ~  paste test.txt copy.txt
  123     123
  456     456
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></span></code></pre>
</li>
<li><code>rm</code>: 用于删除一个文件或目录 <ul>
<li>语法<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">rm [options] name...
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
</li>
<li>参数 -r: 递归删除目录内所有文件 </li>
</ul>
</li>
<li><code>touch</code>: 修改文件或者目录的时间属性，若文件不存在则会创建一个新文件<br>  -语法: <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">  touch [-acfm][-d][-r] [-t][--help][--version][文件或目录…]
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
  -使用示例: <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">  touch file #若文件不存在则创建文件，文件存在则修改文件的事件属性
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
</li>
<li><p><code>which</code>: 查找文件，会在环境变量$PATH设置的目录里查找符合条件的文件</p>
<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">  which conda
  which pip
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span></span></code></pre>
</li>
<li><p><code>cp</code>: 用于复制文件或者目录</p>
<ul>
<li>语法:<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">cp [options] source dest
cp [options] source... directory
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span></span></code></pre>
</li>
<li>参数说明: <ul>
<li>-r: 若给出的源文件是一个目录文件，此时将复制该目录下所有的子目录和文件。</li>
</ul>
</li>
<li>使用示例<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">cp –r test/ newtest   #用户使用该指令复制目录时，必须使用参数 -r 或者 -R
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
</li>
</ul>
</li>
<li><code>scp</code>:  是 secure copy 的缩写, scp 是 linux 系统下基于 ssh 登陆进行安全的远程文件拷贝命令 <ul>
<li>语法: <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">scp [可选参数] file_source file_target
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
</li>
<li>使用示例: <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">#从本机复制到远程 
scp local_file remote_username@remote_ip:remote_folder 
或者 
scp local_file remote_username@remote_ip:remote_file 
或者 
scp local_file remote_ip:remote_folder 
或者 
scp local_file remote_ip:remote_file 
#从远程复制到本地
scp root@www.runoob.com:/home/root/others/music /home/space/music/1.mp3 
scp -r www.runoob.com:/home/root/others/ /home/space/music/
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></span></code></pre>
</li>
</ul>
</li>
</ul>
<h4 id="文档编辑"><a href="#%E6%96%87%E6%A1%A3%E7%BC%96%E8%BE%91" class="headerlink" title="文档编辑"></a>文档编辑</h4><ul>
<li><code>grep</code>: 用于查找文件里符合条件的字符串 <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">  grep "被查找的字符串" 文件名
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
</li>
</ul>
<h4 id="文件传输"><a href="#%E6%96%87%E4%BB%B6%E4%BC%A0%E8%BE%93" class="headerlink" title="文件传输"></a>文件传输</h4><h4 id="磁盘管理"><a href="#%E7%A3%81%E7%9B%98%E7%AE%A1%E7%90%86" class="headerlink" title="磁盘管理"></a>磁盘管理</h4><ul>
<li><code>cd</code>: 切换当前工作目录 </li>
<li><code>df</code>: 显示目前在linux系统上的文件系统磁盘使用情况统计 <ul>
<li>语法: <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">df [选项]... [FILE]...
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
</li>
<li>使用示例: <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">icarus@DESKTOP-7Q4C11S  /home  df icarus
Filesystem     1K-blocks     Used Available Use% Mounted on
/dev/sdc       263174212 35268580 214467476  15% /
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span></span></code></pre>
</li>
</ul>
</li>
<li><code>du</code>: 用于显示目录或文件的大小 <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">  du -s icarus/ #显示该文件夹中所有文件所占的内存和 
  du -a icarus/ #显示目录中个别文件的大小
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span></span></code></pre>
</li>
<li><code>mkdir</code>: 用于创建目录 <ul>
<li>参数说明: -p 确保目录名称存在，不存在就新建一个  </li>
<li>使用示例: <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">mkdir -p /home/projects
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
</li>
</ul>
</li>
<li><p><code>tree</code>: 以树状图列出目录的内容 </p>
</li>
<li><p><code>ls</code>: 显示指定工作目录下的内容 </p>
<ul>
<li>参数 <ul>
<li>-a: 显示所有文件及目录 </li>
<li>-l: 除文件名称外，亦将文件型态、权限、拥有者、文件大小等资讯详细列出 </li>
</ul>
</li>
</ul>
</li>
</ul>
<h4 id="磁盘维护"><a href="#%E7%A3%81%E7%9B%98%E7%BB%B4%E6%8A%A4" class="headerlink" title="磁盘维护"></a>磁盘维护</h4><h4 id="网络通讯"><a href="#%E7%BD%91%E7%BB%9C%E9%80%9A%E8%AE%AF" class="headerlink" title="网络通讯"></a>网络通讯</h4><ul>
<li><code>ifconfig</code>: 用于显示或设置网络设备,查看ip/mac地址 </li>
<li><code>ping</code>: 用于检测主机，</li>
</ul>
<h4 id="系统管理"><a href="#%E7%B3%BB%E7%BB%9F%E7%AE%A1%E7%90%86" class="headerlink" title="系统管理"></a>系统管理</h4><ul>
<li><code>kill</code>: 用于删除执行中的程序或工作<ul>
<li>语法: <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">kill [-s ][程序]　或　kill [-l ]
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
</li>
<li>常用参数: <ul>
<li>-HUP: 重新加载进程 </li>
<li>-KILL: 杀死一个进程 </li>
<li>-TERM: 正常停止一个进程</li>
</ul>
</li>
<li>使用示例: <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">kill 12345
kill -KILL 123456 #强制杀死进程
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span></span></code></pre>
</li>
</ul>
</li>
<li><code>ps</code>: 显示当前进程的状态，类似于windows的任务管理器 <ul>
<li>语法: <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">ps [options] [--help]
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
</li>
<li>参数 <ul>
<li>-a: 列出所有的进程 </li>
</ul>
</li>
<li>使用示例: <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">ps -ef | grep 进程关键字
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
</li>
</ul>
</li>
</ul>
<h4 id="系统设置"><a href="#%E7%B3%BB%E7%BB%9F%E8%AE%BE%E7%BD%AE" class="headerlink" title="系统设置"></a>系统设置</h4><ul>
<li><code>clear</code>: 用于清除屏幕 </li>
<li><code>export</code>: 用于设置或者显示环境变量,效力仅限于该次登陆操作 <ul>
<li>语法: <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">export [-fnp][变量名称]=[变量设置值]
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
</li>
<li>参数名称<ul>
<li>-n: 删除指定的变量 </li>
<li>-p: 列出所有的shell赋予程序的环境变量 </li>
</ul>
</li>
<li>使用示例: <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">export CUDA_VISIBLE_DEVICES=0,3
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
</li>
</ul>
</li>
</ul>
<h4 id="备份压缩"><a href="#%E5%A4%87%E4%BB%BD%E5%8E%8B%E7%BC%A9" class="headerlink" title="备份压缩"></a>备份压缩</h4><h4 id="设备管理"><a href="#%E8%AE%BE%E5%A4%87%E7%AE%A1%E7%90%86" class="headerlink" title="设备管理"></a>设备管理</h4>
{% endraw %}
