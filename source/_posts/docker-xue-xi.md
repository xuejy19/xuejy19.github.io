---
title: docker学习
date: '2022-11-03 12:00:00'
updated: '2022-11-06 12:00:00'
permalink: 2022/11/03/docker-xue-xi/
tags:
- 容器化技术
categories:
- 技术学习
math: false
legacy_html: true
img: /medias/featureimages/2.jpg
summary: '前言从实际场景谈起在没有虚拟机之前，环境的迁移是一项非常复杂的工作，开发人员开发时需要搭建一个环境，服务开发完毕后，测试人员又需要重新搭建一遍环境，而到环境上线时，还需要再搭建一遍环境，这大大浪费了时间，降低了效率。
  大多问题都是在运行环境改变时才显现的，可能是这样一种场景，开发者把代码从开发环境 '
---
{% raw %}
<h3 id="前言"><a href="#%E5%89%8D%E8%A8%80" class="headerlink" title="前言"></a>前言</h3><h4 id="从实际场景谈起"><a href="#%E4%BB%8E%E5%AE%9E%E9%99%85%E5%9C%BA%E6%99%AF%E8%B0%88%E8%B5%B7" class="headerlink" title="从实际场景谈起"></a>从实际场景谈起</h4><p>在没有虚拟机之前，环境的迁移是一项非常复杂的工作，开发人员开发时需要搭建一个环境，服务开发完毕后，测试人员又需要重新搭建一遍环境，而到环境上线时，还需要再搭建一遍环境，这大大浪费了时间，降低了效率。<span id="more"></span></p>
<p>大多问题都是在运行环境改变时才显现的，可能是这样一种场景，开发者把代码从开发环境 push 到测试环境然后到更上层的环境。比如开发者在 Windows 上编写应用代码，但是上层环境（测试、stage 或是生产环境）是基于 Linux 的。在这种情况下，当操作系统变化时，应用的某些功能可能就会出现问题。所以，基本上，当配套软件环境不一样的时候，间歇性故障的几率就会更大。</p>
<p>正如 Docker 创始人 Solomon Hykes 所说，“测试环境使用 Python 2.7，但是生产环境使用 Python 3，那么一些奇怪的事情就会发生。或者你依赖某个特定版本的 SSl 库的行为，但是却安装了另外一个版本。在 Debian 上运行测试环境，但是生产环境使用 Redhat，那么任何奇怪的事情都可能发生。”</p>
<p>改变可能不只是运行环境，也有可能是网络，Hykes 补充道，“网络拓扑可能不同，或者安全策略和存储也有可能不同，但是，软件需要在这些基础设施上面运行。”</p>
<h4 id="虚拟机技术"><a href="#%E8%99%9A%E6%8B%9F%E6%9C%BA%E6%8A%80%E6%9C%AF" class="headerlink" title="虚拟机技术"></a>虚拟机技术</h4><p>在计算机技术中，<strong>虚拟化技术是一种资源管理技术，将计算机的各种实体资源(cpu、内存、磁盘空间、网络适配器等)予以抽象、转换后呈现出来并可供分割、组合为一个或多个电脑组态环境。</strong> 由此，打破实体结构间的不可切割的障碍，使用户可以比原本的组态更好的方式来应用这些电脑硬体资源。</p>
<p>虚拟化技术虽然在一定程度上解决了环境迁移的问题，但其缺点也是显然的，每个虚拟机都有着独立的CMOS、硬盘和操作系统，是一个非常笨重的程序，而我们有时可能想要迁移的只是一个很小的服务，用虚拟化技术来进行迁移显得有些大而无当，那么有没有一种技术可以保留虚拟化技术的优点而避免虚拟化技术的缺点呢？<br>这就到了本篇文章的主角——容器。</p>
<h3 id="容器"><a href="#%E5%AE%B9%E5%99%A8" class="headerlink" title="容器"></a>容器</h3><h4 id="概述"><a href="#%E6%A6%82%E8%BF%B0" class="headerlink" title="概述"></a>概述</h4><p>什么是容器？ 一言以蔽之，<strong>容器是轻量的操作系统级虚拟化</strong>，借助容器，我们可以在一个资源隔离的进程中运行应用及其依赖项，运行应用的所必需的组件都将打包成一个镜像并可以复用。执行镜像时，它运行在一个隔离环境中，并且不会共享主机的内存、CPU以及磁盘，这就保证了容器内进程不能监控容器外任何进程。<br><img src="/.io//%E5%AE%B9%E5%99%A8%E6%8A%80%E6%9C%AF.jpeg" alt></p>
<h4 id="容器与虚拟机的区别"><a href="#%E5%AE%B9%E5%99%A8%E4%B8%8E%E8%99%9A%E6%8B%9F%E6%9C%BA%E7%9A%84%E5%8C%BA%E5%88%AB" class="headerlink" title="容器与虚拟机的区别"></a>容器与虚拟机的区别</h4><p>主要的区别在于虚拟隔离的界限，虚拟机通常包含整个操作系统及其应用程序，同时也需要一个<code>hypervisor</code>(可以理解为一个“元”操作系统，协调硬件资源的访问，在各个虚拟机之间添加防护)来控制虚拟机。<br>因为虚拟机包含操作系统，其大小一般是几个G，这也导致了在使用虚拟机时启动操作系统以及初始化托管应用便已经会花费几分钟的时间；与虚拟机相对，容器更加轻量，一般在M以内，启动时也可以做到秒启动。<br><img src="/.io//%E8%99%9A%E6%8B%9F%E6%9C%BA_vs_%E5%AE%B9%E5%99%A8.png" alt><br>总结来说，容器与虚拟机技术的区别主要有三点: </p>
<ul>
<li>隔离界限: 虚拟机是操作系统层级的隔离，而容器只是应用级别的隔离 </li>
<li>占用空间: 虚拟机一般是G级别，而容器则一般是M级别 </li>
<li>启动响应时间: 虚拟机启动需要耗费几分钟，而容器可以做到秒启动 </li>
</ul>
<div class="table-container">
<table>
<thead>
<tr>
<th style="text-align:left">维度</th>
<th style="text-align:left">虚拟机</th>
<th style="text-align:left">容器</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left">体量</td>
<td style="text-align:left">重量级</td>
<td style="text-align:left">轻量级</td>
</tr>
<tr>
<td style="text-align:left">操作系统</td>
<td style="text-align:left">独立操作系统</td>
<td style="text-align:left">共享操作系统</td>
</tr>
<tr>
<td style="text-align:left">虚拟化边界</td>
<td style="text-align:left">硬件级虚拟化</td>
<td style="text-align:left">OS级虚拟化</td>
</tr>
<tr>
<td style="text-align:left">启动时间</td>
<td style="text-align:left">分钟级别</td>
<td style="text-align:left">毫秒级别</td>
</tr>
<tr>
<td style="text-align:left">隔离级别</td>
<td style="text-align:left">完全隔离</td>
<td style="text-align:left">进程级隔离</td>
</tr>
</tbody>
</table>
</div>
<h4 id="容器的优点"><a href="#%E5%AE%B9%E5%99%A8%E7%9A%84%E4%BC%98%E7%82%B9" class="headerlink" title="容器的优点"></a>容器的优点</h4><ol>
<li>敏捷环境: 容器技术最大的优点是创建容器实例比创建虚拟机实例快得多，容器轻量级的脚本可以从性能和大小方面减少开销。 </li>
<li>提高生产力: 容器通过移除跨服务依赖和冲突提高了开发者的生产力。每个容器都可以看作是一个不同的微服务，因此可以独立升级，而不用担心同步。 </li>
<li>版本控制: 每一个容器的镜像都有版本控制，这样就可以追踪到不同版本的容器，监控版本之间的差异。 </li>
<li>运行环境可移植: 容器分装了所有运行应用程序所必需的相关细节比如应用依赖以及操作系统，这就使得景象从一个环境移植到另一个环境更加灵活。 </li>
<li>标准化: 大多数容器基于开放标准，可以运行在所有主流Linux发行版、Microsoft平台等。 </li>
<li>安全: 容器之间的进程是相互隔离的，其中的基础设施亦是如此，这样其中一个容器的升级或者变化不会影响其他容器。 </li>
</ol>
<h4 id="容器的缺点"><a href="#%E5%AE%B9%E5%99%A8%E7%9A%84%E7%BC%BA%E7%82%B9" class="headerlink" title="容器的缺点"></a>容器的缺点</h4><ol>
<li>复杂性增加: 随着容器及应用数量的增加，同时也伴随着复杂性的增加，在生产环境中管理如此之多的容器是一个极具挑战性的任务，可以使用<code>Kubernetes</code>和<code>Mesos</code>等工具管理具有一定规模数量的容器。 </li>
<li>原生Linux支持: 大多数容器技术，比如<code>Docker</code>,Linux容器(<code>LXC</code>)，相比于在原生Linux中运行容器，在Microsoft环境中运行容器略显笨拙，日常使用也会带来复杂性。</li>
<li>不成熟: 容器技术仍在发展完善中。 </li>
</ol>
<h4 id="容器分类"><a href="#%E5%AE%B9%E5%99%A8%E5%88%86%E7%B1%BB" class="headerlink" title="容器分类"></a>容器分类</h4><p>容器一般分为两类: </p>
<ul>
<li>操作系统容器 </li>
<li>应用容器 </li>
</ul>
<p><img src="/.io//%E6%93%8D%E4%BD%9C%E7%B3%BB%E7%BB%9F%E5%AE%B9%E5%99%A8vs%E5%BA%94%E7%94%A8%E5%AE%B9%E5%99%A8.jpeg" alt></p>
<h5 id="操作系统容器"><a href="#%E6%93%8D%E4%BD%9C%E7%B3%BB%E7%BB%9F%E5%AE%B9%E5%99%A8" class="headerlink" title="操作系统容器"></a>操作系统容器</h5><blockquote>
<p>  操作系统层虚拟化是一种计算机虚拟化技术，这种技术将操作系统内核虚拟化，可以允许多个独立用户空间的存在，而不是只有一个。这些实例有时会被称为容器、虚拟引擎、虚拟专用服务器或是 jails（FreeBSD jail 或者 chroot jail）。从运行在容器中的程序角度来看，这些实例就如同真正的计算机。</p>
</blockquote>
<p>如上所述，容器共享宿主机的内核，但是提供用户空间隔离，我们可以像在宿主机操作系统上一样，在容器中安装、配置以及运行应用程序。<br>当需要配置大量具有相同配置的操作系统时，操作系统容器就会非常有用。要创建操作系统容器，我们可以利用一些容器技术，比如LXC、OpenVZ、Linux VServer等。</p>
<h5 id="应用容器"><a href="#%E5%BA%94%E7%94%A8%E5%AE%B9%E5%99%A8" class="headerlink" title="应用容器"></a>应用容器</h5><blockquote>
<p>应用程序虚拟化是从其所执行的底层操作系统封装计算机程序的软件技术。一个完全虚拟化的应用，尽管仍像原来一样执行，但是并不会进行传统意义上的安装。应用在运行时的行为就像它直接与原始操作系统以及操作系统所管理的所有资源进行交互一样，但可以实现不同程度的隔离或者沙盒化。</p>
</blockquote>
<p>应用程序容器旨在为单个进程进行打包和运行服务，而在OS容器中，可以运行多个服务和进程，常用的应用级别的容器化技术有Docker和Rocket，下面将对Docker进行详细介绍。 </p>
<h3 id="Docker"><a href="#Docker" class="headerlink" title="Docker"></a>Docker</h3><h4 id="Docker简介"><a href="#Docker%E7%AE%80%E4%BB%8B" class="headerlink" title="Docker简介"></a>Docker简介</h4><p><strong>Docker</strong>最初是<code>dotCloud</code>的一个公司内部项目，在2013年开源，主要项目代码在github上进行维护。<br><strong>Docker</strong>自开源后受到广泛的关注和讨论，由于<code>Docker</code>项目的火爆，在2013年底，<code>dotCloud</code>公司决定改名为<code>DocKer</code>，<code>Docker</code>最初是在<code>Ubuntu 12.04</code>上开发实现，后续<code>Red Hat</code>则在<code>RHEL 6.5</code>开始对<code>Docker</code>进行支持；<code>Google</code>也在其<code>Paas</code>铲平中广泛应用<code>Docker</code>。<br><strong>Docker</strong>使用<code>Go</code>语言开发实现，基于Linux内核的<code>cgroup</code>、<code>namespace</code>以及<code>OverlayFS</code>类的<code>Union FS</code>等技术，对进程进行封装隔离，实现<strong>操作系统层面的虚拟化</strong>。 由于隔离的进程独立于诉诸和其它隔离的进程，因此也称其为容器。最初实现是基于<code>LXC</code>，从<code>0.7</code>版本后开始去除<code>LXC</code>，转而使用自行开发的<code>libcontainer</code>，从<code>1.11</code>版本开始，则进一步演进为使用<code>runC</code>和<code>containerd</code>。<br><img src="/.io//docker-on-linux.png" alt><br><strong>Docker</strong>在容器的基础上，进行了进一步的封装，从文件系统、网络互联到进程隔离等等，极大的简化了容器的创建和维护。</p>
<h4 id="Docker基本概念"><a href="#Docker%E5%9F%BA%E6%9C%AC%E6%A6%82%E5%BF%B5" class="headerlink" title="Docker基本概念"></a>Docker基本概念</h4><p><strong>Docker</strong>包括三个基本概念:</p>
<ul>
<li><strong>镜像(Image)</strong> </li>
<li><strong>容器(Container)</strong> </li>
<li><strong>仓库(Repository)</strong> </li>
</ul>
<h5 id="镜像"><a href="#%E9%95%9C%E5%83%8F" class="headerlink" title="镜像"></a>镜像</h5><p><strong>Docker镜像</strong>是一个特殊的文件系统，除了提供容器运行所需的程序、库、资源、配置等文件外，还包含一些为运行时准备的一些配置参数(如匿名卷、环境变量、用户等)，镜像不包含任何动态数据，其内容在构建后也不会被改变。<br>因为镜像包含操作系统完整的root文件系统，其体积往往是庞大的，因此在Docker设计时，就充分利用<code>UnionFS</code>技术，将其设计为分层存储的架构。严格来说，镜像只是一个虚拟的概念，其实际体现并非由一个文件组成，而是由一组文件系统组成，或者说，由<strong>多层文件系统</strong>联合组成，通过分层存储的形式，大大优化了镜像的存储空间。</p>
<h5 id="容器-1"><a href="#%E5%AE%B9%E5%99%A8-1" class="headerlink" title="容器"></a>容器</h5><p>镜像和容器的关系，就像是<code>类</code>和<code>实例</code>一样，镜像时静态的定义，容器时镜像运行时的实体，容器可以被创建、启动、停止、删除、暂停等。<br>容器的实质是进程，但与之结在宿主中执行的进程不同，容器进程运行于属于自己的<strong>独立的命名空间</strong>。因此容器可以拥有自己的root文件系统、自己的网络配置、自己的进程空间，甚至自己的用户ID空间。容器内的进程是运行在一个隔离的环境里，这种特性使得容器封装的应用比直接在宿主运行更加安全。 </p>
<h5 id="仓库"><a href="#%E4%BB%93%E5%BA%93" class="headerlink" title="仓库"></a>仓库</h5><p>镜像构建完成后，可以很容易在当前宿主机上运行，但是，如果需要在其它服务器上使用这个镜像，据需要一个集中的存储、分发镜像的服务，这就是<code>Docker Registry</code>。<br>一个<code>Docker Registry</code>可以包含多个仓库，每个仓库可以包含多个标签，每个标签对应一个镜像。 </p>
<ul>
<li><p><strong>Docker Registry 公开服务</strong><br>开放给用户使用、允许用户管理镜像的Registry服务，一般这类服务允许用户免费上传、下载公开的镜像，并可能提供收费服务供用户管理私有镜像。<br>最常用的Registry公开服务时官方的<a target="_blank" rel="noopener" href="https://hub.docker.com/">DockerHub</a>，这也是默认的Registry，拥有大量的高质量官方镜像。 </p>
</li>
<li><p><strong>私有Docker Registry</strong><br>除了使用公开服务外，用户还可以在本地搭建私有Docker Registry，开源的Docker Registry只提供了<code>Docker Registry API</code>的服务端的实现，但不包含图形界面，以及镜像维护、用户管理、访问控制等高级功能。<br>除了官方的Docker Registry外，还有第三方软件实现了<code>Docker Registry API</code>，甚至提供了用户界面以及一些高级功能，比如<a target="_blank" rel="noopener" href="https://github.com/goharbor/harbor">Harbor</a>和<a target="_blank" rel="noopener" href="https://yeasy.gitbook.io/docker_practice/repository/nexus3_registry">Sonatype Nexus</a></p>
</li>
</ul>
<h4 id="Docker安装"><a href="#Docker%E5%AE%89%E8%A3%85" class="headerlink" title="Docker安装"></a>Docker安装</h4><h5 id="for-mac"><a href="#for-mac" class="headerlink" title="for mac"></a>for mac</h5><p>使用Homebrew Cask进行安装: </p>
<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">brew install --cask docker
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
<h4 id="使用镜像"><a href="#%E4%BD%BF%E7%94%A8%E9%95%9C%E5%83%8F" class="headerlink" title="使用镜像"></a>使用镜像</h4><h5 id="获取镜像"><a href="#%E8%8E%B7%E5%8F%96%E9%95%9C%E5%83%8F" class="headerlink" title="获取镜像"></a>获取镜像</h5><p><a target="_blank" rel="noopener" href="https://hub.docker.com/">DockerHub</a>上有大量高质量镜像可以使用，从Docker镜像仓库获取镜像的命令是<code>docker pull</code>，其命令格式为:</p>
<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">$ docker pull [选项] [Docker Registry 地址[:端口号]/]仓库名[:标签]
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
<p>具体的options可以通过<code>docker pull --help</code>查看，镜像仓库地址和仓库名一般为如下格式: </p>
<ul>
<li>Docker镜像仓库地址: 地址的格式一般是&lt;域名/IP&gt;[:端口号], 默认地址是DockerHub(<code>docker.io</code>) </li>
<li>仓库名: 仓库名一般是两段式名称，即<code>&lt;用户名/软件名&gt;</code>，对于docker hub，如果不给出用户名，则默认为<code>library</code>，也就是官方镜像。 </li>
</ul>
<p>栗子: </p>
<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">$ docker pull ubuntu:18.04
18.04: Pulling from library/ubuntu
a404e5416296: Pull complete
Digest: sha256:ca70a834041dd1bf16cc38dfcd24f0888ec4fa431e09f3344f354cf8d1724499
Status: Downloaded newer image for ubuntu:18.04
docker.io/library/ubuntu:18.04
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span><span></span><span></span></span></code></pre>
<p>这是最简洁的一种命令格式，从官方镜像库下载官方镜像，在下载过程中可以更直观理解镜像分层存储的概念，下载时是一层层去下载，下载结束后，给出该镜像完整的<code>sha256</code>摘要，确保下载的一致性。 </p>
<h5 id="镜像运行"><a href="#%E9%95%9C%E5%83%8F%E8%BF%90%E8%A1%8C" class="headerlink" title="镜像运行"></a>镜像运行</h5><p>以上面下载的<code>ubuntu 18.04</code>镜像为例，如果我们打算启动里面的<code>bash</code>并进行交互操作的话，可以执行下述命令: </p>
<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">$ docker run -it --rm ubuntu:18.04 bash 
root@fb3e34c006ee:/# cat /etc/os-release
NAME="Ubuntu"
VERSION="18.04.6 LTS (Bionic Beaver)"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 18.04.6 LTS"
VERSION_ID="18.04"
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
VERSION_CODENAME=bionic
UBUNTU_CODENAME=bionic
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></span></code></pre>
<p>参数介绍:</p>
<ul>
<li><code>-i</code>: 交互式操作 </li>
<li><code>-t</code>: 终端 </li>
<li><code>--rm</code>: 容器退出后随之将其删除 </li>
<li><code>ubuntu:18.84</code>: 这是指用<code>ubuntu:18.04</code>镜像为基础来启动容器 </li>
<li><code>bash</code>: 放在镜像名后的是命令，这里我们希望有个交互式Shell，这里选用<code>bash</code>。 </li>
</ul>
<h5 id="列出镜像"><a href="#%E5%88%97%E5%87%BA%E9%95%9C%E5%83%8F" class="headerlink" title="列出镜像"></a>列出镜像</h5><p>要想列出已经下载下来的镜像，可以使用<code>docker image ls</code>命令: </p>
<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">$ docker image ls
REPOSITORY               TAG       IMAGE ID       CREATED        SIZE
nginx                    latest    76c69feac34e   9 days ago     142MB
ubuntu                   18.04     71eaf13299f4   10 days ago    63.1MB
docker/getting-started   latest    cb90f98fd791   6 months ago   28.8MB
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span><span></span></span></code></pre>
<p>需要注意的是，由于docker镜像具有多层存储结构，并且可以继承、复用，因此罗列出的镜像体积总和并非所有镜像的实际硬盘消耗，实际镜像硬盘占用空间很可能要比这个列表镜像大小的总和要小得多，可以通过<code>docker system df</code>命令来便捷地查看镜像、容器、数据卷所占空间。 </p>
<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">$ docker system df
TYPE            TOTAL     ACTIVE    SIZE      RECLAIMABLE
Images          3         0         233.7MB   233.7MB (100%)
Containers      0         0         0B        0B
Local Volumes   0         0         0B        0B
Build Cache     0         0         0B        0B
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span><span></span><span></span></span></code></pre>
<p>几种特殊镜像: </p>
<ul>
<li>虚悬镜像: 由于新旧镜像同名，旧镜像名称被取消，从而出现的仓库名、标签均为<code>&lt;none&gt;</code>的镜像，可以用下面的命令删除:<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">$ docker image prune
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
</li>
<li>中间层镜像: 为了加速镜像构建、重复利用资源，docker会利用中间层镜像，<code>docker image ls</code>只会显示顶层镜像，如果希望显示包括中间层镜像的所有镜像的话，需要添加<code>-a</code>参数: <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">$ docker image prune -a
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
</li>
<li>列出部分镜像: 可以在<code>docker image ls</code>后添加仓库名/仓库名:标签形式来列出特定镜像也可以通过添加<code>-f</code>参数通过某些过滤规则来罗列过滤后的镜像。 </li>
</ul>
<h5 id="删除本地镜像"><a href="#%E5%88%A0%E9%99%A4%E6%9C%AC%E5%9C%B0%E9%95%9C%E5%83%8F" class="headerlink" title="删除本地镜像"></a>删除本地镜像</h5><p>如果要删除本地的镜像，可以使用<code>docker image rm</code>命令，其格式为: </p>
<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">$ docker image rm [选项]  [ ...]
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
<p>其中<code>&lt;镜像&gt;</code>可以是<code>镜像名、镜像短ID、镜像长ID、镜像摘要</code></p>
<h5 id="利用commit理解镜像构成"><a href="#%E5%88%A9%E7%94%A8commit%E7%90%86%E8%A7%A3%E9%95%9C%E5%83%8F%E6%9E%84%E6%88%90" class="headerlink" title="利用commit理解镜像构成"></a>利用<code>commit</code>理解镜像构成</h5><blockquote>
<p><strong>注意</strong>: <code>docker commit</code>除了学习之外，还有一些特殊的应用场合，比如入侵后保存现场等。但是一般不使用<code>docker commit</code>定制镜像，定制镜像应当使用<code>Dockerfile</code>来完成。</p>
</blockquote>
<p>镜像是容器的基础，每次执行<code>docker run</code>都需要指定哪个镜像作为容器运行的基础，在之前的例子中，我们所使用的都是来自<code>Docker Hub</code>的镜像，而当这些镜像无法之结满足需求时，就需要进行镜像定制。 </p>
<p>镜像是多层存储，每一层是在前一层的基础上进行的修改；而容器同样也是多层存储，是在以镜像为基础层，在其基础上加一层作为容器运行的存储层。</p>
<p>以定制一个web服务器为栗子，讲解镜像构建: </p>
<ul>
<li><p>运行容器</p>
<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">$ docker run --name webserver -d -p 80:80 nginx
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
<p>这条命令会用<code>nginx</code>镜像启动了一个容器，命名为<code>webserver</code>，并且映射了80端口，这样可以用浏览器去访问<code>nginx</code>服务器。<br><img src="/.io//%E7%95%8C%E9%9D%A2%E6%98%BE%E7%A4%BA.png" alt></p>
</li>
<li><p>修改界面显示内容<br>可以使用<code>docker exec</code>命令进入容器，修改其内容.</p>
<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">$ docker exec -it webserver bash 
root@f979a53d4179:/# echo '<h1>Hello, Docker!</h1>' &gt; /usr/share/nginx/html/index.html
root@f979a53d4179:/# exit
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span></span></code></pre>
<p>我们以交互式终端进入<code>webserver</code>容器，并执行了<code>bash</code>命令，也就是获得一个可操作的Shell。<br>然后替代了<code>index.html</code>中的内容以修改网页显示内容，此时刷新浏览器:<br><img src="/.io//%E9%A1%B5%E9%9D%A2%E6%98%BE%E7%A4%BA_%E6%9B%B4%E6%94%B9%E5%90%8E.png" alt> </p>
</li>
<li><p>查看变化<br>我们修改了容器的文件，也就是改动了容器的存储层，我们可以通过<code>docker diff</code>命令看到具体的改动，在修改了容器后，我们希望能将其保存下来形成镜像。<br>当我们运行一个容器的时候(如果不使用卷的话)，我们做的任何文件修改都会被记录在容器存储层里。而Docker提供了一个<code>docker commit</code>命令，可以将容器的存储层保存下来成为镜像。换句话说，就是在原有镜像的基础上，再叠加上容器的存储层，并构成新的镜像，以后我们运行这个新镜像的时候，就会拥有原有容器最后的文件变化。</p>
</li>
<li><p>生成镜像<br><code>docker commit</code>的语法格式为</p>
<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">docker commit [选项]  [[:]]
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
<p>可以使用下面的命令将容器保存为镜像: </p>
<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">$ docker commit \
  --author "xuejy" \
  --message "修改了默认网页" \
  webserver \
  nginx:v2
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span><span></span></span></code></pre>
<p>其中<code>--author</code>是指定修改的作者，而<code>--message</code>则是记录本次修改的内容</p>
</li>
<li><p>慎用<code>docker commit</code><br>通过<code>docker diff webserver</code>可以看到有大量无关内容被添加进来，导致镜像极为臃肿。<br>此外，使用<code>docker commit</code>意味着所有对镜像的操作都是黑箱操作，生成的镜像也被称为<code>黑箱镜像</code>，换句话说，除了制作镜像的人知道执行过什么命令、怎么生成的镜像，别人根本无从得知。</p>
</li>
</ul>
<h5 id="使用Dockerfile定制镜像"><a href="#%E4%BD%BF%E7%94%A8Dockerfile%E5%AE%9A%E5%88%B6%E9%95%9C%E5%83%8F" class="headerlink" title="使用Dockerfile定制镜像"></a>使用Dockerfile定制镜像</h5><p>镜像的定制实际上就是定制每一层所添加的配置、文件。如果我们可以把每一层修改、安装、构建、操作的命令都写入一个脚本，用这个脚本来构建、定制镜像，那么之前提及的无法重复的问题、镜像构建透明性的问题、体积的问题就都会解决，这个脚本就是dockerfile。<br>Dockerfile是一个文本文件，其包含了一条条的指令，每一条指令构建一层，因此每一条指令的内容就是描述该层应当如何构建。</p>
<ul>
<li>在一个空白目录中，建立一个文本文件，并命名为<code>Dockerfile</code>:<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">$ mkdir mynginx
$ cd mynginx
$ touch Dockerfile
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span></span></code></pre>
</li>
<li><p>其内容为</p>
<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">FROM nginx
RUN echo '<h1>Hello, Docker!</h1>' &gt; /usr/share/nginx/html/index
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span></span></code></pre>
<ul>
<li><strong>FROM指定基础镜像</strong><br>所谓定制镜像，就是以一个镜像为基础，在其上进行定制。<code>FROM</code>就是指定<strong>基础镜像</strong>，基础镜像可以使一些服务类的镜像，比如<code>nginx、redis</code>，也可以是语言类的镜像，比如<code>python、ruby</code>，甚至可以是更为基础的操作系统镜像，如<code>ubuntu、debian</code>等；除了选择现有镜像为基础镜像外，Docker还存在一个特殊的镜像，名为<code>scratch</code>，表示一个空白的镜像。</li>
<li><strong>RUN执行命令</strong><br><code>RUN</code>是用来执行命令行命令的，其格式有两种: <ul>
<li>shell格式: <code>RUN&lt;命令&gt;</code>，就像直接在命令行中输入的命令一样</li>
<li>exec格式: <code>RUN ["可执行文件", "参数1", "参数2"]</code>，这更像是函数调用中的格式<br>需要注意，<code>RUN</code>命令可以像<code>Shell</code>脚本一样执行命令，但并不是要把每个命令对应一个<code>RUN</code>，因为每一个<code>RUN</code>行为都会建立一层镜像，臃肿、非常多层的镜像，不仅仅增加了构建部署的时间，也很容易出错。对于多条命令的单层镜像，正确的写法应当如下: <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">FROM debian:stretch
RUN set -x; buildDeps='gcc libc6-dev make wget' \
&amp;&amp; apt-get update \
&amp;&amp; apt-get install -y $buildDeps \
&amp;&amp; wget -O redis.tar.gz "http://download.redis.io/releases/redis-5.0.3.tar.gz" \
&amp;&amp; mkdir -p /usr/src/redis \
&amp;&amp; tar -xzf redis.tar.gz -C /usr/src/redis --strip-components=1 \
&amp;&amp; make -C /usr/src/redis \
&amp;&amp; make -C /usr/src/redis install \
&amp;&amp; rm -rf /var/lib/apt/lists/* \
&amp;&amp; rm redis.tar.gz \
&amp;&amp; rm -r /usr/src/redis \
&amp;&amp; apt-get purge -y --auto-remove $buildDeps
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></span></code></pre>
<strong>注意在构建之后一定要清理掉无关文件</strong>。</li>
</ul>
</li>
</ul>
</li>
<li><p>构建镜像</p>
<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">$ docker build -t nginx:v3 .
Sending build context to Docker daemon 2.048 kB
Step 1 : FROM nginx
---&gt; e43d811ce2f4
Step 2 : RUN echo '<h1>Hello, Docker!</h1>' &gt; /usr/share/nginx/html/index.html
---&gt; Running in 9cdc27646c7b
---&gt; 44aa4490ce2c
Removing intermediate container 9cdc27646c7b
Successfully built 44aa4490ce2c
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></span></code></pre>
</li>
<li><p>镜像构建上下文<br><code>docker build</code>命令最后的<code>.</code>指定的是上下文路径。<br>Docker在运行时分为Docker引擎(服务端守护进程)和客户端工具，Docker的引擎提供了一组REST API, 被称为<code>Docker remote API</code>，而如<code>docker</code>命令这样的客户端工具，则是通过这组API与Docker引擎交互，从而完成各种功能。因此，虽然表面上我们在本机使用各种<code>docker</code>功能，但实际上，一切的都是使用的远程调用形式在服务端完成。</p>
</li>
</ul>
<p>当我们进行镜像构建时，经常会需要将一些本地文件复制进镜像，而通过指定上下文路径，相当于给定了一个地址锚点，让服务端能够找到这些文件。</p>
<h5 id="Dockerfile-指令详解"><a href="#Dockerfile-%E6%8C%87%E4%BB%A4%E8%AF%A6%E8%A7%A3" class="headerlink" title="Dockerfile 指令详解"></a>Dockerfile 指令详解</h5><p>除了<code>FROM</code>、<code>RUN</code>，还提供了<code>COPY、ADD</code>等十多个指令可供使用 </p>
<h4 id="操作容器"><a href="#%E6%93%8D%E4%BD%9C%E5%AE%B9%E5%99%A8" class="headerlink" title="操作容器"></a>操作容器</h4><p>容器是独立运行的一个或一组应用，以及它们的运行态环境。</p>
<h5 id="容器启动"><a href="#%E5%AE%B9%E5%99%A8%E5%90%AF%E5%8A%A8" class="headerlink" title="容器启动"></a>容器启动</h5><p>启动容器有两种方式，一种是基于镜像新建一个容器并启动，另外一个是将在终止状态的容器重新启动。</p>
<ul>
<li>新建并启动: 所需的命令为<code>docker run</code>，当使用该命令创建容器时，Docker在后台进行的操作包括： <ul>
<li>检查本地是否存在指定的镜像，不存在就从<code>registry</code>下载 </li>
<li>利用镜像创建并启动一个容器 </li>
<li>分配一个文件系统，并在只读的镜像层外面挂载一层可读写层 </li>
<li>从宿主主机配置的网桥接口中桥接一个虚拟接口到容器中去</li>
<li>从地址池配置一个ip地址给容器</li>
<li>执行用户指定的应用程序</li>
<li>执行完毕后容器被终止</li>
</ul>
</li>
<li>启动已终止容器: 可以通过<code>docker container start</code>直接将一个已经终止的容器启动 </li>
</ul>
<p><strong>容器的核心在于所执行的应用程序，所需要的资源都是应用程序运行所必须的，除此之外，并没有其它资源。</strong></p>
<h5 id="守护态运行"><a href="#%E5%AE%88%E6%8A%A4%E6%80%81%E8%BF%90%E8%A1%8C" class="headerlink" title="守护态运行"></a>守护态运行</h5><p>更多的时候，需要让Docker在后台运行而不是直接把执行命令的结果输出在当前宿主机下，此时可以通过添加<code>-d</code>参数来实现，如果想要获取容器的输出信息，可以通过<code>docker container logs [container ID or NAMES]</code>来实现。 </p>
<h5 id="终止"><a href="#%E7%BB%88%E6%AD%A2" class="headerlink" title="终止"></a>终止</h5><p>可以使用<code>docker container stop</code>来终止一个运行中的容器，此外，当Docker容器中指定的应用终结时，容器也自动终止。</p>
<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">docker container start [] # 启动处于终止状态的容器 
docker container restart [] # 终止一个运行态的容器，然后再启动它
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span></span></code></pre>
<h5 id="进入容器"><a href="#%E8%BF%9B%E5%85%A5%E5%AE%B9%E5%99%A8" class="headerlink" title="进入容器"></a>进入容器</h5><p>在使用<code>-d</code>参数时，容器启动后会进入后台，某些时候需要进入容器进行操作，此时可以使用<code>docker attach</code>和<code>docker exec</code>命令。</p>
<ul>
<li><code>docker attach</code>:<br>```shell<br>$ docker run -dit ubuntu<br>243c32535da7d142fb0e6df616a3c3ada0b8ab417937c853a9e1c251f499f550</li>
</ul>
<p>$ docker container ls<br>CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES<br>243c32535da7        ubuntu:latest       “/bin/bash”         18 seconds ago      Up 17 seconds                           nostalgic_hypatia</p>
<p>$ docker attach 243c<br>root@243c32535da7:/#</p>
<pre><code>如果从这个`stdin`中`exit`，会导致容器的停止 
- `docker exec`: 
`docker exec`后可以跟多个参数，常用的有`-i`和`-t`，如果只用`-i`参数，由于没有分配伪终端，界面没有命令行提示符，但命令执行结果仍然可以返回，当`-i -t`一起使用时，则可以看到熟悉的Linux命令符。
```shell
$ docker run -dit ubuntu
69d137adef7a8a689cbcb059e94da5489d3cddd240ff675c640c8d96e84fe1f6

$ docker container ls
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
69d137adef7a        ubuntu:latest       "/bin/bash"         18 seconds ago      Up 17 seconds                           zealous_swirles

$ docker exec -i 69d1 bash
ls
bin
boot
dev
...

$ docker exec -it 69d1 bash
root@69d137adef7a:/#
</code></pre><p>如果从这个stdin中exit，容器仍会继续运行，推荐使用<code>docker exec</code>命令进入容器。 </p>
<h5 id="导出和导入"><a href="#%E5%AF%BC%E5%87%BA%E5%92%8C%E5%AF%BC%E5%85%A5" class="headerlink" title="导出和导入"></a>导出和导入</h5><ul>
<li>导出容器: 如果要导出某个本地容器，可以使用<code>docker export</code>命令 <pre class="line-numbers language-lang-shell"><code class="language-lang-shell">$ docker container ls -a
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS                    PORTS               NAMES
7691a814370e        ubuntu:18.04        "/bin/bash"         36 hours ago        Exited (0) 21 hours ago                       test
$ docker export 7691a814370e &gt; ubuntu.tar
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span></span></code></pre>
</li>
<li>导入容器快照<br>可以使用<code>docker import</code>从容器快照文件中再导入为镜像<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">$ docker import ubuntu.tar test/ubuntu:v1.0
sha256:228a7a889fd20e6d4b4e3bdf8069a0b64d24c1758196919a4c34480841256c36
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span></span></code></pre>
<h5 id="删除"><a href="#%E5%88%A0%E9%99%A4" class="headerlink" title="删除"></a>删除</h5></li>
<li>删除容器: 可以通过<code>docker container rm</code>来删除一个处于终止状态的容器， 如果要删除一个正在运行中的容器，可以添加<code>-f</code>参数 。</li>
<li>清理所有处于终止状态的容器: 通过<code>docker container prune</code>实现 </li>
</ul>
<h4 id="访问仓库"><a href="#%E8%AE%BF%E9%97%AE%E4%BB%93%E5%BA%93" class="headerlink" title="访问仓库"></a>访问仓库</h4><p>仓库(Repository)是集中存放镜像的地方，注册服务器(Registry)是管理仓库的具体服务器，每个服务器上可以有多个仓库，而每个仓库下面有多个镜像，从这方面来说，仓库可以被认为是一个具体的项目或目录，例如对于仓库地址<code>docker.io/ubuntu</code>来说，<code>docker.io</code>是注册服务器地址，<code>ubuntu</code>是仓库名。 </p>
<h5 id="Docker-Hub"><a href="#Docker-Hub" class="headerlink" title="Docker Hub"></a>Docker Hub</h5><p>这是Docker官方维护的一个公共仓库，其中已经包括了数量超过2650000的镜像，大部分需求都可以通过在Docker Hub中直接下载镜像来实现。</p>
<ul>
<li><p>镜像查找<br>可以通过<code>docker search</code>命令来查找官方仓库中的镜像</p>
<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">$ docker search centos
NAME                               DESCRIPTION                                     STARS     OFFICIAL   AUTOMATED
centos                             The official build of CentOS.                   6449      [OK]
ansible/centos7-ansible            Ansible on Centos7                              132                  [OK]
consol/centos-xfce-vnc             Centos container with "headless" VNC session…   126                  [OK]
jdeathe/centos-ssh                 OpenSSH / Supervisor / EPEL/IUS/SCL Repos - …   117                  [OK]
centos/systemd                     systemd enabled base container.                 96                   [OK]
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span><span></span><span></span><span></span></span></code></pre>
</li>
<li><p>下载镜像<br>通过<code>docker pull</code>命令将镜像下载到本地</p>
<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">$ docker pull centos
Using default tag: latest
latest: Pulling from library/centos
7a0437f04f83: Pull complete
Digest: sha256:5528e8b1b1719d34604c87e11dcd1c0a20bedf46e83b5632cdeac91b8c04efc1
Status: Downloaded newer image for centos:latest
docker.io/library/centos:latest
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span><span></span><span></span><span></span></span></code></pre>
</li>
<li>推送镜像<br>用户也可以在登陆后通过<code>docker push</code>将自己的镜像推送到Docker Hub</li>
</ul>
<pre class="line-numbers language-lang-shell"><code class="language-lang-shell">$ docker tag ubuntu:18.04 username/ubuntu:18.04

$ docker image ls

REPOSITORY                                               TAG                    IMAGE ID            CREATED             SIZE
ubuntu                                                   18.04                  275d79972a86        6 days ago          94.6MB
username/ubuntu                                          18.04                  275d79972a86        6 days ago          94.6MB

$ docker push username/ubuntu:18.04

$ docker search username

NAME                      DESCRIPTION                                     STARS               OFFICIAL            AUTOMATED
username/ubuntu
<span aria-hidden="true" class="line-numbers-rows"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></span></code></pre>
<ul>
<li>自动构建(付费服务)<br>自动构建可以自动触发构建镜像，方便升级镜像。自动构建允许用户通过Docker Hub指定跟踪一个目标网站(比如<code>github</code>)上的项目，一旦项目发生新的提交或者创建了新的标签，Docker Hub会自动构建镜像并推送到Docker Hub中。</li>
</ul>
<h6 id="私有仓库"><a href="#%E7%A7%81%E6%9C%89%E4%BB%93%E5%BA%93" class="headerlink" title="私有仓库"></a>私有仓库</h6><h4 id="数据管理"><a href="#%E6%95%B0%E6%8D%AE%E7%AE%A1%E7%90%86" class="headerlink" title="数据管理"></a>数据管理</h4><p><code>数据卷</code>是可供一个或多个容器使用的特殊目录，它绕过<code>UFS</code>，可以提供很多有用的特性:</p>
<ul>
<li><code>数据卷</code>可以在容器之间共享和重用</li>
<li>对<code>数据卷</code>的修改会立马生效</li>
<li>对<code>数据卷</code>的更新不会影响镜像 </li>
<li><code>数据卷</code>默认会一直存在，即使容器被删除 </li>
</ul>
<blockquote>
<p> <code>数据卷</code>的使用，类似于Linux下对目录或文件进行mount,镜像中被指定为挂载点的目录中的文件会复制到数据卷中</p>
</blockquote>
<h5 id="创建一个数据卷"><a href="#%E5%88%9B%E5%BB%BA%E4%B8%80%E4%B8%AA%E6%95%B0%E6%8D%AE%E5%8D%B7" class="headerlink" title="创建一个数据卷"></a>创建一个数据卷</h5><pre class="line-numbers language-lang-shell"><code class="language-lang-shell">$ docker volume create my-vol
<span aria-hidden="true" class="line-numbers-rows"><span></span></span></code></pre>
{% endraw %}
