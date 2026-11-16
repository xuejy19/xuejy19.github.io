---
title: 安全外壳协议-ssh
date: '2022-10-24 12:00:00'
updated: '2022-10-28 12:00:00'
permalink: 2022/10/24/an-quan-wai-ke-xie-yi-ssh/
tags:
- 安全性协议
categories:
- 技术学习
- 论文阅读
- linux
math: false
legacy_html: true
img: /medias/featureimages/6.jpg
summary: 简介(from wiki)安全外壳协议(Secure Shell Protocol,简称ssh)是一种加密的网络传输协议，可在不安全的网络中为网络服务提供安全的传输环境。SSH通过在网络中创建安全隧道来实现SSH客户端与服务器之间的连接;SSH最常见的用途是远程登录系统，人们通常利用SSH来传输命令
---
{% raw %}
<h4 id="简介-from-wiki"><a href="#%E7%AE%80%E4%BB%8B-from-wiki" class="headerlink" title="简介(from wiki)"></a>简介(from wiki)</h4><p>安全外壳协议(Secure Shell Protocol,简称ssh)是一种加密的网络传输协议，可在不安全的网络中为网络服务提供安全的传输环境。SSH通过在网络中创建安全隧道来实现SSH客户端与服务器之间的连接;SSH最常见的用途是远程登录系统，人们通常利用SSH来传输命令行界面和远程执行命令。<br><span id="more"></span></p>
<h4 id="相关文件"><a href="#%E7%9B%B8%E5%85%B3%E6%96%87%E4%BB%B6" class="headerlink" title="相关文件"></a>相关文件</h4><p>SSH服务会在系统的用户目录下生成一个<code>.ssh</code>文件夹，文件夹下一般会有如下文件: </p>
<ul>
<li><code>config</code>: 配置文件，可以为远程主机配置host、登录端口、默认登录用户等，方便用户使用； </li>
<li><code>known_hosts</code>: 记录本机访问过的服务端所提供的host key，下次访问相同主机时会核对host key，如不匹配会发出警告； </li>
<li><code>id_rsa</code>: 保存本机的私钥 </li>
<li><code>id_rsa.pub</code>: 保存本机的公钥 </li>
<li><code>authorized_keys</code>: 本机作为服务端所保存的已授权客户端的公钥</li>
</ul>
<h4 id="非对称加密"><a href="#%E9%9D%9E%E5%AF%B9%E7%A7%B0%E5%8A%A0%E5%AF%86" class="headerlink" title="非对称加密"></a>非对称加密</h4><p>SSH有密码登录和证书登录两种认证方式，都是以非对称加密实现身份验证。不同于对称加密的加密解密使用同一套密钥的方式，非对称加密拥有公钥和私钥两套密钥，其中 公钥负责加密，私钥负责解密 ，<strong>公钥加密后的密文只能通过对应的私钥进行解密，而通过公钥几乎不可能推算出私钥</strong>。另外提一下，数字签名一类是使用私钥编码、公钥解码的，但是这种方式只能称为编解码而不能称作加密解密，因为公钥是公开的，拥有公钥者都可以通过密文解码出明文，加密的意义就不存在。</p>
<h4 id="密码登陆的验证过程"><a href="#%E5%AF%86%E7%A0%81%E7%99%BB%E9%99%86%E7%9A%84%E9%AA%8C%E8%AF%81%E8%BF%87%E7%A8%8B" class="headerlink" title="密码登陆的验证过程"></a>密码登陆的验证过程</h4><ol>
<li>客户端向远程服务端发出登录请求，服务端把自己的公钥发给客户端；</li>
<li>客户端本机使用服务端提供的公钥对密码进行加密，并把密文发送至服务端；</li>
<li>服务端接收密文并使用私钥把密文解密出密码；</li>
<li>服务端通过密码验证用户身份并返回登录结果；</li>
</ol>
<h4 id="证书登陆的验证过程"><a href="#%E8%AF%81%E4%B9%A6%E7%99%BB%E9%99%86%E7%9A%84%E9%AA%8C%E8%AF%81%E8%BF%87%E7%A8%8B" class="headerlink" title="证书登陆的验证过程"></a>证书登陆的验证过程</h4><ol>
<li>客户端事先生成本机的公钥和私钥，并把公钥追加到服务端主机的authorized_keys文件中；</li>
<li>客户端发出登录请求，服务端在authorized_keys中匹配该客户端对应的公钥；</li>
<li>服务端生成随机数，并使用客户端的公钥进行加密，然后把密文发送给客户端；</li>
<li>客户端通过本机私钥解密密文获得随机数；</li>
<li>客户端利用随机数和会话密钥session key通过MD5生成摘要Digest，并发送给服务端；</li>
<li>服务端使用同样的算法生成另一份Digest；</li>
<li>服务端对比生成的Digest和客户端发送过来的Digest，返回登录结果；</li>
</ol>
{% endraw %}
