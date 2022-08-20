---
title: 初识Docker
date: 2020-10-28 10:20:16
tags: 容器 
categories: 开发知识 
toc: true 
mathjax: true 
---
近期因为在外出差，没咋看论文，为了让时间得到充分利用，自己打算学习一些开发相关的知识，毕竟技多不压身。 这部分就参考[Docker官方文档](https://docs.docker.com/get-started/)来进行Docker基本知识的学习。 
该部分分为以下三个章节进行组织:
- 概念与设置 
- 构建并运行镜像
- 在Docker Hub 上共享镜像 
<!--more-->
### 概念与设置 
#### Docker概念 
Docker是供开发人员和系统管理员 使用容器构建，运行和共享应用程序的平台。使用容器部署应用程序称为容器化。容器不是新的，但用于轻松部署应用程序的容器却是新的。
容器化越来越受欢迎，这是因为容器具有如下几点优点: 
- 灵活：即使最复杂的应用程序也可以容器化。
- 轻量级：容器利用并共享主机内核，在系统资源方面比虚拟机更有效。 
- 可移植：您可以在本地构建，部署到云并在任何地方运行。
- 松散耦合：容器是高度自给自足并封装的，可让您在不破坏其他容器的情况下更换或升级它们。
- 可扩展：您可以在数据中心内增加并自动分布容器副本。
- 安全：容器将积极的约束和隔离应用于流程，而用户无需进行任何配置。 

#### 镜像与容器 
从根本上说，一个容器不过是一个**正在运行的进程**，并对其应用了一些附加的封装功能，以使其与主机和其他容器隔离。容器隔离的最重要方面之一是每个容器都与自己的专用文件系统进行交互。该文件系统由**Docker映像**提供。映像包括运行应用程序所需的所有内容-代码或二进制文件，运行时，依赖关系以及所需的任何其他文件系统对象。

#### 容器和虚拟机 
一个容器本地运行于Linux系统上，并与其他容器共享主机内核。一个容器运行一个离散进程，占用的内存不超过其他任何可执行文件，因此其比较轻量。 
相比之下，虚拟机（VM）运行一个全面的“来宾”操作系统，通过hypervisor对主机资源进行虚拟访问。一般来说，除了应用程序逻辑所消耗的外，vm会产生很多开销。  
![容器与虚拟机](https://raw.githubusercontent.com/xuejy19/xuejy19.github.io/source/Img/Container.png)
 


#### Docker环境设置 
参考[docker桌面版安装及测试](https://docs.docker.com/get-started/)来进行安装，运行:
```bash
docker run hello-world 
```
如果返回:
```bash
Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/get-started/
```
则说明已经在机器上成功安装docker desktop 并进行了快速测试，成功运行了第一个容器化应用`hello-world`. 

### 构建并运行镜像 
#### 简介
通过上面的测试，我们已经设置好了开发环境，接下来我们便可以开发容器化的应用程序，通常，开发流程如下: 
- 首先创建Docker映像，为应用程序的每个组件创建和测试单独的容器。
- 将您的容器和支持基础结构组装成一个完整的应用程序。
- 测试，共享和部署完整的容器化应用程序。 

在本章节中，我们将工作重心放在第一个步骤，创建容器将基于的镜像。请记住，Docker映像捕获了将在其中运行容器化进程的私有文件系统；您需要创建一个图像，其中包含您的应用程序需要运行的内容。
#### 设置 
首先下载`node-bulletin-board`项目，这是一个用Node.js编写的简单公告版应用程序,首先通过`git`下clone该项目:
```bash
git clone https://github.com/dockersamples/node-bulletin-board
cd node-bulletin-board/bulletin-board-app
```

#### 使用dockerfile定义一个容器 
在下载整个项目后，使用找到`bulletin board`应用中的`dockerfile`文件，`dockerfile`描述了如何为容器组装私有文件系统，并且还包含一些元数据来描述如何基于该映像运行容器。 
下面给出该示例项目的`dockerfile`的注释文件: 
```bash
# Use the official image as a parent image.
FROM node:current-slim

# Set the working directory.
WORKDIR /usr/src/app

# Copy the file from your host to your current location.
COPY package.json .

# Run the command inside your image filesystem.
RUN npm install

# Add metadata to the image to describe which port the container is listening on at runtime.
EXPOSE 8080

# Run the specified command within the container.
CMD [ "npm", "start" ]

# Copy the rest of your app's source code from your host to your image filesystem.
COPY . .
```
下面给出该`dockerfile`命令逻辑解释: 
- 通过`FROM`指定一个基础映像-`node:current-slim`。这是由`node.js`公司构建的官方镜像并且已由`Docker`验证为包含`Node.js`长期支持解释器和基本依赖项的高质量镜像。
- 使用`WORKDIR`指定所有后续操作都应该从映像文件系统中的`/usr/src/app`目录（而不是主机的文件系统）执行。
- 使用`COPY`将`package.json`文件从主机复制到当前目录`(.)`,在本例子中，该当前目录是`/usr/src/app/package.json`
- 使用`RUN`命令在你的镜像文件系统中运行`npm install`命令，它将读取`package.json`确定应用程序的节点依赖项并安装它们。
- 使用`COPY`将应用的剩余资源代码从主机复制到你的镜像文件系统。

上面的步骤构建了我们的映像文件系统，但同时还包含一些其它命令:
- 使用`CMD`命令来指定了如何基于该映像运行容器，该命令意味着该镜像要支持的容器化进程是`npm start` 
- `EXPOSURE 8080`则是通知`Docker`容器会实时监听端口`8080` 

#### 构建并测试镜像
运行以下命令来构建公告板映像:
```bash
docker build --tag bulletinboard:1.0 . 
```
运行成功后会显示： 
```bash
Successfully tagged bulletinboard:1.0
```
#### 将镜像作为容器运行
1. 运行以下命令以基于新映像启动容器 
```bash
docker run --publish 8000:8080 --detach --name bb bulletinboard:1.0 
```
这里有几个常见的标志: 
- `--publish`要求Docker将主机端口8000上传入的流量转发到容器的端口8080。容器具有自己的专用端口集，因此，如果要从网络访问某个端口，则必须以这种方式将流量转发到该端口。否则，作为默认的安全状态，防火墙规则将阻止所有网络流量到达您的容器。 
- `--detach`要求docker在后台运行此程序 
- `--name`指定一个名称，在后续命令中，可以使用该名称引用容器，本命令中指定为`bb`

2. 在浏览器中访问应用程序`localhost:8000`,可以看到公告版应用程序已启动并且正在运行，在这一步，您通常会尽一切可能确保容器按预期方式工作。例如，现在是运行单元测试的时候了。
3. 对公告板容器正常工作感到满意后可以将其删除：
```bash
docker rm --force bb 
```
该`--force`选项停止一个正在运行中的容器，也可以采用`docker stop bb`来停止一个容器，这样就不需要使用`--force`选项来`rm`容器。

### 在Docker Hub 中分享镜像 
#### 介绍
开发容器化程序的最后一步是在[docker hub](https://hub.docker.com/)之类的注册标上共享镜像，以便可以轻松下载他们，并在任何目标计算机上运行他们。 

#### 创建一个Docker hub 存储库
在创建完docker id之后，可以创建一个docker镜像仓库

