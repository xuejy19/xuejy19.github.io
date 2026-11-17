# 风之声

薛建业的个人主页与「风之声」博客。使用 **Hexo 8.1.2 + 独立学术主页 + 原 Matery 博客主题**。

`/` 是清爽的个人主页，`/blog/` 是原 Matery 博客。文章、标签、分类和归档保持原地址；旧 `/page/2/` 等分页会跳转到新的博客分页。

## 本地预览

要求 Node.js 22 或以上；`.nvmrc` 记录了主版本。使用 nvm 时，先执行 `nvm install && nvm use`。

```sh
npm ci
npm run build
npm run check
npm run server
```

打开 <http://127.0.0.1:4000/>。预览服务只监听本机，不会发布到 GitHub。

## 个人主页

- `source/_data/profile.yml` 管理姓名、简介、学术兴趣、职业经历、教育经历、代表成果、联系方式和精选笔记。资料根据用户授权查阅的简历、研究材料及公开论文整理；任职采用经历表述。论文按出版信息展示，技术报告可用 `type` 和 `contribution` 标明类型及个人贡献。
- `homepage/index.ejs` 是独立主页模板，`source/css/academic.css` 管理主页样式，与 Matery 阅读页分开。
- 首页使用的猫头像在 `source/medias/avatar-cat-warm-wise-v1.png`；未采用的头像保存在 `homepage/alternatives/`，不生成到网站中。
- `scripts/academic-home.js` 生成根首页及旧博客分页跳转。主页不加载第三方字体或 JavaScript，手机上可直接使用导航和博客入口。

## 写作与配置

```sh
npx hexo new post "文章标题"
```

- 文章在 `source/_posts/`，分类和标签写在文章开头的 YAML 中。
- `_config.yml` 管理站点、链接和部署；`themes/hexo-theme-matery/_config.yml` 管理主题。
- `source/Img/` 保存正文原图，`source/medias/` 保存头像、封面等；使用 `/Img/文件名` 或 `/medias/文件名` 引用。
- 数学文章设置 `math: true`；公式在构建时由 KaTeX 渲染，不依赖旧 MathJax CDN。
- `img` 为文章列表与文章页的封面。
- 自定义 CSS 在 `source/css/custom.css`。页面模板和样式位于 `themes/hexo-theme-matery/`。
- `drafts/` 保存旧项目未发布的文档，不在 Hexo 的输入目录中。

## 迁移记录

- 源码基于原 `source` 分支；`master` 保存已生成的网站。
- `migration/published-manifest.json` 记录原站 65 个文章入口和 663 个标题锚点，用于防止链接丢失。
- 恢复了 13 个缺源码页面、3 篇较新的正文、1 篇受保护文章，以及1篇旧渲染器下的特殊排版文章，共 18 篇使用原发布 HTML 保存正文。
- 其余文章继续使用原 Markdown。恢复文章保留 `.md` 文件及元数据，正文中的 `{% raw %}` 包含 HTML，编辑时应保留这对标记；原 Markdown 可从 Git 的 `origin/source` 查阅。
- 加密文章保留原密文、解密脚本和密码提示；未重新设置密码或公开原文。
- 保留 `/about/`、`/contact/`、`/friends/`、`/timeline/` 等入口。动态页仍连接原 Artitalk 服务。
- 核心样式、脚本、图标字体、公式字体与已有图片由本站提供。Valine 评论与访问统计仍依赖对应外部服务。
- 旧域名链接已改为 `xuejy19.github.io`，构建不会生成 `CNAME`。

`tools/restore-published.py` 是一次性恢复记录，依赖 Python 的 PyYAML、lxml。正常写作和构建不需要它；**不要在编辑恢复文章后重新运行**，否则会用旧发布正文覆盖新编辑。

## 校验与浏览器截图

`npm run check` 检查个人主页、博客入口及分页兼容，并检查文章入口、旧锚点、公式、加密内容、搜索条目、核心资源及旧域名。搜索只应包含 65 篇已发布文章。

本地服务运行时，可以执行：

```sh
npm run check:browser
```

浏览器检查涵盖封面、搜索、目录、公式、加密界面、代码和手机导航，并在 `.cache/previews/` 保存截图。Mac 默认使用已安装的 Chrome；其他环境可设置 `CHROME_PATH`，或运行 `npx playwright install chromium` 安装测试浏览器。可用 `PREVIEW_URL` 指定其他本地端口。

浏览器校验会阻止外部请求，以确认主要阅读功能可独立工作；它不验证评论后端、访问统计或 Artitalk 服务，也不会发表测试评论。

## 尚需补充的旧图片

`migration/image-review.json` 记录 49 个历史遗留图片地址：48 个 `/.io/` 路径，以及一张缺失的 `time.png`。迁移时能找到原图的 131 处引用已改为同站路径。其余图片保留原引用，等待找到原图后补入；校验不会把这些历史问题计为新增资源故障。

## 发布

先查看本地效果，再执行：

```sh
sh deploy.sh
```

此脚本重新构建、运行静态校验，成功后才将产物发布到远程 `master`。根地址 <https://xuejy19.github.io/> 展示个人主页，<https://xuejy19.github.io/blog/> 展示博客；不需要另建仓库或更换域名。它会更新公开网站，需要可用的 GitHub SSH 身份。`npm run deploy` 会校验已有构建结果，再以普通 Git 提交发布；不会强制改写发布分支历史。

源码与生成结果分开保存：远端 `source` 保存 Hexo 源码，`master` 保存生成的网站。修改后先审查并提交源码，再用 `git push origin HEAD:source` 同步源码分支，最后执行 `sh deploy.sh` 发布网站。当前本地分支是 `theme/matery`，跟踪 `origin/source`；旧 `push.sh` 固定推送本地 `source`，不适用于此分支。

已按个人偏好恢复原 Matery 模板。Fluid 调研仅作为历史记录，见 [THEME_MIGRATION_RESEARCH.md](THEME_MIGRATION_RESEARCH.md)。
