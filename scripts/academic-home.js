'use strict';

const path = require('node:path');

hexo.extend.generator.register('academic-home', function(locals) {
  const profile = locals.data.profile;
  if (!profile) throw new Error('Missing source/_data/profile.yml');
  const html = this.render.renderSync({
    path: path.join(this.base_dir, 'homepage/index.ejs')
  }, { profile, canonical: this.config.url });
  const routes = [{ path: 'index.html', data: html }];

  // Keep bookmarks to the former blog pagination working after moving its index.
  const perPage = this.config.index_generator.per_page;
  const pages = perPage > 0 ? Math.ceil(locals.posts.length / perPage) : 1;
  for (let number = 2; number <= pages; number++) {
    const destination = `/blog/page/${number}/`;
    routes.push({
      path: `page/${number}/index.html`,
      data: `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>博客分页已迁移 · 风之声</title><meta http-equiv="refresh" content="0;url=${destination}"><link rel="canonical" href="${new URL(destination, this.config.url).href}"></head><body><p>博客分页已迁移，<a href="${destination}">继续阅读第 ${number} 页</a>。</p></body></html>`
    });
  }
  return routes;
});
