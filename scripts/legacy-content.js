'use strict';

const fs = require('node:fs');
const path = require('node:path');
const cheerio = require('cheerio');
const katex = require('katex');
const manifest = require('../migration/published-manifest.json');

// Recovered HTML keeps the published content, including MathJax 2 script blocks.
// Render those formulas at build time so no obsolete MathJax CDN is needed.
function renderMath(tex, displayMode) {
  return katex.renderToString(tex, {
    displayMode, throwOnError: false, strict: 'ignore', trust: false
  });
}

hexo.extend.filter.register('before_post_render', data => {
  data.content = data.content.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f]/g, '');
  return data;
});

hexo.extend.filter.register('after_post_render', data => {
  const $ = cheerio.load(data.content, null, false);
  if (data.legacy_html && !data.legacy_encrypted) {
    $('script[type^="math/tex"]').each((_, node) => {
      $(node).replaceWith(renderMath($(node).text(), /mode=display/.test($(node).attr('type'))));
    });
    const visit = node => {
      for (const child of [...(node.children || [])]) {
        if (child.type === 'text' && child.data.includes('$')) {
          const escaped = child.data.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
          const rendered = escaped.replace(/(?<!\\)\$([^$\n]+?)(?<!\\)\$/g,
            (_, tex) => renderMath(cheerio.load(tex, null, false).text(), false));
          if (rendered !== escaped) $(child).replaceWith(rendered);
        } else if (child.type !== 'text' && !['pre', 'code', 'script', 'style', 'math', 'annotation'].includes(child.name)
          && !$(child).hasClass('katex')) visit(child);
      }
    };
    visit($.root()[0]);
  }

  // Preserve incoming heading links when changing the Markdown renderer.
  const entry = manifest.articles.find(article => article.source === `source/${data.source}`);
  if (entry && entry.heading_details) {
    const remaining = [...entry.heading_details];
    const normalize = text => text.replace(/[$\s]+/g, '').replace(/[‘’]/g, "'")
      .replace(/[“”]/g, '"').toLowerCase();
    $('h1,h2,h3,h4,h5,h6').each((_, node) => {
      const label = $(node).clone();
      label.find('.katex').each((_, math) => $(math).replaceWith($(math).find('annotation').text()));
      const title = normalize(label.text());
      const index = remaining.findIndex(item => normalize(item.text) === title);
      if (index >= 0) {
        $(node).attr('id', remaining[index].id);
        remaining.splice(index, 1);
      }
    });
  }
  data.content = $.html();
  return data;
}, 30);

// Serve core libraries from the same origin. Package versions are in the lockfile.
hexo.extend.generator.register('local-vendor-assets', function() {
  const routes = [];
  const add = (source, destination) => routes.push({
    path: `vendor/${destination}`, data: () => fs.createReadStream(source)
  });
  const walk = (directory, destination) => {
    for (const item of fs.readdirSync(directory, { withFileTypes: true })) {
      const source = path.join(directory, item.name);
      if (item.isDirectory()) walk(source, `${destination}/${item.name}`);
      else if (!item.name.endsWith('.map')) add(source, `${destination}/${item.name}`);
    }
  };
  const modules = path.join(this.base_dir, 'node_modules');
  walk(path.join(modules, 'katex/dist'), 'katex');
  add(path.join(modules, 'typed.js/lib/typed.min.js'), 'typed/typed.min.js');
  const fontDir = path.join(modules, '@fontsource-variable/noto-sans-sc');
  add(path.join(fontDir, 'index.css'), 'fonts/noto-sans-sc/index.css');
  add(path.join(fontDir, 'LICENSE'), 'fonts/noto-sans-sc/LICENSE.txt');
  walk(path.join(fontDir, 'files'), 'fonts/noto-sans-sc/files');
  return routes;
});

hexo.extend.generator.register('search-index', function(locals) {
  const cdata = value => `<![CDATA[${String(value || '').replace(/\]\]>/g, ']]]]><![CDATA[>')}]]>`;
  const entries = locals.posts.sort('-date').map(post => {
    const $ = cheerio.load(post.content || '', null, false);
    $('script,style,.katex-mathml').remove();
    const text = post.legacy_encrypted ? '' : $.text().replace(/[\x00-\x08\x0b\x0c\x0e-\x1f]/g, '');
    return `<entry><title>${cdata(post.title)}</title><url>${cdata(`/${post.path}`)}</url><content>${cdata(text)}</content></entry>`;
  }).join('\n');
  return { path: 'search.xml', data: `<?xml version="1.0" encoding="utf-8"?>\n<search>\n${entries}\n</search>\n` };
});
