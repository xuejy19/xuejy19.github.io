'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const cheerio = require('cheerio');
const manifest = require('../migration/published-manifest.json');
const imageReview = require('../migration/image-review.json');
const root = path.resolve(__dirname, '../public');
const knownMissingImages = new Set(Object.keys(imageReview.unresolved));
let headings = 0;
let formulas = 0;
const read = route => fs.readFileSync(path.join(root, route.replace(/^\//, '')), 'utf8');

for (const article of manifest.articles) {
  const html = read(`${article.url}index.html`);
  const $ = cheerio.load(html);
  assert.ok($('title').text().includes(article.title), `Wrong article at ${article.url}`);
  assert.ok($('#articleContent').length, `Missing body at ${article.url}`);
  assert.equal($('.katex-error').length, 0, `Invalid math at ${article.url}`);
  assert.equal($('script[type^="math/tex"]').length, 0, `Legacy math not rendered at ${article.url}`);
  for (const id of article.headings) {
    assert.ok($('[id]').toArray().some(node => node.attribs.id === id), `Lost heading #${id} at ${article.url}`);
    headings++;
  }
  if (article.encrypted) {
    assert.equal($('#hexo-blog-encrypt').length, 1, 'Encrypted article was exposed or lost');
    if (article.cipher_sha256) assert.equal(crypto.createHash('sha256').update($('#hbeData').text()).digest('hex'), article.cipher_sha256);
  }
  formulas += $('.katex').length;
}

const htmlFiles = fs.readdirSync(root, { recursive: true }).filter(file => file.endsWith('.html'));
for (const file of htmlFiles) {
  const html = read(file);
  assert.ok(!html.includes('soundofwind.top'), `Retired domain in ${file}`);
  assert.ok(!html.includes('cdn.jsdelivr.net/gh/xuejy19/xuejy19.github.io'), `Legacy asset CDN in ${file}`);
  const $ = cheerio.load(html);
  for (const node of $('link[rel="stylesheet"][href],script[src],img[src]').toArray()) {
    const url = node.attribs.src || node.attribs.href;
    if (!url.startsWith('/') || url.startsWith('//') || knownMissingImages.has(url)) continue;
    const local = decodeURIComponent(url.split(/[?#]/)[0]);
    assert.ok(fs.existsSync(path.join(root, local)), `Missing local asset ${url} in ${file}`);
  }
}

assert.ok(!fs.existsSync(path.join(root, 'CNAME')), 'Custom domain would be reintroduced');
assert.ok(!fs.readdirSync(root, { recursive: true }).some(file => file.includes('神秘信件')), 'Excluded article was published');
const xml = read('/search.xml');
const $xml = cheerio.load(xml, { xmlMode: true });
assert.equal($xml('entry').length, manifest.articles.length, 'Search index omitted articles');
assert.ok(!/[\x00-\x08\x0b\x0c\x0e-\x1f]/.test(xml), 'Invalid XML control characters');
for (const route of ['about', 'friends', 'contact', 'timeline', 'tags', 'categories', 'archives']) {
  assert.ok(fs.existsSync(path.join(root, route, 'index.html')), `Lost page /${route}/`);
}
assert.ok(formulas > 100, 'Formula output is unexpectedly small');
console.log(`Verified ${manifest.articles.length} article URLs, ${headings} heading links, ${formulas} formulas and ${htmlFiles.length} pages.`);
console.log(`Historical image URLs still requiring originals: ${knownMissingImages.size} (migration/image-review.json).`);
