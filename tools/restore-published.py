#!/usr/bin/env python3
"""One-time recovery from the pre-migration master. Requires PyYAML and lxml.

Published HTML remains the recovery source; original Markdown remains in Git.
Run only before editing recovered posts. Never imports unpublished post content.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import yaml
from lxml import html as lh

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "11f64d77aed7e6b0f9ec83b14a0430a06e7da661"
if (ROOT / 'migration/published-manifest.json').exists():
    raise SystemExit('Recovery already completed. Refusing to overwrite migrated articles.')

def blob(path):
    return subprocess.check_output(["git", "show", BASELINE + ":" + path], cwd=str(ROOT))

def norm(text):
    return re.sub(r"\s+", "", text).lower()

def split_post(text):
    parts = text.split("---", 2)
    return (yaml.safe_load(parts[1]) or {}, parts[2]) if text.startswith("---") else ({}, text)

existing = {}
for p in (ROOT / "source/_posts").glob("*.md"):
    fm, body = split_post(p.read_text())
    existing[norm(str(fm.get("title", p.stem)))] = (p, fm, body)

paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASELINE], cwd=str(ROOT)).decode().splitlines()
entries = []
for path in paths:
    if not re.fullmatch(r"20\d\d/\d\d/\d\d/[^/]+/index.html", path):
        continue
    text = blob(path).decode()
    doc = lh.fromstring(text)
    title = doc.findtext(".//title").rsplit(" | ", 1)[0]
    content = doc.get_element_by_id("articleContent")
    body = (content.text or "") + "".join(lh.tostring(c, encoding="unicode") for c in content)
    date = "-".join(path.split("/")[:3])
    updated = re.search(r"更新日期:[\s\S]{0,100}?(\d{4}-\d\d-\d\d)", text)
    updated = updated.group(1) if updated else date
    url = path[:-len("index.html")]
    found = existing.get(norm(title))
    if found:
        p, fm, original = found
    else:
        p = ROOT / "source/_posts" / (path.split("/")[-2] + ".md")
        fm, original = {}, ""
    tags = doc.xpath('//div[contains(concat(" ",normalize-space(@class)," ")," article-tag ")]')
    tags = [a.text_content().strip() for a in tags[0].xpath('.//a')] if tags else []
    cats = [a.text_content().strip() for a in doc.xpath('//a[contains(concat(" ",normalize-space(@class)," ")," post-category ")]')]
    cover = doc.xpath('//div[contains(concat(" ",normalize-space(@class)," ")," post-cover ")]/@style')
    cover = re.search(r"url\([\"\']?(.*?)[\"\']?\)", cover[0]).group(1) if cover else "/medias/banner/1.jpg"
    cover = cover.replace("https://cdn.jsdelivr.net/gh/xuejy19/xuejy19.github.io", "")
    fm.update(title=title, date=fm.get("date", date + " 12:00:00"), updated=updated + " 12:00:00", permalink=url, tags=list(dict.fromkeys(tags)), categories=list(dict.fromkeys(cats)), index_img=cover, banner_img=cover)
    fm.pop("tag", None)
    fm.pop("img", None)
    fm["math"] = bool(fm.pop("mathjax", False) or 'type="math/tex' in body)
    encrypted = 'id="hexo-blog-encrypt"' in body
    recover = not found or updated > "2022-08-20" or encrypted or title == '星轨拍摄与延时视频'
    if recover:
        # Keep the rendered body instead of a lossy HTML-to-Markdown conversion.
        # Raw blocks protect literal braces and code from Hexo's template parser.
        original = "\n{% raw %}\n" + body.strip() + "\n{% endraw %}\n"
        fm["legacy_html"] = True
        if encrypted:
            fm.pop("password", None)
            fm["legacy_encrypted"] = True
            fm["index_excerpt"] = "这篇文章需要密码才能阅读。"
        else:
            summary = content.text_content().strip()
            fm["index_excerpt"] = re.sub(r"\s+", " ", summary)[:150]
    p.write_text("---\n" + yaml.safe_dump(fm, allow_unicode=True, sort_keys=False).strip() + "\n---\n" + original.lstrip("\n"))
    headings = content.xpath('.//*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6]/@id')
    entry = dict(title=title, url="/" + url, source=str(p.relative_to(ROOT)), recovered_html=recover, encrypted=encrypted, headings=headings)
    entry['heading_details'] = [dict(id=n.get('id'), text=n.text_content()) for n in content.xpath('.//*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6]') if n.get('id')]
    if encrypted:
        entry['cipher_sha256'] = hashlib.sha256(doc.get_element_by_id('hbeData').text.encode()).hexdigest()
    entries.append(entry)

# The active theme must not be the only owner of published assets.
for path in paths:
    if path.startswith("medias/") or path in ["favicon.jpg", "favicon.png", "lib/hbe.js", "css/hbe.style.css"]:
        out = ROOT / "source" / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(blob(path))

(ROOT / "migration/published-manifest.json").write_text(json.dumps(dict(baseline=BASELINE, articles=entries), ensure_ascii=False, indent=2) + "\n")
print("Preserved", len(entries), "published routes; restored", sum(e["recovered_html"] for e in entries), "HTML bodies.")
