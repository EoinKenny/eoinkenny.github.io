"""Build the blog listing and RSS feed using only Python's standard library."""
from datetime import datetime, timezone
from email.utils import format_datetime
from html import escape
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://eoinkenny.github.io"


def main():
    posts = json.loads((ROOT / "blog/posts.json").read_text())
    posts.sort(key=lambda post: post["date"], reverse=True)
    cards = []
    seen = set()
    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")
    for tag, value in [("title", "Eoin Kenny — Blog"), ("link", BASE + "/blog/"),
                       ("description", "Research notes on AI and interpretability."), ("language", "en")]:
        ET.SubElement(channel, tag).text = value
    for post in posts:
        slug = post["slug"]
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug) or slug in seen:
            raise ValueError(f"Invalid or duplicate slug: {slug}")
        seen.add(slug)
        directory = ROOT / "blog" / slug
        if not (directory / "index.html").is_file():
            raise FileNotFoundError(directory / "index.html")
        image = directory / post["image"]
        if not image.is_file() or not image.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"Missing or invalid image: {image}")
        date = datetime.strptime(post["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        label = f"{date.day} {date:%B %Y}"
        url = f"/blog/{slug}/"
        tags = "".join(f"<span>{escape(tag)}</span>" for tag in post["tags"])
        cards.append(f'''<li class="post"><article><a class="post-link" href="{url}">
<img src="{url}{escape(post['image'], quote=True)}" alt="{escape(post['image_alt'], quote=True)}" width="2418" height="820">
<div><p class="post-meta"><time datetime="{post['date']}">{label}</time> · {int(post['minutes'])} min read</p>
<h2>{escape(post['title'])}</h2><p class="post-description">{escape(post['description'])}</p>
<div class="tags">{tags}</div><span class="read-post">Read the post →</span></div>
</a></article></li>''')
        item = ET.SubElement(channel, "item")
        for tag, value in [("title", post["title"]), ("link", BASE + url), ("guid", BASE + url),
                           ("description", post["description"]), ("pubDate", format_datetime(date))]:
            ET.SubElement(item, tag).text = value
    html = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blog — Eoin Kenny</title><meta name="description" content="Research notes on AI, interpretability, and what helps models make better decisions, by Eoin Kenny.">
<link rel="canonical" href="https://eoinkenny.github.io/blog/">
<link rel="alternate" type="application/rss+xml" title="Eoin Kenny — Blog" href="/blog/feed.xml">
<link rel="stylesheet" href="/css/navigation.css"><link rel="stylesheet" href="style.css"></head><body>
<nav class="site-nav" aria-label="Main navigation"><a class="site-name" href="/">Eoin M. Kenny</a><div class="site-links"><a href="/">Home</a><a href="/background/">Background</a><a href="/blog/" aria-current="page">Blog post</a></div></nav>
<main><p class="eyebrow">Research notes</p><h1>Blog</h1>
<p class="intro">Experiments, ideas, and things I learned about AI along the way.</p>
<ol class="posts">''' + "\n".join(cards) + '''</ol></main>
<footer><span>Eoin Kenny</span><a href="/blog/feed.xml">RSS feed</a></footer></body></html>
'''
    (ROOT / "blog/index.html").write_text(html)
    ET.indent(rss)
    ET.ElementTree(rss).write(ROOT / "blog/feed.xml", encoding="utf-8", xml_declaration=True)
    print(f"Built blog index and RSS feed for {len(posts)} post(s).")


if __name__ == "__main__":
    main()
