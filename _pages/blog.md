---
permalink: /blog/
title: "Blog"
excerpt: "Research notes, project writeups, and blogpost on interesting findings."
author_profile: true
redirect_from:
  - /posts/
---

<style>
  .page__title { display: none; }

  .blog-wrap { max-width: 760px; }

  .blog-hero { padding: 4px 0 18px; }
  .blog-hero h1 {
    margin: 0 0 6px;
    font-size: clamp(1.7rem, 1.6vw + 1rem, 2.2rem);
    font-weight: 700;
    letter-spacing: -0.01em;
    color: #1f2328;
  }
  .blog-hero p {
    color: #6b7280;
    font-size: 1rem;
    margin: 0;
  }

  .post-list {
    list-style: none;
    padding: 0;
    margin: 18px 0 0;
  }
  .post-item {
    padding: 18px 0;
    border-bottom: 1px solid #eef0f3;
  }
  .post-item:last-child { border-bottom: 0; }
  .post-item .post-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    color: #6b7280;
    font-size: 0.85rem;
    margin-bottom: 6px;
    font-variant-numeric: tabular-nums;
  }
  .post-item .post-meta .read-time::before,
  .post-item .post-meta .kind::before {
    content: "·";
    margin-right: 10px;
    color: #c4c8ce;
  }
  .post-item .post-meta .kind {
    color: #1d4ed8;
    background: #eff5ff;
    border: 1px solid #dbe7ff;
    border-radius: 999px;
    padding: 0 9px;
    font-size: 0.75rem;
    letter-spacing: 0.01em;
  }
  .post-item .post-meta .kind::before {
    content: "";
    margin: 0;
  }
  .post-item h2 {
    margin: 0 0 6px;
    font-size: 1.25rem;
    font-weight: 700;
    line-height: 1.35;
    letter-spacing: -0.005em;
  }
  .post-item h2 a { color: #1f2328; }
  .post-item h2 a:hover { color: #1d4ed8; }
  .post-item .post-excerpt {
    color: #4b5563;
    font-size: 0.97rem;
    line-height: 1.6;
    margin: 0 0 8px;
  }
  .post-item .tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 6px;
  }
  .post-item .tag {
    background: #f1f4f8;
    color: #4b5563;
    border-radius: 999px;
    padding: 2px 9px;
    font-size: 0.78rem;
  }
  .empty-state {
    color: #6b7280;
    font-size: 0.95rem;
    border: 1px dashed #d4d8de;
    border-radius: 10px;
    padding: 26px;
    text-align: center;
    margin-top: 12px;
  }
</style>

<div class="blog-wrap">

<div class="blog-hero">
  <h1>Blog</h1>
  <p>Research notes, project writeups, and blogpost on interesting findings.</p>
</div>

{% assign visible_posts = site.posts | where_exp: "post", "post.hidden != true" %}

<ul class="post-list">

  <!-- Project page: DSR -->
  <li class="post-item">
    <div class="post-meta">
      <time datetime="2026-05-01">May 2026</time>
      <span class="kind">Project page</span>
    </div>
    <h2><a href="/dit-register/">Taming Outlier Tokens in Diffusion Transformers</a></h2>
    <p class="post-excerpt">
      Outlier patch tokens hurt both ViT encoders and diffusion transformers in RAE-DiT pipelines. Our
      Dual-Stage Registers (DSR) patch both sides and improve ImageNet-256 FID from 5.89 → 4.58 at 80 epochs.
    </p>
    <div class="tags">
      <span class="tag">diffusion</span>
      <span class="tag">DiT</span>
      <span class="tag">registers</span>
    </div>
  </li>

  <!-- Project page: Uni-Instruct -->
  <li class="post-item">
    <div class="post-meta">
      <time datetime="2025-09-01">Sep 2025</time>
      <span class="kind">Project page</span>
    </div>
    <h2><a href="/uni-instruct/">Uni-Instruct: One-step Diffusion through Unified Divergence Instruction</a></h2>
    <p class="post-excerpt">
      A single <em>f</em>-divergence framework that subsumes 10+ one-step diffusion distillation methods —
      and a new SoTA one-step FID of 1.02 on ImageNet 64×64. NeurIPS 2025.
    </p>
    <div class="tags">
      <span class="tag">diffusion</span>
      <span class="tag">distillation</span>
      <span class="tag">NeurIPS 2025</span>
    </div>
  </li>

  {% for post in visible_posts %}
  <li class="post-item">
    <div class="post-meta">
      <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%b %-d, %Y" }}</time>
      {% if post.read_time != false %}<span class="read-time">{% include read-time.html %}</span>{% endif %}
      <span class="kind">Blogpost</span>
    </div>
    <h2><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
    {% if post.excerpt %}
      <p class="post-excerpt">{{ post.excerpt | strip_html | truncate: 220 }}</p>
    {% endif %}
    {% if post.tags and post.tags.size > 0 %}
      <div class="tags">
        {% for tag in post.tags %}<span class="tag">{{ tag }}</span>{% endfor %}
      </div>
    {% endif %}
  </li>
  {% endfor %}

</ul>

</div>
