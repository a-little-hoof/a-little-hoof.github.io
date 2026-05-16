---
title: "Hello, world"
date: 2026-05-11
permalink: /blog/2026/05/hello-world/
excerpt: "A short note marking the start of this blog — what to expect, and the template I'll use for posts going forward."
tags:
  - meta
  - notes
hidden: false
---

<style>
  /* Hide the academicpages page__title — we render our own header below */
  .page__title { display: none; }

  .post-wrap { max-width: 720px; }
  .post-wrap p,
  .post-wrap li {
    color: #2f3338;
    line-height: 1.75;
    font-size: 1.02rem;
  }
  .post-wrap h2 {
    font-size: 1.35rem;
    font-weight: 700;
    margin: 36px 0 10px;
    letter-spacing: -0.005em;
  }
  .post-wrap h3 {
    font-size: 1.1rem;
    font-weight: 600;
    margin: 24px 0 8px;
  }

  .post-header { margin-bottom: 22px; }
  .post-header .meta {
    color: #6b7280;
    font-size: 0.88rem;
    letter-spacing: 0.02em;
    margin-bottom: 8px;
  }
  .post-header h1 {
    margin: 0 0 10px;
    font-size: clamp(1.7rem, 1.8vw + 1rem, 2.3rem);
    font-weight: 700;
    line-height: 1.2;
    letter-spacing: -0.01em;
    color: #1f2328;
  }
  .post-header .post-tags {
    display: flex; flex-wrap: wrap; gap: 6px;
    margin-top: 6px;
  }
  .post-header .post-tags .tag {
    background: #f1f4f8;
    color: #4b5563;
    border-radius: 999px;
    padding: 2px 9px;
    font-size: 0.78rem;
  }

  /* Lead paragraph — a slightly bigger intro */
  .post-wrap .lead {
    font-size: 1.12rem;
    color: #2f3338;
    line-height: 1.65;
  }

  /* Block quote */
  .post-wrap blockquote {
    border-left: 3px solid #d4d8de;
    padding: 4px 16px;
    color: #4b5563;
    font-style: italic;
    margin: 18px 0;
    background: #fafbfc;
    border-radius: 0 6px 6px 0;
  }

  /* Code */
  .post-wrap code {
    background: #f1f4f8;
    color: #1f2328;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 0.92em;
  }
  .post-wrap pre {
    background: #0f172a;
    color: #e2e8f0;
    border-radius: 8px;
    padding: 14px 16px;
    overflow-x: auto;
    font-size: 0.88rem;
    line-height: 1.55;
    margin: 14px 0;
  }
  .post-wrap pre code {
    background: transparent;
    color: inherit;
    padding: 0;
  }

  /* Figures */
  .post-figure {
    margin: 22px 0;
    text-align: center;
  }
  .post-figure img {
    max-width: 100%;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
  }
  .post-figure figcaption {
    color: #6b7280;
    font-size: 0.88rem;
    margin-top: 8px;
    text-align: left;
  }

  /* "Back to blog" link */
  .post-back {
    display: inline-block;
    margin-top: 36px;
    padding-top: 18px;
    border-top: 1px solid #eef0f3;
    color: #1d4ed8;
    font-size: 0.92rem;
  }
</style>

<article class="post-wrap">

<header class="post-header">
  <div class="meta">
    <time datetime="2026-05-11">May 11, 2026</time>
  </div>
  <h1>Hello, world</h1>
  <div class="post-tags">
    <span class="tag">meta</span>
    <span class="tag">notes</span>
  </div>
</header>

<p class="lead">
  This is a placeholder post — the first one in this blog. It exists to lock in the layout and to act as a copy-paste
  template for future writing. Replace this paragraph with the actual lead-in for the post.
</p>

<h2>What I'm planning to write here</h2>

<p>
  Short notes that are too long for a tweet but too small for a paper. Two flavours mostly:
</p>

<ul>
  <li><b>Paper digests</b> — what I learned from a recent paper, in a few hundred words.</li>
  <li><b>Research notes</b> — derivations, plots, or experiments that I want to remember in a few months.</li>
  <li>Occasionally a non-research post — running, hiking, books.</li>
</ul>

<h2>A quick tour of what this template supports</h2>

<h3>Block quotes</h3>
<blockquote>
  "Everything should be made as simple as possible, but no simpler." — usually attributed to Einstein.
</blockquote>

<h3>Inline and block code</h3>
<p>
  Inline code looks like <code>register_tokens</code>. Fenced code blocks render with a dark theme:
</p>

<pre><code class="language-python">def hello(world: str) -> str:
    return f"hello, {world}"

print(hello("world"))
</code></pre>

<h3>Figures with captions</h3>
<p>
  Drop an image into <code>images/</code> and reference it as a centred figure:
</p>
<figure class="post-figure">
  <img src="/images/500x300.png" alt="placeholder figure" />
  <figcaption><b>Fig. 1.</b> A placeholder figure — replace with something meaningful.</figcaption>
</figure>

<h3>Math</h3>
<p>
  Inline math (when MathJax is set up): <em>L = ½‖x − μ‖²</em>. Display math:
</p>
<p style="text-align:center; font-style: italic;">
  L(θ) = 𝔼<sub>x∼p</sub> [ ‖f<sub>θ</sub>(x) − x‖² ]
</p>

<h2>How to add a new post</h2>

<ol>
  <li>Copy this file to <code>_posts/YYYY-MM-DD-slug.md</code>.</li>
  <li>Update the front matter (<code>title</code>, <code>date</code>, <code>tags</code>, <code>excerpt</code>,
    <code>permalink</code>).</li>
  <li>Rewrite the content below the front matter.</li>
  <li>If you want to hide a post from the blog index, set <code>hidden: true</code> in the front matter.</li>
</ol>

<a class="post-back" href="/blog/">← Back to blog</a>

</article>
