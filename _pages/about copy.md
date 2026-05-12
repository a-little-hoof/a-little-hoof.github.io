---
permalink: /
title: "Yifei Wang"
excerpt: "Yifei Wang — second-year PhD student at Rice University, working on generative models."
author_profile: true
redirect_from: 
  - /About/
  - /About.html
---

<style>
  /* Hide the default academicpages page title — we use a custom hero instead */
  .page__title { display: none; }

  .home-wrap { max-width: 760px; }
  .home-wrap p { color: #2f3338; line-height: 1.7; }

  /* HERO */
  .home-hero { padding: 4px 0 8px; }
  .home-hero h1 {
    margin: 0 0 6px;
    font-size: clamp(1.9rem, 1.8vw + 1.1rem, 2.4rem);
    font-weight: 700;
    letter-spacing: -0.01em;
    color: #1f2328;
  }
  .home-hero .role {
    margin: 0 0 14px;
    color: #6b7280;
    font-size: 1rem;
  }
  .home-hero .lead {
    font-size: 1.05rem;
    color: #2f3338;
    margin: 0 0 4px;
  }

  /* Section headings */
  .home-section {
    margin-top: 38px;
  }
  .home-section > h2 {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: -0.005em;
    color: #1f2328;
    margin: 0 0 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #e5e7eb;
  }

  /* News */
  .home-news ul {
    list-style: none;
    padding: 0;
    margin: 0;
  }
  .home-news li {
    display: grid;
    grid-template-columns: 110px 1fr;
    gap: 14px;
    padding: 8px 0;
    border-bottom: 1px dashed #eef0f3;
    align-items: baseline;
    font-size: 0.97rem;
  }
  .home-news li:last-child { border-bottom: 0; }
  .home-news time {
    color: #6b7280;
    font-variant-numeric: tabular-nums;
    font-size: 0.9rem;
    letter-spacing: 0.01em;
  }
  .home-news .badge {
    display: inline-block;
    background: #eff5ff;
    color: #1d4ed8;
    border: 1px solid #dbe7ff;
    border-radius: 999px;
    padding: 1px 8px;
    font-size: 0.72rem;
    margin-right: 6px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    vertical-align: 1px;
  }

  /* Pub cards */
  .pub-cards {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin: 14px 0;
  }
  .pub-card {
    display: flex;
    flex-direction: column;
    overflow: hidden;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    background: #fafbfc;
    transition: border-color .15s ease, transform .15s ease, box-shadow .15s ease;
  }
  .pub-card:hover {
    border-color: #cbd5e1;
    box-shadow: 0 2px 12px rgba(15, 23, 42, .05);
  }
  .pub-card .thumb {
    display: block;
    width: 100%;
    aspect-ratio: 2 / 1;
    background: #fff;
    border-bottom: 1px solid #e5e7eb;
    overflow: hidden;
  }
  .pub-card .thumb img {
    width: 100%; height: 100%; object-fit: cover; display: block;
  }
  .pub-card .body {
    padding: 14px 16px 16px;
  }
  .pub-card h3 {
    margin: 0 0 6px;
    font-size: 1.02rem;
    font-weight: 600;
    line-height: 1.35;
  }
  .pub-card h3 a { color: #1f2328; }
  .pub-card h3 a:hover { color: #1d4ed8; }
  .pub-card .venue {
    font-size: 0.82rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0 0 8px;
  }
  .pub-card .desc {
    color: #4b5563;
    font-size: 0.93rem;
    margin: 0 0 10px;
    line-height: 1.55;
  }
  .pub-card .authors {
    font-size: 0.85rem; color: #6b7280; margin: 0 0 8px;
  }
  .pub-card .pub-links a {
    font-size: 0.85rem;
    margin-right: 12px;
    color: #1d4ed8;
  }
  .see-all {
    display: inline-block;
    margin-top: 6px;
    font-size: 0.92rem;
    color: #1d4ed8;
  }
  @media (max-width: 600px) {
    .pub-cards { grid-template-columns: 1fr; }
    .home-news li { grid-template-columns: 90px 1fr; gap: 10px; }
  }

  /* About paragraphs */
  .home-about p { margin: 0 0 12px; font-size: 0.98rem; }
</style>

<div class="home-wrap">

<div class="home-hero">
  <h1>Hi, I'm Yifei Wang</h1>
  <p class="role">Second-year PhD student · Rice University</p>
  <p class="lead">
    I work on generative models — diffusion, flow matching, and the building blocks that make them more
    efficient and more controllable.
  </p>
</div>

<div class="home-section home-news">
  <h2>News</h2>
  <ul>
    <li>
      <time>May 2026</time>
      <span><span class="badge">new</span> Joining <a href="https://www.cs.jhu.edu/~ayuille/">Alan Yuille</a>'s lab at JHU as a visiting student for the summer.</span>
    </li>
    <li>
      <time>May 2026</time>
      <span><span class="badge">new</span> Released <a href="/dit-register/">DSR</a> (with Apple).</span>
    </li>
    <li>
      <time>Sep 2025</time>
      <span><a href="/uni-instruct/">Uni-Instruct</a> (with Xiaohongshu Inc.) accepted to <b>NeurIPS 2025</b>.</span>
    </li>
    <li>
      <time>Aug 2025</time>
      <span>Started PhD at Rice University, working with <a href="https://weichen582.github.io/">Chen Wei</a>.</span>
    </li>
    <li>
      <time>Sep 2024</time>
      <span><a href="https://arxiv.org/abs/2407.01014">EM-Diffusion</a> accepted to <b>NeurIPS 2024</b>.</span>
    </li>
  </ul>
</div>

<div class="home-section">
  <h2>Selected work</h2>

  <div class="pub-cards">

    <div class="pub-card">
      <a class="thumb" href="/dit-register/">
        <img src="/dit-register/static/images/pca_map_2.jpg" alt="DSR PCA map visualization" />
      </a>
      <div class="body">
        <h3><a href="/dit-register/">DSR</a></h3>
        <p class="venue">Preprint · 2026</p>
        <p class="authors">Xiaoyu Wu*, <b>Yifei Wang*</b>, Tsu-Jui Fu, Liang-Chieh Chen, Zhe Gan, Chen Wei</p>
        <p class="desc">
          Outlier patch tokens hurt both ViT encoders and diffusion transformers in RAE-DiT pipelines.
          Our <em>Dual-Stage Registers</em> patch both sides and improve ImageNet-256 FID from
          5.89&nbsp;→&nbsp;4.58 at 80 epochs.
        </p>
        <div class="pub-links">
          <a href="https://arxiv.org/abs/2605.05206">arXiv</a>
          <a href="/dit-register/">project page</a>
        </div>
      </div>
    </div>

    <div class="pub-card">
      <a class="thumb" href="/uni-instruct/">
        <img src="/uni-instruct/static/images/uni_instruct_overview.png" alt="Uni-Instruct overview" />
      </a>
      <div class="body">
        <h3><a href="/uni-instruct/">Uni-Instruct: One-step Diffusion through Unified Divergence Instruction</a></h3>
        <p class="venue">NeurIPS · 2025</p>
        <p class="authors"><b>Yifei Wang</b>, Weimin Bai, Colin Zhang, Debing Zhang, Weijian Luo, He Sun</p>
        <p class="desc">
          A single <em>f</em>-divergence framework that subsumes 10+ one-step diffusion distillation methods
          (Diff-Instruct, DMD, SiD, SIM, …) — and a new SoTA one-step FID of <b>1.02</b> on ImageNet&nbsp;64×64,
          beating the 35-NFE EDM teacher.
        </p>
        <div class="pub-links">
          <a href="https://arxiv.org/abs/2505.20755">arXiv</a>
          <a href="/uni-instruct/">project page</a>
          <a href="https://github.com/a-little-hoof/Uni_Instruct">code</a>
        </div>
      </div>
    </div>

  </div>

  <a class="see-all" href="/Research/">See all publications →</a>
</div>

<div class="home-section home-about">
  <h2>About</h2>
  <p>
    I'm a second-year PhD student at Rice University, where I am working with
    <a href="https://weichen582.github.io/">Chen Wei</a>. In May 2026 I'll be visiting
    <a href="https://www.cs.jhu.edu/~ayuille/">Alan Yuille</a>'s lab at Johns Hopkins University.
    Before Rice I received my B.S. from Peking University, advised by
    <a href="https://ai4imaging.github.io/">Prof. He Sun</a>. My research focuses on generative modeling —
    primarily diffusion models — with an emphasis on the theory and the practical bottlenecks that govern
    their training.
  </p>
  <p>
    Outside the lab, I run and hike a lot. I've finished several marathons, and once spent a summer doing
    ecological field research in Saihanba and Xihaigu. I also write Chinese-language blogs on
    <a href="https://www.zhihu.com/people/cameron-78-28">Zhihu</a>.
  </p>
</div>

</div>
