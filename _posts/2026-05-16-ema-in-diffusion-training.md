---
title: "Ignoring EMA may lead to unfair comparison"
date: 2026-05-16
permalink: /blog/2026/05/ema-in-diffusion-training/
excerpt: "We benchmark EMA decay across pixel-, latent-, and representation-space diffusion models and show that the popular 0.9999 default silently trades recall for precision, and can quietly re-order leaderboards in convergence-speed comparisons."
tags:
  - diffusion
  - training-dynamics
  - EMA
hidden: false
---

<!-- Open Sans for body text — matches the samacquaviva.com flow-evals aesthetic -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap" rel="stylesheet">

<style>
  /* Hide the default academicpages page title — we render our own header below */
  .page__title { display: none; }

  /* Color tokens (mirrors flow-evals) */
  :root {
    --fg: #1a1a1a;
    --fg-muted: #666;
    --accent: #52adc8;
    --rule: #ddd;
    --rule-soft: #e6e5e0;
    --code-bg: #f5f4f0;
    --highlight: #fdf6e3;
  }

  /* Core layout */
  .post-wrap {
    max-width: 760px;
    font-family: "Open Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 16.5px;
    line-height: 1.7;
    color: var(--fg);
  }
  .post-wrap p,
  .post-wrap li {
    color: var(--fg);
    line-height: 1.7;
    font-size: 16.5px;
  }
  .post-wrap a { color: var(--accent); text-decoration: none; }
  .post-wrap a:hover { text-decoration: underline; }
  .post-wrap strong, .post-wrap b { font-weight: 600; color: var(--fg); }

  /* Headings — H2 has a thin top rule, acting as section divider.
     Explicitly reset bottom/background/text-decoration so the academicpages
     theme defaults (which add their own border-bottom) don't bleed through. */
  .post-wrap h2 {
    font-family: inherit;
    font-size: 1.55rem;
    font-weight: 600;
    margin: 3.5rem 0 1rem;
    padding: 1.5rem 0 0;
    border-top: 1px solid var(--rule);
    border-bottom: none;
    background: transparent;
    text-decoration: none;
    box-shadow: none;
    color: var(--fg);
    letter-spacing: -0.005em;
    scroll-margin-top: 1.25rem;
  }
  .post-wrap h3 {
    font-family: inherit;
    font-size: 1.15rem;
    font-weight: 600;
    margin: 2rem 0 0.7rem;
    padding: 0;
    border: none;
    background: transparent;
    text-decoration: none;
    color: var(--fg);
    scroll-margin-top: 1.25rem;
  }

  /* Post header (h1 + tags) */
  .post-header { margin-bottom: 1.5rem; }
  .post-header .meta {
    color: var(--fg-muted);
    font-size: 0.88rem;
    margin-bottom: 0.5rem;
  }
  .post-header h1 {
    font-family: inherit;
    margin: 0 0 0.75rem;
    font-size: clamp(1.5rem, 1.2vw + 0.9rem, 2rem);
    font-weight: 700;
    line-height: 1.2;
    letter-spacing: -0.015em;
    color: var(--fg);
  }
  .post-header .post-authors {
    color: var(--fg);
    font-size: 1rem;
    line-height: 1.5;
    margin: 0.25rem 0 0.1rem;
  }
  .post-header .equal-contrib {
    color: var(--fg-muted);
    font-size: 0.82rem;
    margin-bottom: 0.5rem;
  }
  .post-header .post-tags {
    display: flex; flex-wrap: wrap; gap: 6px;
    margin-top: 0.5rem;
  }
  .post-header .post-tags .tag {
    background: var(--code-bg);
    color: #4b5563;
    border-radius: 999px;
    padding: 2px 9px;
    font-size: 0.78rem;
  }

  .post-wrap .lead {
    font-size: 1rem;
    color: #555;
    line-height: 1.65;
    text-wrap: balance;
  }
  /* Balance lines on short, important blocks: H1 title, H2 section headers, subtitle */
  .post-header h1,
  .post-wrap h2 {
    text-wrap: balance;
  }
  /* Long-form blocks (paragraphs, figure captions): use text-wrap: pretty, which
     fixes single-word orphans on the last line without the line-count limit of balance */
  .post-wrap p,
  .post-figure figcaption,
  .pair-figure + figcaption,
  .footnotes li {
    text-wrap: pretty;
  }

  /* Blockquote — subtle tinted background, no border-radius corner-cut */
  .post-wrap blockquote {
    margin: 1.5rem 0;
    padding: 1rem 1.25rem;
    background: var(--code-bg);
    border-left: 4px solid #9a9a92;
    border-radius: 6px;
    color: #333;
    font-size: 0.95rem;
  }

  /* Inline code — tinted box, mono */
  .post-wrap code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, "Courier New", monospace;
    background: var(--code-bg);
    color: var(--fg);
    padding: 0.15em 0.4em;
    border-radius: 4px;
    font-size: 0.88em;
  }

  /* Figures */
  .post-figure {
    margin: 1.5rem 0;
    text-align: center;
  }
  .post-figure img { max-width: 100%; }
  .post-figure figcaption {
    color: var(--fg-muted);
    font-size: 0.88rem;
    line-height: 1.55;
    margin-top: 8px;
    text-align: left;
  }
  .post-figure figcaption b,
  .post-figure figcaption strong { color: var(--fg); }

  /* Paired figures (toy example side-by-side) */
  .pair-figure {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin: 1.5rem 0 0.5rem;
  }
  .pair-figure .col { text-align: center; }
  .pair-figure img { max-width: 100%; }
  .pair-figure .label {
    font-size: 0.82rem; font-weight: 600; color: #4b5563;
    text-transform: uppercase; letter-spacing: 0.04em;
    margin-bottom: 4px;
  }
  .pair-figure + figcaption {
    color: var(--fg-muted);
    font-size: 0.88rem;
    margin-top: 0;
    text-align: left;
  }
  @media (max-width: 580px) {
    .pair-figure { grid-template-columns: 1fr; }
  }

  /* Tables — minimal: bold header underlined, faint inter-row rules, no row-color blocks */
  table.ema-sweep {
    width: 100%;
    border-collapse: collapse;
    margin: 1.2rem 0;
    font-size: 0.92rem;
  }
  table.ema-sweep th, table.ema-sweep td {
    padding: 8px 10px;
    border-bottom: 1px solid var(--rule-soft);
    text-align: center;
  }
  table.ema-sweep th {
    background: transparent;
    border-bottom: 2px solid var(--fg);
    font-weight: 600;
    color: var(--fg);
    font-size: 0.86rem;
  }
  table.ema-sweep td.left, table.ema-sweep th.left { text-align: left; }
  /* Row classes — keep the markup, drop the colored blocks; signal via subtler cues */
  table.ema-sweep tr.row-best td { font-weight: 600; }
  table.ema-sweep tr.row-default td { color: var(--accent); }
  table.ema-sweep tr.row-collapse td { color: #999; font-style: italic; }
  .legend-note { font-size: 0.82rem; color: var(--fg-muted); margin-top: -4px; }

  /* TL;DR / callout — left-border treatment, no coloured box */
  .callout {
    max-width: 720px;
    margin: 1.4rem 0 1.5rem;
    padding: 0.15rem 0 0.15rem 1.1rem;
    background: transparent;
    border: none;
    border-left: 4px solid #9a9a92;
    border-radius: 0;
    color: var(--fg);
    font-size: 1rem;
  }
  .callout b { color: var(--fg); font-weight: 600; }

  /* Back-to-blog link */
  .post-back {
    display: inline-block;
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--rule);
    color: var(--accent);
    font-size: 0.92rem;
  }

  /* Footnote citations — dotted underline on the cite, no colored superscript box */
  sup.footnote-ref a {
    font-size: 0.72em;
    vertical-align: super;
    line-height: 0;
    margin-left: 1px;
    text-decoration: none;
    color: var(--accent);
  }
  sup.footnote-ref a:hover { text-decoration: underline; }
  .footnotes {
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--rule);
    font-size: 0.92rem;
    color: var(--fg-muted);
  }
  .footnotes h3 {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--fg-muted);
    margin: 0 0 0.5rem;
    border-top: none;
    padding-top: 0;
  }
  .footnotes ol { padding-left: 22px; margin: 0; }
  .footnotes li { margin: 0.5rem 0; line-height: 1.6; }
  .footnotes li[id]:target,
  sup[id]:target a { background: var(--highlight); border-radius: 4px; padding: 1px 4px; }
  .footnote-back {
    text-decoration: none;
    margin-left: 6px;
    color: #999;
    font-size: 0.95em;
  }
  .footnote-back:hover { color: var(--accent); }

  /* Right-side fixed table of contents — visible only on wide screens */
  #toc {
    display: none;  /* hidden by default; the media query below shows it on ≥1180px */
    position: fixed;
    top: 140px;
    /* Anchor just to the right of the article column (760px wide, centered),
       with a small gap. Falls back to a viewport-edge offset on narrow screens. */
    left: max(24px, calc(50% + 380px + 24px));
    width: 200px;
    max-height: calc(100vh - 180px);
    overflow-y: auto;
    flex-direction: column;
    gap: 2px;
    z-index: 5;
    font-family: "Open Sans", -apple-system, BlinkMacSystemFont, sans-serif;
    padding: 0 4px;
  }
  @media (min-width: 1180px) {
    #toc { display: flex; }
  }
  #toc .toc-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--fg-muted);
    padding: 0 10px 8px;
    margin-bottom: 4px;
    border-bottom: 1px solid var(--rule-soft);
  }
  #toc a {
    display: block;
    padding: 3px 10px;
    border-left: 2px solid transparent;
    color: var(--fg-muted);
    font-size: 0.78rem;
    line-height: 1.45;
    text-decoration: none;
    transition: color 0.15s ease, border-color 0.15s ease;
  }
  #toc a:hover {
    color: var(--fg);
    text-decoration: none;
  }
  #toc a.toc-active {
    color: var(--fg);
    border-left-color: var(--fg);
    font-weight: 600;
  }
</style>

<!-- MathJax: enables LaTeX math via $$...$$ (display) and \(...\) (inline). Loaded only on this post. -->
<script>
window.MathJax = {
  tex: {
    inlineMath: [['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    packages: { '[+]': ['noerrors'] }
  },
  loader: { load: ['[tex]/noerrors'] }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" id="MathJax-script" async></script>

<nav id="toc" aria-label="Table of contents">
  <div class="toc-title">Contents</div>
</nav>

<article class="post-wrap">

<header class="post-header">
  <h1>Ignoring EMA may lead to unfair comparison</h1>
  <div class="post-authors">
    <a href="https://a-little-hoof.github.io/">Yifei Wang</a><sup>*</sup>, <a href="https://nicholas0228.github.io/">Xiaoyu Wu</a><sup>*</sup>, <a href="https://weichen582.github.io/">Chen Wei</a>
  </div>
  <div class="equal-contrib"><sup>*</sup>Equal contribution</div>
  <div class="post-tags">
    <span class="tag">diffusion</span>
    <span class="tag">training-dynamics</span>
    <span class="tag">EMA</span>
  </div>
</header>

<p class="lead">
  Diffusion transformers are now competing on convergence speed: who can hit a target FID in
  the fewest training epochs. Yet you can climb that leaderboard almost for free, by quietly
  tuning a hyperparameter almost nobody talks about — the EMA decay.
</p>

<figure class="post-figure">
  <img src="/images/blog/ema/ema_sweep_curves.png" alt="gFID vs EMA decay at 80 epochs for RAE-DiT (DINOv2-B), with horizontal references for raw, LightningDiT, and the published RAE number" />
  <figcaption>
    <b>Same model architecture, same data, same 80-epoch training run — only the EMA decay
    changes.</b> RAE-DiT-XL<sup class="footnote-ref" id="fnref:rae"><a href="#fn:rae">1</a></sup> (DINOv2-B) on ImageNet 256.
    At the community-default \(\beta = 0.9999\), our sweep gives FID = 4.14 (the RAE paper
    reports 4.28 for the same configuration) — essentially tied with LightningDiT's
    4.29 at the same \(\beta\).
    The interesting case is the same paper's DiT-DH variant, which they report at
    FID = 2.16 using a <em>non-default</em> \(\beta = 0.9995\). Our DiT-XL sweep shows
    that switching from \(\beta = 0.9999\) to \(\beta = 0.9995\) alone is worth about a full
    FID point at this training stage (4.14 → 3.29 in our sweep).
  </figcaption>
</figure>

<p>
  This post isn't a recipe for exploiting that loophole — it's an attempt to make the phenomenon 
  visible and to bring EMA decay into the community's attention as a first-class hyperparameter. 
  Almost every modern diffusion-model paper uses an exponential moving average (EMA) of the weights and reports
  results from the EMA checkpoint, yet the decay value is usually inherited from a parent
  codebase rather than tuned or even reported. This post shows that the choice matters more than
  the field acknowledges.
</p>

<div class="callout">
  <b>TL;DR.</b> EMA decay matters far more than the community treats it. On the same 80-epoch RAE-DiT run, sweeping decay alone moves gFID from <b>3.21</b> to <b>4.14</b> — a swing larger than the gap between most "SOTA" diffusion-model claims. A deeper investigation finds that EMA scale corresponds to a precision–recall trade-off.
</div>

<h2>Background: what is EMA?</h2>

<p>
Exponential moving average (EMA) keeps a slowly-updated copy of the model weights alongside the
training weights. At every optimizer step \(t\), the EMA weights \(\theta'\) are updated as
</p>

<p>
$$\theta'_t \;=\; \beta \, \theta'_{t-1} \;+\; (1 - \beta)\, \theta_t$$
</p>

<p>
where \(\beta\) is the EMA <em>decay</em> (the community default is \(\beta = 0.9999\)). At evaluation
time, samples are drawn from \(\theta'\) instead of \(\theta\) — the running average rather than the latest
optimizer step.
</p>

<p>
EMA was introduced to diffusion models in NCSNv2<sup class="footnote-ref" id="fnref:ncsnv2"><a href="#fn:ncsnv2">2</a></sup>,
where it was proposed as a fix for a very visible failure mode: <em>color shift</em>. Without
EMA, samples generated near the end of training would drift in colour statistics — overall
brightness, white balance, hue — even when the score loss kept decreasing. Averaging the weights
over recent optimizer steps smoothed out high-frequency fluctuations in the trajectory and
brought the sample statistics back in line.
</p>

<figure class="post-figure">
  <img src="/images/blog/ema/ncsnv2_color_shift.png"
       alt="CIFAR-10 FID over training iterations for NCSN with and without EMA, with sample insets showing color-shifted samples from the raw model and clean samples from the EMA model"
       style="max-width: 500px; display: block; margin: 0 auto;" />
  <figcaption>
    <b>NCSN with vs. without EMA on CIFAR-10.</b> The raw NCSN checkpoint (dashed blue)
    oscillates wildly in FID and produces visibly colour-shifted, off-distribution samples;
    the EMA-averaged checkpoint (orange) trains stably and recovers correct image statistics.
    Figure taken from Song &amp; Ermon, <em>Improved Techniques for Training Score-Based
    Generative Models</em> (NeurIPS 2020)<sup class="footnote-ref"><a href="#fn:ncsnv2">2</a></sup>;
    credit to the original authors.
  </figcaption>
</figure>

<p>
The fix works, and it generalizes well beyond the colour-shift problem: an EMA-smoothed
checkpoint always achieves lower FID compared to the raw checkpoint, across pixel-,
latent-, and representation-space diffusion. This empirical regularity is the reason EMA has
become a near-universal default in modern diffusion training.
</p>

<h2>Why care about EMA decay at all?</h2>

<h3>EMA decay is part of the evaluation protocol</h3>

<p>
Because almost every paper reports the EMA checkpoint, the reported number is not determined
only by the model architecture, training objective, optimizer, or training budget. It also
depends on the EMA decay used to construct the checkpoint. When EMA decay is not explicitly 
controlled, it becomes an implicit part of the evaluation protocol.
</p>

<h3>Convergence-speed comparisons are EMA-dependent</h3>

<p>
The way of comparing diffusion model performance has changed from converged model FID score to convergence speed. The previous way to argue "method X is better than method Y" in diffusion papers was to compare FID until convergence. This comparison is impractical not only because training diffusion models takes weeks to converge, but also because it provides limited signal: incremental improvements in absolute FID value mean little when every method can achieve an FID score lower than 2. What we actually care about is how much training the method needs to be useful.
</p>

<p>
A cleaner alternative according to RAEv2<sup class="footnote-ref" id="fnref:raev2"><a href="#fn:raev2">3</a></sup> is to invert the
axis. Fix a quality threshold <em>k</em>, and report <em>EP<sub>FID@k</sub></em> — the number of
training epochs required for the unguided gFID to first drop below <em>k</em>. It rewards
real reductions in compute rather than fractional FID improvements, and its interpretation does
not depend on where in training the snapshot was taken.
</p>

<figure class="post-figure">
  <div style="display: grid; grid-template-columns: 1.3fr 1fr; gap: 18px; align-items: center; margin: 8px 0;">
    <img src="/images/blog/ema/convergence.jpg"
         alt="FID-vs-epoch convergence curves for different methods, showing when each crosses the target quality threshold k"
         style="max-width: 100%; display: block;" />
    <table class="ema-sweep" style="margin: 0;">
      <thead>
        <tr>
          <th class="left">Method</th>
          <th>EP<sub>FID@7.9</sub></th>
        </tr>
      </thead>
      <tbody>
        <tr><td class="left">SiT</td><td>1400</td></tr>
        <tr><td class="left">REPA</td><td>80</td></tr>
      </tbody>
    </table>
  </div>
  <figcaption>
    <b>Convergence-style reporting with EP<sub>FID@k</sub>.</b> Left: FID-vs-epoch curves for each
    method; the horizontal line at <em>k</em> marks the target quality. Right: the number of
    epochs each method takes to first cross <em>k</em>. The metric rewards genuine reductions in
    training compute and is robust to where the comparison snapshot is taken.
    Left figure adapted from REPA<sup class="footnote-ref" id="fnref:repa"><a href="#fn:repa">4</a></sup>; credit to the original authors.
  </figcaption>
</figure>

<p>
But EP<sub>FID@k</sub> is still EMA-dependent. The FID curve a paper reports is not the curve of
the underlying optimizer; it is the curve of the <em>EMA-smoothed</em> checkpoint, and the
smoothing decay shapes that curve. Within a single training run, different decays produce
different FID-vs-step curves, hit different minima, and therefore cross any given threshold
<em>k</em> at different points. So "method X reaches gFID = <em>k</em> faster than Y" can mean
one of two very different things: (i) X's underlying training really is faster, or (ii) X
happens to sit closer to its FID-optimal EMA decay for that particular <em>k</em>.
</p>

<figure class="post-figure">
  <img src="/images/blog/ema/convergence_threshold.png"
       alt="FID vs. training epoch for one training run, with one curve per EMA decay and a horizontal threshold k = 4.0" />
  <figcaption>
    <b>FID vs. training progress for a single RAE-DiT (DINOv2) run, one curve per EMA decay.</b>
    The shape and slope of the convergence curve all depend on which EMA decay
    is applied.
  </figcaption>
</figure>

<figure class="post-figure">
  <table class="ema-sweep" style="max-width: 460px; margin: 0 auto;">
    <thead>
      <tr>
        <th class="left">EMA decay</th>
        <th>EP<sub>FID@4</sub> (epochs)</th>
      </tr>
    </thead>
    <tbody>
      <tr><td class="left">raw (no EMA)</td><td>64</td></tr>
      <tr><td class="left">\(\beta = 0.9\)</td><td>64</td></tr>
      <tr><td class="left">\(\beta = 0.99\)</td><td>48</td></tr>
      <tr class="row-best"><td class="left">\(\beta = 0.999\)</td><td>32</td></tr>
      <tr><td class="left">\(\beta = 0.9995\)</td><td>32</td></tr>
      <tr><td class="left">\(\beta = 0.9997\)</td><td>48</td></tr>
      <tr class="row-default"><td class="left">\(\beta = 0.9999\)</td><td>&gt; 80</td></tr>
    </tbody>
  </table>
  <figcaption>
    <b>EP<sub>FID@4</sub> for each EMA decay.</b> \(\beta = 0.999\) reaches FID ≤ 4 in 32
    epochs; the community-default \(\beta = 0.9999\) never crosses 4 inside the 80-epoch
    budget — at least a 2.5× difference in "convergence speed", bought purely by the EMA
    setting.
  </figcaption>
</figure>

<p>
The practical consequence: <b>EP<sub>FID@k</sub> without a matched EMA sweep is under-specified.</b>
Two methods can swap rankings just because of which decay each one defaults to. The minimum bar
for convergence-speed claims is to report EP<sub>FID@k</sub> together with the EMA decay used,
and ideally either (a) sweep EMA per method under a shared protocol, or (b) report EP both for
the EMA checkpoint and for the raw checkpoint, so the smoothing effect can be separated from
the underlying training dynamics.
</p>

<h3>EMA settings are often hard to audit</h3>

<p>
This reporting bar is hard to enforce in practice. Many papers inherit their training code from a
parent codebase, lock the EMA decay to its inherited value, and never note that the decay was
not separately tuned for the method being proposed. Some papers do not state the decay at all.
The result is that an EP<sub>FID@k</sub> claim conflates two factors that we now have evidence
behave very differently: the convergence of the underlying method, and the placement of the EMA
decay on the precision–recall tradeoff at the training stage where the threshold is crossed.
</p>

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>Parent codebase</th>
      <th>Reported EMA decay</th>
      <th>Epoch 40</th>
      <th>Epoch 80</th>
      <th>EP<sub>FID@3</sub></th>
      <th>Open sourced?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DiT</td>
      <td>-</td>
      <td>0.9999</td>
      <td>39</td>
      <td>16</td>
      <td>&gt;800</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>SiT</td>
      <td>DiT</td>
      <td>0.9999</td>
      <td>34</td>
      <td>15</td>
      <td>&gt;800</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>JiT</td>
      <td>DiT</td>
      <td>0.9999</td>
      <td>-</td>
      <td>42.93</td>
      <td>&gt;800</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>REPA</td>
      <td>DiT, SiT</td>
      <td>0.9999</td>
      <td>10.5</td>
      <td>7.9</td>
      <td>&gt;800</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>RAE</td>
      <td>LightningDiT</td>
      <td>0.9995</td>
      <td>6.7</td>
      <td>4.3</td>
      <td>200</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>RJF</td>
      <td>RAE</td>
      <td>0.9995</td>
      <td>-</td>
      <td>3.6</td>
      <td>80~85</td>
      <td>No</td>
    </tr>
    <tr>
      <td>FAE</td>
      <td>RAE</td>
      <td>-</td>
      <td>2.8</td>
      <td>2.1</td>
      <td>35~40</td>
      <td>No</td>
    </tr>
    <tr>
      <td>RAEv2</td>
      <td>RAE</td>
      <td>0.9995</td>
      <td>2.75</td>
      <td>2.25</td>
      <td>&lt;20</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>PAE</td>
      <td>RAE</td>
      <td>0.9999</td>
      <td>5.8</td>
      <td>1.9</td>
      <td>45~50</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>

<p>
This table is not meant to claim that any particular result is invalid. Rather, it highlights a reporting issue: when EMA decay is not searched or clearly reported, an intermediate FID can reflect both the convergence of the method and the choice of EMA decay.
</p>

<h2>A controlled EMA sweep: EMA causes precision-recall trade-off</h2>

<p>
  So that's the headline: EMA decay alone can re-order a leaderboard. The rest of this post
  asks the deeper question — <em>how?</em> What is EMA actually doing to the model's learned
  distribution that lets a single hyperparameter swing FID by several points? To answer that,
  we benchmark three diffusion models that span the typical input-space spectrum —
  JiT<sup class="footnote-ref" id="fnref:jit"><a href="#fn:jit">5</a></sup> (pixel space),
  SiT<sup class="footnote-ref" id="fnref:sit"><a href="#fn:sit">6</a></sup> (latent space), and
  RAE<sup class="footnote-ref"><a href="#fn:rae">1</a></sup> (representation space) — and for
  each one we run a full EMA sweep. For JiT and RAE the grid is
  <code>{0.5, 0.9, 0.99, 0.999, 0.9993, 0.9995, 0.9997, 0.9999, 0.99999, 0.999999}</code>;
  for SiT we use a denser grid concentrated near 1,
  <code>{0.999, 0.9993, 0.9995, 0.9997, 0.9998, 0.9999, 0.99991, 0.99993, 0.99995, 0.99997}</code>.
  Each sweep also includes the raw online checkpoint as a baseline. We train each model for
  around 80 epochs and report FID<sup class="footnote-ref" id="fnref:fid"><a href="#fn:fid">7</a></sup>,
  Inception Score, precision, and recall, following each repo's provided sampling strategy.
</p>

<h2 data-toc-skip="true">Finding 1: EMA decay slides you along a precision–recall curve</h2>

<p>
  Here's what the RAE sweep looks like at 80 epochs. Every row uses the <em>same</em> checkpoint — they only
  differ in which EMA decay we apply on top of it.
</p>

<figure class="post-figure">
  <img src="/images/blog/ema/rae_precision_recall.png"
       alt="Precision-recall trade-off curve for the RAE EMA sweep at 80 epochs: each point is one EMA decay value; as decay grows from raw to 0.9999, precision rises and recall falls" />
  <figcaption>
    <b>RAE-DiT-XL at 80 epochs: EMA decay traces a precision–recall trade-off.</b>
    Each point is the same training checkpoint with a different EMA decay applied.
    As \(\beta\) grows from raw (no EMA) toward the community default \(\beta = 0.9999\),
    samples gain precision<sup class="footnote-ref" id="fnref:pr"><a href="#fn:pr">9</a></sup>
    but lose recall.
  </figcaption>
</figure>

<p>
  Two things jump out from the curve:
</p>

<ol>
  <li>
    <b>EMA decay traces a clean precision–recall trade-off.</b> Moving along the curve from raw to
    \(\beta = 0.9999\), samples gain precision (0.68 → 0.79) while losing recall (0.61 → 0.53).
    No single decay wins on both axes.
  </li>
  <li>
    <b>The community default \(\beta = 0.9999\) sits at the precision-extreme end of the curve.</b>
    It gives the highest precision in the sweep (0.787), but recall has fallen all the way to 0.531
    — the trade-off is steep. (Finding 2 picks up what this does to FID and IS.)
  </li>
</ol>

<p>
  In other words: EMA isn't only smoothing optimization noise. It's also dialing the model along a
  fidelity–coverage trade-off.
</p>

<h3>The same pattern holds in pixel and latent spaces</h3>

<p>
  RAE is not a special case. The same precision-up / recall-down trajectory shows up in
  pixel-space (JiT) and latent-space (SiT) diffusion, and the same collapse
  cliff appears at very large decays.
</p>

<figure class="post-figure">
  <img src="/images/blog/ema/jit_precision_recall.png"
       alt="Precision-recall trade-off curve for JiT (pixel space) at 80 epochs: as EMA decay grows, precision rises and recall falls, with a sharp hook at the community default beta=0.9999 where precision drops back below the beta=0.9997 value" />
  <figcaption>
    <b>JiT (pixel space) at 80 epochs: same trade-off as RAE.</b>
    Precision rises and recall falls as \(\beta\) grows. The community default \(\beta = 0.9999\)
    sits at the high-precision / low-recall end of the curve.
  </figcaption>
</figure>

<figure class="post-figure">
  <img src="/images/blog/ema/sit_precision_recall.png"
       alt="Precision-recall trade-off curve for SiT (latent space) at 90 epochs: a smooth precision-up, recall-down trajectory across the dense decay grid, with the community default beta=0.9999 near the FID-best point" />
  <figcaption>
    <b>SiT (latent space) at 90 epochs (CFG = 1.5): a subtler version of the same trade-off.</b>
  </figcaption>
</figure>

<h2 data-toc-skip="true">Finding 2: The community default β = 0.9999 is not the optimal choice</h2>

<p>
  Modern diffusion codebases overwhelmingly default to \(\beta = 0.9999\) for EMA, usually
  inherited from an earlier paper rather than re-tuned per method. Across the three diffusion
  model families we swept, this default is suboptimal on FID — the metric the field actually
  reports — in two of three cases. In one of three (JiT) it is <em>strictly dominated</em>:
  another decay produces lower FID, higher Inception Score, higher precision, and higher recall
  simultaneously. There is no metric you can invoke to defend the default for JiT.
</p>

<figure class="post-figure">
  <img src="/images/blog/ema/fid_is_3panel.png"
       alt="3-panel chart of FID and Inception Score vs EMA decay for RAE-DiT-XL, JiT, and SiT. The community default beta=0.9999 is highlighted in red on each curve. RAE: 0.9999 is FID-suboptimal but wins IS. JiT: 0.9999 is worse on both FID and IS than beta=0.9995. SiT: 0.9999 ties for FID-best but IS keeps climbing past it." />
  <figcaption>
    <b>FID and IS vs EMA decay across three model families.</b>
    Each panel: FID (blue, left axis), Inception Score<sup class="footnote-ref" id="fnref:is"><a href="#fn:is">8</a></sup>
    (purple, right axis). The red marker on each curve is the community default
    \(\beta = 0.9999\). The RAE panel also shows three published FID baselines for context
    (dashed lines: raw RAE, LightningDiT, and the RAE codebase's own number at \(\beta = 0.9995\)).
  </figcaption>
</figure>

<ol>
  <li>
    <b>RAE</b> (representation space): FID bottoms out at \(\beta = 0.999\) (3.21). At the
    default \(\beta = 0.9999\), FID has climbed back to 4.14 — almost a full point worse. IS
    does peak at the default, so \(\beta = 0.9999\) wins one metric — but not the one the
    leaderboard runs on.
  </li>
  <li>
    <b>JiT</b> (pixel space): both FID and IS are best at \(\beta = 0.9995\) (FID 8.53, IS
    45.11). At the default \(\beta = 0.9999\), FID is 9.42 and IS is 44.14 — both worse.
    \(\beta = 0.9995\) <em>Pareto-dominates</em> the default on FID, IS, precision, and recall
    simultaneously. There is no metric to defend \(\beta = 0.9999\) here.
  </li>
  <li>
    <b>SiT</b> (latent space): the mildest case. The default ties \(\beta = 0.99993\) for
    best FID. But IS continues to climb past it, peaking at \(\beta = 0.99997\) — so even here
    the default is one of several near-optima rather than <em>the</em> optimum, and this is the
    strongest defense of \(\beta = 0.9999\) we could find.
  </li>
</ol>

<p>
  Three models, one pattern: \(\beta = 0.9999\) is rarely the metric-optimal choice — and in
  JiT it has no leg to stand on.
</p>

<h3>Why this matters: rankings depend on β</h3>

<p>
  Most diffusion papers compare new methods at the default \(\beta = 0.9999\), often without
  even reporting the value. If that default is FID-suboptimal for most models, every such
  comparison is happening at a non-optimal decay — and headline rankings can shift purely from
  EMA mismatches. The RAE panel above makes this concrete: the same RAE-DiT-XL training run sits
  just under LightningDiT's 4.29 line at the default \(\beta = 0.9999\) (4.14, barely beating
  it), but <em>well</em> below it at \(\beta = 0.999\) (3.21, beating it by a full point). When
  the same paper's DiT-DH variant reports FID 2.16 using \(\beta = 0.9995\) and is compared to
  LightningDiT's 4.29 at \(\beta = 0.9999\), part of the headline 2.13-FID gap is the
  architecture and part is the EMA mismatch — and without a matched sweep we can't tell how
  much.
</p>

<p>
  <b>Caveat.</b> These sweeps are at ~80-epoch training horizons. At much longer training,
  optimal \(\beta\) likely drifts toward 1, and \(\beta = 0.9999\) may become defensible. The
  argument isn't "use \(\beta = 0.999\) instead" — it is that the default needs to be tuned
  per method and per training stage, not inherited.
</p>

<p>
  Our practical recommendation is mild but firm: at minimum, papers should report the EMA decay
  they used. Ideally, comparisons should either tune EMA per method under a shared protocol, or
  include the raw-checkpoint numbers alongside EMA ones so the smoothing effect can be
  disentangled from the underlying method.
</p>

<h2>What does this <em>look like</em>? A 2D toy</h2>

<p>
  To make the effect visible, we ran a controlled experiment on a 2D tree-structured Gaussian mixture (from
  autoguidance<sup class="footnote-ref" id="fnref:autoguidance"><a href="#fn:autoguidance">10</a></sup>). The ground truth is a hierarchical tree of branches; some branches are dense and "popular,"
  others are sparse "tail" branches. We trained a small MLP score model, then compared the raw model with
  EMA versions at decays <code>0.99 / 0.997 / 0.998 / 0.999</code>.
</p>

<figure class="post-figure">
  <img src="/images/blog/ema/soft-overconcentration.jpg" alt="Soft over-concentration figure" />
  <figcaption>
    <b>Soft mode collapse, visualised.</b> Top: final samples overlaid on the true tree. Bottom: trajectories
    from a fixed seed grid, with tail-branch paths highlighted. As the EMA decay grows, samples drift toward
    the dominant branches and tail branches lose coverage. At decay <code>0.999</code> the averaged model
    becomes stale enough that trajectories leave the manifold entirely.
  </figcaption>
</figure>

<p>
  Zooming in on the <em>raw model</em> vs <em>EMA 0.998</em>:
</p>

<div class="pair-figure">
  <div class="col">
    <div class="label">Raw model</div>
    <img src="/images/blog/ema/model.jpg" alt="Raw model samples on the tree distribution" />
  </div>
  <div class="col">
    <div class="label">EMA 0.998</div>
    <img src="/images/blog/ema/ema0998.jpg" alt="EMA 0.998 samples on the tree distribution" />
  </div>
</div>
<figcaption style="color:#6b7280; font-size:0.88rem;">
  Same seeds, same training run. EMA 0.998 produces samples that hug the main trunk and visibly under-cover
  the nearby tail branches — even though the underlying training is identical.
</figcaption>

<p>
  This isn't a hard collapse to a single mode. It's something subtler — a <em>soft over-concentration</em>
  toward the dominant branches, with rare branches gradually under-served. On big image benchmarks, the same
  mechanism shows up as cleaner-looking samples with lower recall: typical patterns get sharpened, rare
  configurations quietly disappear.
</p>

<h2>Closing remarks</h2>

<p>
EMA decay is one of those quiet hyperparameters that almost everyone uses but almost nobody tunes. 
We hope this post makes a case for treating it like learning rate,
batch size, or guidance scale — a first-class design choice that materially shapes both the
model's learned distribution and the numbers we publish about it.
</p>

<p>
If you take one thing away from this post, let it be the habit of asking, whenever you see an
intermediate-checkpoint number: <em>what EMA decay produced it, and would it survive a sweep?</em>
That single question, asked routinely, would already remove a large fraction of the apparent
"progress" the field reports today.
</p>

<h2>Acknowledgments</h2>

<p>
We thank <a href="https://tsujuifu.github.io/">Tsu-Jui Fu</a>, <a href="http://liangchiehchen.com/">Liang-Chieh Chen</a>, and <a href="https://zhegan27.github.io/">Zhe Gan</a> for fruitful discussions.
</p>

<section class="footnotes">
  <h3>References</h3>
  <ol>
    <li id="fn:rae">
      Zheng, Boyang, Ma, Tong, &amp; Xie.
      <em>Diffusion Transformers with Representation Autoencoders.</em>
      arXiv preprint arXiv:2510.11690, 2025.
      <a href="https://arxiv.org/abs/2510.11690">arXiv:2510.11690</a>.
      <a href="#fnref:rae" class="footnote-back" title="back to text">↩︎</a>
    </li>
    <li id="fn:ncsnv2">
      Song &amp; Ermon.
      <em>Improved Techniques for Training Score-Based Generative Models.</em>
      NeurIPS 2020.
      <a href="https://papers.neurips.cc/paper_files/paper/2020/file/92c3b916311a5517d9290576e3ea37ad-Paper.pdf">paper</a>.
      <a href="#fnref:ncsnv2" class="footnote-back" title="back to text">↩︎</a>
    </li>
    <li id="fn:raev2">
      Singh et al.
      <em>Improved Baselines with Representation Autoencoders.</em>
      arXiv preprint arXiv:2605.18324, 2026.
      <a href="https://arxiv.org/abs/2605.18324">arXiv:2605.18324</a>.
      <a href="#fnref:raev2" class="footnote-back" title="back to text">↩︎</a>
    </li>
    <li id="fn:repa">
      Yu, Kwak, Jang, Jeong, Huang, Shin, &amp; Xie.
      <em>Representation Alignment for Generation: Training Diffusion Transformers is Easier than You Think.</em>
      arXiv preprint arXiv:2410.06940, 2024.
      <a href="https://arxiv.org/abs/2410.06940">arXiv:2410.06940</a>.
      <a href="#fnref:repa" class="footnote-back" title="back to text">↩︎</a>
    </li>
    <li id="fn:jit">
      Li &amp; He. <em>Back to Basics: Let Denoising Generative Models Denoise.</em> arXiv 2025.
      <a href="https://arxiv.org/abs/2511.13720">arXiv:2511.13720</a>.
      <a href="#fnref:jit" class="footnote-back" title="back to text">↩︎</a>
    </li>
    <li id="fn:sit">
      Ma, Goldstein, Albergo, Boffi, Vanden-Eijnden, &amp; Xie.
      <em>SiT: Exploring Flow- and Diffusion-based Generative Models with Scalable Interpolant Transformers.</em>
      ECCV 2024.
      <a href="#fnref:sit" class="footnote-back" title="back to text">↩︎</a>
    </li>
    <li id="fn:fid">
      Heusel, Ramsauer, Unterthiner, Nessler, &amp; Hochreiter.
      <em>GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.</em>
      NeurIPS 2017. (FID.)
      <a href="#fnref:fid" class="footnote-back" title="back to text">↩︎</a>
    </li>
    <li id="fn:is">
      Salimans, Goodfellow, Zaremba, Cheung, Radford, &amp; Chen.
      <em>Improved Techniques for Training GANs.</em>
      NeurIPS 2016. (Inception Score.)
      <a href="#fnref:is" class="footnote-back" title="back to text">↩︎</a>
    </li>
    <li id="fn:pr">
      Kynkäänniemi, Karras, Laine, Lehtinen, &amp; Aila.
      <em>Improved Precision and Recall Metric for Assessing Generative Models.</em>
      NeurIPS 2019.
      <a href="#fnref:pr" class="footnote-back" title="back to text">↩︎</a>
    </li>
    <li id="fn:autoguidance">
      Karras, Aittala, Kynkäänniemi, Lehtinen, Aila, &amp; Laine.
      <em>Guiding a Diffusion Model with a Bad Version of Itself.</em>
      NeurIPS 2024.
      <a href="#fnref:autoguidance" class="footnote-back" title="back to text">↩︎</a>
    </li>
  </ol>
</section>

<a class="post-back" href="/blog/">← Back to blog</a>

</article>

<script>
/* Auto-build the right-side TOC from h2 headings in the post and run a
   scroll-spy. Uses block comments only — Jekyll compress_html strips
   newlines, which would turn // line comments into "everything is a comment". */
(function () {
  var article = document.querySelector('.post-wrap');
  var toc = document.querySelector('#toc');
  if (!article || !toc) return;
  var headings = Array.from(article.querySelectorAll('h2')).filter(function (h) { return h.dataset.tocSkip !== 'true'; });
  if (headings.length === 0) { toc.style.display = 'none'; return; }
  function slugify(s) { return s.toLowerCase().replace(/[^a-z0-9 \-]/g, '').trim().replace(/\s+/g, '-').slice(0, 60); }
  var linkByHeading = new Map();
  headings.forEach(function (h, i) {
    if (!h.id) h.id = slugify(h.textContent) || ('sec-' + (i + 1));
    var a = document.createElement('a');
    a.href = '#' + h.id;
    a.textContent = h.textContent;
    a.addEventListener('click', function () {
      document.querySelectorAll('#toc a').forEach(function (l) { l.classList.remove('toc-active'); });
      a.classList.add('toc-active');
    });
    toc.appendChild(a);
    linkByHeading.set(h, a);
  });
  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) {
        document.querySelectorAll('#toc a').forEach(function (l) { l.classList.remove('toc-active'); });
        linkByHeading.get(entry.target).classList.add('toc-active');
      }
    });
  }, { rootMargin: '-25% 0px -65% 0px', threshold: 0 });
  headings.forEach(function (h) { observer.observe(h); });
})();
</script>
