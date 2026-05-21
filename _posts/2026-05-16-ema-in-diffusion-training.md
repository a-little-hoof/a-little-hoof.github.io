---
title: "Cheating at diffusion benchmarks with one hyperparameter"
date: 2026-05-16
permalink: /blog/2026/05/ema-in-diffusion-training/
excerpt: "We benchmark EMA decay across pixel-, latent-, and representation-space diffusion models and show that the popular 0.9999 default silently trades recall for precision, and can quietly re-order leaderboards in convergence-speed comparisons."
tags:
  - diffusion
  - training-dynamics
  - EMA
hidden: false
---

<style>
  /* Hide the default academicpages page title — we render our own header below */
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

  .post-wrap .lead {
    font-size: 1.12rem;
    color: #2f3338;
    line-height: 1.65;
  }

  .post-wrap blockquote {
    border-left: 3px solid #d4d8de;
    padding: 4px 16px;
    color: #4b5563;
    margin: 18px 0;
    background: #fafbfc;
    border-radius: 0 6px 6px 0;
  }

  .post-wrap code {
    background: #f1f4f8;
    color: #1f2328;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 0.92em;
  }

  .post-figure {
    margin: 22px 0;
    text-align: center;
  }
  .post-figure img {
    max-width: 100%;
  }
  .post-figure figcaption {
    color: #6b7280;
    font-size: 0.88rem;
    margin-top: 8px;
    text-align: left;
  }
  .pair-figure {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin: 22px 0 6px;
  }
  .pair-figure .col { text-align: center; }
  .pair-figure img { max-width: 100%; }
  .pair-figure .label {
    font-size: 0.82rem; font-weight: 600; color: #4b5563;
    text-transform: uppercase; letter-spacing: 0.04em;
    margin-bottom: 4px;
  }
  .pair-figure + figcaption {
    color: #6b7280;
    font-size: 0.88rem;
    margin-top: 0;
    text-align: left;
  }
  @media (max-width: 580px) {
    .pair-figure { grid-template-columns: 1fr; }
  }

  /* Result table */
  table.ema-sweep {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 0.93rem;
  }
  table.ema-sweep th, table.ema-sweep td {
    padding: 7px 10px;
    border-bottom: 1px solid #eef0f3;
    text-align: center;
  }
  table.ema-sweep th {
    background: #f6f7f9;
    font-weight: 600;
    color: #1f2328;
    font-size: 0.85rem;
    letter-spacing: 0.01em;
  }
  table.ema-sweep td.left, table.ema-sweep th.left { text-align: left; }
  table.ema-sweep tr.row-best td { background: #f3f8ff; font-weight: 600; }
  table.ema-sweep tr.row-default td { background: #fff7ed; }
  table.ema-sweep tr.row-collapse td { color: #b91c1c; }
  .legend-note { font-size: 0.82rem; color: #6b7280; margin-top: -4px; }

  .callout {
    background: #eff5ff;
    border: 1px solid #dbe7ff;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 18px 0;
    color: #1f2328;
    font-size: 0.97rem;
  }
  .callout b { color: #1d4ed8; }

  .post-back {
    display: inline-block;
    margin-top: 36px;
    padding-top: 18px;
    border-top: 1px solid #eef0f3;
    color: #1d4ed8;
    font-size: 0.92rem;
  }

  /* Footnote citations */
  sup.footnote-ref a {
    font-size: 0.72em;
    vertical-align: super;
    line-height: 0;
    margin-left: 1px;
    text-decoration: none;
    color: #1d4ed8;
  }
  sup.footnote-ref a:hover { text-decoration: underline; }
  .footnotes {
    margin-top: 42px;
    padding-top: 16px;
    border-top: 1px solid #eef0f3;
    font-size: 0.92rem;
    color: #4b5563;
  }
  .footnotes h3 {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6b7280;
    margin: 0 0 8px;
  }
  .footnotes ol { padding-left: 22px; margin: 0; }
  .footnotes li { margin: 8px 0; line-height: 1.6; }
  .footnotes li[id]:target,
  sup[id]:target a { background: #fff7cc; border-radius: 4px; padding: 1px 4px; }
  .footnote-back {
    text-decoration: none;
    margin-left: 6px;
    color: #9aa1a8;
    font-size: 0.95em;
  }
  .footnote-back:hover { color: #1d4ed8; }
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

<article class="post-wrap">

<header class="post-header">
  <h1>Cheating at diffusion benchmarks with one hyperparameter</h1>
  <div class="post-tags">
    <span class="tag">diffusion</span>
    <span class="tag">training-dynamics</span>
    <span class="tag">EMA</span>
  </div>
</header>

<p class="lead">
  Diffusion transformers are now competing on <em>convergence speed</em> — who can hit a target
  FID in the fewest training epochs. Here is an uncomfortable thing about that competition: with
  the way the community currently reports results, you can cheat at it almost for free, by
  quietly tuning a single hyperparameter that almost nobody noticed — the EMA decay.
</p>

<figure class="post-figure">
  <img src="/images/blog/ema/ema_sweep_curves.png" alt="gFID vs EMA decay at 40 and 80 epochs for RAE-DiT (DINOv2)" />
  <figcaption>
    <b>Same model architecture, same data, same 80-epoch training run — only the EMA decay
    changes.</b> RAE-DiT<sup class="footnote-ref" id="fnref:rae"><a href="#fn:rae">1</a></sup> (DINOv2) on ImageNet 256. The RAE paper reports its 80-epoch number
    at the <em>unusual</em> EMA decay \(\beta = 0.9995\) (FID \(\approx\) <b>3.29</b>), and compares it to
    LightningDiT at the community-default \(\beta = 0.9999\) (FID = <b>4.29</b>) — a
    <b>~1.00 FID</b> apparent improvement. But our sweep shows that simply switching RAE<sup class="footnote-ref"><a href="#fn:rae">1</a></sup>
    to the same 0.9999 collapses it to FID = <b>4.14</b> — essentially tied with LightningDiT
    (\(\Delta \approx\) <b>0.15</b>). So most of the headline "improvement" is bought by an EMA decay that
    isn't shared between the two methods, not by anything in the underlying RAE algorithm.
    Pick the right column and you "converge faster" than the previous SOTA without changing
    anything that should matter.
  </figcaption>
</figure>

<p>
  This post isn't a recipe for exploiting that loophole — it's an attempt to make the phenomenon 
  visible and to bring EMA decay into the community's attention as a first-class hyperparameter. 
  Almost every modern diffusion-model paper uses an exponential moving average (EMA) of the weights and reports
  results from the EMA checkpoint, yet the decay value is usually inherited from a parent
  codebase rather than tuned or even reported. Our experiments show this matters more than the
  field acknowledges.
</p>

<div class="callout">
  <b>TL;DR.</b> EMA decay matters far more than the community treats it. On the same 80-epoch RAE-DiT run, sweeping decay alone moves gFID from <b>3.21</b> to <b>4.14</b> — a swing larger than the gap between most "SOTA" diffusion-model claims. More fundamentally, EMA decay is not just a smoother: it is a distribution-shaping knob that moves the model along a precision–recall tradeoff. Larger decays often improve precision while suppressing recall, producing a soft form of mode collapse. Since different decays correspond to different points on this tradeoff, fixing one decay implicitly favors one kind of model behavior — and using <em>different</em> decays per method can silently re-order a leaderboard. The community-default 0.9999 is therefore not a neutral choice.
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
The way of comparing diffusion model performance has changed from converged model FID score to convergence speed. Previous way to argue "method X is better than method Y" in diffusion papers is to compare FID until converged. This comparison is impractical not only because training diffusion models takes weeks to converged, but also provide limited signal with incremental improvements in the absolute FID value when every method can achieve FID score lower than 2. What we actually care about is how much training the method needs to be useful.
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
Two consequences follow:
</p>
<ul>
  <li><b>EP<sub>FID@k</sub> without a matched EMA sweep is under-specified.</b> Two methods can
  swap rankings just because of which decay each one defaults to. We should either strictly follow the baseline EMA values or sweep the best EMA value for each method.</li>
</ul>

<p>
For convergence-speed claims, the minimum bar is: report EP<sub>FID@k</sub> together with the
EMA decay used, and ideally either (a) sweep EMA per method under a shared protocol, or (b)
report EP both for the EMA checkpoint and for the raw checkpoint, so that the smoothing effect
can be separated from the underlying training dynamics.
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
      <td>LighteningDiT</td>
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

<h2>A controlled EMA sweep</h2>

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
  around 80 epochs and report FID, Inception Score, precision, and recall, following each
  repo's provided sampling strategy.
</p>

<h2>Finding 1: EMA decay slides you along a precision–recall curve</h2>

<p>
  Here's what the RAE sweep looks like at 80 epochs. Every row uses the <em>same</em> checkpoint — they only
  differ in which EMA decay we apply on top of it.
</p>

<table class="ema-sweep">
  <thead>
    <tr>
      <th class="left">EMA decay</th>
      <th>FID<sup class="footnote-ref" id="fnref:fid"><a href="#fn:fid">7</a></sup> ↓</th>
      <th>IS<sup class="footnote-ref" id="fnref:is"><a href="#fn:is">8</a></sup> ↑</th>
      <th>Precision<sup class="footnote-ref" id="fnref:pr"><a href="#fn:pr">9</a></sup> ↑</th>
      <th>Recall ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="left">raw (no EMA)</td><td>4.46</td><td>159.6</td><td>0.684</td><td>0.607</td></tr>
    <tr><td class="left">0.5</td><td>4.38</td><td>160.0</td><td>0.680</td><td>0.611</td></tr>
    <tr><td class="left">0.9</td><td>3.88</td><td>164.3</td><td>0.687</td><td><b>0.615</b></td></tr>
    <tr><td class="left">0.99</td><td>3.33</td><td>179.3</td><td>0.709</td><td>0.604</td></tr>
    <tr class="row-best"><td class="left">0.999</td><td><b>3.20</b></td><td>200.0</td><td>0.741</td><td>0.585</td></tr>
    <tr><td class="left">0.9993</td><td>3.24</td><td>204.5</td><td>0.744</td><td>0.586</td></tr>
    <tr><td class="left">0.9995</td><td>3.28</td><td>207.6</td><td>0.749</td><td>0.579</td></tr>
    <tr><td class="left">0.9997</td><td>3.38</td><td>212.3</td><td>0.758</td><td>0.569</td></tr>
    <tr class="row-default"><td class="left">0.9999 <small>(community default)</small></td><td>4.16</td><td><b>234.8</b></td><td><b>0.787</b></td><td>0.531</td></tr>
    <tr class="row-collapse"><td class="left">0.99999</td><td>444.78</td><td>1.23</td><td>0.000</td><td>0.000</td></tr>
    <tr class="row-collapse"><td class="left">0.999999</td><td>328.03</td><td>1.22</td><td>0.000</td><td>0.000</td></tr>
  </tbody>
</table>
<p class="legend-note">Blue row = best FID. Orange row = the community-default 0.9999. Red rows = full collapse.</p>

<p>
  Three things jump out:
</p>

<ol>
  <li>
    <b>The best-FID, best-precision, and best-recall settings are <em>different EMA values</em>.</b> Recall
    peaks at 0.9, FID at 0.999, IS and precision at 0.9999. There is no single decay that wins on every metric.
  </li>
  <li>
    <b>The community default of 0.9999 is not the FID-optimal choice</b> at this training stage. It gives the
    highest precision and the highest IS, but FID is worse than every decay between 0.99 and 0.9997 — because
    recall has dropped from 0.61 to 0.53.
  </li>
  <li>
    <b>Pushing the decay further (0.99999, 0.999999) is a cliff.</b> The averaged model becomes so stale that
    the generative distribution falls off the data manifold entirely.
  </li>
</ol>

<p>
  In other words: EMA isn't only smoothing optimization noise. It's also dialing the model along a
  fidelity–coverage tradeoff. Reporting only quality-flavored metrics (FID, IS, precision) systematically
  hides half of what EMA is doing.
</p>

<h3>The same pattern holds in pixel and latent spaces</h3>

<p>
  RAE is not a special case. The same precision-up / recall-down trajectory shows up in
  <em>pixel-space</em> (JiT) and <em>latent-space</em> (SiT) diffusion, and the same collapse
  cliff appears at very large decays. What differs across models is <em>where on the curve</em>
  the community-default \(0.9999\) lands: sub-optimal for RAE and JiT, near-optimal for SiT
  at this training stage. The point isn't that \(0.9999\) is universally wrong; it's that
  picking a decay without sweeping is a coin flip whose outcome depends on the model,
  the input space, <em>and</em> the training stage.
</p>

<p><b>JiT (pixel space) at 80 epochs.</b></p>

<table class="ema-sweep">
  <thead>
    <tr>
      <th class="left">EMA decay</th>
      <th>FID ↓</th>
      <th>IS ↑</th>
      <th>Precision ↑</th>
      <th>Recall ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="left">raw (no EMA)</td><td>13.10</td><td>40.06</td><td>0.520</td><td><b>0.528</b></td></tr>
    <tr><td class="left">0.5</td><td>12.59</td><td>40.54</td><td>0.527</td><td><b>0.528</b></td></tr>
    <tr><td class="left">0.9</td><td>11.51</td><td>41.29</td><td>0.550</td><td>0.522</td></tr>
    <tr><td class="left">0.99</td><td>10.08</td><td>42.96</td><td>0.582</td><td>0.503</td></tr>
    <tr><td class="left">0.999</td><td>8.73</td><td>44.20</td><td>0.614</td><td>0.494</td></tr>
    <tr><td class="left">0.9993</td><td>8.59</td><td>45.02</td><td>0.617</td><td>0.494</td></tr>
    <tr><td class="left">0.9995</td><td>8.53</td><td>45.11</td><td>0.621</td><td>0.487</td></tr>
    <tr class="row-best"><td class="left">0.9997</td><td><b>8.42</b></td><td>45.35</td><td><b>0.682</b></td><td>0.487</td></tr>
    <tr class="row-default"><td class="left">0.9999 <small>(community default)</small></td><td>9.42</td><td>44.14</td><td>0.618</td><td>0.462</td></tr>
    <tr class="row-collapse"><td class="left">0.99999</td><td>360.62</td><td>1.20</td><td>0.078</td><td>0.000</td></tr>
    <tr class="row-collapse"><td class="left">0.999999</td><td>422.26</td><td>1.03</td><td>0.000</td><td>0.000</td></tr>
  </tbody>
</table>
<p class="legend-note">Blue = best FID (\(\beta = 0.9997\)). Orange = community-default \(0.9999\). Red = full collapse. As with RAE, the default is <em>not</em> FID-optimal here.</p>

<p><b>SiT (latent space) at 90 epochs (classifier-free guidance scale 1.5).</b></p>

<table class="ema-sweep">
  <thead>
    <tr>
      <th class="left">EMA decay</th>
      <th>FID ↓</th>
      <th>IS ↑</th>
      <th>Precision ↑</th>
      <th>Recall ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="left">raw (no EMA)</td><td>6.93</td><td>136.73</td><td>0.634</td><td><b>0.573</b></td></tr>
    <tr><td class="left">0.999</td><td>5.73</td><td>152.28</td><td>0.674</td><td>0.563</td></tr>
    <tr><td class="left">0.9993</td><td>5.58</td><td>154.54</td><td>0.676</td><td>0.563</td></tr>
    <tr><td class="left">0.9995</td><td>5.55</td><td>154.65</td><td>0.684</td><td>0.564</td></tr>
    <tr><td class="left">0.9997</td><td>5.41</td><td>156.62</td><td>0.690</td><td>0.558</td></tr>
    <tr><td class="left">0.9998</td><td>5.24</td><td>160.88</td><td>0.698</td><td>0.549</td></tr>
    <tr class="row-default"><td class="left">0.9999 <small>(community default)</small></td><td><b>5.17</b></td><td>163.73</td><td>0.707</td><td>0.548</td></tr>
    <tr><td class="left">0.99991</td><td>5.20</td><td>164.51</td><td>0.708</td><td>0.547</td></tr>
    <tr class="row-best"><td class="left">0.99993</td><td><b>5.17</b></td><td>162.94</td><td>0.713</td><td>0.540</td></tr>
    <tr><td class="left">0.99995</td><td>5.21</td><td>164.70</td><td>0.717</td><td>0.536</td></tr>
    <tr><td class="left">0.99997</td><td>5.51</td><td>164.76</td><td><b>0.723</b></td><td>0.527</td></tr>
  </tbody>
</table>
<p class="legend-note">For SiT, the community-default \(0.9999\) ties the FID-best entry (\(0.99993\)) — but recall continues to fall as \(\beta\) increases, so the precision–recall tradeoff is still active even though the default happens to be near-optimal on FID here.</p>

<h2>Finding 2: EMA changes the ranking, not just the score</h2>

<p>
  Because the precision–recall tradeoff is real, two models that look essentially tied under their raw
  checkpoints can <em>separate</em> once you apply different EMA decays — and vice versa. Conversely, a method
  that looks "best" under <code>0.9999</code> may stop being best when you sweep the decay. We've already seen
  one example of this at the top of the post: RAE-DiT vs LightningDiT shows a <b>~1.00</b> FID gap when each
  method uses its own EMA decay (\(\beta = 0.9995\) vs \(\beta = 0.9999\)), but the gap shrinks to just
  <b>~0.15</b> once both are evaluated at the same \(\beta = 0.9999\). The "improvement" was an EMA artefact,
  not an architecture or objective advantage. The lesson generalizes: if a benchmark fixes EMA to 0.9999 for
  everyone — or worse, leaves it inherited and undocumented — part of the resulting ordering is an EMA
  artefact too.
</p>

<p>
  Our practical recommendation is mild but firm: at minimum, papers should report the EMA decay they used.
  Ideally, comparisons should either tune EMA per method under a shared protocol, or include the raw-checkpoint
  numbers alongside EMA ones so the smoothing effect can be disentangled from the underlying method.
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
