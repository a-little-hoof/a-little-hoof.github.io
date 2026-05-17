---
title: "Your EMA decay is a hyperparameter, not a convention"
date: 2026-05-16
permalink: /blog/2026/05/ema-in-diffusion-training/
excerpt: "We benchmark EMA decay across pixel-, latent-, and representation-space diffusion models and find that the popular 0.9999 default silently trades recall for precision — and that the optimal decay shifts as training progresses."
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

<article class="post-wrap">

<header class="post-header">
  <div class="meta">
    <time datetime="2026-05-16">May 16, 2026</time> · 6 min read
  </div>
  <h1>Your EMA decay is a hyperparameter, not a convention</h1>
  <div class="post-tags">
    <span class="tag">diffusion</span>
    <span class="tag">training-dynamics</span>
    <span class="tag">EMA</span>
  </div>
</header>

<p class="lead">
  Almost every modern diffusion-model paper uses an exponential moving average of the weights and reports results from the EMA checkpoint. However, the EMA decay is often treated as an implementation detail: inherited from a previous codebase or chosen once, but rarely studied systematically. This matters because many recent papers compare intermediate checkpoints to demonstrate faster convergence, with 80-epoch results becoming a common evaluation setting. Our experiments show that EMA decay can strongly affect performance at this stage. As a result, an apparent improvement at 80 epochs may partly reflect a better EMA choice rather than a genuinely faster-converging model. For fair comparison, EMA decay should be searched and reported as part of the experimental protocol.
</p>

<div class="callout">
  <b>TL;DR.</b> EMA decay is not just a smoother. It is a distribution-shaping knob that moves the model along a precision–recall tradeoff. Larger decays often improve precision while suppressing recall, producing a soft form of mode collapse. Since different decays correspond to different points on this tradeoff, fixing one decay implicitly favors one kind of model behavior. Therefore, the community-default 0.9999 is not a neutral choice, especially when comparing models before convergence.
</div>

<h2>Why care about EMA decay at all?</h2>

<h3>EMA decay is part of the evaluation protocol</h3>

<p>
Almost every modern diffusion-model paper reports results from an EMA checkpoint. This means the reported number is not determined only by the model architecture, training objective, optimizer, or training budget. It also depends on the EMA decay used to construct the checkpoint. If this decay is inherited, fixed, or chosen differently across methods, then EMA becomes an implicit part of the evaluation protocol.
</p>

<h3>Intermediate-checkpoint comparisons make this more serious</h3>

<p>
This issue is especially important for intermediate evaluations. Many recent papers report results at 80 epochs to demonstrate faster convergence. But at this stage, EMA decay can strongly affect the reported performance. An apparent improvement at 80 epochs may therefore partly reflect a better EMA choice rather than a genuinely faster-converging model.
</p>

<p>
For fair comparison, EMA decay should be searched and reported, especially when comparing methods before convergence.
</p>

<h3>A concrete example: method rankings can change after EMA tuning</h3>

<p>
To see why this matters, consider an 80-epoch comparison between different diffusion training methods. If each method is evaluated with its inherited or default EMA decay, the reported ranking appears to measure which method converges faster. However, after sweeping the EMA decay for each method, the picture changes: some methods gain substantially more from EMA tuning than others, and the relative gaps between methods can shrink, grow, or even reverse.
</p>

<p>
This means that an 80-epoch number is not only a property of the model or training objective. It is also a property of the EMA decay used to report the checkpoint. Without an EMA sweep, a method can look better simply because its chosen EMA decay is better matched to that training stage.
</p>

<p>
  We benchmark three diffusion models that span the typical input-space spectrum — JiT<sup class="footnote-ref" id="fnref:jit"><a href="#fn:jit">1</a></sup> (pixel space), SiT<sup class="footnote-ref" id="fnref:sit"><a href="#fn:sit">2</a></sup>
  (latent space), and RAE<sup class="footnote-ref" id="fnref:rae"><a href="#fn:rae">3</a></sup> (representation space) — and for each one we run a full EMA sweep:
  <code>{0.5, 0.9, 0.99, 0.999, 0.9993, 0.9995, 0.9997, 0.9999, 0.99999, 0.999999}</code>, plus the raw
  online checkpoint, at multiple training stages. Same training pipeline, same sampling protocol; only the
  decay changes.
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
      <th>FID<sup class="footnote-ref" id="fnref:fid"><a href="#fn:fid">4</a></sup> ↓</th>
      <th>IS<sup class="footnote-ref" id="fnref:is"><a href="#fn:is">5</a></sup> ↑</th>
      <th>Precision<sup class="footnote-ref" id="fnref:pr"><a href="#fn:pr">6</a></sup> ↑</th>
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

<h2>Finding 2: EMA changes the ranking, not just the score</h2>

<p>
  Because the precision–recall tradeoff is real, two models that look essentially tied under their raw
  checkpoints can <em>separate</em> once you apply different EMA decays — and vice versa. Conversely, a method
  that looks "best" under <code>0.9999</code> may stop being best when you sweep the decay. We've seen rankings
  flip more than once across our experiments. So if a benchmark fixes EMA to 0.9999 for everyone, part of the
  resulting ordering is an EMA artefact, not an architecture or objective advantage.
</p>

<p>
  Our practical recommendation is mild but firm: at minimum, papers should report the EMA decay they used.
  Ideally, comparisons should either tune EMA per method under a shared protocol, or include the raw-checkpoint
  numbers alongside EMA ones so the smoothing effect can be disentangled from the underlying method.
</p>

<h2>Finding 3: the optimal EMA depends on the training stage</h2>

<p>
  The most consequential of our findings: there is no <em>single</em> EMA decay that is optimal across the
  whole training trajectory. Early on, when the online model is changing rapidly, a long EMA horizon is
  averaging over qualitatively different model states — it doesn't stabilise the current model so much as
  mix in stale behaviour. Later in training, once optimization is more stable, longer horizons start to
  behave like genuine smoothers and become helpful.
</p>

<p>
  Empirically this means the metric-optimal EMA decay <em>drifts</em> as training proceeds, and the gap
  between "stage-aware decay" and "fixed 0.9999" is largest precisely at the partial-training stages where
  most empirical comparisons are made.
</p>

<h2>What does this <em>look like</em>? A 2D toy</h2>

<p>
  To make the effect visible, we ran a controlled experiment on a 2D tree-structured Gaussian mixture (from
  autoguidance<sup class="footnote-ref" id="fnref:autoguidance"><a href="#fn:autoguidance">7</a></sup>). The ground truth is a hierarchical tree of branches; some branches are dense and "popular,"
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

<h2>A mechanistic intuition</h2>

<p>
  EMA averages weights over time, which means it preferentially retains structures that are <em>stable</em>
  over time. Frequent, high-density modes are reinforced; rare, fine-grained, still-evolving structures get
  attenuated. Pair that with a long horizon (large decay) and a non-stationary optimization trajectory, and
  you get soft over-concentration almost by construction.
</p>

<p>
  Phrased that way, the precision–recall tradeoff and the soft mode collapse are two views of the same
  phenomenon, not two independent effects.
</p>

<h2>What about post-hoc EMA?</h2>

<p>
  If the best decay drifts during training, the natural escape hatch is <em>post-hoc EMA</em>: keep dense
  checkpoints, then reconstruct any EMA horizon offline at evaluation time. We tried this. It's a useful
  approximation — but not a free one. Post-hoc reconstruction only has access to a discrete set of
  checkpoints; between them, the model's parameters are unknown. When the training trajectory is rapidly
  non-stationary (early training, learning rare structures), the approximation is noticeably worse than the
  corresponding online EMA. The accuracy of post-hoc EMA is, ironically, controlled by exactly the same
  non-stationarity that makes choosing online EMA difficult in the first place.
</p>

<h2>What we'd like to see going forward</h2>

<ul>
  <li><b>Report EMA decay</b> in diffusion-model papers, the way we report learning rate and batch size.</li>
  <li><b>Sweep EMA per method</b> when running a benchmark, or at the very least include raw-checkpoint numbers.</li>
  <li><b>Pair quality metrics with coverage metrics.</b> FID alone hides the recall side of the tradeoff.</li>
  <li><b>Treat EMA as stage-dependent</b>, not as a fixed convention inherited from upstream papers.</li>
</ul>

<p>
  The paper has the full results, including the per-stage EMA sensitivity curves and the inter-model ranking
  flips. We'll link the arXiv version here once it's up.
</p>

<section class="footnotes">
  <h3>References</h3>
  <ol>
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
    <li id="fn:rae">
      Zheng, Ma, Tong, &amp; Xie. <em>Diffusion Transformers with Representation Autoencoders.</em> arXiv 2025.
      <a href="https://arxiv.org/abs/2510.11690">arXiv:2510.11690</a>.
      <a href="#fnref:rae" class="footnote-back" title="back to text">↩︎</a>
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
