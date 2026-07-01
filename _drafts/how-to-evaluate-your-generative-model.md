---
title: "Struggling Toward Generative Model Evaluation"
layout: ema-note
permalink: /blog/how-to-evaluate-your-generative-model/
excerpt: "A blog post on why generative model evaluation is hard, how representation-based metrics became the default language, and why benchmarks are useful but never final."
tags:
  - generative-models
  - evaluation
  - diffusion
  - blog
hidden: false
---

<style>
.post-wrap h2 {
  font-size: 1.65rem;
}
.post-wrap h3 {
  font-size: 1.35rem;
}
.post-wrap h4 {
  font-size: 1.15rem;
  margin: 1.25rem 0 0.45rem;
}
@media (max-width: 640px) {
  .post-wrap h2 {
    font-size: 1.42rem;
  }
  .post-wrap h3 {
    font-size: 1.22rem;
  }
  .post-wrap h4 {
    font-size: 1.08rem;
  }
}
</style>

<details class="draft-reference-notes" open>
  <summary><strong>Temporary draft reference notes - delete before publishing</strong></summary>

  <p>This section is only for tracking which references support each part of the blog post. Remove the whole block once the writing is finalized.</p>

  <h3>1. What are generative models and why is evaluation hard?</h3>
  <ul>
    <li>References to add:</li>
  </ul>

  <h3>2. Discriminative models, Inception Score, precision, and recall</h3>
  <ul>
    <li>Reference: Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen, "Improved Techniques for Training GANs," arXiv:1606.03498, <a href="https://arxiv.org/abs/1606.03498">https://arxiv.org/abs/1606.03498</a></li>
    <li>Reference: Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, Timo Aila, "Improved Precision and Recall Metric for Assessing Generative Models," arXiv:1904.06991, <a href="https://arxiv.org/abs/1904.06991">https://arxiv.org/abs/1904.06991</a></li>
  </ul>

  <h3>3. FID and variants: KID, FD-DINO, FVD, text-to-image, and video</h3>
  <ul>
    <li>Reference: Gretton course / ICLR 2026 PDF, <a href="https://www.gatsby.ucl.ac.uk/~gretton/coursefiles/iclr26.pdf">https://www.gatsby.ucl.ac.uk/~gretton/coursefiles/iclr26.pdf</a></li>
    <li>Reference: Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter, "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium," arXiv:1706.08500, <a href="https://arxiv.org/abs/1706.08500">https://arxiv.org/abs/1706.08500</a></li>
    <li>Reference: Mikołaj Bińkowski, Dougal J. Sutherland, Michael Arbel, Arthur Gretton, "Demystifying MMD GANs," arXiv:1801.01401, <a href="https://arxiv.org/abs/1801.01401">https://arxiv.org/abs/1801.01401</a></li>
    <li>Reference: George Stein, Jesse C. Cresswell, Rasa Hosseinzadeh, Yi Sui, Brendan Leigh Ross, Valentin Villecroze, Zhaoyan Liu, Anthony L. Caterini, J. Eric T. Taylor, Gabriel Loaiza-Ganem, "Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models," arXiv:2306.04675, <a href="https://arxiv.org/abs/2306.04675">https://arxiv.org/abs/2306.04675</a></li>
  </ul>

  <h3>4. Problems with FID</h3>
  <ul>
    <li>Reference: Mario Lucic, Karol Kurach, Marcin Michalski, Sylvain Gelly, Olivier Bousquet, "Are GANs Created Equal? A Large-Scale Study," arXiv:1711.10337, <a href="https://arxiv.org/abs/1711.10337">https://arxiv.org/abs/1711.10337</a></li>
    <li>Reference: Min Jin Chong, David Forsyth, "Effectively Unbiased FID and Inception Score and where to find them," arXiv:1911.07023, <a href="https://arxiv.org/abs/1911.07023">https://arxiv.org/abs/1911.07023</a></li>
    <li>Reference: Gaurav Parmar, Richard Zhang, Jun-Yan Zhu, "On Aliased Resizing and Surprising Subtleties in GAN Evaluation," arXiv:2104.11222, <a href="https://arxiv.org/abs/2104.11222">https://arxiv.org/abs/2104.11222</a></li>
    <li>Reference: Jiawei Yang, Zhengyang Geng, Xuan Ju, Yonglong Tian, Yue Wang, "Representation Fréchet Loss for Visual Generation," arXiv:2604.28190, <a href="https://arxiv.org/abs/2604.28190">https://arxiv.org/abs/2604.28190</a></li>
    <li>Reference: Nicolas Dufour, Alexei A. Efros, Patrick Pérez, "The FID Lottery: Quantifying Hidden Randomness in Generative-Model Evaluation," arXiv:2606.20536, <a href="https://arxiv.org/abs/2606.20536">https://arxiv.org/abs/2606.20536</a></li>
    <li>Reference: Xingjian Leng, Jaskirat Singh, Zhanhao Liang, Ethan Smith, Martin Bell, Aninda Saha, Yuhui Yuan, Liang Zheng, "DiffusionBench: On Holistic Evaluation of Diffusion Transformers," arXiv:2606.24888, <a href="https://arxiv.org/abs/2606.24888">https://arxiv.org/abs/2606.24888</a></li>
    <li>Notes:</li>
  </ul>

  <h3>5. Benchmarks as a new standard</h3>
  <ul>
    <li>References to add:</li>
  </ul>

  <h3>6. Closing thought</h3>
  <ul>
    <li>References to add:</li>
  </ul>
</details>

> Evaluation is where generative modeling becomes honest. Training asks whether we can sample something. Evaluation asks whether the samples mean what we think they mean.

Generative models have become unusually good at producing things that look convincing: images, videos, speech, music, molecules, layouts, 3D scenes, and text. But the better these models become, the harder evaluation gets. When outputs are obviously broken, evaluation is easy. When outputs are plausible, diverse, and sometimes beautiful, the question shifts from "does it work?" to "what exactly improved?"

This blog post is a rough storyline for how I think the field has struggled toward generative model evaluation: from the basic difficulty, to representation-based metrics, to FID and its many descendants, to benchmark-driven evaluation, and finally to the uncomfortable fact that no single number gets to be the truth.

## 1. Introduction

### 1.1 What Are Generative Models?

A generative model learns a distribution. In the simplest unconditional case, we want

$$
p_\theta(x) \approx p_{\mathrm{data}}(x).
$$

For conditional generation, the target is instead

$$
p_\theta(x \mid c) \approx p_{\mathrm{data}}(x \mid c),
$$

where $c$ might be a class label, text prompt, input image, pose, depth map, audio track, camera trajectory, or editing instruction.

### 1.2 Why Is Evaluation Hard?

This sounds clean until we ask what "approximately equal" should mean. Pixel-wise distance is not enough: two images can be nearly identical to humans but far apart in pixel space after a small shift, crop, or color change. Likelihood is principled but not always aligned with perceptual quality. Human preference is important but expensive, noisy, and hard to reproduce.

So generative evaluation is hard for several reasons at once:

- **Quality and diversity are different goals.** A model can produce beautiful samples while missing rare modes.
- **Perception is not pixel distance.** The metric needs to understand objects, textures, composition, and semantics.
- **Conditional correctness is multi-dimensional.** A text-to-image sample can be realistic but ignore the prompt.
- **Sampling is part of the system.** Guidance scale, sampler, number of steps, candidate count, and reranking can change the result.
- **Benchmarks leak into training decisions.** Once a metric becomes a target, models start adapting to it.

This is why "Model A is better than Model B" is almost never a complete evaluation statement. Better at what? Under which prompt distribution? With how many generated candidates? Compared to which baseline? Using which feature space? Those details are not footnotes. They are the evaluation.

## 2. Discriminative Models As A Natural Thought

The natural escape from pixel space is representation space. Instead of comparing images directly, pass them through a pretrained vision model and compare the resulting features.

The hope is simple: a good representation model maps visually and semantically similar images to nearby vectors. If this is true, then evaluation in feature space should correlate better with human perception than evaluation in raw pixels.

This idea appears in many forms:

- use an ImageNet classifier to judge whether generated images look like recognizable objects;
- use its intermediate features to compare real and generated distributions;
- use CLIP-like features to evaluate image-text alignment;
- use self-supervised features such as DINO for more general visual similarity;
- use video encoders for temporal and motion-aware evaluation.

The representation model becomes a measuring device. It is not the generator, but it defines what the evaluation can see.

### 2.1 Inception Score

Inception Score was one of the early popular representation-based metrics for image generation. It uses a pretrained Inception classifier and rewards two things:

1. each generated image should produce a confident class prediction;
2. the full generated set should cover many classes.

Written roughly, it is

$$
\mathrm{IS} =
\exp\left(
\mathbb{E}_{x \sim p_g}
D_{\mathrm{KL}}(p(y \mid x) \,\|\, p(y))
\right).
$$

The intuition is attractive. If each image is sharp and recognizable, $p(y \mid x)$ should have low entropy. If the generator is diverse, the marginal class distribution $p(y)$ should have high entropy. So the score tries to reward both fidelity and diversity.

But Inception Score also shows the danger of representation-based evaluation. It does not compare to the real data distribution directly. It mostly cares about classifier-recognizable ImageNet categories. It can reward samples that are easy for the classifier while missing other aspects of realism. It also has little to say about conditional generation unless the condition is aligned with the classifier labels.

Still, it introduced an important pattern: use a strong pretrained model as a proxy for human visual judgment.

### 2.2 Precision And Recall

Precision and recall for generative models make the quality-diversity tradeoff more explicit.

- **Precision** asks whether generated samples lie near the real data manifold.
- **Recall** asks whether the generated distribution covers the real data manifold.

This is useful because a single model can be good in one direction and weak in the other. A model with high precision but low recall produces realistic samples but misses modes. A model with high recall but low precision covers many regions but often produces unrealistic samples.

Many generation knobs move along this frontier:

| Knob | Often increases | Often decreases |
| --- | --- | --- |
| classifier-free guidance | alignment, precision | diversity, recall |
| truncation | typicality, sharpness | rare modes |
| stronger filtering | average sample quality | distribution coverage |
| long EMA decay | smoothness, precision | adaptation, recall |
| lower temperature | consistency | variety |

This framing is valuable because it refuses to collapse everything into one number. Sometimes the most important result is not that Model A has a better score, but that it lands in a different place on the precision-recall plane.

## 3. One Score That Rules Them All

Then came FID.

### 3.1 What Is FID?

Frechet Inception Distance compares real and generated samples in Inception feature space. Instead of asking whether individual samples are classified confidently, it fits a Gaussian to the real features and another Gaussian to the generated features:

$$
\mathrm{FID} =
\lVert \mu_r - \mu_g \rVert_2^2
+
\mathrm{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2}).
$$

Here $(\mu_r, \Sigma_r)$ are the mean and covariance of real features, and $(\mu_g, \Sigma_g)$ are the mean and covariance of generated features. Lower is better.

FID became popular for good reasons. It compares generated samples to real data. It is simple to compute. It is more sensitive to visual quality than Inception Score. It gives a convenient scalar for tables. For a long time, if you trained an unconditional image generator, FID was the number everyone looked for first.

It also fit the rhythm of research. You could train a model, generate 50k samples, compute FID, and know where to put your method in the landscape.

### 3.2 Variants Of FID

Once FID became the default form, variants naturally appeared.

Kernel Inception Distance, or KID, keeps the same broad idea of comparing real and generated features, but uses a polynomial-kernel maximum mean discrepancy estimate. One practical advantage is that KID has an unbiased estimator, while FID can be biased for finite sample sizes.

FD-DINO changes the feature extractor. Instead of Inception features, it uses DINO or DINOv2 representations. This is a natural update: if the representation model is the measuring device, then a better or more general representation model may give a better measurement. DINO-style features are often more semantically rich and less tied to supervised ImageNet classification.

The common template is:

1. choose a representation model $\phi(\cdot)$;
2. embed real and generated samples;
3. compare the two embedded distributions;
4. report one distance.

### 3.3 Extending FID Beyond Image Generation

That template has been used far beyond unconditional image generation. Text-to-image papers report FID on COCO-like prompts. Video generation papers report FVD or related video feature distances. Editing and controllable generation papers often combine feature distances with task-specific scores.

For video, the analogous metric is often FVD, Frechet Video Distance. It compares real and generated videos in a video feature space, often using an action-recognition or video-understanding network. The goal is to measure not just frame quality, but temporal coherence and motion realism.

Even when the names change, the underlying dream is the same: find a representation where distribution distance becomes meaningful.

## 4. The Problem With FID

FID is useful, but it is not perfect. For a long time, Frechet distance mostly lived on the evaluation side of the generative modeling pipeline: train a model, generate samples, compute FID, and use the number to compare methods. As generative models become more powerful, the problems of FID have become harder to ignore: the score is useful, influential, and still only a proxy.

### 4.1 How Stable Is FID?

The first problem is stability. FID can move for reasons that have little to do with the conceptual quality of the generator.

#### 4.1.1 Randomness In Training

Random seeds, data order, initialization, augmentation randomness, and hardware-level nondeterminism can all change the final model. If the reported improvement is small, it may be hard to know whether the method improved the generator or simply landed on a better run.

This is why FID should ideally be reported with multiple seeds or confidence intervals, especially when the comparison is close.

#### 4.1.2 Differences In Image Processing

FID is also sensitive to image processing details:

- the exact feature extractor;
- image resolution and preprocessing;
- resizing and cropping implementation;
- color space and normalization;
- image format and compression;
- choice of reference set.

This matters because a reported FID is not just a property of the model. It is a property of the whole evaluation pipeline.

#### 4.1.3 Differences In Experiment Protocol

Even when the metric implementation is fixed, the experiment protocol can change the number:

- number of generated samples;
- whether samples are filtered;
- random seeds and sampling budget;
- guidance scale, sampler, and number of denoising steps.

All of these details matter because FID is often used to compare papers, but papers do not always compare the same sampling procedure.

### 4.2 When FID Score Becomes Unsuggestive

The second problem is meaning. Even when FID is computed correctly and stably, it can become less suggestive about the behavior we actually care about.

#### 4.2.1 Do FID Gains On ImageNet Transfer?

FID is a feature-space metric. If the feature extractor cannot see a failure, FID may not punish it. Inception features are good for many natural-image semantics, but they are not a universal model of human perception. They may underweight counting, spatial relations, text rendering, fine-grained attributes, unusual styles, or domain-specific details.

This becomes especially important when FID gains are measured on ImageNet-like distributions and then treated as evidence for broader generative quality. A method that improves class-conditional ImageNet FID may not automatically improve text-to-image alignment, editing faithfulness, video motion, or long-tail creative use cases.

#### 4.2.2 Small FID Absolute Value Loss Its Meaning

FID compresses a distribution into a Gaussian mean and covariance. This is a drastic summary. Two distributions can share similar first and second moments in feature space while differing in important ways. Mode dropping, rare categories, and compositional failures can be hidden by a good average distance.

Small absolute FID differences are therefore hard to interpret without context. A change from 2.5 to 2.3 may be statistically unstable, visually irrelevant, or meaningful only under a specific protocol. Conversely, a model can improve FID by producing safer, more typical samples while losing diversity or interesting long-tail behavior. For some applications, that is acceptable. For open-ended creative generation, it may be the opposite of what users want.

#### 4.2.3 FID Score On Other Domains Is Even Worse

The problem becomes sharper outside natural-image generation. Inception features were not designed for medical images, satellite imagery, diagrams, text rendering, videos, 3D assets, or highly stylized domains. In those settings, a low FID may mostly say that generated samples match the biases of an ImageNet-trained representation, not that they satisfy the domain's actual criteria.

This is why domain-specific generation often needs domain-specific evaluators: clinical validity for medical images, temporal coherence for video, geometric consistency for 3D, OCR or layout correctness for documents, and instruction faithfulness for editing.

FID also becomes less informative as the field optimizes against it. Once a metric becomes the leaderboard, it becomes part of the training loop, even if indirectly. Hyperparameters are chosen because they improve FID. Architectural decisions are validated by FID. Samplers are tuned for FID. Over time, the metric starts measuring not only generative quality, but also how well the community has learned to satisfy the metric.

This is not a reason to discard FID. It is a reason to treat FID as one instrument in a lab, not as the lab itself.

## 5. Benchmarks Become The New Standard

As generative models became conditional and instruction-driven, one distribution metric was no longer enough. Text-to-image models need prompt following. Video models need temporal consistency. Editing models need locality and preservation. 3D models need geometry. Multimodal models need reasoning across inputs.

### 5.1 What Benchmarks Does

Benchmarks became the next standard.

Instead of asking only whether generated samples match a broad reference distribution, benchmarks ask whether models solve curated tasks:

- generate an image with two objects and the right attributes;
- place one object left of another;
- render text correctly;
- preserve identity during editing;
- follow a multi-step instruction;
- maintain temporal consistency across frames;
- obey camera motion or pose control;
- avoid unsafe or biased outputs.

For text-to-image, this led to prompt suites and automatic evaluators for compositionality, counting, spatial relations, color binding, aesthetics, and image-text alignment. For video generation, benchmarks test motion quality, frame consistency, subject preservation, and text-video alignment. For editing, benchmarks often separate edit success from background preservation.

This is progress. Benchmarks make failure modes visible. They turn vague complaints into testable categories. They also help evaluation move closer to actual use cases.

### 5.2 Benchmarks Are Not Perfect

But benchmarks are not perfect either.

They can be too small, too artificial, or too easy to overfit. They can privilege the evaluator model's worldview. If a benchmark uses an automatic VQA model or captioning model, then the benchmark inherits that model's blind spots. If prompts are public, models and pipelines can be tuned around them. If the scoring rule is too narrow, models may improve the benchmark without improving the user experience.

Human evaluation helps, but it is not magic. Human studies depend on prompt selection, annotator instructions, pairwise versus absolute rating, interface design, randomization, and statistical power. A beautiful image may ignore the prompt. A faithful image may look ugly. A safe image may be boring. A benchmark has to decide what it values.

So we end up with a layered evaluation stack:

| Layer | What it helps answer | Typical failure |
| --- | --- | --- |
| Distribution metrics | Do generated samples resemble real data globally? | hides conditional failures |
| Precision/recall | What is the quality-diversity tradeoff? | depends on feature space |
| Task benchmarks | Does the model solve specific capabilities? | can be overfit or narrow |
| Human preference | Which output do people prefer? | noisy and protocol-dependent |
| Failure audits | How does the model break? | hard to summarize in one number |

This stack is messier than a single leaderboard number, but it is closer to the truth.

## 6. Closing Thought

The history of generative model evaluation is a history of proxies. Pixels were too brittle, so we moved to representations. Inception Score was too indirect, so we moved to FID. FID was too coarse, so we built variants with better features and video encoders. Distribution metrics missed conditional behavior, so we built benchmarks. Benchmarks became targets, so we added human evaluation and failure audits.

Each step fixes something and exposes something else.

I do not think there will be one final metric for generative models. The object we are evaluating is too rich: realism, diversity, controllability, faithfulness, novelty, usefulness, safety, efficiency, and taste all matter, and they do not reduce cleanly to one axis.

The best evaluation is therefore not the most elegant one. It is the one that makes the model legible from several angles. A good paper, report, or internal model card should say:

- what distribution or task the model is meant for;
- which baselines it is compared against;
- which representation metrics improved;
- which benchmark capabilities improved;
- where the model still fails;
- what sampling budget was used;
- what the examples look like without cherry-picking.

As the old saying goes, **beauty is in the eye of the beholders**. For generative models, humans are still the most meaningful evaluators of whether an output feels good, useful, faithful, surprising, or wrong. Automatic metrics are valuable because they are cheap, repeatable, and scalable, but they are still proxies for judgments that ultimately come back to human perception and human use.

When a generative model really improves, it should survive being measured in more than one way. And when it only improves under one metric, the interesting question is not "how good is the model?" It is "what did that metric teach the model to become?"

<section class="references-section">
  <h4>References</h4>
  <ol>
    <li id="fn:is">
      Salimans, Goodfellow, Zaremba, Cheung, Radford, &amp; Chen.
      <em>Improved Techniques for Training GANs.</em>
      NeurIPS 2016.
      <a href="https://arxiv.org/abs/1606.03498">arXiv:1606.03498</a>.
    </li>
    <li id="fn:pr">
      Kynkäänniemi, Karras, Laine, Lehtinen, &amp; Aila.
      <em>Improved Precision and Recall Metric for Assessing Generative Models.</em>
      NeurIPS 2019.
      <a href="https://arxiv.org/abs/1904.06991">arXiv:1904.06991</a>.
    </li>
    <li id="fn:gretton-iclr26">
      Gretton.
      <em>ICLR 2026 course PDF.</em>
      <a href="https://www.gatsby.ucl.ac.uk/~gretton/coursefiles/iclr26.pdf">PDF</a>.
    </li>
    <li id="fn:fid">
      Heusel, Ramsauer, Unterthiner, Nessler, &amp; Hochreiter.
      <em>GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.</em>
      NeurIPS 2017.
      <a href="https://arxiv.org/abs/1706.08500">arXiv:1706.08500</a>.
    </li>
    <li id="fn:kid">
      Bińkowski, Sutherland, Arbel, &amp; Gretton.
      <em>Demystifying MMD GANs.</em>
      ICLR 2018.
      <a href="https://arxiv.org/abs/1801.01401">arXiv:1801.01401</a>.
    </li>
    <li id="fn:fd-dino">
      Stein, Cresswell, Hosseinzadeh, Sui, Ross, Villecroze, Liu, Caterini, Taylor, &amp; Loaiza-Ganem.
      <em>Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models.</em>
      NeurIPS 2023.
      <a href="https://arxiv.org/abs/2306.04675">arXiv:2306.04675</a>.
    </li>
    <li id="fn:gans-equal">
      Lucic, Kurach, Michalski, Gelly, &amp; Bousquet.
      <em>Are GANs Created Equal? A Large-Scale Study.</em>
      NeurIPS 2018.
      <a href="https://arxiv.org/abs/1711.10337">arXiv:1711.10337</a>.
    </li>
    <li id="fn:unbiased-fid">
      Chong &amp; Forsyth.
      <em>Effectively Unbiased FID and Inception Score and where to find them.</em>
      CVPR 2020.
      <a href="https://arxiv.org/abs/1911.07023">arXiv:1911.07023</a>.
    </li>
    <li id="fn:aliased-resizing">
      Parmar, Zhang, &amp; Zhu.
      <em>On Aliased Resizing and Surprising Subtleties in GAN Evaluation.</em>
      CVPR 2022.
      <a href="https://arxiv.org/abs/2104.11222">arXiv:2104.11222</a>.
    </li>
    <li id="fn:fd-loss">
      Yang, Geng, Ju, Tian, &amp; Wang.
      <em>Representation Fréchet Loss for Visual Generation.</em>
      arXiv 2026.
      <a href="https://arxiv.org/abs/2604.28190">arXiv:2604.28190</a>.
    </li>
    <li id="fn:fid-lottery">
      Dufour, Efros, &amp; Pérez.
      <em>The FID Lottery: Quantifying Hidden Randomness in Generative-Model Evaluation.</em>
      arXiv 2026.
      <a href="https://arxiv.org/abs/2606.20536">arXiv:2606.20536</a>.
    </li>
    <li id="fn:diffusionbench">
      Leng, Singh, Liang, Smith, Bell, Saha, Yuan, &amp; Zheng.
      <em>DiffusionBench: On Holistic Evaluation of Diffusion Transformers.</em>
      arXiv 2026.
      <a href="https://arxiv.org/abs/2606.24888">arXiv:2606.24888</a>.
    </li>
  </ol>
</section>

<section class="end-section cite-section">
  <h4>Please Cite</h4>
  <p>If this post is useful for your work, please cite it as:</p>
  <pre class="bibtex"><code>@misc{wang2026struggling,
  title = {Struggling Toward Generative Model Evaluation},
  author = {Wang, Yifei},
  year = {2026},
  url = {https://a-little-hoof.github.io/blog/how-to-evaluate-your-generative-model/},
  note = {Blog post}
}</code></pre>
</section>
