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
reading_time_minutes: 34
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
.post-contents {
  margin: 1.45rem 0 2.1rem;
  padding-left: 1.65rem;
}
.post-contents,
.post-contents ol {
  line-height: 1.8;
}
.post-contents a {
  font-weight: 600;
  text-decoration: none;
}
.post-contents a:hover {
  text-decoration: underline;
}
.post-contents ol {
  margin: 0.15rem 0 0.25rem;
  padding-left: 1.35rem;
}
.benchmark-table {
  margin: 1.15rem 0 1.4rem;
  width: 100%;
}
.benchmark-table table {
  width: 100%;
  table-layout: fixed;
  font-size: 0.86rem;
  line-height: 1.35;
}
.benchmark-table th,
.benchmark-table td {
  vertical-align: top;
  padding: 0.48rem 0.5rem;
  overflow-wrap: anywhere;
}
.benchmark-table col.area-col {
  width: 13%;
}
.benchmark-table col.benchmark-col {
  width: 16%;
}
.benchmark-table col.prompt-col {
  width: 20%;
}
.benchmark-table col.evaluator-col {
  width: 26%;
}
.benchmark-table col.focus-col {
  width: 25%;
}
.benchmark-table .area-cell {
  font-weight: 700;
  color: #2b313a;
}
.ref-back {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.35em;
  height: 1.35em;
  margin-left: 0.35em;
  border-radius: 999px;
  color: #24788d;
  text-decoration: none;
  font-weight: 700;
  line-height: 1;
}
.ref-back:hover {
  background: #e9f4f6;
  text-decoration: none;
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
  .benchmark-table table {
    font-size: 0.78rem;
  }
  .benchmark-table th,
  .benchmark-table td {
    padding: 0.38rem 0.35rem;
  }
}
</style>

> Evaluating generative models is difficult not because current models fail obviously, but because many failures become subtle as generation quality improves. A model may produce realistic samples while still suffering from problems such as poor diversity, weak prompt adherence, or memorization. Therefore, the central challenge is not only determining whether a model works, but understanding which aspects of generation have actually improved.

This blog post traces the development of generative model evaluation: from the basic difficulty, to representation-based metrics, to FID and its many descendants, to benchmark-driven evaluation, and finally to the uncomfortable fact that no single number gets to be the truth.

Below is a clickable table of contents. Click any item to jump directly to that section.

<ol class="post-contents">
  <li><a href="#introduction">Introduction</a>
    <ol>
      <li><a href="#what-are-generative-models">What are generative models?</a></li>
      <li><a href="#why-is-evaluation-hard">Why is evaluation hard?</a></li>
    </ol>
  </li>
  <li><a href="#discriminative-models">Discriminative Models as Evaluators</a>
    <ol>
      <li><a href="#inception-score">Inception Score</a></li>
      <li><a href="#precision-and-recall">Precision and recall</a></li>
    </ol>
  </li>
  <li><a href="#one-score">One score that rules them all</a>
    <ol>
      <li><a href="#what-is-fid">What is FID?</a></li>
      <li><a href="#variants-of-fid">Variants of FID</a></li>
      <li><a href="#beyond-image-generation">Extending FID beyond image generation</a></li>
    </ol>
  </li>
  <li><a href="#problem-with-fid">The problem with FID</a>
    <ol>
      <li><a href="#fid-stability">How stable is FID?</a></li>
      <li><a href="#fid-unsuggestive">When FID score becomes unsuggestive</a></li>
    </ol>
  </li>
  <li><a href="#benchmarks">Benchmarks become the new standard</a>
    <ol>
      <li><a href="#what-benchmarks-does">What benchmarks do</a></li>
      <li><a href="#pass-k-coverage">Pass@k reveals distribution coverage</a></li>
    </ol>
  </li>
  <li><a href="#closing-thought">Closing thought</a></li>
  <li><a href="#references">References</a></li>
</ol>

## 1. Introduction
{: #introduction}

### 1.1 What are generative models?
{: #what-are-generative-models}

Creating noise from data is easy; creating data from noise is generative modeling.<sup class="footnote-ref"><a href="#fn:score-sde">1</a></sup> A generative model learns a distribution. In the simplest unconditional case, we want

$$
p_\theta(x) \approx p_{\mathrm{data}}(x).
$$

For conditional generation, the target is instead

$$
p_\theta(x \mid c) \approx p_{\mathrm{data}}(x \mid c),
$$

where $c$ might be a class label, text prompt, input image, pose, depth map, audio track, camera trajectory, or editing instruction. Sampling $x$ from a trained model $p_\theta$ is generation.

### 1.2 Why is evaluation hard?
{: #why-is-evaluation-hard}

The difficulty of evaluating the quality of generative models fundamentally stems from the fact that generation tasks do not have a single correct answer. For image reconstruction, the output can be directly compared with a ground-truth image, making pointwise metrics such as PSNR and SSIM reasonable choices.<sup class="footnote-ref"><a href="#fn:ssim">2</a></sup> In contrast, for image generation, there may be many plausible outputs under the same condition. Therefore, the evaluation objective is no longer to determine whether a single sample matches a standard answer, but rather to assess whether the generated distribution is close to the real data distribution.

<figure>
  <img src="/images/blog/evaluation/generation-no-single-answer.png" alt="Illustration contrasting reconstruction with a single target and generation with many plausible outputs" style="width: 100%; max-width: 760px; display: block; margin: 1.2rem auto 0.4rem;">
  <figcaption style="color: #66707a; font-size: 0.9rem; line-height: 1.5; text-align: center;">
    Reconstruction has a natural target. Generation has a distribution of plausible answers.
  </figcaption>
</figure>

#### Likelihood evaluation

Log-likelihood is a tempting evaluation criterion because a generative model is supposed to assign high probability to realistic data. For conditional generation, the analogous quantity is

$$
\log p_\theta(x \mid c),
$$

where $x$ is an image and $c$ is the text prompt. If likelihood were a good prompt-alignment metric, images with higher likelihood under a prompt should also receive higher CLIP scores for that prompt.

A simple SD3.5-Large experiment tests this directly. For each prompt, multiple images are generated, each image is scored by latent likelihood $\log p_\theta(x \mid c)$, and the same image is scored by CLIP. The relevant statistic is the within-prompt Spearman correlation between likelihood and CLIP score, because raw pooling across prompts is confounded by prompt-specific likelihood offsets.

| Images | Generation/scoring regime | Within-prompt Spearman: likelihood vs CLIP |
| :---: | :---: | :---: |
| 8 prompts x 16 images | without CFG | +0.109 (p=0.22) |
| 8 prompts x 16 images | with CFG | +0.104 (p=0.24) |
| 2 prompts x 96 images | without CFG | +0.078 (p=0.28) |
| 2 prompts x 96 images | with CFG | +0.203 (p=0.0047) |

The correlations are small. Even in the statistically positive setting, the effect size is weak. This means CLIP score and likelihood are measuring different things: CLIP measures text-image alignment in a learned representation space, while likelihood measures density under the model's latent distribution. Higher likelihood does not reliably imply better CLIP alignment.

Therefore, likelihood is useful as a density diagnostic, but it should not be treated as an automatic evaluation metric for prompt following or perceptual quality.

Human evaluation remains the most direct way to assess perceptual quality, but large-scale studies are expensive and difficult to reproduce. This motivates the development of automatic metrics that are scalable while still correlating well with human judgments.

## 2. Discriminative Models as Evaluators
{: #discriminative-models}

A common approach is to evaluate generative models in a learned representation space rather than directly in pixel space. Pixel-level distances are often poorly aligned with human perception: small changes in translation, cropping, lighting, or texture can produce large pixel differences even when the images remain perceptually similar. Moreover, generation is not a reconstruction problem. The goal is not to reproduce a specific target image, but to measure whether generated samples match the statistics and semantics of the real data distribution.

Pretrained discriminative models provide a practical way to construct such representation spaces. Instead of comparing raw pixels, generated and real images are mapped into feature embeddings, where distances are expected to reflect higher-level visual properties. However, the resulting evaluation is inherently dependent on the chosen representation: the feature extractor determines which aspects of similarity are captured and which are ignored.

<figure>
  <img src="/images/blog/evaluation/inception-v3-fid-is-pipeline.svg?v=1" alt="Schematic of the Inception-v3 architecture used as a feature extractor for generative model evaluation" style="width: 100%; max-width: 860px; display: block; margin: 1.2rem auto 0.4rem;">
  <figcaption style="color: #66707a; font-size: 0.9rem; line-height: 1.5; text-align: center;">
    Inception-v3 maps an image to both class probabilities and intermediate feature embeddings, which makes it useful for Inception Score and FID-style evaluation.
  </figcaption>
</figure>

Once images are represented as features, several evaluation strategies become possible. We can ask whether each generated image is confidently recognized by a classifier, which leads to Inception Score.<sup class="footnote-ref"><a href="#fn:is">3</a></sup> We can ask whether generated samples lie close to the real data manifold and whether they cover it, which leads to precision and recall.<sup class="footnote-ref"><a href="#fn:pr">4</a></sup> Later, we can compare the full real and generated feature distributions, which leads to FID.<sup class="footnote-ref"><a href="#fn:fid">7</a></sup>

This idea appears in many forms:

- use an ImageNet classifier to judge whether generated images look like recognizable objects.<sup class="footnote-ref"><a href="#fn:is">3</a></sup>
- use its intermediate features to compare real and generated distributions.<sup class="footnote-ref"><a href="#fn:fid">7</a></sup>
- use CLIP-like features to evaluate image-text alignment.<sup class="footnote-ref"><a href="#fn:clip">16</a></sup>
- use self-supervised features such as DINO for more general visual similarity.<sup class="footnote-ref"><a href="#fn:fd-dino">10</a></sup>
- use video encoders for temporal and motion-aware evaluation.<sup class="footnote-ref"><a href="#fn:fvd">17</a></sup>

The representation model becomes a measuring device. It is not the generator, but it defines what the evaluation can see.

### 2.1 Inception Score
{: #inception-score}

Inception Score<sup class="footnote-ref"><a href="#fn:is">3</a></sup> was one of the early popular representation-based metrics for image generation. It uses a pretrained Inception classifier and rewards two things:

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

### 2.2 Precision and recall
{: #precision-and-recall}

Precision and recall for generative models<sup class="footnote-ref"><a href="#fn:pr">4</a></sup> take the real data distribution into consideration.

- **Precision** asks whether generated samples lie near the real data manifold.
- **Recall** asks whether the generated distribution covers the real data manifold.

The original precision-recall-distribution method starts by embedding real and generated samples into a feature space, usually with a pretrained classifier. It then clusters the pooled embeddings into bins and estimates two discrete distributions: $p_b$ for real data and $q_b$ for generated data. By sweeping a slope parameter $\lambda$, it obtains a precision-recall curve:

$$
\mathrm{precision}(\lambda)
= \sum_b \min(\lambda p_b, q_b),
\qquad
\mathrm{recall}(\lambda)
= \sum_b \min(p_b, q_b / \lambda).
$$

<figure>
  <img src="/images/blog/evaluation/prd-clustering-bins-histograms.svg?v=1" alt="Schematic of precision-recall distributions using clustered feature-space bins and real/generated histograms" style="width: 100%; max-width: 840px; display: block; margin: 1.2rem auto 0.4rem;">
  <figcaption style="color: #66707a; font-size: 0.9rem; line-height: 1.5; text-align: center;">
    The original PRD method clusters pooled real and generated features, turns cluster assignments into two histograms, and computes precision-recall tradeoffs from the resulting discrete distributions.
  </figcaption>
</figure>

This curve is useful, but it depends on density estimates after clustering. The improved precision and recall metric<sup class="footnote-ref"><a href="#fn:improved-pr">5</a></sup> instead builds explicit non-parametric approximations of the real and generated manifolds in feature space. Let $R=\lbrace r_i \rbrace$ be real features and $G=\lbrace g_j \rbrace$ be generated features. For each real feature $r_i$, let $\rho_i$ be the distance from $r_i$ to its $k$th nearest neighbor among real features, and define the real manifold estimate as

$$
\mathcal{M}_R = \bigcup_i B(r_i, \rho_i).
$$

The generated manifold estimate $\mathcal{M}_G$ is defined the same way using generated features and their $k$th-neighbor radii. Precision and recall are then simple membership tests:

$$
\mathrm{precision}
= \frac{1}{|G|}
\sum_{g_j \in G}
\mathbf{1}[g_j \in \mathcal{M}_R],
\qquad
\mathrm{recall}
= \frac{1}{|R|}
\sum_{r_i \in R}
\mathbf{1}[r_i \in \mathcal{M}_G].
$$

<figure>
  <img src="/images/blog/evaluation/knn-manifold-estimation-precision-recall.svg?v=1" alt="Schematic of k-nearest-neighbor manifold estimation for improved precision and recall" style="width: 100%; max-width: 840px; display: block; margin: 1.2rem auto 0.4rem;">
  <figcaption style="color: #66707a; font-size: 0.9rem; line-height: 1.5; text-align: center;">
    Improved precision and recall approximate the data manifold with adaptive kNN balls. Precision checks generated samples against the real manifold estimate; recall swaps the roles and checks real samples against the generated manifold estimate.
  </figcaption>
</figure>

So high precision means most generated samples fall inside the real-data manifold estimate. High recall means the generated manifold covers most real-data samples. The important point is that the two numbers separate sample realism from distribution coverage instead of forcing both questions into one scalar.

## 3. One score that rules them all
{: #one-score}

FID, introduced in the TTUR paper<sup class="footnote-ref"><a href="#fn:fid">7</a></sup>, later became the de facto standard metric for evaluating image generative models.

### 3.1 What is FID?
{: #what-is-fid}

Frechet Inception Distance compares real and generated samples in Inception feature space. Instead of asking whether individual samples are classified confidently, it fits a Gaussian to the real features and another Gaussian to the generated features:

$$
\mathrm{FID} =
\lVert \mu_r - \mu_g \rVert_2^2
+
\mathrm{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2}).
$$

Here $(\mu_r, \Sigma_r)$ are the mean and covariance of real features, and $(\mu_g, \Sigma_g)$ are the mean and covariance of generated features. Lower is better. Three characteristics of FID are worth keeping in mind:

1. **Finite-sample estimate.** FID is not computed from the full model distribution. In practice, people usually generate 50k samples and use their features to represent the generated distribution. This makes FID easy to standardize, but it also means the score can depend on sample count, random seed, and which finite sample set happens to be evaluated.
2. **Gaussian assumption.** FID reduces each feature distribution to a mean and covariance, then computes the distance between the two fitted Gaussians. This makes the metric simple and stable to report, but it also means that any structure beyond second-order statistics is invisible to the score.
3. **Fréchet Distance.** After the Gaussian approximation, FID uses the Fréchet distance between the two Gaussian distributions, also known as the 2-Wasserstein distance in this setting. The mean term measures how far the centers of the real and generated feature clouds are from each other, while the covariance term measures how different their spreads and correlations are.

### 3.2 Variants of FID
{: #variants-of-fid}

<strong>FD<sub>&infin;</sub></strong><sup class="footnote-ref"><a href="#fn:unbiased-fid">8</a></sup> challenges the finite-sample behavior of FID. In practice, FID is computed from a finite number of real and generated samples, so the estimated means and covariances introduce bias. The idea of FD<sub>&infin;</sub> is to view the reported score as a finite-sample estimate $\mathrm{FD}_N$ and extrapolate toward the value that would be obtained with infinitely many samples:

$$
\mathrm{FD}_\infty = \lim_{N \to \infty} \mathrm{FD}_N.
$$

The conclusion is stronger than "FID needs enough samples." Finite-sample FID is biased, and the bias depends on the model being evaluated. Therefore, fixing the protocol at 50k samples does not fully remove the issue: one model may look better partly because its finite-sample bias is smaller. FD<sub>&infin;</sub> improves the estimator, but it does not change the underlying assumptions of FID: Gaussian feature distributions and a fixed representation space.

**Kernel Inception Distance**, or **KID**<sup class="footnote-ref"><a href="#fn:kid">9</a></sup>, challenges the Gaussian assumption behind FID. It keeps the same broad idea of comparing real and generated samples in Inception feature space, but replaces the Gaussian Fréchet distance with a kernel maximum mean discrepancy (MMD). If $u=\phi(x)$ is a real feature and $v=\phi(\tilde{x})$ is a generated feature, then the squared MMD is

$$
\mathrm{MMD}^2(P, Q)
=
\mathbb{E}_{u,u' \sim P}[k(u,u')]
-
2\mathbb{E}_{u \sim P, v \sim Q}[k(u,v)]
+
\mathbb{E}_{v,v' \sim Q}[k(v,v')].
$$

KID usually uses a polynomial kernel on Inception features:

$$
k(u,v)=
\left(
\frac{1}{d}u^\top v + 1
\right)^3,
$$

where $d$ is the feature dimension. Given real features $\lbrace u_i \rbrace_{i=1}^m$ and generated features $\lbrace v_j \rbrace_{j=1}^n$, an unbiased finite-sample estimator is

$$
\widehat{\mathrm{KID}}
=
\frac{1}{m(m-1)}
\sum_{i \ne i'} k(u_i,u_{i'})
-
\frac{2}{mn}
\sum_{i,j} k(u_i,v_j)
+
\frac{1}{n(n-1)}
\sum_{j \ne j'} k(v_j,v_{j'}).
$$

This unbiased estimator is one reason KID is useful when the number of evaluation samples is limited, while FID can be biased for finite sample sizes.

**FD-DINO**<sup class="footnote-ref"><a href="#fn:fd-dino">10</a></sup> challenges a different part of FID: the feature extractor. Standard FID computes Fréchet distance in Inception feature space, so the metric inherits the priorities and blind spots of a supervised ImageNet classifier. FD-DINO keeps the same Fréchet-distance calculation, but replaces Inception features with DINO or DINOv2 representations.

<figure>
  <img src="/images/blog/evaluation/fd-dino-feature-extraction.svg?v=2" alt="Schematic comparing Inception feature extraction with DINO feature extraction for Fréchet distance evaluation" style="width: 100%; max-width: 840px; display: block; margin: 1.2rem auto 0.4rem;">
  <figcaption style="color: #66707a; font-size: 0.9rem; line-height: 1.5; text-align: center;">
    FD-DINO keeps the Fréchet-distance calculation but changes the representation space from supervised Inception features to DINO-style self-supervised visual features.
  </figcaption>
</figure>

The motivation is that self-supervised visual features can capture semantic similarity without being as tightly tied to ImageNet classification labels. In this view, the distance formula is not the only thing that matters; the representation space defines what the metric can see. If DINO places semantically similar images closer together, then a Fréchet distance in DINO feature space may better reflect perceptual similarity for modern generators, especially diffusion models.

FD-DINO is therefore less a new distance than a reminder that evaluation metrics are pipelines. Changing the encoder can change the meaning of the same mathematical score.

### 3.3 Extending FID beyond image generation
{: #beyond-image-generation}

The same idea has been used far beyond unconditional image generation. In text-to-image evaluation, **MS COCO**<sup class="footnote-ref"><a href="#fn:coco">18</a></sup> became a common reference point because it contains natural images paired with human-written captions. The common zero-shot setting is usually called **FID-30K** on COCO: sample 30,000 captions from the MS COCO 2014 validation split, generate one image for each caption, and compare those 30,000 generated images against the reference images from the full COCO validation set, which contains 40,504 images.<sup class="footnote-ref"><a href="#fn:imagen">19</a></sup> This makes the result easy to compare across papers, but it is still a distribution-level image metric: a model can improve COCO FID without necessarily improving prompt faithfulness for every individual caption.

For video, the analogous metric is often **FVD**, Fréchet Video Distance<sup class="footnote-ref"><a href="#fn:fvd">17</a></sup>. There is no single COCO-like prompt set for FVD, because the reference set depends on the video task: video prediction, unconditional video generation, text-to-video generation, or domain-specific generation may all use different datasets. In the original FVD paper, the authors evaluate on datasets such as BAIR robot pushing, KTH actions, and their StarCraft 2 Videos benchmark. They use I3D video features, especially the logits of an I3D model trained on Kinetics-400, and report FVD with 256 validation samples for BAIR and 1024 samples for KTH and SCV. Many later video-generation papers use 2048 real and generated clips, so the number of clips, clip length, resolution, and feature backbone should always be reported together with the score.

Instead of embedding single images with Inception, FVD embeds short video clips with a video understanding network, then computes the same Gaussian Fréchet distance in that video feature space:

$$
\mathrm{FVD}
=
\lVert \mu_r - \mu_g \rVert_2^2
+
\mathrm{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2}).
$$

Here the means and covariances are computed from video features rather than image features. This matters because video generation is not only about whether each frame looks realistic. The model also has to maintain identity, geometry, motion, and temporal coherence across frames, so a frame-level FID can miss failures such as flicker, frozen motion, or inconsistent object dynamics.

## 4. The problem with FID
{: #problem-with-fid}

Despite its success, FID has important limitations. For years, generative model evaluation often followed a simple pipeline: train a model, generate samples, compute FID, and use the score as the primary basis for comparison. As generative models have improved, these limitations have become increasingly apparent: a single distribution-level metric cannot fully capture the different dimensions of generation quality.

### 4.1 How stable is FID?
{: #fid-stability}

The first problem is stability. FID can move for reasons that have little to do with the conceptual quality of the generator.

#### 4.1.1 Randomness in training

Random seeds, data order, initialization, augmentation randomness, and hardware-level nondeterminism can all change the final model. If the reported improvement is small, it may be hard to know whether the method improved the generator or simply landed on a better run.

The FID Lottery makes this problem explicit.<sup class="footnote-ref"><a href="#fn:fid-lottery">13</a></sup> The paper treats FID as a random variable over two axes: the training seed used to obtain the model and the generation seed used to draw the evaluation samples. On class-conditional ImageNet 256x256, it finds that retraining the same recipe with a different seed moves FID much more than simply resampling from a fixed network. The spread comes from ordinary sources of training randomness, including initialization, data order, and per-step Gaussian noise in the flow-matching loss. The conclusion is practical: a single FID number from one model checkpoint is a lottery ticket. Small gaps should be treated as inconclusive unless they are larger than the measured variation, and strong claims should report error bars over multiple training seeds.

| Condition | N | Between-seed std | Within-seed std | Between-seed CoV | Range | Share of vary-all |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Vary all | 25 | 0.438 | 0.137 | 1.26% | 1.66 | 100.0% |
| Vary noise | 25 | 0.336 | 0.144 | 0.97% | 1.33 | 76.7% |
| Vary initialization | 25 | 0.294 | 0.150 | 0.85% | 1.09 | 67.1% |
| Vary data order | 24 | 0.221 | 0.150 | 0.64% | 0.82 | 50.5% |

The between-seed standard deviation is about three times larger than the within-seed sampling noise, so evaluating more samples from one checkpoint cannot remove the dominant uncertainty. The model itself has to be retrained under multiple seeds.

#### 4.1.2 Differences in evaluation pipeline

FID is also sensitive to image processing details:

- the exact feature extractor;
- image resolution and preprocessing;
- resizing and cropping implementation;
- color space and normalization;
- image format and compression;
- choice of reference set.

This matters because a reported FID is not just a property of the model. It is a property of the whole evaluation pipeline.

Parmar, Zhang, and Zhu make this point concrete in *On Aliased Resizing and Surprising Subtleties in GAN Evaluation*.<sup class="footnote-ref"><a href="#fn:aliased-resizing">12</a></sup> Their paper shows that low-level preprocessing choices such as resizing and compression can create surprisingly large FID changes. The key issue is aliasing: when images are downsampled, the prefilter should change with the downsampling factor, but common bilinear and bicubic resizing implementations often use a fixed-width filter. That can introduce aliasing artifacts, which then corrupt the Inception features used by FID. They also show that JPEG compression can affect the downstream score; if the real training images are compressed, compressing the generated images can even improve FID without making the generator better. The lesson is simple: FID is not only measuring the generator. It is also measuring the hidden image-processing pipeline around the generator.

The reference set is another hidden source of variation. FID compares generated samples to a chosen set of real-image features, so changing that real set changes the target distribution. This is not just a constant offset. In one ImageNet-256 check, the exact same generated samples were scored against three reference sets: 50k validation images, a random 50k subset of training images, and a larger training-set reference. Moving from the validation reference to a training-set reference changed different models in different directions: for one model, FID improved because its samples were closer to the training distribution; for another, FID worsened because its samples happened to sit closer to the validation distribution.

| Input space | Val-50k | Train-50k | Large train reference |
| :---: | :---: | :---: | :---: |
| RAE<sup class="footnote-ref"><a href="#fn:rae">33</a></sup> | 4.295 | 4.008 | 3.535 |
| PAE<sup class="footnote-ref"><a href="#fn:pae">34</a></sup> | 2.976 | 3.539 | 2.889 |
| VA-VAE<sup class="footnote-ref"><a href="#fn:vavae">35</a></sup> | 4.204 | 5.524 | 4.859 |

This example separates two effects. First, reference size matters: going from 50k training images to a much larger training reference lowered FID consistently. Second, reference distribution matters: validation images and training images are not the same target, and the shift is model-dependent. The consequence can be severe enough to flip rankings. In the table above, VA-VAE looks slightly better than RAE against the validation reference, but RAE looks better against the training-based references. Therefore, a paper should not only report "FID"; it should say exactly which reference statistics were used.

### 4.2 When FID score becomes unsuggestive
{: #fid-unsuggestive}

The second problem is meaning. Today, many ImageNet generation papers compete over very small FID gaps, sometimes smaller than 0.2. At that scale, the number can look precise while saying very little about what changed perceptually: did the model become more diverse, more faithful, more useful, or simply better aligned with the evaluation pipeline?

#### 4.2.1 Do FID gains on ImageNet transfer?

FID is a feature-space metric. If the feature extractor cannot see a failure, FID may not punish it. Inception features are good for many natural-image semantics, but they are not a universal model of human perception. They may underweight counting, spatial relations, text rendering, fine-grained attributes, unusual styles, or domain-specific details.

<a href="https://end2end-diffusion.github.io/diffusion-bench/">DiffusionBench</a> makes this transfer problem concrete.<sup class="footnote-ref"><a href="#fn:diffusionbench">14</a></sup> The paper argues that DiT research has become overly centered on class-conditional ImageNet, where methods are often compared by FID and nearby distribution metrics. To test whether that progress transfers, it introduces NanoGen, a unified training and evaluation framework that can train comparable ImageNet and text-to-image DiT models with only a small configuration change. After training 21 latent diffusion models across VAE, RAE, pixel-space, and MeanFlow-style settings, the paper finds no strong positive relationship between ImageNet performance and text-to-image performance. In fact, the reported Pearson correlations between ImageNet and T2I rankings are negative, between -0.377 and -0.580 across three metrics.

<figure>
  <img src="/images/blog/evaluation/diffusionbench-imagenet-t2i-correlation.png?v=1" alt="DiffusionBench scatter plots comparing ImageNet FID with GenEval, DPG-Bench, and GenAIBench scores" style="width: 100%; max-width: 860px; display: block; margin: 1.2rem auto 0.4rem;">
  <figcaption style="text-align: center; color: #66707c; font-size: 0.92rem;">ImageNet FID does not strongly predict text-to-image benchmark performance. Figure taken from <a href="https://end2end-diffusion.github.io/diffusion-bench/">DiffusionBench</a>.</figcaption>
</figure>

The conclusion is not that ImageNet FID is useless. It is that ImageNet FID is local evidence. A method can improve class-conditional ImageNet generation without improving prompt-following, compositionality, or the behavior users actually care about in text-to-image systems. If the claim is broad progress in generative modeling, ImageNet FID alone is too narrow a witness.

#### 4.2.2 When Small FID Gaps Stop Meaning Much

The FID Lottery already shows one reason small gaps are dangerous: retraining the same recipe with different random seeds can move FID more than the improvement claimed by many papers.<sup class="footnote-ref"><a href="#fn:fid-lottery">13</a></sup> Even before asking whether the metric reflects human preference, we have to ask whether the reported gap is larger than the noise induced by training randomness, sampling randomness, and evaluation details.

Even if the model and reference distribution are fixed, a finite 50k reference has its own sampling variance. In 100 independent random 50k draws from ImageNet training images, the estimated FID floor varied as follows:

| Reference | n | Mean | Std | Variance | Range |
| :---: | :---: | :---: | :---: | :---: | :---: |
| vs Val | 100 | 2.223 | 0.029 | 0.00084 | [2.137, 2.276] |
| vs VIRTUAL | 100 | 0.857 | 0.011 | 0.00013 | [0.831, 0.880] |
| vs JiT | 100 | 0.754 | 0.011 | 0.00012 | [0.726, 0.777] |

The variance is small, but not zero. Around model FIDs of 3 to 5, this corresponds to fluctuations on the order of a few hundredths. Combined with the FID Lottery result, this makes tiny FID gaps hard to interpret. A difference of 0.05, 0.1, or even 0.2 may reflect the seed, the reference set, preprocessing, or sample count more than a meaningful change in generative behavior. In this regime, a small FID gap cannot suggest much by itself.

#### 4.2.3 FID Is Weaker in Conditional Domains

The problem becomes sharper once generation becomes conditional. Text-to-image is the simplest example. In class-conditional ImageNet generation, the condition is a single label, and FID can at least ask whether the generated image distribution looks like the ImageNet distribution. In text-to-image, the condition is open-ended language. The model has to follow arbitrary prompts faithfully and produce visually convincing images at the same time. <a href="https://peppaking8.github.io/#/post/minit2i">MiniT2I</a> makes this distinction explicit: it reports MSCOCO-30K FID as a distribution-level realism metric, but notes that this metric does not ask whether the generated image actually followed the prompt.<sup class="footnote-ref"><a href="#fn:minit2i">36</a></sup>

<figure>
  <img src="/images/blog/evaluation/minit2i-t2i-examples.png?v=1" alt="MiniT2I examples showing generated images with their text prompts" style="width: 100%; max-width: 860px; display: block; margin: 1.2rem auto 0.4rem;">
  <figcaption style="text-align: center; color: #66707c; font-size: 0.92rem;">Text-to-image evaluation must ask whether the image follows the prompt, not only whether it looks realistic. Examples from <a href="https://peppaking8.github.io/#/post/minit2i">MiniT2I</a>.</figcaption>
</figure>

This is exactly where FID becomes weak. A generated image can be sharp, natural, and close to the COCO image distribution while still missing the requested object, count, spatial relation, text, style, or rare concept. Conversely, an image can follow a strange prompt faithfully while looking less typical under an Inception feature distribution. For T2I, the central question is not only "does this look like a real image?" It is also "does this image satisfy this sentence?" FID only sees the first question.

## 5. Benchmarks become the new standard
{: #benchmarks}

This is why text-to-image evaluation quickly moves beyond FID toward prompt-following benchmarks, VQA-based checks, OCR, image-text alignment, and human preference. FID can still be useful as a realism signal, but for T2I it is no longer enough to describe the model's behavior.

### 5.1 What benchmarks do
{: #what-benchmarks-does}

Instead of asking only whether generated samples match a broad reference distribution, benchmarks ask whether models solve curated tasks. In practice, a benchmark usually contains a prompt suite, a generation protocol, and an automatic evaluator. The evaluator is often another pretrained model: an object detector, segmentation model, VQA model, captioning model, CLIP-like encoder, video encoder, optical-flow estimator, or MLLM judge. So the score is not an abstract measurement of "quality"; it is a task-specific measurement of whether the generated output satisfies what the benchmark and its verifier know how to check.

The table below is not exhaustive, but it shows how the evaluator changes with the task.

<div class="benchmark-table">
  <table>
    <colgroup>
      <col class="area-col">
      <col class="benchmark-col">
      <col class="prompt-col">
      <col class="evaluator-col">
      <col class="focus-col">
    </colgroup>
    <thead>
      <tr>
        <th>Area</th>
        <th>Benchmark</th>
        <th>Prompt set</th>
        <th>Automatic evaluator / detector</th>
        <th>Main evaluation focus</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td rowspan="5" class="area-cell">Text-to-image</td>
        <td>GenEval<sup class="footnote-ref"><a href="#fn:geneval">20</a></sup></td>
        <td>553 prompts over six object-centric tasks</td>
        <td>Mask2Former instance segmentation; CLIP ViT-L/14 color classifier</td>
        <td>Object presence, counting, relative position, color, and attribute binding</td>
      </tr>
      <tr>
        <td>DPG-Bench<sup class="footnote-ref"><a href="#fn:dpg-bench">21</a></sup></td>
        <td>1,065 long dense prompts; four images per prompt</td>
        <td>mPLUG-large MLLM judge over DSG-style questions and graphs</td>
        <td>Dense prompt following with multiple objects, attributes, and relationships</td>
      </tr>
      <tr>
        <td>GenEval 2<sup class="footnote-ref"><a href="#fn:geneval2">22</a></sup></td>
        <td>800 prompts with varying compositionality</td>
        <td>Soft-TIFA, a VQA-based atom-level evaluator</td>
        <td>Primitive concepts, counts, relations, compositionality, and benchmark drift</td>
      </tr>
      <tr>
        <td>GenAI-Bench<sup class="footnote-ref"><a href="#fn:genai-bench">23</a></sup></td>
        <td>1,600 real-world prompts; 800-prompt video coreset</td>
        <td>VQAScore and human ratings</td>
        <td>Text-to-visual alignment for basic and advanced compositional reasoning</td>
      </tr>
      <tr>
        <td>T2I-CompBench<sup class="footnote-ref"><a href="#fn:t2i-compbench">24</a></sup></td>
        <td>6,000 prompts across three categories and six subcategories</td>
        <td>BLIP-VQA, UniDet, CLIPScore, 3-in-1 metric, MiniGPT-4-CoT</td>
        <td>Attribute binding, spatial/non-spatial relations, and complex compositions</td>
      </tr>
      <tr>
        <td rowspan="2" class="area-cell">Video</td>
        <td>VBench<sup class="footnote-ref"><a href="#fn:vbench">25</a></sup></td>
        <td>946 dimension prompts, plus 800 category prompts; typically five videos per prompt</td>
        <td>DINO, CLIP, RAFT, LAION aesthetic predictor, MUSIQ, GRiT, UMT, Tag2Text, ViCLIP</td>
        <td>Fine-grained video quality and video-condition consistency across 16 dimensions</td>
      </tr>
      <tr>
        <td>T2V-CompBench<sup class="footnote-ref"><a href="#fn:t2v-compbench">26</a></sup></td>
        <td>1,400 prompts across seven compositional categories</td>
        <td>MLLM-based, detection-based, and tracking-based metrics</td>
        <td>Attribute binding, motion binding, action binding, interactions, and numeracy in video</td>
      </tr>
      <tr>
        <td rowspan="2" class="area-cell">Editing</td>
        <td>I2EBench<sup class="footnote-ref"><a href="#fn:i2ebench">27</a></sup></td>
        <td>2,000+ images and 4,000+ original/diverse instructions</td>
        <td>GPT-4V for high-level edits; CLIP/SSIM for style and low-level edits</td>
        <td>Instruction-based image editing across 16 high- and low-level dimensions</td>
      </tr>
      <tr>
        <td>GIE-Bench<sup class="footnote-ref"><a href="#fn:gie-bench">28</a></sup></td>
        <td>1,000+ editing tasks from 800+ images across 20 categories</td>
        <td>VQA-style multiple-choice questions; Grounded SAM masks; object-aware preservation score</td>
        <td>Functional correctness and content preservation in non-edited regions</td>
      </tr>
      <tr>
        <td rowspan="2" class="area-cell">3D</td>
        <td>T3Bench<sup class="footnote-ref"><a href="#fn:t3bench">29</a></sup></td>
        <td>300 prompts across single object, surroundings, and multi-object scenes</td>
        <td>Multi-view rendering with CLIP/ImageReward quality scoring; captioning plus GPT-4 alignment scoring</td>
        <td>Text-to-3D quality, text alignment, and view consistency</td>
      </tr>
      <tr>
        <td>3DGen-Bench<sup class="footnote-ref"><a href="#fn:3dgen-bench">30</a></sup></td>
        <td>1,020 text/image prompts; 11,220 generated 3D assets</td>
        <td>3DGen-Score and 3DGen-Eval trained from human preferences</td>
        <td>Geometry plausibility, geometry detail, texture quality, geometry-texture coherence, prompt-asset alignment</td>
      </tr>
      <tr>
        <td class="area-cell">Multimodal</td>
        <td>MMMU<sup class="footnote-ref"><a href="#fn:mmmu">31</a></sup></td>
        <td>11.5K multimodal questions across six disciplines and 30 subjects</td>
        <td>Multiple-choice / open-ended answer accuracy over expert problems</td>
        <td>Reasoning across images, charts, diagrams, tables, text, and domain knowledge</td>
      </tr>
    </tbody>
  </table>
</div>

Benchmarks represent an important shift toward more diagnostic evaluation. Rather than relying only on global distribution metrics, they define specific tasks that expose concrete failure modes and better reflect practical requirements.

However, benchmark-based evaluation introduces its own biases. GenEval provides a useful example: by measuring object presence, counting, spatial relationships, color, and attribute binding through a fixed detector-and-classifier pipeline, it makes text-to-image failures more explicit. At the same time, this fixed evaluation protocol creates opportunities for benchmark-specific optimization. For example, a model trained on BLIP-3o data achieves a GenEval score of 0.67, higher than SD3.5-M's score of 0.63. This improvement may indicate better prompt following, but it may also partially reflect adaptation to GenEval's prompt distribution and evaluator.

### 5.2 Pass@k reveals distribution coverage
{: #pass-k-coverage}

Benchmark scores usually ask whether one generated output follows the prompt, satisfies a verifier, or passes a task-specific check. In that sense, they are often alignment metrics: they tell us whether a sample matches what the prompt or benchmark asks for.

Pass@k pushes this idea closer to distribution-level evaluation by asking how the score changes as we draw more samples from the same model. A single benchmark score mainly measures the success probability of a typical draw; pass@k asks whether the model's conditional distribution contains many different successful outputs, or whether probability mass has concentrated on a narrow set of evaluator-favored samples.<sup class="footnote-ref"><a href="#fn:pass-k-diffusion">32</a></sup>

<figure>
  <img src="/images/blog/evaluation/sd35-turbo-passk-curves.png?v=2" alt="Pass@k curves comparing SD3.5-Large and SD3.5-L-Turbo on GenEval, GenEval2, and DPG-Bench" style="width: 100%; max-width: 920px; display: block; margin: 1.2rem auto 0.4rem;">
  <figcaption style="color: #66707a; font-size: 0.9rem; line-height: 1.5; text-align: center;">
    SD3.5-Large and SD3.5-L-Turbo pass@k curves. Turbo improves low-k success, while the multi-step teacher catches up as the sample budget grows.
  </figcaption>
</figure>

For each prompt, generate $n$ independent samples and let $c$ be the number that pass the benchmark verifier. The pass@k estimator asks whether at least one sample in a size-$k$ subset succeeds:

$$
\mathrm{pass@}k
=
\mathbb{E}_{\mathrm{prompts}}
\left[
1 -
\frac{\binom{n-c}{k}}{\binom{n}{k}}
\right].
$$

Pass@1 recovers the ordinary single-sample benchmark score under the same evaluator. Larger $k$ asks a different question: if we sample more from the same model, does the distribution contain more successful generations for the prompt? If a method improves pass@1 but the pass@k curve saturates early, the method may be improving one-shot sampling efficiency or prompt-following alignment while reducing distribution coverage. If the curve keeps rising at larger $k$, it suggests that the model assigns probability to a broader set of successful outputs.

## 6. Closing thought
{: #closing-thought}

So we end up with a layered evaluation stack:

| Layer | What it helps answer | Typical failure |
| --- | --- | --- |
| Likelihood and pointwise metrics | Does the model assign probability to real data, or match a reference output? | weak alignment with perception; generation has many valid answers |
| Discriminative features | Do samples look recognizable in a pretrained representation space? | inherits the feature extractor's blind spots |
| IS and precision/recall | Are samples confident, realistic, and covering the real-data manifold? | separates quality and coverage, but still depends on representation space |
| FID and FD variants | Are real and generated feature distributions close? | sensitive to Gaussian assumptions, sample size, preprocessing, and domain |
| Task benchmarks | Does the model satisfy specific prompt-following, video, editing, 3D, or multimodal checks? | can be hacked, overfit, or limited by the verifier |
| Pass@k | Does success grow as we draw more samples from the same conditional distribution? | depends on verifier choice and sampling budget |

The history of generative model evaluation is a history of proxies. Pixels were too brittle, so we moved to representations. Inception Score was too indirect, so we moved to FID. FID was too coarse, so we built variants with better features and video encoders. Distribution metrics missed conditional behavior, so we built benchmarks. Benchmarks became targets, so we added human evaluation and failure audits.

Each step fixes something and exposes something else.

No single metric captures all aspects of generative model quality. Different evaluation methods measure different properties: distribution-level metrics assess similarity to the data distribution, precision and recall diagnose fidelity and coverage, benchmarks test specific capabilities, and qualitative analysis reveals failure modes that automatic metrics may miss.

Future evaluation will likely require a combination of complementary metrics rather than a single universal score. Automatic metrics can improve scalability and reproducibility, but they remain proxies for the properties we ultimately care about. The challenge is not to find one perfect metric, but to design evaluation protocols that reveal why one model performs better than another.

<section class="references-section" id="references">
  <h4>References</h4>
  <ol>
    <li id="fn:score-sde">
      Song, Sohl-Dickstein, Kingma, Kumar, Ermon, &amp; Poole.
      <em>Score-Based Generative Modeling through Stochastic Differential Equations.</em>
      ICLR 2021.
      <a href="https://arxiv.org/abs/2011.13456">arXiv:2011.13456</a>.
    </li>
    <li id="fn:ssim">
      Wang, Bovik, Sheikh, &amp; Simoncelli.
      <em>Image quality assessment: from error visibility to structural similarity.</em>
      IEEE Transactions on Image Processing, 13(4):600-612, 2004.
      <a href="https://doi.org/10.1109/TIP.2003.819861">doi:10.1109/TIP.2003.819861</a>.
    </li>
    <li id="fn:is">
      Salimans, Goodfellow, Zaremba, Cheung, Radford, &amp; Chen.
      <em>Improved Techniques for Training GANs.</em>
      NeurIPS 2016.
      <a href="https://arxiv.org/abs/1606.03498">arXiv:1606.03498</a>.
    </li>
    <li id="fn:pr">
      Sajjadi, Bachem, Lucic, Bousquet, &amp; Gelly.
      <em>Assessing Generative Models via Precision and Recall.</em>
      NeurIPS 2018.
      <a href="https://arxiv.org/abs/1806.00035">arXiv:1806.00035</a>.
    </li>
    <li id="fn:improved-pr">
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
    <li id="fn:unbiased-fid">
      Chong &amp; Forsyth.
      <em>Effectively Unbiased FID and Inception Score and where to find them.</em>
      CVPR 2020.
      <a href="https://arxiv.org/abs/1911.07023">arXiv:1911.07023</a>.
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
    <li id="fn:clip">
      Radford, Kim, Hallacy, Ramesh, Goh, Agarwal, Sastry, Askell, Mishkin, Clark, Krueger, &amp; Sutskever.
      <em>Learning Transferable Visual Models From Natural Language Supervision.</em>
      ICML 2021.
      <a href="https://arxiv.org/abs/2103.00020">arXiv:2103.00020</a>.
    </li>
    <li id="fn:fvd">
      Unterthiner, van Steenkiste, Kurach, Marinier, Michalski, &amp; Gelly.
      <em>Towards Accurate Generative Models of Video: A New Metric &amp; Challenges.</em>
      arXiv 2018.
      <a href="https://arxiv.org/abs/1812.01717">arXiv:1812.01717</a>.
    </li>
    <li id="fn:coco">
      Lin, Maire, Belongie, Bourdev, Girshick, Hays, Perona, Ramanan, Zitnick, &amp; Dollár.
      <em>Microsoft COCO: Common Objects in Context.</em>
      ECCV 2014.
      <a href="https://arxiv.org/abs/1405.0312">arXiv:1405.0312</a>.
    </li>
    <li id="fn:imagen">
      Saharia, Chan, Saxena, Li, Whang, Denton, Ghasemipour, Gontijo Lopes, Karagol Ayan, Salimans, Ho, Fleet, &amp; Norouzi.
      <em>Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding.</em>
      NeurIPS 2022.
      <a href="https://papers.nips.cc/paper_files/paper/2022/hash/ec795aeadae0b7d230fa35cbaf04c041-Abstract-Conference.html">NeurIPS paper</a>.
    </li>
    <li id="fn:geneval">
      Ghosh, Hajishirzi, &amp; Schmidt.
      <em>GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment.</em>
      NeurIPS 2023.
      <a href="https://arxiv.org/abs/2310.11513">arXiv:2310.11513</a>.
    </li>
    <li id="fn:dpg-bench">
      Hu, Wang, Fang, Fu, Cheng, &amp; Yu.
      <em>ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment.</em>
      ECCV 2024.
      <a href="https://arxiv.org/abs/2403.05135">arXiv:2403.05135</a>.
    </li>
    <li id="fn:geneval2">
      Kamath, Chang, Krishna, Zettlemoyer, Hu, &amp; Ghazvininejad.
      <em>GenEval 2: Addressing Benchmark Drift in Text-to-Image Evaluation.</em>
      arXiv 2025.
      <a href="https://arxiv.org/abs/2512.16853">arXiv:2512.16853</a>.
    </li>
    <li id="fn:genai-bench">
      Li, Lin, Pathak, Li, Fei, Wu, Ling, Xia, Zhang, Neubig, &amp; Ramanan.
      <em>GenAI-Bench: Evaluating and Improving Compositional Text-to-Visual Generation.</em>
      CVPR Workshops 2024.
      <a href="https://arxiv.org/abs/2406.13743">arXiv:2406.13743</a>.
    </li>
    <li id="fn:t2i-compbench">
      Huang, Sun, Xie, Li, &amp; Liu.
      <em>T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation.</em>
      NeurIPS 2023.
      <a href="https://arxiv.org/abs/2307.06350">arXiv:2307.06350</a>.
    </li>
    <li id="fn:vbench">
      Huang, He, Yu, Zhang, Si, Jiang, Zhang, Wu, Jin, Chanpaisit, Wang, Chen, Wang, Lin, Qiao, &amp; Liu.
      <em>VBench: Comprehensive Benchmark Suite for Video Generative Models.</em>
      CVPR 2024.
      <a href="https://arxiv.org/abs/2311.17982">arXiv:2311.17982</a>.
    </li>
    <li id="fn:t2v-compbench">
      Sun, Huang, Liu, Wu, Xu, Li, &amp; Liu.
      <em>T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation.</em>
      CVPR 2025.
      <a href="https://arxiv.org/abs/2407.14505">arXiv:2407.14505</a>.
    </li>
    <li id="fn:i2ebench">
      Ma, Ji, Ye, Lin, Wang, Zheng, Zhou, Sun, &amp; Ji.
      <em>I2EBench: A Comprehensive Benchmark for Instruction-based Image Editing.</em>
      NeurIPS 2024.
      <a href="https://arxiv.org/abs/2408.14180">arXiv:2408.14180</a>.
    </li>
    <li id="fn:gie-bench">
      Qian, Lu, Fu, Wang, Chen, Yang, Hu, &amp; Gan.
      <em>GIE-Bench: Towards Grounded Evaluation for Text-Guided Image Editing.</em>
      arXiv 2025.
      <a href="https://arxiv.org/abs/2505.11493">arXiv:2505.11493</a>.
    </li>
    <li id="fn:t3bench">
      He, Bai, Lin, Zhao, Hu, Sheng, Yi, Li, &amp; Liu.
      <em>T3Bench: Benchmarking Current Progress in Text-to-3D Generation.</em>
      arXiv 2023.
      <a href="https://arxiv.org/abs/2310.02977">arXiv:2310.02977</a>.
    </li>
    <li id="fn:3dgen-bench">
      Zhang, Zhang, Wu, Wang, Wetzstein, Lin, &amp; Liu.
      <em>3DGen-Bench: Comprehensive Benchmark Suite for 3D Generative Models.</em>
      arXiv 2025.
      <a href="https://arxiv.org/abs/2503.21745">arXiv:2503.21745</a>.
    </li>
    <li id="fn:mmmu">
      Yue et al.
      <em>MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI.</em>
      CVPR 2024.
      <a href="https://arxiv.org/abs/2311.16502">arXiv:2311.16502</a>.
    </li>
    <li id="fn:pass-k-diffusion">
      Wang, Wu, Fu, Chen, Chen, Gan, &amp; Wei.
      <em>Sharper, Not Broader: Pass@k Reveals Mode Collapse in Guided and Distilled Diffusion.</em>
      Draft manuscript, 2026.
    </li>
    <li id="fn:rae">
      Zheng, Ma, Tong, &amp; Xie.
      <em>Diffusion Transformers with Representation Autoencoders.</em>
      arXiv 2025.
      <a href="https://arxiv.org/abs/2510.11690">arXiv:2510.11690</a>.
    </li>
    <li id="fn:pae">
      Yue, Hu, Chen, Zhang, Pan, Liu, Wang, Lan, Zhu, Zheng, &amp; Wang.
      <em>What Matters for Diffusion-Friendly Latent Manifold? Prior-Aligned Autoencoders for Latent Diffusion.</em>
      arXiv 2026.
      <a href="https://arxiv.org/abs/2605.07915">arXiv:2605.07915</a>.
    </li>
    <li id="fn:vavae">
      Yao &amp; Wang.
      <em>Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models.</em>
      arXiv 2025.
      <a href="https://arxiv.org/abs/2501.01423">arXiv:2501.01423</a>.
    </li>
    <li id="fn:minit2i">
      Wang et al.
      <em>A Minimalist Baseline for Text-to-Image Generation.</em>
      Blog post, 2026.
      <a href="https://peppaking8.github.io/#/post/minit2i">https://peppaking8.github.io/#/post/minit2i</a>.
    </li>
  </ol>
</section>

<script>
document.addEventListener("DOMContentLoaded", function () {
  var firstRefs = {};
  document.querySelectorAll(".footnote-ref a[href^='#fn:']").forEach(function (link) {
    var key = link.getAttribute("href").slice(1);
    if (!firstRefs[key]) {
      var backId = "back-" + key;
      link.id = backId;
      firstRefs[key] = backId;
    }
  });

  Object.keys(firstRefs).forEach(function (key) {
    var item = document.getElementById(key);
    if (!item || item.querySelector(".ref-back")) return;

    var back = document.createElement("a");
    back.className = "ref-back";
    back.href = "#" + firstRefs[key];
    back.setAttribute("aria-label", "Back to citation");
    back.textContent = "↩";
    item.appendChild(document.createTextNode(" "));
    item.appendChild(back);
  });
});
</script>

<section class="end-section cite-section">
  <h4>Please Cite</h4>
  <p>If this post is useful for your work, please cite it as:</p>
  <pre class="bibtex"><code>@misc{wang2026struggling,
  title = {Struggling Toward Generative Model Evaluation},
  author = {Wang, Yifei and Wu, Xiaoyu and Wei, Chen},
  year = {2026},
  url = {https://a-little-hoof.github.io/blog/how-to-evaluate-your-generative-model/},
  note = {Blog post}
}</code></pre>
</section>
