---
title: "Notes on Diffusion Models"
permalink: /blog/notes-on-diffusion-models/
excerpt: "A working note collecting the core definitions, training objectives, sampler choices, and practical questions around diffusion models."
tags:
  - diffusion
  - generative-models
  - notes
hidden: false
---

> Working draft. The goal is to connect the main views of diffusion models: generative modeling basics, diffusion formulations, samplers, guidance, and few-step generation.

<style>
.page__content #draft-toc {
  margin: 1.2rem 0 2rem;
  padding: 0.75rem 0 0.75rem 0.9rem;
  border-left: 1px solid #dfe4e8;
  color: #66707a;
}
.page__content #draft-toc .toc-title {
  margin-bottom: 0.45rem;
  color: #252932;
  font-size: 0.78rem;
  font-weight: 700;
  text-transform: uppercase;
}
.page__content #draft-toc a {
  display: block;
  padding: 0.28rem 0.55rem;
  border-left: 3px solid transparent;
  border-radius: 0 5px 5px 0;
  color: #66707a;
  font-size: 0.86rem;
  font-weight: 500;
  line-height: 1.35;
  text-decoration: none;
}
.page__content #draft-toc a:hover {
  color: #252932;
  background: #f5f7f8;
  text-decoration: none;
}
.page__content #draft-toc a.toc-active {
  color: #252932;
  border-left-color: #24788d;
}
@media (min-width: 1460px) {
  .page__content #draft-toc {
    position: fixed;
    top: 140px;
    left: max(24px, calc(50% + 380px + 44px));
    width: 360px;
    max-height: calc(100vh - 180px);
    overflow-y: auto;
    overflow-x: hidden;
    z-index: 5;
    margin: 0;
  }
}
</style>

<nav id="draft-toc" aria-label="Table of contents">
  <div class="toc-title">Contents</div>
</nav>

## Generative Modeling Basics
### VAE
- Explain the Evidence Lower Bound (ELBO).

### GAN
- What should the optimal discriminator satisfy?

## Diffusion Formulations
> Diffusion model family.

### DDPM

<details class="ddim-block" open>
  <summary>
    <span class="ddim-block__title"><strong>1.</strong> Forward and reverse processes</span>
  </summary>
  <div class="ddim-block__content">
    <figure>
      <img src="/images/blog/diffusion/ddpm-forward-reverse.png" alt="DDPM forward noising process and learned reverse denoising process" style="width: 100%; max-width: 760px; display: block; margin: 0.75rem auto 0.35rem;">
      <figcaption style="color: #66707a; font-size: 0.9rem; line-height: 1.5; text-align: center;">
        DDPM defines a fixed forward noising chain \(q(x_t\mid x_{t-1})\) and learns a reverse denoising chain \(p_\theta(x_{t-1}\mid x_t)\).
      </figcaption>
    </figure>

    <p>The forward process is fixed: it gradually adds Gaussian noise to data \(x_0\) until \(x_T\) is close to standard normal noise. With variance schedule \(\beta_t\), define \(\alpha_t=1-\beta_t\):</p>

$$
q(x_t\mid x_{t-1})
=
\mathcal{N}\left(
\sqrt{\alpha_t}x_{t-1},
(1-\alpha_t)I
\right)
=
\mathcal{N}\left(
\sqrt{1-\beta_t}x_{t-1},
\beta_t I
\right).
$$

    <p>Because the forward process is linear Gaussian, we can sample \(x_t\) directly from \(x_0\). Let \(\bar{\alpha}_t=\prod_{s=1}^t\alpha_s\). Then</p>

$$
q(x_t\mid x_0)
=
\mathcal{N}\left(
\sqrt{\bar{\alpha}_t}x_0,
(1-\bar{\alpha}_t)I
\right),
\qquad
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+ \sqrt{1-\bar{\alpha}_t}\epsilon,
\quad
\epsilon\sim\mathcal{N}(0,I).
$$

    <p>The reverse process is what we learn. Starting from noise \(x_T\sim\mathcal{N}(0,I)\), the model denoises one step at a time:</p>

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal{N}\left(
\mu_\theta(x_t,t),
\Sigma_\theta(x_t,t)
\right).
$$

    <p>In the common noise-prediction parameterization, the network predicts \(\epsilon_\theta(x_t,t)\). This prediction gives an estimate of the clean sample</p>

$$
\hat{x}_0(x_t,t)
=
\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t)}
{\sqrt{\bar{\alpha}_t}},
$$

    <p>which is then used to construct the reverse mean \(\mu_\theta(x_t,t)\). The learning problem is therefore: train a denoiser that can infer the noise added at each time \(t\), then use it to walk backward from \(x_T\) to \(x_0\).</p>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>2.</strong> Deriving the DDPM loss</span>
  </summary>
  <div class="ddim-block__content">
    <p>The training objective starts from the likelihood of a clean datapoint \(x_0\). Since the reverse model contains latent variables \(x_1,\ldots,x_T\), we marginalize over the whole reverse trajectory:</p>

    <p>Insert the known forward process \(q(x_{1:T}\mid x_0)\) as an importance distribution, then apply Jensen's inequality:</p>

$$
\begin{aligned}
\log p_\theta(x_0)
&=
\log \int p_\theta(x_{0:T})\,dx_{1:T} \\
&=
\log \int q(x_{1:T}\mid x_0)
\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}
\,dx_{1:T} \\
&=
\log \mathbb{E}_{q(x_{1:T}\mid x_0)}
\left[
\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}
\right] \\
&\ge
\mathbb{E}_{q(x_{1:T}\mid x_0)}
\left[
\log
\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}
\right].
\end{aligned}
$$

    <p>So maximizing likelihood can be replaced by maximizing this variational lower bound. Equivalently, we minimize the negative ELBO:</p>

$$
L
=
\mathbb{E}_q
\left[
-\log
\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}
\right].
$$

    <p>Now use the Markov factorizations of the learned reverse chain and the fixed forward chain:</p>

$$
p_\theta(x_{0:T})
=
p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t),
\qquad
q(x_{1:T}\mid x_0)
=
\prod_{t=1}^{T}q(x_t\mid x_{t-1}).
$$

    <p>The denominator can be rewritten using the posterior \(q(x_{t-1}\mid x_t,x_0)\), which is analytically Gaussian:</p>

$$
\begin{aligned}
L
&=
\mathbb{E}_q
\left[
-\log p(x_T)
-
\sum_{t>1}
\log
\frac{p_\theta(x_{t-1}\mid x_t)}
{q(x_t\mid x_{t-1})}
-
\log
\frac{p_\theta(x_0\mid x_1)}
{q(x_1\mid x_0)}
\right] \\
&=
\mathbb{E}_q
\left[
-\log
\frac{p(x_T)}{q(x_T\mid x_0)}
-
\sum_{t>1}
\log
\frac{p_\theta(x_{t-1}\mid x_t)}
{q(x_{t-1}\mid x_t,x_0)}
-
\log p_\theta(x_0\mid x_1)
\right].
\end{aligned}
$$

    <p>This gives the standard DDPM loss decomposition:</p>

    <p>The negative ELBO becomes a prior matching term, many denoising KL terms, and a reconstruction term:</p>

$$
\begin{aligned}
L
=
\mathbb{E}_q
\Big[
&D_{\mathrm{KL}}\!\left(q(x_T\mid x_0)\,\|\,p(x_T)\right) \\
&+
\sum_{t>1}
D_{\mathrm{KL}}\!\left(
q(x_{t-1}\mid x_t,x_0)
\,\|\,
p_\theta(x_{t-1}\mid x_t)
\right) \\
&-
\log p_\theta(x_0\mid x_1)
\Big].
\end{aligned}
$$

    <p>The first term asks the terminal forward distribution to look like the prior \(p(x_T)\), the middle terms train the learned reverse transition to match the true denoising posterior, and the last term reconstructs \(x_0\) from \(x_1\).</p>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>3.</strong> Calculating DDPM loss</span>
  </summary>
  <div class="ddim-block__content">
    <p>Start from the ELBO decomposition above. For \(t>1\), the learnable part is the KL matching term</p>

$$
L_{t-1}
=
\mathbb{E}_q
\left[
D_{\mathrm{KL}}\!\left(
q(x_{t-1}\mid x_t,x_0)
\,\|\,
p_\theta(x_{t-1}\mid x_t)
\right)
\right].
$$

    <p>The other terms are usually not what produces the simple DDPM training loss: \(D_{\mathrm{KL}}(q(x_T\mid x_0)\|p(x_T))\) is ignored because the noise schedule makes \(q(x_T\mid x_0)\) close to \(\mathcal{N}(0,I)\) and it does not train the denoiser directly; the reconstruction term \(-\log p_\theta(x_0\mid x_1)\) is often treated separately or dropped in the simplified objective.</p>

    <p>The fixed forward process gives</p>

$$
\alpha_t := 1-\beta_t,
\qquad
\bar{\alpha}_t := \prod_{s=1}^{t}\alpha_s.
$$

    <p>and therefore the exact one-step posterior is Gaussian:</p>

$$
q(x_{t-1}\mid x_t,x_0)
=
\mathcal{N}\left(
x_{t-1};
\tilde{\mu}_t(x_t,x_0),
\tilde{\beta}_t I
\right),
$$

$$
\tilde{\mu}_t(x_t,x_0)
:=
\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0
+
\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t,
\qquad
\tilde{\beta}_t
:=
\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t.
$$

    <p>We choose the learned reverse transition to also be Gaussian, with a fixed variance \(\sigma_t^2 I\):</p>

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal{N}\left(
x_{t-1};
\mu_\theta(x_t,t),
\sigma_t^2 I
\right).
$$

    <p>Now the KL between two Gaussians with fixed variance reduces to a weighted mean-squared error between their means, plus constants independent of \(\theta\):</p>

$$
L_{t-1}
=
\mathbb{E}_q
\left[
\frac{1}{2\sigma_t^2}
\left\|
\tilde{\mu}_t(x_t,x_0)-\mu_\theta(x_t,t)
\right\|_2^2
\right]
+
\mathrm{const}.
$$

    <p>This is where reparameterization enters. Instead of sampling \(x_t\) abstractly from \(q(x_t\mid x_0)\), write it using a standard noise variable:</p>

$$
q(x_t\mid x_0)
=
\mathcal{N}\left(
x_t;
\sqrt{\bar{\alpha}_t}x_0,
(1-\bar{\alpha}_t)I
\right),
$$

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

    <p>Solving this equation for \(x_0\) gives</p>

$$
x_0
=
\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon}
{\sqrt{\bar{\alpha}_t}}.
$$

    <p>Substitute this \(x_0\) into the true posterior mean \(\tilde{\mu}_t(x_t,x_0)\). The posterior mean can now be written in terms of the actual noise \(\epsilon\):</p>

$$
\tilde{\mu}_t(x_t,x_0)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon
\right).
$$

    <p>Then parameterize the learned mean by predicting the noise:</p>

$$
\mu_\theta(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)
\right).
$$

    <p>So mean matching is equivalent to noise matching:</p>

$$
\left\|
\tilde{\mu}_t(x_t,x_0)-\mu_\theta(x_t,t)
\right\|_2^2
=
\frac{\beta_t^2}{\alpha_t(1-\bar{\alpha}_t)}
\left\|
\epsilon-\epsilon_\theta(x_t,t)
\right\|_2^2.
$$

    <p>The full variational objective would keep the timestep-dependent weight \(\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar{\alpha}_t)}\). The simplified DDPM loss drops this weight and the constants, leaving the practical objective:</p>

$$
L_{\mathrm{simple}}
=
\mathbb{E}_{t,x_0,\epsilon}
\left[
\left\|
\epsilon
-
\epsilon_\theta
\left(
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
t
\right)
\right\|_2^2
\right],
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

    <p>Intuitively, the model is trained to answer one question: given a noisy sample \(x_t\) and the timestep \(t\), what Gaussian noise \(\epsilon\) was added to \(x_0\)?</p>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>4.</strong> Reparameterization</span>
  </summary>
  <div class="ddim-block__content">
    <p>The reparameterization trick separates randomness from the quantity we want the network to learn. Instead of treating \(x_t\) as an opaque Gaussian sample from \(q(x_t\mid x_0)\), we write it as a deterministic function of \(x_0\), \(t\), and a standard Gaussian noise variable:</p>

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

    <p>This is useful because the training target is now explicit: the model sees \(x_t\) and \(t\), and tries to predict the exact noise \(\epsilon\) that produced \(x_t\).</p>

$$
\epsilon_\theta(x_t,t)
\approx
\epsilon.
$$

    <p>Once the model predicts \(\epsilon_\theta(x_t,t)\), we can also recover an estimate of the clean sample:</p>

$$
\hat{x}_0(x_t,t)
=
\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t)}
{\sqrt{\bar{\alpha}_t}}.
$$

    <p>Plugging this prediction into the Gaussian posterior mean gives the DDPM reverse mean in the noise-prediction parameterization:</p>

$$
\mu_\theta(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}
\epsilon_\theta(x_t,t)
\right).
$$

    <p>So the learned reverse transition becomes</p>

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal{N}\left(
x_{t-1};
\mu_\theta(x_t,t),
\sigma_t^2 I
\right),
$$

    <p>and sampling is just repeatedly applying this denoising step, with fresh noise added according to \(\sigma_t\).</p>
  </div>
</details>

<details class="ddim-block ddim-algorithm">
  <summary>
    <span class="ddim-block__title"><strong>5.</strong> Pseudocode: DDPM Training and Sampling</span>
  </summary>
  <div class="ddim-block__content">
    <div class="ddim-algorithm-grid">
      <div class="ddim-algorithm-panel">
        <div class="ddim-algorithm__require"><strong>Algorithm 1</strong> Training</div>
        <div class="ddim-algorithm__body">
          <div class="ddim-algorithm__row">
            <div class="ddim-algorithm__num">1:</div>
            <div class="ddim-algorithm__code"><strong>repeat</strong></div>
            <div class="ddim-algorithm__comment"></div>
          </div>
          <div class="ddim-algorithm__row">
            <div class="ddim-algorithm__num">2:</div>
            <div class="ddim-algorithm__code">\(\quad x_0 \sim q(x_0)\)</div>
            <div class="ddim-algorithm__comment"></div>
          </div>
          <div class="ddim-algorithm__row">
            <div class="ddim-algorithm__num">3:</div>
            <div class="ddim-algorithm__code">\(\quad t \sim \mathrm{Uniform}(\{1,\ldots,T\})\)</div>
            <div class="ddim-algorithm__comment"></div>
          </div>
          <div class="ddim-algorithm__row">
            <div class="ddim-algorithm__num">4:</div>
            <div class="ddim-algorithm__code">\(\quad \epsilon \sim \mathcal{N}(0,I)\)</div>
            <div class="ddim-algorithm__comment"></div>
          </div>
          <div class="ddim-algorithm__row">
            <div class="ddim-algorithm__num">5:</div>
            <div class="ddim-algorithm__code">\(\quad\)Take gradient descent step on</div>
            <div class="ddim-algorithm__comment"></div>
          </div>
          <div class="ddim-algorithm__row">
            <div class="ddim-algorithm__num"></div>
            <div class="ddim-algorithm__code">\(\quad\nabla_\theta\left\|\epsilon-\epsilon_\theta\left(\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,t\right)\right\|^2\)</div>
            <div class="ddim-algorithm__comment"></div>
          </div>
          <div class="ddim-algorithm__row">
            <div class="ddim-algorithm__num">6:</div>
            <div class="ddim-algorithm__code"><strong>until converged</strong></div>
            <div class="ddim-algorithm__comment"></div>
          </div>
        </div>
      </div>

      <div class="ddim-algorithm-panel">
        <div class="ddim-algorithm__require"><strong>Algorithm 2</strong> Sampling</div>
        <div class="ddim-algorithm__body">
          <div class="ddim-algorithm__row">
            <div class="ddim-algorithm__num">1:</div>
            <div class="ddim-algorithm__code">\(x_T \sim \mathcal{N}(0,I)\)</div>
            <div class="ddim-algorithm__comment"></div>
          </div>
          <div class="ddim-algorithm__row">
            <div class="ddim-algorithm__num">2:</div>
            <div class="ddim-algorithm__code"><strong>for</strong> \(t=T,\ldots,1\) <strong>do</strong></div>
            <div class="ddim-algorithm__comment"></div>
          </div>
          <div class="ddim-algorithm__row">
            <div class="ddim-algorithm__num">3:</div>
            <div class="ddim-algorithm__code">\(\quad z \sim \mathcal{N}(0,I)\) if \(t>1\), else \(z=0\)</div>
            <div class="ddim-algorithm__comment"></div>
          </div>
          <div class="ddim-algorithm__row">
            <div class="ddim-algorithm__num">4:</div>
            <div class="ddim-algorithm__code">\(\quad x_{t-1}=\dfrac{1}{\sqrt{\alpha_t}}\left(x_t-\dfrac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)\right)+\sigma_t z\)</div>
            <div class="ddim-algorithm__comment"></div>
          </div>
          <div class="ddim-algorithm__row">
            <div class="ddim-algorithm__num">5:</div>
            <div class="ddim-algorithm__code"><strong>end for</strong></div>
            <div class="ddim-algorithm__comment"></div>
          </div>
          <div class="ddim-algorithm__row">
            <div class="ddim-algorithm__num">6:</div>
            <div class="ddim-algorithm__code"><strong>return</strong> \(x_0\)</div>
            <div class="ddim-algorithm__comment"></div>
          </div>
        </div>
      </div>
    </div>

    <p>Here \(\sigma_t\) is the sampling noise scale. In the original DDPM sampler, it is typically chosen from the reverse-process variance, for example \(\sigma_t^2=\tilde{\beta}_t\) or \(\sigma_t^2=\beta_t\), depending on the variance parameterization.</p>
  </div>
</details>

### Score-based SDEs

### Flow Matching

### Q&A

## Deep dive in diffusion models
> Math related properties.

### Itô integral

### Fokker-Planck equations

### Tweedie's formula

### Denosing score matching

## Samplers
> How do we sample efficiently from a pretrained denoiser?

### DDIM<sup class="footnote-ref" id="fnref:ddim"><a href="#fn:ddim">1</a></sup>

<style>
.page__content .ddim-block {
  margin: 1rem 0 1.35rem;
  border-top: 2px solid #1f2328;
  border-bottom: 1px solid #1f2328;
  color: #252932;
  font-size: 0.96rem;
  line-height: 1.55;
}
.page__content .ddim-block summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.45rem 0;
  cursor: pointer;
  list-style: none;
}
.page__content .ddim-block summary::-webkit-details-marker {
  display: none;
}
.page__content .ddim-block summary::after {
  content: "+";
  color: #66707a;
  font-size: 1.1rem;
  font-weight: 600;
  line-height: 1;
}
.page__content .ddim-block[open] summary::after {
  content: "-";
}
.page__content .ddim-block__title {
  font-size: 1.05rem;
}
.page__content .ddim-block__content {
  padding: 0.35rem 0 0.85rem;
  border-top: 1px solid #8b949e;
}
.page__content .ddim-block__content p:first-child {
  margin-top: 0.45rem;
}
.page__content .ddim-algorithm__require {
  padding: 0.45rem 0 0.35rem;
  font-size: 0.98rem;
}
.page__content .ddim-algorithm__body {
  padding-bottom: 0.45rem;
}
.page__content .ddim-algorithm-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 0.85rem;
  padding: 0.45rem 0 0.25rem;
}
.page__content .ddim-algorithm-panel {
  min-width: 0;
}
.page__content .ddim-algorithm-panel .ddim-algorithm__require {
  border-bottom: 1px solid #8b949e;
  margin-bottom: 0.35rem;
}
.page__content .ddim-algorithm-panel .ddim-algorithm__row {
  grid-template-columns: 2rem minmax(0, 1fr);
}
.page__content .ddim-algorithm-panel .ddim-algorithm__comment {
  display: none;
}
.page__content .ddim-algorithm__row {
  display: grid;
  grid-template-columns: 2.2rem minmax(0, 1fr) minmax(12rem, auto);
  gap: 0.5rem;
  align-items: baseline;
  min-height: 1.65rem;
}
.page__content .ddim-algorithm__num {
  color: #66707a;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.page__content .ddim-algorithm__code {
  min-width: 0;
}
.page__content .ddim-algorithm__comment {
  color: #66707a;
  font-size: 0.92rem;
  text-align: right;
}
@media (max-width: 760px) {
  .page__content .ddim-algorithm__row {
    grid-template-columns: 2rem minmax(0, 1fr);
  }
  .page__content .ddim-algorithm__comment {
    grid-column: 2;
    text-align: left;
  }
}
</style>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>1.</strong> What is DDIM?</span>
  </summary>
  <div class="ddim-block__content">
    <p>DDIM starts from the same trained denoiser as DDPM, but changes the sampling process. The key point is that DDPM training only uses the marginal noising distribution</p>

$$
q(x_t \mid x_0) =
\mathcal{N}(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I),
$$

    <p>not the full Markov structure of the forward chain. Therefore, after training, we can choose a different reverse-time process as long as it is consistent with these marginals.</p>

    <p>Given a noisy sample $x_t$ and a noise predictor $\epsilon_\theta(x_t, t)$, first estimate the clean sample:</p>

$$
\hat{x}_0(x_t, t)
=
\frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t)}
{\sqrt{\bar{\alpha}_t}}.
$$

    <p>DDIM then writes the next sample $x_{t-1}$ as a combination of three pieces: the predicted clean sample, the direction pointing back toward $x_t$, and optional fresh Gaussian noise.</p>

$$
\begin{aligned}
x_{t-1}
=&
\sqrt{\bar{\alpha}_{t-1}}
\underbrace{
\left(
\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta^{(t)}(x_t)}
{\sqrt{\bar{\alpha}_t}}
\right)
}_{\text{predicted }x_0}
+ \underbrace{
\sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\epsilon_\theta^{(t)}(x_t)
}_{\text{direction pointing to }x_t}
+ \underbrace{\sigma_t\epsilon_t}_{\text{random noise}},
\quad \epsilon_t \sim \mathcal{N}(0,I).
\end{aligned}
$$

    <p>The parameter $\sigma_t$ controls how stochastic the sampler is:</p>

$$
\sigma_t
=
\eta
\sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}}
\sqrt{1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}.
$$

    <p>When $\eta = 1$, the sampler is close to the DDPM reverse process. When $\eta = 0$, the noise term disappears and the update becomes deterministic:</p>

$$
x_{t-1}
=
\sqrt{\bar{\alpha}_{t-1}}\hat{x}_0
+ \sqrt{1-\bar{\alpha}_{t-1}}\epsilon_\theta(x_t, t).
$$

    <p>The acceleration comes from using a subsequence of timesteps. Instead of sampling all $T$ steps, choose</p>

$$
\tau_0 < \tau_1 < \cdots < \tau_S,
\quad S \ll T,
$$

    <p>and apply the same update from $\tau_i$ to $\tau_{i-1}$. In practice this can reduce sampling from hundreds or thousands of denoiser calls to tens of calls, without retraining the model.</p>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>2.</strong> Theoretical explanation of DDIM: probability flow ODE</span>
  </summary>
  <div class="ddim-block__content">
    <p>The deterministic DDIM case, $\eta=0$, maps the initial noise to the final sample through a fixed trajectory.</p>

    <p>In continuous time, a diffusion process can be written as an SDE</p>

$$
d x = f(x,t)\,dt + g(t)\,d w.
$$

    <p>The corresponding reverse-time SDE uses the score $\nabla_x \log p_t(x)$:</p>

$$
d x =
\left[f(x,t)-g(t)^2\nabla_x\log p_t(x)\right]dt
+ g(t)\,d\bar{w}.
$$

    <p>The probability flow ODE removes the stochastic term but preserves the same marginal distributions $p_t(x)$:</p>

$$
d x =
\left[f(x,t)-\frac{1}{2}g(t)^2\nabla_x\log p_t(x)\right]dt.
$$

    <p>DDIM is the discrete-time analogue of this idea. Once the model predicts the noise, we can convert it into a score-like direction using</p>

$$
s_\theta(x_t,t)
\approx
-\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar{\alpha}_t}}.
$$

    <p>The deterministic DDIM update follows this learned direction without injecting new noise at every step. Under an ideal score model, the trajectory has the same endpoint distribution as the stochastic sampler, but it is much easier to skip timesteps because the path is deterministic.</p>
  </div>
</details>

<details class="ddim-block ddim-algorithm">
  <summary>
    <span class="ddim-block__title"><strong>3.</strong> Pseudocode: DDIM Sampling</span>
  </summary>
  <div class="ddim-block__content">
    <div class="ddim-algorithm__require"><strong>Require:</strong> trained noise predictor \(\epsilon_\theta\), number of steps \(S\), noise schedule \(\bar{\alpha}\)</div>
    <div class="ddim-algorithm__body">
      <div class="ddim-algorithm__row">
        <div class="ddim-algorithm__num">1:</div>
        <div class="ddim-algorithm__code">Sample \(x_T \sim \mathcal{N}(0,I)\)</div>
        <div class="ddim-algorithm__comment"></div>
      </div>
      <div class="ddim-algorithm__row">
        <div class="ddim-algorithm__num">2:</div>
        <div class="ddim-algorithm__code">Create timestep subsequence \([\tau_S,\tau_{S-1},\ldots,\tau_1]\) from \([T,\ldots,1]\)</div>
        <div class="ddim-algorithm__comment">&#9657; e.g., \([1000,900,800,\ldots]\)</div>
      </div>
      <div class="ddim-algorithm__row">
        <div class="ddim-algorithm__num">3:</div>
        <div class="ddim-algorithm__code"><strong>for</strong> \(i=S,S-1,\ldots,1\) <strong>do</strong></div>
        <div class="ddim-algorithm__comment"></div>
      </div>
      <div class="ddim-algorithm__row">
        <div class="ddim-algorithm__num">4:</div>
        <div class="ddim-algorithm__code">\(\quad t \leftarrow \tau_i\)</div>
        <div class="ddim-algorithm__comment"></div>
      </div>
      <div class="ddim-algorithm__row">
        <div class="ddim-algorithm__num">5:</div>
        <div class="ddim-algorithm__code">\(\quad t_{\mathrm{prev}} \leftarrow \tau_{i-1}\) (or \(0\) if \(i=1\))</div>
        <div class="ddim-algorithm__comment"></div>
      </div>
      <div class="ddim-algorithm__row">
        <div class="ddim-algorithm__num">6:</div>
        <div class="ddim-algorithm__code">\(\quad \epsilon \leftarrow \epsilon_\theta(x_t,t)\)</div>
        <div class="ddim-algorithm__comment">&#9657; predict noise using the trained DDPM</div>
      </div>
      <div class="ddim-algorithm__row">
        <div class="ddim-algorithm__num">7:</div>
        <div class="ddim-algorithm__code">\(\quad \hat{x}_0 \leftarrow \dfrac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}}\)</div>
        <div class="ddim-algorithm__comment">&#9657; predict clean image</div>
      </div>
      <div class="ddim-algorithm__row">
        <div class="ddim-algorithm__num">8:</div>
        <div class="ddim-algorithm__code">\(\quad x_{t_{\mathrm{prev}}} \leftarrow \sqrt{\bar{\alpha}_{t_{\mathrm{prev}}}}\hat{x}_0+\sqrt{1-\bar{\alpha}_{t_{\mathrm{prev}}}}\epsilon\)</div>
        <div class="ddim-algorithm__comment">&#9657; deterministic DDIM step</div>
      </div>
      <div class="ddim-algorithm__row">
        <div class="ddim-algorithm__num">9:</div>
        <div class="ddim-algorithm__code"><strong>end for</strong></div>
        <div class="ddim-algorithm__comment"></div>
      </div>
      <div class="ddim-algorithm__row">
        <div class="ddim-algorithm__num">10:</div>
        <div class="ddim-algorithm__code"><strong>return</strong> \(x_0\)</div>
        <div class="ddim-algorithm__comment"></div>
      </div>
    </div>
  </div>
</details>

### ODE solvers

### DPM solvers

### Q&A

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>Q1.</strong> How is \(\sigma_t\) defined in DDIM, and where does it come from?</span>
  </summary>
  <div class="ddim-block__content">
    <p>For the adjacent-step notation, DDIM defines</p>

$$
\sigma_t
=
\eta
\sqrt{
\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}
}
\sqrt{
1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}
}.
$$

    <p>For a skipped timestep schedule, replace \(t-1\) by the previous selected timestep \(t_{\mathrm{prev}}\):</p>

$$
\sigma_t
=
\eta
\sqrt{
\frac{1-\bar{\alpha}_{t_{\mathrm{prev}}}}{1-\bar{\alpha}_t}
}
\sqrt{
1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t_{\mathrm{prev}}}}
}.
$$

    <p>It comes from choosing the conditional sampling distribution</p>

$$
q_\sigma(x_{t-1}\mid x_t,x_0)
=
\mathcal{N}
\left(
\sqrt{\bar{\alpha}_{t-1}}x_0
+ \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}
\frac{x_t-\sqrt{\bar{\alpha}_t}x_0}{\sqrt{1-\bar{\alpha}_t}},
\sigma_t^2 I
\right).
$$

    <p>The mean has two parts: a clean-sample component and a residual noise-direction component. The variance \(\sigma_t^2\) is then set so the reverse step remains compatible with the forward marginals. The scalar \(\eta\) is a free knob: \(\eta=0\) gives deterministic DDIM, while \(\eta=1\) recovers the DDPM-style stochastic variance for the same adjacent timestep.</p>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>Q2.</strong> If \(\eta=1\) and we do not skip steps, is DDIM the same as DDPM?</span>
  </summary>
  <div class="ddim-block__content">
    <p>Distributionally, yes: with all adjacent timesteps and \(\eta=1\), DDIM chooses</p>

$$
\sigma_t^2
=
\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}
\left(
1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}
\right).
$$

    <p>which is exactly the DDPM posterior variance \(\tilde{\beta}_t\):</p>

$$
\tilde{\beta}_t
=
\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t.
$$

    <p>Since \(\alpha_t=\bar{\alpha}_t/\bar{\alpha}_{t-1}\), we have</p>

$$
1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}
=
1-\alpha_t
=
\beta_t.
$$

    <p>so \(\sigma_t^2=\tilde{\beta}_t\). The DDIM mean also reduces to the usual DDPM reverse mean when the model predicts noise \(\epsilon_\theta(x_t,t)\). Therefore, as a Markov transition distribution, \(\eta=1\) DDIM with no timestep skipping matches DDPM sampling.</p>

    <p>The practical caveat is that "same" means same conditional distribution, not necessarily the exact same sample path. To get identical individual samples, the code must use the same variance convention, timestep indexing, clipping rules, and random noise draws. Otherwise the two samplers should be statistically equivalent but not bitwise identical.</p>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>Q3.</strong> Why is DDIM the discrete-time analogue of probability flow ODE?</span>
  </summary>
  <div class="ddim-block__content">
    <p>The probability flow ODE is the deterministic counterpart of the reverse-time SDE. It removes the Brownian noise term while preserving the same marginal distributions \(p_t(x)\) as the stochastic diffusion process.</p>

$$
d x =
\left[
f(x,t)-\frac{1}{2}g(t)^2\nabla_x\log p_t(x)
\right]dt.
$$

    <p>DDIM does the same thing in discrete time. Start from the DDPM marginal parameterization</p>

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+ \sqrt{1-\bar{\alpha}_t}\epsilon.
$$

    <p>After the model predicts \(\epsilon_\theta(x_t,t)\), DDIM estimates the clean sample and then moves to a lower-noise time using the same predicted noise direction:</p>

$$
\hat{x}_0
=
\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t)}
{\sqrt{\bar{\alpha}_t}},
\qquad
x_{t_{\mathrm{prev}}}
=
\sqrt{\bar{\alpha}_{t_{\mathrm{prev}}}}\hat{x}_0
+ \sqrt{1-\bar{\alpha}_{t_{\mathrm{prev}}}}\epsilon_\theta(x_t,t).
$$

    <p>This is deterministic when \(\eta=0\): no new noise is injected. The sampler follows a learned direction field from high noise to low noise, just as the probability flow ODE follows a deterministic vector field induced by the score.</p>

    <p>The link to the score comes from the VP/DDPM identity</p>

$$
\nabla_{x_t}\log p_t(x_t)
\approx
-\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar{\alpha}_t}}.
$$

    <p>So, informally: probability flow ODE is the continuous-time deterministic sampler; DDIM with \(\eta=0\) is its discrete-time version built from the same trained noise predictor and the same marginal noise schedule.</p>
  </div>
</details>

## Architecture
> Coding part.

### Transformers

- Diffusion transformers
- Multimodel diffusion transformers

## Guidance
> Guidance, a cheat code for diffusion models.<sup class="footnote-ref" id="fnref:guidance-cheat-code"><a href="#fn:guidance-cheat-code">4</a></sup>

## Distillation
> How to train one-step and few-step diffusion models.

### Distribution matching distillation

### Trajectory distillation

## Image and Video Diffusion Models

### Image Diffusion Models

### Video Diffusion Models

## Study Checklist

### Core Diffusion Questions

#### Foundations
- Write down the forward SDE and the reverse SDE.
- Write down the Fokker-Planck equation.
- Derive the probability flow ODE from the reverse SDE using the Fokker-Planck equation.
- Write down the VP-SDE and VE-SDE, and explain their connections to DDPM and NCSN.
- Prove the equivalence between score matching (SM) and denoising score matching (DSM).
- Explain the maximum likelihood training of diffusion models.
- What is the relationship between SM/DSM and maximum likelihood training?
- Explain Tweedie's formula.
- Use Tweedie's formula to transform epsilon prediction into x_0 prediction.
- Derive the optimal denoiser.
- Explain how DDIM accelerates the sampling procedure.
- Explain how DPM-Solver accelerates the sampling procedure. Why is DDIM a special case of DPM-Solver?

#### Sampling
- In diffusion models, more sampling steps do not necessarily mean better results.
- Training and sampling of diffusion models doesn't require the same noise schedule.

### Schrödinger Bridges
- What is Doob's h-transform?

### Flow Matching Questions
- Write down the close form formula of velosity.

### One-Step/Few-Step Questions
- Explain the idea of consistency models (CM).
- Explain the idea of MeanFlow (MF) and how to calculate the mean velocity during training.
- What is the relationship between MF and CM?

### Reading List
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)
- [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)
- [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)
- [A Connection Between Score Matching and Denoising Autoencoders](https://ieeexplore.ieee.org/abstract/document/6795935)
- [Interpretation and generalization of score matching](https://arxiv.org/pdf/1205.2629)
- [Maximum Likelihood Training of Score-Based Diffusion Models](https://arxiv.org/abs/2101.09258)
- [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)
- [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447)
- [Consistency Models](https://arxiv.org/abs/2303.01469)
- [Improved Techniques for Training Consistency Models](https://arxiv.org/abs/2310.14189)
- [Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models](https://arxiv.org/abs/2410.11081)

<style>
.page__content .references-section {
  margin-top: 3rem;
  padding-top: 0;
  border-top: none;
  color: #66707a;
}
.page__content .references-section h2 {
  margin-top: 3.5rem;
}
.page__content .references-section ol {
  padding-left: 22px;
  margin: 0;
}
.page__content .references-section li {
  margin: 0.5rem 0;
  color: #66707a;
  font-size: 0.95rem;
  line-height: 1.65;
}
.page__content .references-section li em {
  color: #66707a;
}
.page__content .references-section li a {
  color: #24788d;
  text-decoration-color: rgba(36, 120, 141, 0.2);
}
</style>

<section class="footnotes references-section">
  <h2>References</h2>
  <ol>
    <li id="fn:ddim">
      Song, Meng, &amp; Ermon.
      <em>Denoising Diffusion Implicit Models.</em>
      ICLR 2021.
      <a href="https://arxiv.org/abs/2010.02502">arXiv:2010.02502</a>.
      <a href="#fnref:ddim" class="footnote-back" title="back to text">↩︎</a>
    </li>
    <li id="fn:mit-6s982">
      MIT 6.S982 staff.
      <em>Diffusion Models: From Theory to Practice (6.S982): Spring '25.</em>
      Course notes, 2025.
      <a href="https://hackmd.io/vHEEDpPOTmex1O4zFeMYTw">HackMD notes</a>.
    </li>
    <li id="fn:cmu-10799">
      Kelly Yutong He.
      <em>CMU 10-799 Diffusion &amp; Flow Matching.</em>
      Course page, Spring 2026.
      <a href="https://kellyyutonghe.github.io/10799S26/">course website</a>.
    </li>
    <li id="fn:guidance-cheat-code">
      Dieleman.
      <em>Guidance: a cheat code for diffusion models.</em>
      Blog post, 2022.
      <a href="https://benanne.github.io/2022/05/26/guidance.html">benanne.github.io</a>.
      <a href="#fnref:guidance-cheat-code" class="footnote-back" title="back to text">↩︎</a>
    </li>
  </ol>
</section>

<script>
/* Auto-build a Contents list from h2 sections, matching the EMA post behavior. */
(function () {
  function ready(fn) {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', fn);
    else fn();
  }
  ready(function () {
    var article = document.querySelector('.page__content');
    var toc = document.querySelector('#draft-toc');
    if (!article || !toc) return;
    var headings = Array.from(article.querySelectorAll('h2')).filter(function (h) {
      return h.closest('.references-section') === null && h.dataset.tocSkip !== 'true';
    });
    if (headings.length === 0) {
      toc.style.display = 'none';
      return;
    }
    function slugify(s) {
      return s.toLowerCase().replace(/[^a-z0-9 \-]/g, '').trim().replace(/\s+/g, '-').slice(0, 60);
    }
    var linkByHeading = new Map();
    headings.forEach(function (h, i) {
      if (!h.id) h.id = slugify(h.textContent) || ('sec-' + (i + 1));
      var a = document.createElement('a');
      a.href = '#' + h.id;
      a.textContent = h.textContent;
      a.addEventListener('click', function () {
        document.querySelectorAll('#draft-toc a').forEach(function (l) { l.classList.remove('toc-active'); });
        a.classList.add('toc-active');
      });
      toc.appendChild(a);
      linkByHeading.set(h, a);
    });
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        document.querySelectorAll('#draft-toc a').forEach(function (l) { l.classList.remove('toc-active'); });
        linkByHeading.get(entry.target).classList.add('toc-active');
      });
    }, { rootMargin: '-25% 0px -65% 0px', threshold: 0 });
    headings.forEach(function (h) { observer.observe(h); });
  });
})();
</script>
