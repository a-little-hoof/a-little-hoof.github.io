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
.page__content .design-space-box {
  margin: 1.25rem 0 1.5rem;
  padding: 1rem;
  border: 1.5px solid #252932;
  border-radius: 18px;
  background: #f5f6f7;
}
.page__content .design-space-title {
  margin-bottom: 0.85rem;
  color: #252932;
  font-size: 1.12rem;
  font-weight: 700;
  text-align: center;
}
.page__content .design-space-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.75rem;
}
.page__content .design-space-card {
  min-width: 0;
  padding: 0.85rem 0.95rem;
  border: 1.5px solid #252932;
  border-radius: 14px;
}
.page__content .design-space-card--training {
  background: #ffd4d1;
}
.page__content .design-space-card--model {
  background: #ffe2a3;
}
.page__content .design-space-card--sampling {
  background: #bdeef2;
}
.page__content .design-space-card h3 {
  margin: 0 0 0.55rem;
  color: #111827;
  font-size: 1rem;
  text-align: center;
}
.page__content .design-space-card ul {
  margin: 0;
  padding-left: 1.05rem;
}
.page__content .design-space-card li {
  margin: 0.42rem 0;
  line-height: 1.45;
}
@media (max-width: 760px) {
  .page__content .design-space-grid {
    grid-template-columns: 1fr;
  }
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

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>1.</strong> What is VAE?</span>
  </summary>
  <div class="ddim-block__content">
    <figure>
      <img src="/images/blog/diffusion/vae-encoder-decoder.png" alt="VAE encoder maps X to latent Z and decoder reconstructs X" style="width: 100%; max-width: 860px; display: block; margin: 0.75rem auto 0.35rem;">
      <figcaption style="color: #66707a; font-size: 0.9rem; line-height: 1.5; text-align: center;">
        VAE encodes data \(X\) into a latent variable \(Z\), then decodes \(Z\) back into a reconstruction of \(X\).
      </figcaption>
    </figure>

    <p>A VAE starts from a simple autoencoder picture. The encoder \(E_\phi\) maps data \(X\) into a latent variable \(Z\), and the decoder \(D_\theta\) maps \(Z\) back to a reconstruction of \(X\).</p>

    <p>We are trying to do two things at the same time. First, we want the model to explain the data well, so we want to maximize the likelihood of \(X\). Second, we want the \(Z\) we get from encoding \(X\) to actually decode back into the same \(X\), so the latent code must preserve useful information for reconstruction.</p>

    <p>The difficulty is that we can design the prior \(p_\theta(z)\), the encoder \(q_\phi(z\mid x)\), and the decoder \(p_\theta(x\mid z)\). But we do not directly have the marginal likelihood \(p_\theta(x)\), and we also do not directly have the true posterior \(p_\theta(z\mid x)\).</p>

    <p>So the problem becomes: we want \(p_\theta(x)\) so the model assigns high likelihood to data, and we want \(q_\phi(z\mid x)\) to be close to \(p_\theta(z\mid x)\) so the encoder's latent distribution is a good stand-in for the true posterior. The ELBO is the tool that connects these pieces into something trainable.</p>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>2.</strong> Deriving ELBO</span>
  </summary>
  <div class="ddim-block__content">
    <p>In a VAE, we want to maximize the marginal likelihood \(p_\theta(x)\), but the true posterior \(p_\theta(z\mid x)\) is usually intractable. So we introduce an approximate posterior \(q_\phi(z\mid x)\).</p>

$$
\begin{aligned}
\log p_\theta(x)
&=
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log p_\theta(x)
\right] \\
&=
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log
\frac{p_\theta(x,z)}{p_\theta(z\mid x)}
\right] \\
&=
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log
\left(
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\frac{q_\phi(z\mid x)}{p_\theta(z\mid x)}
\right)
\right] \\
&=
\underbrace{
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right]
}_{\mathcal{L}_{\theta,\phi}(x)\ \text{(ELBO)}}
+
\underbrace{
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log
\frac{q_\phi(z\mid x)}{p_\theta(z\mid x)}
\right]
}_{D_{\mathrm{KL}}\left(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\right)}.
\end{aligned}
$$

    <p>Since KL divergence is nonnegative, the first term is a lower bound on the log likelihood:</p>

$$
\mathcal{L}_{\theta,\phi}(x)
\le
\log p_\theta(x).
$$

    <p>Now expand the ELBO using \(p_\theta(x,z)=p_\theta(z)p_\theta(x\mid z)\):</p>

$$
\begin{aligned}
\mathcal{L}_{\theta,\phi}(x)
&=
\mathbb{E}_{z\sim q_\phi(z\mid x)}
\left[
\log
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right] \\
&=
\mathbb{E}_{z\sim q_\phi(z\mid x)}
\left[
\log p_\theta(x,z)
-
\log q_\phi(z\mid x)
\right] \\
&=
\mathbb{E}_{z\sim q_\phi(z\mid x)}
\left[
\log p_\theta(z)
+
\log p_\theta(x\mid z)
-
\log q_\phi(z\mid x)
\right] \\
&=
\mathbb{E}_{z\sim q_\phi(z\mid x)}
\left[
\log p_\theta(x\mid z)
\right]
-
\mathbb{E}_{z\sim q_\phi(z\mid x)}
\left[
\log q_\phi(z\mid x)
-
\log p_\theta(z)
\right] \\
&=
\underbrace{
\mathbb{E}_{z\sim q_\phi(z\mid x)}
\left[
\log p_\theta(x\mid z)
\right]
}_{\text{reconstruction term}}
-
\underbrace{
D_{\mathrm{KL}}\left(q_\phi(z\mid x)\,\|\,p_\theta(z)\right)
}_{\text{regularization term}}.
\end{aligned}
$$

    <p>So the VAE objective balances two forces: reconstruct \(x\) well from latent \(z\), while keeping the approximate posterior \(q_\phi(z\mid x)\) close to the prior \(p_\theta(z)\).</p>

    <p>The same lower bound can also be derived directly with Jensen's inequality. Insert \(q_\phi(z\mid x)\) into the marginal likelihood:</p>

$$
\begin{aligned}
\log p_\theta(x)
&=
\log \int p_\theta(x,z)\,dz \\
&=
\log \int
\frac{q_\phi(z\mid x)}{q_\phi(z\mid x)}
p_\theta(x,z)\,dz \\
&=
\log
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right].
\end{aligned}
$$

    <p>Because \(\log(\cdot)\) is concave, Jensen's inequality gives \(\log \mathbb{E}[Y]\ge \mathbb{E}[\log Y]\). Therefore</p>

$$
\begin{aligned}
\log p_\theta(x)
&\ge
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right] \\
&=
\mathbb{E}_{z\sim q_\phi(z\mid x)}
\left[
\log p_\theta(x\mid z)
\right]
-
D_{\mathrm{KL}}\left(q_\phi(z\mid x)\,\|\,p_\theta(z)\right)
=
\mathcal{L}_{\theta,\phi}(x).
\end{aligned}
$$

    <p>The only inequality step is Jensen's inequality: for convex \(f\), \(\mathbb{E}[f(X)]\ge f(\mathbb{E}[X])\); for concave \(f\), \(f(\mathbb{E}[X])\ge \mathbb{E}[f(X)]\). Here \(f=\log\), so it is concave.</p>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>3.</strong> Calculating VAE loss and reparameterization</span>
  </summary>
  <div class="ddim-block__content">
    <p>In implementation, the encoder outputs a Gaussian approximate posterior:</p>

$$
q_\phi(z\mid x)
=
\mathcal{N}\left(
z;
\mu_\phi(x),
\mathrm{diag}\left(\sigma_\phi^2(x)\right)
\right).
$$

    <p>To sample \(z\) while keeping the computation differentiable with respect to \(\phi\), use the reparameterization trick:</p>

$$
z
=
\mu_\phi(x)
+
\sigma_\phi(x)\odot\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

    <p>Training usually minimizes the negative ELBO:</p>

$$
L(\phi,\theta;x)
=
\underbrace{
-\mathbb{E}_{z\sim q_\phi(z\mid x)}
\left[
\log p_\theta(x\mid z)
\right]
}_{\text{reconstruction loss}}
+
\underbrace{
D_{\mathrm{KL}}\left(q_\phi(z\mid x)\,\|\,p_\theta(z)\right)
}_{\text{latent regularization}}
$$

    <p>For the common choice \(p_\theta(z)=\mathcal{N}(0,I)\) and diagonal Gaussian \(q_\phi(z\mid x)\), the KL term has a closed form:</p>

$$
D_{\mathrm{KL}}\left(q_\phi(z\mid x)\,\|\,p_\theta(z)\right)
=
\frac{1}{2}
\sum_d
\left(
\mu_d(x;\phi)^2
+
\sigma_d(x;\phi)^2
-
1
-
2\log\sigma_d(x;\phi)
\right).
$$

    <p>If the decoder likelihood is modeled with a squared-error reconstruction term, the practical VAE loss becomes</p>

$$
L(\phi,\theta;x)
=
\left\|x-\mu_\theta(z)\right\|^2
+
\frac{1}{2}
\sum_d
\left(
\mu_d(x;\phi)^2
+
\sigma_d(x;\phi)^2
-
1
-
2\log\sigma_d(x;\phi)
\right).
$$
  </div>
</details>

<details class="ddim-block ddim-algorithm">
  <summary>
    <span class="ddim-block__title"><strong>4.</strong> Pseudocode: VAE Training and Sampling</span>
  </summary>
  <div class="ddim-block__content">
    <div class="ddim-algorithm__require"><strong>Require:</strong> encoder parameters \(\phi\), decoder parameters \(\theta\)</div>
    <div class="ddim-algorithm-grid">
      <div class="ddim-algorithm-panel">
        <div class="ddim-algorithm__require"><strong>Training</strong> For existing data \(x\)</div>
        <div class="ddim-algorithm__body">
          <div class="ddim-algorithm__row">
            <div class="ddim-algorithm__num">1:</div>
            <div class="ddim-algorithm__code">Encode \(x\) and get \(\mu_\phi(x)\) and \(\sigma_\phi^2(x)\)</div>
            <div class="ddim-algorithm__comment"></div>
          </div>
          <div class="ddim-algorithm__row">
            <div class="ddim-algorithm__num">2:</div>
            <div class="ddim-algorithm__code">Sample \(\epsilon\sim\mathcal{N}(0,I)\)</div>
            <div class="ddim-algorithm__comment"></div>
          </div>
          <div class="ddim-algorithm__row">
            <div class="ddim-algorithm__num">3:</div>
            <div class="ddim-algorithm__code">\(z=\mu_\phi(x)+\sigma_\phi(x)\epsilon\)</div>
            <div class="ddim-algorithm__comment"></div>
          </div>
          <div class="ddim-algorithm__row">
            <div class="ddim-algorithm__num">4:</div>
            <div class="ddim-algorithm__code">Calculate the loss</div>
            <div class="ddim-algorithm__comment"></div>
          </div>
        </div>

$$
L(\phi,\theta;x)
=
\left\|x-\mu_\theta(z)\right\|^2
+
\frac{1}{2}
\sum_d
\left(
\mu_d(x;\phi)^2
+
\sigma_d(x;\phi)^2
-
1
-
2\log\sigma_d(x;\phi)
\right).
$$
      </div>

      <div class="ddim-algorithm-panel">
        <div class="ddim-algorithm__require"><strong>Sampling</strong></div>
        <div class="ddim-algorithm__body">
          <div class="ddim-algorithm__row">
            <div class="ddim-algorithm__num">1:</div>
            <div class="ddim-algorithm__code">Sample \(z\sim\mathcal{N}(0,I)\)</div>
            <div class="ddim-algorithm__comment"></div>
          </div>
          <div class="ddim-algorithm__row">
            <div class="ddim-algorithm__num">2:</div>
            <div class="ddim-algorithm__code">Get \(x=\mathrm{Decoder}(z)\)</div>
            <div class="ddim-algorithm__comment"></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</details>

### GAN

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>1.</strong> What is GAN?</span>
  </summary>
  <div class="ddim-block__content">
    <figure>
      <img src="/images/blog/diffusion/gan-minimax.png" alt="GAN minimax objective with generator and discriminator roles" style="width: 100%; max-width: 860px; display: block; margin: 0.75rem auto 0.35rem;">
      <figcaption style="color: #66707a; font-size: 0.9rem; line-height: 1.5; text-align: center;">
        A GAN trains a generator \(G\) and discriminator \(D\) in a minimax game.
      </figcaption>
    </figure>

    <p>A generative adversarial network has two players. The generator \(G\) starts from an easy-to-sample noise variable \(z\), such as Gaussian noise, and transforms it into a fake sample \(G(z)\). The discriminator \(D\) receives a sample and tries to decide whether it came from the real data distribution or from the generator.</p>

    <p>The discriminator tries to become better at distinguishing real samples from fake ones. The generator tries to make fake samples more realistic so that they fool the discriminator. This gives the minimax objective</p>

$$
\min_G \max_D V(D,G)
=
\mathbb{E}_{x\sim p_{\mathrm{data}}(x)}
\left[
\log D(x)
\right]
+
\mathbb{E}_{z\sim p_z(z)}
\left[
\log(1-D(G(z)))
\right].
$$

    <p>Intuitively, \(D(x)\) should be high for real data, while \(D(G(z))\) should be low for generated data. The generator improves by moving \(G(z)\) toward the target data distribution, while the discriminator keeps pressure on the generator by identifying the remaining differences.</p>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>2.</strong> Optimal discriminator</span>
  </summary>
  <div class="ddim-block__content">
    <p>For a fixed generator \(G\), let \(p_g\) be the distribution induced by generated samples \(G(z)\). The discriminator maximizes</p>

$$
V(D,G)
=
\mathbb{E}_{x\sim p_{\mathrm{data}}}
\left[
\log D(x)
\right]
+
\mathbb{E}_{x\sim p_g}
\left[
\log(1-D(x))
\right].
$$

    <p>Writing the expectations as integrals, the objective becomes</p>

$$
V(D,G)
=
\int
\left[
p_{\mathrm{data}}(x)\log D(x)
+
p_g(x)\log(1-D(x))
\right]dx.
$$

    <p>Since \(D(x)\) appears independently for each \(x\), optimize the integrand pointwise. Let \(y=D(x)\), \(a=p_{\mathrm{data}}(x)\), and \(b=p_g(x)\):</p>

$$
f(y)=a\log y+b\log(1-y).
$$

    <p>Set the derivative to zero:</p>

$$
\frac{df}{dy}
=
\frac{a}{y}
-
\frac{b}{1-y}
=0.
$$

$$
a(1-y)=by
\quad\Longrightarrow\quad
a=(a+b)y.
$$

    <p>Therefore the optimal discriminator is</p>

$$
D^*(x)
=
\frac{p_{\mathrm{data}}(x)}
{p_{\mathrm{data}}(x)+p_g(x)}.
$$

    <p>Interpretation: the discriminator estimates the probability that a sample came from the data distribution rather than the generator. At equilibrium, when \(p_g=p_{\mathrm{data}}\), we get \(D^*(x)=\frac{1}{2}\) everywhere.</p>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>3.</strong> Training the generator is minimizing JS divergence</span>
  </summary>
  <div class="ddim-block__content">
    <p>Plug the optimal discriminator back into the GAN value function. Let \(p_d=p_{\mathrm{data}}\). Since</p>

$$
D^*(x)
=
\frac{p_d(x)}{p_d(x)+p_g(x)},
\qquad
1-D^*(x)
=
\frac{p_g(x)}{p_d(x)+p_g(x)},
$$

    <p>we get</p>

$$
\begin{aligned}
V(D^*,G)
&=
\mathbb{E}_{x\sim p_d}
\left[
\log
\frac{p_d(x)}{p_d(x)+p_g(x)}
\right]
+
\mathbb{E}_{x\sim p_g}
\left[
\log
\frac{p_g(x)}{p_d(x)+p_g(x)}
\right] \\
&=
\int
p_d(x)\log
\frac{p_d(x)}{p_d(x)+p_g(x)}
\,dx
+
\int
p_g(x)\log
\frac{p_g(x)}{p_d(x)+p_g(x)}
\,dx.
\end{aligned}
$$

    <p>Define the mixture distribution</p>

$$
m(x)=\frac{1}{2}\left(p_d(x)+p_g(x)\right).
$$

    <p>Then \(p_d(x)+p_g(x)=2m(x)\), so</p>

$$
\begin{aligned}
V(D^*,G)
&=
\int
p_d(x)\log
\frac{p_d(x)}{2m(x)}
\,dx
+
\int
p_g(x)\log
\frac{p_g(x)}{2m(x)}
\,dx \\
&=
\int
p_d(x)\log
\frac{p_d(x)}{m(x)}
\,dx
+
\int
p_g(x)\log
\frac{p_g(x)}{m(x)}
\,dx
-
2\log 2 \\
&=
D_{\mathrm{KL}}(p_d\,\|\,m)
+
D_{\mathrm{KL}}(p_g\,\|\,m)
-
\log 4.
\end{aligned}
$$

    <p>The Jensen-Shannon divergence is</p>

$$
D_{\mathrm{JS}}(p_d\,\|\,p_g)
=
\frac{1}{2}
D_{\mathrm{KL}}(p_d\,\|\,m)
+
\frac{1}{2}
D_{\mathrm{KL}}(p_g\,\|\,m).
$$

    <p>Therefore</p>

$$
V(D^*,G)
=
-\log 4
+
2D_{\mathrm{JS}}(p_{\mathrm{data}}\,\|\,p_g).
$$

    <p>So after the discriminator is optimized, training the generator is equivalent to minimizing the Jensen-Shannon divergence between the data distribution and the generator distribution. The minimum is reached when \(p_g=p_{\mathrm{data}}\).</p>
  </div>
</details>

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

<details class="ddim-block ddim-algorithm">
  <summary>
    <span class="ddim-block__title"><strong>4.</strong> Pseudocode: DDPM Training and Sampling</span>
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

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>Q1.</strong> What is the reparameterization trick in DDPM, and why do we need it?</span>
  </summary>
  <div class="ddim-block__content">
    <p>The reparameterization trick in DDPM is to rewrite a noisy sample from \(q(x_t\mid x_0)\) as a deterministic function of the clean data \(x_0\), the timestep \(t\), and a standard Gaussian noise variable \(\epsilon\). Instead of treating \(x_t\) as an opaque Gaussian sample, write it as</p>

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

    <p>We need this because it isolates all randomness in \(\epsilon\), making the training target explicit: given \(x_t\) and \(t\), predict the noise that created \(x_t\).</p>

$$
\epsilon_\theta(x_t,t)\approx\epsilon.
$$

    <p>This is also where the simplified loss comes from. After the Gaussian KL term in the ELBO is reduced to matching reverse means, substitute the reparameterized \(x_t\) into the posterior mean. Mean matching becomes proportional to</p>

$$
\left\|
\epsilon-\epsilon_\theta(x_t,t)
\right\|_2^2.
$$

    <p>DDPM then drops the timestep-dependent weighting and constants, giving the practical objective</p>

$$
L_{\mathrm{simple}}
=
\mathbb{E}_{t,x_0,\epsilon}
\left[
\left\|
\epsilon-\epsilon_\theta
\left(
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
t
\right)
\right\|_2^2
\right].
$$
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>Q2.</strong> What are the three common DDPM prediction targets?</span>
  </summary>
  <div class="ddim-block__content">
    <p>A denoising model can be parameterized to predict different but equivalent quantities. Given \(x_t\) and \(t\), the network can predict the clean sample \(x_0\), the previous sample \(x_{t-1}\), or the added noise \(\epsilon\).</p>

    <p><strong>1. Predict \(x_0\).</strong> The model directly estimates the clean data:</p>

$$
\hat{x}_0 = f_\theta(x_t,t).
$$

    <p>Then use \(\hat{x}_0\) inside the true posterior mean \(\tilde{\mu}_t(x_t,\hat{x}_0)\) to sample \(x_{t-1}\).</p>

    <p><strong>2. Predict \(x_{t-1}\).</strong> The model directly predicts the next denoised step, usually through the reverse mean:</p>

$$
\mu_\theta(x_t,t)\approx x_{t-1},
\qquad
p_\theta(x_{t-1}\mid x_t)
=
\mathcal{N}\left(x_{t-1};\mu_\theta(x_t,t),\sigma_t^2I\right).
$$

    <p>This matches the Markov reverse-chain view most directly, but it is less common as the raw neural-network target.</p>

    <p><strong>3. Predict noise \(\epsilon\).</strong> The model predicts the Gaussian noise used to create \(x_t\):</p>

$$
\epsilon_\theta(x_t,t)\approx\epsilon.
$$

    <p>From the noise prediction, recover the clean sample estimate</p>

$$
\hat{x}_0
=
\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t)}
{\sqrt{\bar{\alpha}_t}},
$$

    <p>or equivalently the reverse mean</p>

$$
\mu_\theta(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}
\epsilon_\theta(x_t,t)
\right).
$$

    <p>The original DDPM simplified objective uses noise prediction because the training target \(\epsilon\) is known exactly after reparameterization, and the loss becomes a simple mean-squared error.</p>
  </div>
</details>

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

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>1.</strong> Euler solver</span>
  </summary>
  <div class="ddim-block__content">
    <figure>
      <img src="/images/blog/diffusion/euler-solver-step.png" alt="Euler solver step follows the local velocity from x_t to x_{t plus delta t}" style="width: 100%; max-width: 820px; display: block; margin: 0.75rem auto 0.35rem;">
      <figcaption style="color: #66707a; font-size: 0.9rem; line-height: 1.5; text-align: center;">
        Euler's method follows the local velocity \(v_\theta(x_t,t)\) for a small time step \(\Delta t\).
      </figcaption>
    </figure>

    <p>For a deterministic sampler, we can view generation as solving an ODE from noise to data. Suppose the learned model gives a velocity field \(v_\theta(x_t,t)\):</p>

$$
\frac{d x_t}{dt}
=
v_\theta(x_t,t).
$$

    <p>Euler's method is the simplest numerical solver for this ODE. At time \(t\), it approximates the curve by a straight line in the current velocity direction:</p>

$$
x_{t+\Delta t}
=
x_t
+
\Delta t\,v_\theta(x_t,t).
$$

    <p>In diffusion sampling, the sign of \(\Delta t\) depends on the time convention. If time runs from data to noise, generation integrates backward from high noise to low noise, so the update is often written as</p>

$$
x_{t-\Delta t}
=
x_t
-
\Delta t\,v_\theta(x_t,t).
$$

    <p>The benefit is simplicity: one model evaluation gives one step. The downside is that Euler is only first-order accurate, so using too few steps can drift away from the true trajectory. Higher-order solvers improve this by using better approximations to the same underlying ODE path.</p>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>2.</strong> Midpoint solver</span>
  </summary>
  <div class="ddim-block__content">
    <figure>
      <img src="/images/blog/diffusion/midpoint-solver-step.png" alt="Midpoint solver estimates the midpoint velocity before taking the full step" style="width: 100%; max-width: 820px; display: block; margin: 0.75rem auto 0.35rem;">
      <figcaption style="color: #66707a; font-size: 0.9rem; line-height: 1.5; text-align: center;">
        The midpoint method first estimates \(x_{t+\frac{1}{2}\Delta t}\), then uses the midpoint velocity for the full step.
      </figcaption>
    </figure>

    <p>The midpoint solver improves over Euler by evaluating the velocity at an estimated middle point rather than only at the beginning of the interval.</p>

$$
\frac{d x_t}{dt}
=
v_\theta(x_t,t).
$$

    <p>First, take a half Euler step to estimate the midpoint:</p>

$$
x_{t+\frac{1}{2}\Delta t}
=
x_t
+
\frac{1}{2}\Delta t\,v_\theta(x_t,t).
$$

    <p>Then evaluate the velocity at that midpoint and use it for the full update:</p>

$$
x_{t+\Delta t}
=
x_t
+
\Delta t\,
v_\theta
\left(
x_{t+\frac{1}{2}\Delta t},
t+\frac{1}{2}\Delta t
\right).
$$

    <p>Compared with Euler, this uses one extra model evaluation per step, but it usually follows the curved trajectory more accurately. In diffusion sampling, this is useful when the denoising path bends noticeably between two selected timesteps.</p>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>3.</strong> Second-order Heun solver</span>
  </summary>
  <div class="ddim-block__content">
    <figure>
      <img src="/images/blog/diffusion/heun-solver-step.png" alt="Second-order Heun solver averages beginning and endpoint velocities" style="width: 100%; max-width: 820px; display: block; margin: 0.75rem auto 0.35rem;">
      <figcaption style="color: #66707a; font-size: 0.9rem; line-height: 1.5; text-align: center;">
        Heun's method predicts an endpoint, evaluates the velocity there, then averages the two velocities.
      </figcaption>
    </figure>

    <p>Heun's method is another second-order solver. It first uses Euler to predict where the next point would be:</p>

$$
\hat{x}_{t+\Delta t}
=
x_t
+
\Delta t\,v_\theta(x_t,t).
$$

    <p>Then it evaluates the velocity at the predicted endpoint:</p>

$$
v_{\mathrm{end}}
=
v_\theta
\left(
\hat{x}_{t+\Delta t},
t+\Delta t
\right).
$$

    <p>Finally, it averages the starting velocity and the endpoint velocity:</p>

$$
x_{t+\Delta t}
=
x_t
+
\frac{\Delta t}{2}
\left[
v_\theta(x_t,t)
+
v_\theta
\left(
\hat{x}_{t+\Delta t},
t+\Delta t
\right)
\right].
$$

    <p>Compared with midpoint, Heun also uses two velocity evaluations, but the second one is taken at the predicted endpoint rather than the middle. This makes it a predictor-corrector method: predict with Euler, then correct using the average slope.</p>
  </div>
</details>

### DPM solvers

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>1.</strong> DPM-Solver</span>
  </summary>
  <div class="ddim-block__content">
    <p>DPM-Solver is a diffusion-specific ODE solver. The key observation is that the probability flow ODE has a known linear part and a learned nonlinear part:</p>

$$
\frac{d x_t}{dt}
=
f(t)x_t
-
\frac{1}{2}g^2(t)\nabla_x\log q_t(x_t).
$$

    <p>Here the first term \(f(t)x_t\) is linear and known from the noise schedule. The second term contains the learned score. In the noise-prediction parameterization, the score term can be written using \(\epsilon_\theta(x_t,t)\). For VP-style parameterization,</p>

$$
f(t)=\frac{d\log\alpha_t}{dt},
\qquad
g^2(t)
=
\frac{d\sigma_t^2}{dt}
-
2\frac{d\log\alpha_t}{dt}\sigma_t^2
=
2\sigma_t^2
\left(
\frac{d\log\sigma_t}{dt}
-
\frac{d\log\alpha_t}{dt}
\right).
$$

    <p>Because the linear part is known, we can solve it analytically and only approximate the learned noise term:</p>

$$
x_t
=
e^{\int_s^t f(\tau)\,d\tau}x_s
+
\int_s^t
\left(
e^{\int_\tau^t f(r)\,dr}
\frac{g^2(\tau)}{2\sigma_\tau}
\epsilon_\theta(x_\tau,\tau)
\right)
d\tau.
$$

    <p>DPM-Solver then changes variables to log-SNR time \(\lambda\). One update from \(t_{i-1}\) to \(t_i\) can be written as</p>

$$
x_{t_{i-1}\to t_i}
=
\frac{\alpha_{t_i}}{\alpha_{t_{i-1}}}\tilde{x}_{t_{i-1}}
-
\alpha_{t_i}
\int_{\lambda_{t_{i-1}}}^{\lambda_{t_i}}
e^{-\lambda}
\hat{\epsilon}_\theta(\hat{x}_\lambda,\lambda)
d\lambda.
$$

    <p>The remaining difficult part is the learned noise function \(\hat{\epsilon}_\theta(\hat{x}_\lambda,\lambda)\). DPM-Solver approximates it with a Taylor expansion around the previous time:</p>

$$
\hat{\epsilon}_\theta(\hat{x}_\lambda,\lambda)
=
\sum_{n=0}^{k-1}
\frac{(\lambda-\lambda_{t_{i-1}})^n}{n!}
\hat{\epsilon}_\theta^{(n)}
\left(
\hat{x}_{\lambda_{t_{i-1}}},
\lambda_{t_{i-1}}
\right)
+
O\left((\lambda-\lambda_{t_{i-1}})^k\right).
$$

    <p>Substituting this expansion into the integral gives</p>

$$
\begin{aligned}
x_{t_{i-1}\to t_i}
=&
\frac{\alpha_{t_i}}{\alpha_{t_{i-1}}}\tilde{x}_{t_{i-1}}
-
\alpha_{t_i}
\sum_{n=0}^{k-1}
\hat{\epsilon}_\theta^{(n)}
\left(
\hat{x}_{\lambda_{t_{i-1}}},
\lambda_{t_{i-1}}
\right) \\
&\quad\cdot
\int_{\lambda_{t_{i-1}}}^{\lambda_{t_i}}
e^{-\lambda}
\frac{(\lambda-\lambda_{t_{i-1}})^n}{n!}
d\lambda
+
O(h_i^{k+1}).
\end{aligned}
$$

    <p>The important split is: the derivatives of the learned noise term are estimated, while the integral involving \(e^{-\lambda}\) can be calculated analytically. For \(k=1\), we keep only the zeroth-order term and get the first-order DPM-Solver update:</p>

$$
\tilde{x}_{t_i}
=
\frac{\alpha_{t_i}}{\alpha_{t_{i-1}}}
\tilde{x}_{t_{i-1}}
-
\sigma_{t_i}
\left(e^{h_i}-1\right)
\epsilon_\theta(\tilde{x}_{t_{i-1}},t_{i-1}),
\qquad
h_i=\lambda_{t_i}-\lambda_{t_{i-1}}.
$$

    <p>So compared with generic ODE solvers, DPM-Solver uses the special structure of diffusion ODEs: integrate the known linear dynamics exactly, and approximate only the learned denoising term.</p>
  </div>
</details>

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

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>Q4.</strong> Why do midpoint and Heun have smaller error than Euler?</span>
  </summary>
  <div class="ddim-block__content">
    <p>For an ODE</p>

$$
\frac{dx}{dt}=f(x,t),
$$

    <p>one solver step is trying to approximate the true integral</p>

$$
x(t+\Delta t)
=
x(t)
+
\int_t^{t+\Delta t} f(x(s),s)\,ds.
$$

    <p>So the real question is: how well does the solver approximate the average slope over the interval?</p>

    <p><strong>Euler</strong> uses the left endpoint slope:</p>

$$
x_{t+\Delta t}^{\mathrm{Euler}}
=
x_t
+
\Delta t\,f(x_t,t).
$$

    <p>But the true solution has the Taylor expansion</p>

$$
x(t+\Delta t)
=
x(t)
+
\Delta t f
+
\frac{(\Delta t)^2}{2}f'
+
O(\Delta t^3),
$$

    <p>where \(f\) and \(f'\) are evaluated along the trajectory at time \(t\). Euler keeps only the first slope term, so it misses the second-order correction. This gives Euler a local truncation error of order \(O(\Delta t^2)\).</p>

    <p><strong>Midpoint</strong> instead estimates the slope halfway through the interval. The average slope over the interval satisfies</p>

$$
\frac{1}{\Delta t}
\int_t^{t+\Delta t} f(x(s),s)\,ds
=
f(t)
+
\frac{\Delta t}{2}f'(t)
+
O(\Delta t^2).
$$

    <p>The midpoint slope has the matching expansion</p>

$$
f\left(
t+\frac{1}{2}\Delta t
\right)
=
f
+
\frac{\Delta t}{2}f'
+
O(\Delta t^2).
$$

    <p>That is the key: midpoint matches the average slope through the first variation term. It cancels Euler's leading local bias, so the local truncation error improves from \(O(\Delta t^2)\) to \(O(\Delta t^3)\).</p>

    <p><strong>Heun</strong> is better for a similar reason. It first predicts an endpoint with Euler, then evaluates the slope again at that endpoint. Instead of trusting only the left endpoint slope, Heun averages the beginning slope and the predicted endpoint slope:</p>

$$
x_{t+\Delta t}^{\mathrm{Heun}}
=
x_t
+
\frac{\Delta t}{2}
\left[
f(x_t,t)
+
f(\hat{x}_{t+\Delta t},t+\Delta t)
\right],
\qquad
\hat{x}_{t+\Delta t}=x_t+\Delta t f(x_t,t).
$$

    <p>The predicted endpoint slope has the Taylor expansion</p>

$$
f(\hat{x}_{t+\Delta t},t+\Delta t)
=
f
+
\Delta t\,f'
+
O(\Delta t^2).
$$

    <p>Therefore the average of the starting slope and predicted endpoint slope is</p>

$$
\frac{1}{2}
\left[
f(x_t,t)
+
f(\hat{x}_{t+\Delta t},t+\Delta t)
\right]
=
f
+
\frac{\Delta t}{2}f'
+
O(\Delta t^2).
$$

    <p>This average slope matches the same first-order change of the vector field as midpoint, so it cancels the same leading Euler error term. Midpoint samples the slope in the middle; Heun averages the slope at the beginning and the predicted end. Both are second-order methods.</p>

    <table>
      <thead>
        <tr>
          <th>Method</th>
          <th>Slope approximation</th>
          <th>Local error</th>
          <th>Global error</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Euler</td>
          <td>left endpoint slope</td>
          <td>\(O(\Delta t^2)\)</td>
          <td>\(O(\Delta t)\)</td>
        </tr>
        <tr>
          <td>Midpoint</td>
          <td>midpoint slope</td>
          <td>\(O(\Delta t^3)\)</td>
          <td>\(O(\Delta t^2)\)</td>
        </tr>
        <tr>
          <td>Heun</td>
          <td>average of start and predicted-end slopes</td>
          <td>\(O(\Delta t^3)\)</td>
          <td>\(O(\Delta t^2)\)</td>
        </tr>
      </tbody>
    </table>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>Q5.</strong> Why do people use DPM-Solver less for flow matching models?</span>
  </summary>
  <div class="ddim-block__content">
    <p>DPM-Solver is designed for diffusion probability flow ODEs. It uses the special diffusion parameterization to split the ODE into a known linear part and a learned noise-prediction part:</p>

$$
\frac{dx_t}{dt}
=
\underbrace{f(t)x_t}_{\text{known linear part}}
+
\underbrace{\text{learned denoising term}}_{\epsilon_\theta\ \text{or score term}}.
$$

    <p>Because the linear part is known, DPM-Solver can integrate that part analytically and only approximate the learned term. This is very useful for DDPM/score-based models, where the sampler is built around the noise schedule, log-SNR time, and a noise or score prediction network.</p>

    <p>Flow matching changes the setup. The model is trained to directly predict a velocity field:</p>

$$
\frac{dx_t}{dt}
=
v_\theta(x_t,t).
$$

    <p>So there is no special diffusion linear term that must be separated out. The learned object is already the ODE velocity. This makes generic ODE solvers such as Euler, midpoint, Heun, or Runge-Kutta natural choices.</p>

    <p>In short: DPM-Solver is powerful because it exploits diffusion-specific structure. Flow matching removes much of that structure by directly learning the transport velocity, so the advantage of a diffusion-specific solver becomes less central.</p>
  </div>
</details>

## Design Space
> How to define a diffusion model?

<div class="design-space-box">
  <div class="design-space-title">The design space of diffusion models</div>
  <div class="design-space-grid">
    <div class="design-space-card design-space-card--training">
      <h3>Training</h3>
      <ul>
        <li>Prefixed noise schedule</li>
        <li>Training noise sampling schedule</li>
        <li>Loss weighting w.r.t. time</li>
      </ul>
    </div>
    <div class="design-space-card design-space-card--model">
      <h3>Model</h3>
      <ul>
        <li>Reparameterization</li>
        <li>Input/output scaling</li>
        <li>How to do time conditioning</li>
      </ul>
    </div>
    <div class="design-space-card design-space-card--sampling">
      <h3>Sampling</h3>
      <ul>
        <li>Solver</li>
        <li>Sampling-time noise schedule</li>
        <li>Number of time steps</li>
      </ul>
    </div>
  </div>
</div>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>1.</strong> Prefixed noise schedule</span>
  </summary>
  <div class="ddim-block__content">
    <figure>
      <img src="/images/blog/diffusion/snr-linear-cosine-schedule.png" alt="Linear versus cosine noise schedule, cumulative signal, and signal-to-noise ratio curves" style="width: 100%; max-width: 920px; display: block; margin: 0.75rem auto 0.35rem;">
      <figcaption style="color: #66707a; font-size: 0.9rem; line-height: 1.5; text-align: center;">
        Linear and cosine schedules produce different \(\beta_t\), \(\bar{\alpha}_t\), and SNR trajectories.
      </figcaption>
    </figure>

    <p>The three y-axes correspond to the quantities in the forward noising process:</p>

$$
q(x_t \mid x_{t-1})
=
\mathcal{N}\!\left(x_t;\sqrt{1-\beta_t}\,x_{t-1},\beta_t I\right),
$$

$$
\alpha_t = 1-\beta_t,
\qquad
\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s,
$$

$$
q(x_t \mid x_0)
=
\mathcal{N}\!\left(x_t;\sqrt{\bar{\alpha}_t}\,x_0,(1-\bar{\alpha}_t)I\right),
\qquad
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
$$

$$
\mathrm{SNR}(t)
=
\frac{\text{signal variance}}{\text{noise variance}}
=
\frac{\bar{\alpha}_t}{1-\bar{\alpha}_t},
\qquad
\mathrm{SNR}_{\mathrm{dB}}(t)
=
10\log_{10}\!\left(\frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}\right).
$$

    <p><strong>Linear scheduler</strong> chooses the noise rates \(\beta_t\) by linear interpolation:</p>

$$
\beta_t
=
\beta_{\min}
+
\frac{t-1}{T-1}
\left(\beta_{\max}-\beta_{\min}\right).
$$

    <p><strong>Cosine scheduler</strong> instead defines the cumulative signal \(\bar{\alpha}_t\) with a cosine-shaped curve, then derives \(\beta_t\) from it:</p>

$$
\bar{\alpha}_t
=
\frac{f(t)}{f(0)},
\qquad
f(t)
=
\cos^2\!\left(
\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}
\right),
$$

$$
\beta_t
=
1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}.
$$

    <figure>
      <img src="/images/blog/diffusion/snr-forward-linear-cosine.png" alt="Forward noising comparison for linear and cosine schedules across timesteps" style="width: 100%; max-width: 920px; display: block; margin: 0.75rem auto 0.35rem;">
      <figcaption style="color: #66707a; font-size: 0.9rem; line-height: 1.5; text-align: center;">
        The cosine schedule preserves visible signal longer, while the linear schedule destroys the image earlier.
      </figcaption>
    </figure>

    <p>At small \(t\), SNR is high: \(x_t\) still looks close to data. At large \(t\), SNR is low: \(x_t\) is mostly noise. A linear \(\beta_t\) schedule can make SNR drop too aggressively, so many late timesteps contain almost no useful signal. A cosine scheduler is designed to control the SNR decay more smoothly, preserving signal for longer and making the denoising tasks across timesteps better balanced.</p>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>2.</strong> Training noise sampling schedule</span>
  </summary>
  <div class="ddim-block__content">
    <p>Variational Diffusion Models<sup class="footnote-ref" id="fnref:vdm"><a href="#fn:vdm">5</a></sup> frame diffusion training as likelihood-based variational learning. Their key observation is that the continuous-time VLB can be written in terms of the signal-to-noise ratio, so the noise schedule can be optimized jointly with the denoising model. In practice, they learn a log-SNR schedule that reduces the variance of the VLB estimator, making optimization more stable and efficient.</p>

    <figure>
      <img src="/images/blog/diffusion/vdm-learned-snr-schedule.png" alt="Learned log-SNR schedule and variance of VLB estimate from Variational Diffusion Models" style="width: 100%; max-width: 720px; display: block; margin: 0.75rem auto 0.35rem;">
      <figcaption style="color: #66707a; font-size: 0.9rem; line-height: 1.5; text-align: center;">
        The learned log-SNR schedule lowers the variance of the VLB estimate compared with hand-designed schedules.
      </figcaption>
    </figure>

    <p>After choosing the noising process, training still needs to decide which timestep \(t\) to sample for each data point. The simplest choice is uniform sampling:</p>

$$
t \sim \mathrm{Uniform}\{1,\ldots,T\}.
$$

    <p>But not every timestep is equally useful. Some noise levels may be too easy, while others may dominate the gradient. A more general view is to train under a timestep sampling distribution:</p>

$$
t \sim p_{\mathrm{train}}(t),
\qquad
\mathcal{L}
=
\mathbb{E}_{t \sim p_{\mathrm{train}},x_0,\epsilon}
\left[
w(t)
\left\|
\epsilon-\epsilon_\theta(x_t,t)
\right\|_2^2
\right].
$$

    <p>If the schedule itself is learned, one convenient parameterization is the log-SNR curve:</p>

$$
\gamma_\eta(t)
=
\log \mathrm{SNR}_\eta(t)
=
\log\frac{\bar{\alpha}_\eta(t)}{1-\bar{\alpha}_\eta(t)}.
$$

    <p>Once \(\gamma_\eta(t)\) is learned, it determines the signal and noise scales:</p>

$$
\bar{\alpha}_\eta(t)
=
\mathrm{sigmoid}\!\left(\gamma_\eta(t)\right),
\qquad
1-\bar{\alpha}_\eta(t)
=
\mathrm{sigmoid}\!\left(-\gamma_\eta(t)\right).
$$

$$
x_t
=
\sqrt{\bar{\alpha}_\eta(t)}x_0
+
\sqrt{1-\bar{\alpha}_\eta(t)}\epsilon.
$$

    <p>The key idea is that the model does not only care what noise levels exist; it also cares how often training visits each noise level.</p>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>3.</strong> Loss weighting w.r.t. time</span>
  </summary>
  <div class="ddim-block__content">
    <p>Even with the same timestep sampler, we can change how much each timestep contributes to the objective by adding a time-dependent weight:</p>

$$
\mathcal{L}
=
\mathbb{E}_{t,x_0,\epsilon}
\left[
w(t)
\left\|
\epsilon-\epsilon_\theta(x_t,t)
\right\|_2^2
\right].
$$

    <p>Another equivalent view is: sample the more difficult timesteps more frequently during training time. If a certain noise level produces larger error, we can either increase its loss weight \(w(t)\), or sample that \(t\) more often through \(p_{\mathrm{train}}(t)\).</p>

    <figure>
      <img src="/images/blog/diffusion/loss-weighting-time-sampling.png" alt="Loss as a function of noise scale with a sampling distribution over sigma" style="width: 100%; max-width: 560px; display: block; margin: 0.75rem auto 0.35rem;">
      <figcaption style="color: #66707a; font-size: 0.9rem; line-height: 1.5; text-align: center;">
        Observed initial (green) and final loss per noise level, representative of the 32×32 (blue) and 64×64 (orange) models. The shaded regions represent the standard deviation over 10k random samples. Our proposed training sample density is shown by the dashed red curve.
      </figcaption>
    </figure>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>4.</strong> Reparameterization</span>
  </summary>
  <div class="ddim-block__content">
    <p>The same noised sample can be written as a mixture of clean signal and noise:</p>

$$
x_t
=
\alpha_t x_0+\sigma_t\epsilon.
$$

    <p>So the denoiser can be trained to predict different but equivalent targets:</p>

$$
x_{0,\theta}(x_t,t),
\qquad
\epsilon_\theta(x_t,t),
\qquad
v_\theta(x_t,t).
$$

$$
\begin{array}{c|c|c}
\textbf{Prediction target}
&
\textbf{Easy regime}
&
\textbf{Hard regime}
\\
\hline
x_{0,\theta}(x_t,t)
&
\text{low noise / high SNR}
&
\text{high noise / low SNR}
\\
\epsilon_\theta(x_t,t)
&
\text{high noise / low SNR}
&
\text{low noise / high SNR}
\\
v_\theta(x_t,t)
&
\text{balanced}
&
\text{balanced}
\end{array}
$$

    <p>Progressive Distillation for Fast Sampling of Diffusion Models<sup class="footnote-ref" id="fnref:progressive-distillation"><a href="#fn:progressive-distillation">6</a></sup> uses velocity prediction as a better-balanced parameterization:</p>

$$
v
=
\alpha_t\epsilon-\sigma_t x_0.
$$
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>5.</strong> Input/output scaling</span>
  </summary>
  <div class="ddim-block__content">
    <p>At different noise levels, \(x_t\) can have different magnitudes. Input/output scaling normalizes the signal seen by the network and rescales the prediction back to the desired target:</p>

$$
\hat{x}_t
=
c_{\mathrm{in}}(t)x_t,
\qquad
\hat{y}_\theta
=
c_{\mathrm{out}}(t)F_\theta(\hat{x}_t,t).
$$

    <p>This is a model design choice: the mathematical diffusion process can be fixed, while the neural network interface is rescaled for easier learning.</p>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>6.</strong> How to do time conditioning</span>
  </summary>
  <div class="ddim-block__content">
    <p>The denoiser needs to know the current noise level. Instead of feeding raw \(t\), many models embed either \(t\), \(\sigma_t\), or log-SNR:</p>

$$
\lambda_t
=
\log\frac{\alpha_t}{\sigma_t},
\qquad
e_t
=
\mathrm{Embed}(t)
\quad\text{or}\quad
\mathrm{Embed}(\lambda_t).
$$

    <p>The embedding can be injected through additive bias, adaptive normalization, or attention conditioning.</p>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>7.</strong> Solver</span>
  </summary>
  <div class="ddim-block__content">
    <p>See sampler section.</p>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>8.</strong> Sampling-time noise schedule</span>
  </summary>
  <div class="ddim-block__content">
    <p>The schedule used during sampling does not have to visit every training timestep. In EDM, Karras et al.<sup class="footnote-ref" id="fnref:edm"><a href="#fn:edm">7</a></sup> study this as a discretization problem: for a fixed number of sampling steps, choose the noise levels \(\{\sigma_i\}\) so the numerical solver has smaller truncation error. A more detailed analysis can be found in Appendix D.1 of EDM.</p>

$$
\sigma_0=\sigma_{\max}
>
\sigma_1
>
\cdots
>
\sigma_N=0.
$$

    <p>A convenient choice is to sample uniformly after applying a polynomial warp. For \(i<N\),</p>

$$
\sigma_i
=
\left(
\sigma_{\max}^{1/\rho}
+
\frac{i}{N-1}
\left(
\sigma_{\min}^{1/\rho}
-
\sigma_{\max}^{1/\rho}
\right)
\right)^\rho,
\qquad
\sigma_N=0.
$$

    <p>When \(\rho=1\), this is uniform spacing in \(\sigma\). As \(\rho\) increases, more sampling points are allocated to low-noise levels, where Euler and Heun steps can otherwise accumulate larger visible error.</p>

    <figure>
      <img src="/images/blog/diffusion/sampling-noise-rho-schedule.png" alt="Truncation error and FID for different polynomial noise schedules" style="width: 100%; max-width: 920px; display: block; margin: 0.75rem auto 0.35rem;">
      <figcaption style="color: #66707a; font-size: 0.9rem; line-height: 1.5; text-align: center;">
        Different \(\rho\) values redistribute sampling steps across noise levels. Larger \(\rho\) places more steps near low noise; the right plot shows that this affects FID.
      </figcaption>
    </figure>

    <p>The design question is therefore not just “how many steps?”, but also “where should those steps be placed?” A good sampling-time schedule spends more resolution where solver error matters most.</p>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>9.</strong> Number of time steps</span>
  </summary>
  <div class="ddim-block__content">
    <table style="width: 100%; border-collapse: collapse; margin: 0.9rem 0; font-size: 0.95rem;">
      <thead>
        <tr>
          <th style="border-bottom: 1px solid #d6dde3; padding: 0.45rem; text-align: left;">Model / sampler family</th>
          <th style="border-bottom: 1px solid #d6dde3; padding: 0.45rem; text-align: left;">Typical steps</th>
          <th style="border-bottom: 1px solid #d6dde3; padding: 0.45rem; text-align: left;">Use case</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="padding: 0.45rem; border-bottom: 1px solid #edf1f4;">Original DDPM-style ancestral sampling</td>
          <td style="padding: 0.45rem; border-bottom: 1px solid #edf1f4;">hundreds to \(1000\)</td>
          <td style="padding: 0.45rem; border-bottom: 1px solid #edf1f4;">slow reference sampling</td>
        </tr>
        <tr>
          <td style="padding: 0.45rem; border-bottom: 1px solid #edf1f4;">DDIM / PLMS / classic Stable Diffusion samplers</td>
          <td style="padding: 0.45rem; border-bottom: 1px solid #edf1f4;">\(20\)-\(50\)</td>
          <td style="padding: 0.45rem; border-bottom: 1px solid #edf1f4;">standard image generation</td>
        </tr>
        <tr>
          <td style="padding: 0.45rem; border-bottom: 1px solid #edf1f4;">DPM-Solver / DPM++ / Euler / Heun</td>
          <td style="padding: 0.45rem; border-bottom: 1px solid #edf1f4;">\(10\)-\(30\)</td>
          <td style="padding: 0.45rem; border-bottom: 1px solid #edf1f4;">fast high-quality sampling</td>
        </tr>
        <tr>
          <td style="padding: 0.45rem; border-bottom: 1px solid #edf1f4;">Flow-matching / rectified-flow image models</td>
          <td style="padding: 0.45rem; border-bottom: 1px solid #edf1f4;">\(20\)-\(50\)</td>
          <td style="padding: 0.45rem; border-bottom: 1px solid #edf1f4;">current large text-to-image models</td>
        </tr>
        <tr>
          <td style="padding: 0.45rem;">Distilled / turbo / few-step models</td>
          <td style="padding: 0.45rem;">\(1\)-\(8\)</td>
          <td style="padding: 0.45rem;">real-time or interactive generation</td>
        </tr>
      </tbody>
    </table>
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

## A industry level text-to-image diffusion pipeline
> Study notes on Krea 2 Technical Report<sup class="footnote-ref" id="fnref:krea2"><a href="#fn:krea2">8</a></sup>

## A industry level video diffusion pipeline

## Reading list

### Foundations
- [A Connection Between Score Matching and Denoising Autoencoders](https://ieeexplore.ieee.org/abstract/document/6795935)
- [Interpretation and generalization of score matching](https://arxiv.org/pdf/1205.2629)
- [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)
- [Maximum Likelihood Training of Score-Based Diffusion Models](https://arxiv.org/abs/2101.09258)

### DDPM, DDIM, and score-based models
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)
- [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)

### Samplers
- [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927)

### Flow matching
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)

### One-step and few-step models
- [Consistency Models](https://arxiv.org/abs/2303.01469)
- [Improved Techniques for Training Consistency Models](https://arxiv.org/abs/2310.14189)
- [Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models](https://arxiv.org/abs/2410.11081)
- [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447)

## Coding list

### Warmup
- [Tensor Puzzles](https://colab.research.google.com/github/srush/Tensor-Puzzles/blob/main/Tensor%20Puzzlers.ipynb): practice tensor indexing, broadcasting, and vectorization.

- https://github.com/KellyYutongHe/cmu-10799-diffusion

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
    <li id="fn:vdm">
      Kingma, Salimans, Poole, &amp; Ho.
      <em>Variational Diffusion Models.</em>
      NeurIPS 2021.
      <a href="https://arxiv.org/abs/2107.00630">arXiv:2107.00630</a>.
      <a href="#fnref:vdm" class="footnote-back" title="back to text">↩︎</a>
    </li>
    <li id="fn:progressive-distillation">
      Salimans &amp; Ho.
      <em>Progressive Distillation for Fast Sampling of Diffusion Models.</em>
      ICLR 2022.
      <a href="https://arxiv.org/abs/2202.00512">arXiv:2202.00512</a>.
      <a href="#fnref:progressive-distillation" class="footnote-back" title="back to text">↩︎</a>
    </li>
    <li id="fn:edm">
      Karras, Aittala, Aila, &amp; Laine.
      <em>Elucidating the Design Space of Diffusion-Based Generative Models.</em>
      NeurIPS 2022.
      <a href="https://arxiv.org/abs/2206.00364">arXiv:2206.00364</a>.
      <a href="#fnref:edm" class="footnote-back" title="back to text">↩︎</a>
    </li>
    <li id="fn:krea2">
      Krea.
      <em>Krea 2 Technical Report.</em>
      Technical report, 2026.
      <a href="https://www.krea.ai/blog/krea-2-technical-report">krea.ai</a>.
      <a href="#fnref:krea2" class="footnote-back" title="back to text">↩︎</a>
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
