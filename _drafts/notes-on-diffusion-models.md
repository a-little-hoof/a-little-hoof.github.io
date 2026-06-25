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
.page__content #draft-toc a.toc-subsection {
  padding-left: 1.1rem;
  color: #7a8490;
  font-size: 0.78rem;
  font-weight: 450;
}
.page__content #draft-toc a.toc-subsection::before {
  content: "· ";
}
.page__content #draft-toc a.toc-section {
  margin-top: 0.22rem;
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

<details class="ddim-block">
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

### Score-based diffusion models

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>1.</strong> What are score-based diffusion models?</span>
  </summary>
  <div class="ddim-block__content">
    <p>Score-based diffusion models describe diffusion as a continuous-time stochastic process. The <strong>forward process</strong> gradually turns data into noise, and the <strong>reverse process</strong> uses a learned score function to denoise samples back into data.</p>

    <figure>
      <img src="/images/blog/diffusion/score-sde-forward-reverse.png" alt="Forward diffusion process and reverse denoising process" style="width: 100%; max-width: 920px; display: block; margin: 0.75rem auto 0.35rem;">
      <figcaption style="color: #66707a; font-size: 0.9rem; line-height: 1.5; text-align: center;">
        The fixed <strong>forward process</strong> maps data to noise. The generative <strong>reverse process</strong> maps noise back to data.
      </figcaption>
    </figure>

    <p>The <strong>forward process</strong> is controlled by a forward SDE:</p>

$$
d x
=
f(x,t)\,dt
+
g(t)\,dw,
$$

    <p>where \(f(x,t)\) is the drift, \(g(t)\) is the diffusion coefficient, and \(w\) is standard Brownian motion. This SDE defines the noisy marginal distribution \(p_t(x)\) at every time \(t\).</p>

    <p>The <strong>reverse process</strong> is also an SDE. Its drift depends on the score function \(\nabla_x\log p_t(x)\):</p>

$$
d x
=
\left[
f(x,t)
-
g(t)^2\nabla_x\log p_t(x)
\right]dt
+
g(t)\,d\bar{w}.
$$

    <p>So the model's central job is to learn the score function</p>

$$
s_\theta(x,t)
\approx
\nabla_x\log p_t(x).
$$

    <p>In principle, this can be trained with score matching on the noisy marginal:</p>

$$
J_{\mathrm{SM}}(\theta)
=
\mathbb{E}_{q_t(x_t)}
\left[
\frac{1}{2}
\left\|
s_\theta(x_t,t)
-
\nabla_{x_t}\log q_t(x_t)
\right\|_2^2
\right].
$$

    <p>In practice, people usually use denoising score matching: sample a clean data point, add known Gaussian noise, and train the model to match the known conditional score of that noising process.</p>

$$
J_{\mathrm{DSM}}(\theta)
=
\mathbb{E}_{q_t(x_0,x_t)}
\left[
\frac{1}{2}
\left\|
s_\theta(x_t,t)
-
\nabla_{x_t}\log q_t(x_t\mid x_0)
\right\|_2^2
\right].
$$
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>2.</strong> Mathematical foundation</span>
  </summary>
  <div class="ddim-block__content">
    <p>This part is <strong>not important</strong> in understanding score-based diffusion models, but it explains where the reverse SDE and probability flow ODE come from.</p>

    <p>The current storyline is: Brownian motion gives the continuous-time noise source; Itô calculus tells us how functions of stochastic paths evolve; Fokker-Planck describes the evolution of densities; and time reversal explains why the reverse sampler needs the score.</p>

    <details class="ddim-block foundation-subblock">
      <summary>
        <span class="ddim-block__title">Brownian motion and Itô integral</span>
      </summary>
      <div class="ddim-block__content">

    <p><strong>Brownian motion.</strong> A standard Brownian motion \(w_t\) is the continuous-time noise source. It starts at zero, has independent increments, and satisfies</p>

$$
w_{t+\Delta t}-w_t
\sim
\mathcal{N}(0,\Delta t I).
$$

    <p>Informally, over a tiny interval \(dt\), this means</p>

$$
dw_t
\sim
\mathcal{N}(0,dt\,I),
\qquad
dw_t
\approx
\sqrt{dt}\,\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

    <p>So \(dw_t\) has typical size \(\sqrt{dt}\), but it is still random; it is not literally equal to \(\sqrt{dt}\). This is why the drift term is order \(dt\), while the noise term is order \(\sqrt{dt}\):</p>

$$
f(x_t,t)\,dt=O(dt),
\qquad
g(t)\,dw_t=O(\sqrt{dt}).
$$

    <p><strong>Key intuition:</strong> because \((dw_t)^2\) is order \(dt\), the quadratic Brownian term survives in Itô's formula.</p>

    <p>So Brownian motion is not differentiable in the ordinary sense. Instead of writing a classical derivative \(dw_t/dt\), we define integrals with respect to Brownian motion.</p>

    <p><strong>Itô integral.</strong> For a deterministic function \(h(t)\), the stochastic integral is the limit of weighted Brownian increments:</p>

$$
\int_0^t h(s)\,dw_s
\approx
\sum_{i=1}^n
h(t_{i-1})
\left(w_{t_i}-w_{t_{i-1}}\right).
$$

    <p>Because each Brownian increment is Gaussian, the integral is also Gaussian:</p>

$$
\int_0^t h(s)\,dw_s
\sim
\mathcal{N}
\left(
0,
\int_0^t h(s)^2\,ds
\right).
$$

        <p><strong>Conclusion:</strong> stochastic calculus lets us give precise meaning to continuous-time noise injection.</p>
      </div>
    </details>

    <details class="ddim-block foundation-subblock">
      <summary>
        <span class="ddim-block__title">Itô process and Itô formula</span>
      </summary>
      <div class="ddim-block__content">

    <p><strong>Itô process.</strong> An SDE combines an ordinary integral and an Itô integral:</p>

$$
d x_t
=
b(x_t,t)\,dt
+
g(t)\,dw_t.
$$

    <p>Equivalently, in integral form,</p>

$$
x_t-x_0
=
\int_0^t b(x_s,s)\,ds
+
\int_0^t g(s)\,dw_s.
$$

    <p>Here \(b(x,t)\) is the drift and \(g(t)\) is the diffusion coefficient. The drift transports samples; the diffusion term spreads them out. <strong>The forward SDE defines both random sample paths and a family of marginal densities \(p_t(x)\).</strong></p>

    <p><strong>Itô's formula.</strong> If \(f(t,x_t)\) is a smooth function of an Itô process, the usual chain rule gets an extra second-order term:</p>

$$
d\,f(t,x_t)
=
\left[
\partial_t f
+
\nabla_x f^\top b
+
\frac{1}{2}g(t)^2\Delta_x f
\right]dt
+
g(t)\nabla_x f^\top dw_t.
$$

    <p>One quick way to see where this comes from is to Taylor expand \(f(t,x_t)\):</p>

$$
df
=
\partial_t f\,dt
+
\nabla_x f^\top dx_t
+
\frac{1}{2}dx_t^\top\nabla_x^2 f\,dx_t
+
\text{higher order terms}.
$$

    <p>Substitute the SDE \(dx_t=b(x_t,t)dt+g(t)dw_t\). The stochastic scaling rules are</p>

$$
dt^2\approx 0,
\qquad
dt\,dw_t\approx 0,
\qquad
dw_t\,dw_t^\top\approx I\,dt.
$$

    <p>Therefore</p>

$$
dx_tdx_t^\top
=
\left(b\,dt+g\,dw_t\right)
\left(b\,dt+g\,dw_t\right)^\top
\approx
g(t)^2I\,dt.
$$

    <p>The second-order Taylor term becomes</p>

$$
\frac{1}{2}dx_t^\top\nabla_x^2 f\,dx_t
=
\frac{1}{2}g(t)^2\Delta_x f\,dt.
$$

    <p>The first-order term becomes</p>

$$
\nabla_x f^\top dx_t
=
\nabla_x f^\top b\,dt
+
g(t)\nabla_x f^\top dw_t.
$$

    <p>Compare this with an ODE. If there is no Brownian noise, the dynamics are</p>

$$
dx_t
=
b(x_t,t)\,dt.
$$

    <p>The ordinary chain rule gives</p>

$$
df(t,x_t)
=
\partial_t f\,dt
+
\nabla_x f^\top dx_t
=
\left[
\partial_t f
+
\nabla_x f^\top b
\right]dt.
$$

    <p>So the contrast is</p>

$$
\text{ODE:}\qquad
df
=
\left[
\partial_t f+\nabla_x f^\top b
\right]dt.
$$

$$
\text{SDE:}\qquad
df
=
\left[
\partial_t f+\nabla_x f^\top b
+
\frac{1}{2}g(t)^2\Delta_x f
\right]dt
+
g(t)\nabla_x f^\top dw_t.
$$

        <p><strong>Conclusion:</strong> the extra \(\frac{1}{2}g(t)^2\Delta_x f\) term is the signature of Brownian noise. It is the reason SDEs evolve densities differently from ODEs.</p>
      </div>
    </details>

    <details class="ddim-block foundation-subblock">
      <summary>
        <span class="ddim-block__title">Solving the OU process</span>
      </summary>
      <div class="ddim-block__content">

    <p>For example, the Ornstein-Uhlenbeck process from the lecture is</p>

$$
dX_t
=
-X_t\,dt
+
\sqrt{2}\,dw_t.
$$

    <p>To solve it with Itô's formula, choose</p>

$$
f(t,x)=e^t x.
$$

    <p>Then</p>

$$
\partial_t f=e^t x,
\qquad
\partial_x f=e^t,
\qquad
\partial_{xx}f=0.
$$

    <p>Apply Itô's formula to \(f(t,X_t)=e^tX_t\):</p>

$$
\begin{aligned}
d(e^tX_t)
&=
\left(
\partial_t f
+
\partial_x f(-X_t)
+
\frac{1}{2}\partial_{xx}f\cdot 2
\right)dt
+
\partial_x f\sqrt{2}\,dw_t \\
&=
\left(e^tX_t-e^tX_t+0\right)dt
+
\sqrt{2}e^t\,dw_t \\
&=
\sqrt{2}e^t\,dw_t.
\end{aligned}
$$

    <p>Integrate from \(0\) to \(t\):</p>

$$
e^tX_t-X_0
=
\sqrt{2}
\int_0^t e^s\,dw_s.
$$

    <p>Therefore</p>

$$
X_t
=
e^{-t}X_0
+
\sqrt{2}
\int_0^t e^{-(t-s)}\,dw_s.
$$

    <p>Since the Itô integral is Gaussian with variance \(1-e^{-2t}\), the distributional form is</p>

$$
X_t
\overset{d}{=}
e^{-t}X_0
+
\sqrt{1-e^{-2t}}\,\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

        <p>This is exactly the diffusion-model intuition: the signal coefficient decays, while the noise coefficient grows.</p>
      </div>
    </details>

    <details class="ddim-block foundation-subblock">
      <summary>
        <span class="ddim-block__title">Fokker-Planck and time reversal</span>
      </summary>
      <div class="ddim-block__content">

    <p><strong>Fokker-Planck equation.</strong> To understand the distribution rather than one sample path, apply Itô's formula to a test function \(\varphi(x_t)\), take expectation, and use integration by parts. This gives the density PDE:</p>

$$
\frac{\partial p_t(x)}{\partial t}
=
-\nabla_x\cdot\left(b(x,t)p_t(x)\right)
+
\frac{1}{2}g(t)^2\Delta_x p_t(x).
$$

    <p>The first term transports probability mass according to the drift. The second term spreads probability mass because of noise. <strong>Fokker-Planck is the bridge from sample dynamics to distribution dynamics.</strong></p>

    <p><strong>Time reversal.</strong> In diffusion models, the forward process starts from data and ends near noise. Generation needs the opposite direction. If the forward SDE produces marginals \(p_t(x)\), then the reverse-time process must have marginals \(p_{T-t}(x)\).</p>

    <p>The lecture derives the reverse process by forcing the reverse density to satisfy the correct Fokker-Planck equation. The drift that makes the equations match contains the score:</p>

$$
d x_t
=
\left[
b(x_t,t)
-
g(t)^2\nabla_x\log p_t(x_t)
\right]dt
+
g(t)\,d\bar{w}_t.
$$

    <p>The new term is</p>

$$
\nabla_x\log p_t(x).
$$

        <p><strong>This is the key conclusion: reversing a diffusion requires the score of the noisy marginal distribution.</strong> The forward SDE is chosen by the model designer, but the reverse SDE is unknown until we learn or estimate \(\nabla_x\log p_t(x)\).</p>
      </div>
    </details>

    <details class="ddim-block foundation-subblock">
      <summary>
        <span class="ddim-block__title">Probability flow ODE</span>
      </summary>
      <div class="ddim-block__content">

    <p><strong>Probability flow ODE.</strong> The lecture then asks whether sampling must be stochastic. Surprisingly, there is a deterministic ODE with the same one-time marginal densities:</p>

$$
d x_t
=
\left[
b(x_t,t)
-
\frac{1}{2}g(t)^2\nabla_x\log p_t(x_t)
\right]dt.
$$

    <p><strong>The reverse SDE and probability flow ODE can have the same marginal distributions, but they do not have the same paths.</strong> The SDE path is random and non-smooth; the ODE path is deterministic and smooth once the initial noise sample is fixed.</p>

    <p>For the variance-exploding case</p>

$$
d x_t
=
g(t)\,dw_t,
$$

    <p>the probability flow ODE becomes</p>

$$
d x_t
=
-\frac{1}{2}g(t)^2\nabla_x\log p_t(x_t)\,dt.
$$

        <p>Running this ODE backward gives a deterministic sampler. <strong>This is the conceptual bridge from score-based SDEs to ODE samplers such as probability-flow sampling and many modern diffusion solvers.</strong></p>
      </div>
    </details>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>3.</strong> Why we can use denoising score matching (Gradient perspective)</span>
  </summary>
  <div class="ddim-block__content">
    <p>This derivation follows Vincent's denoising score matching view: instead of estimating the unknown marginal score directly, train against the conditional score of a known noising process.<sup class="footnote-ref" id="fnref:smdae"><a href="#fn:smdae">12</a></sup></p>

    <p>Fix a timestep \(t\). Let \(q_{\mathrm{data}}(x_0)\) be the data distribution, \(q_t(x_t\mid x_0)\) be the forward noising kernel, and \(q_t(x_0,x_t)=q_{\mathrm{data}}(x_0)q_t(x_t\mid x_0)\) be the joint distribution of clean and noisy samples. The noisy marginal is</p>

$$
q_t(x_t)
=
\int q_{\mathrm{data}}(x_0)q_t(x_t\mid x_0)\,dx_0
$$

    <p>The score model tries to learn the marginal score</p>

$$
s_\theta(x_t,t)
\approx
\nabla_{x_t}\log q_t(x_t).
$$

    <p>The explicit score matching objective is</p>

$$
J_{\mathrm{SM}}(\theta)
=
\mathbb{E}_{q_t(x_t)}
\left[
\frac{1}{2}
\left\|
s_\theta(x_t,t)
-
\nabla_{x_t}\log q_t(x_t)
\right\|_2^2
\right].
$$

    <p>Expand the square:</p>

$$
\begin{aligned}
J_{\mathrm{SM}}(\theta)
&=
\mathbb{E}_{q_t(x_t)}
\left[
\frac{1}{2}\|s_\theta(x_t,t)\|_2^2
\right]
-
\mathbb{E}_{q_t(x_t)}
\left[
\left\langle
s_\theta(x_t,t),
\nabla_{x_t}\log q_t(x_t)
\right\rangle
\right]
+
C_1,
\end{aligned}
$$

    <p>where \(C_1\) does not depend on \(\theta\). The difficult term contains the unknown marginal score. Rewrite that term using the definition of \(q_t(x_t)\):</p>

$$
\begin{aligned}
&\mathbb{E}_{q_t(x_t)}
\left[
\left\langle
s_\theta(x_t,t),
\nabla_{x_t}\log q_t(x_t)
\right\rangle
\right] \\
&=
\int q_t(x_t)
\left\langle
s_\theta(x_t,t),
\frac{\nabla_{x_t}q_t(x_t)}{q_t(x_t)}
\right\rangle
dx_t \\
&=
\int
\left\langle
s_\theta(x_t,t),
\nabla_{x_t}q_t(x_t)
\right\rangle
dx_t \\
&=
\int
\left\langle
s_\theta(x_t,t),
\nabla_{x_t}
\int q_{\mathrm{data}}(x_0)q_t(x_t\mid x_0)\,dx_0
\right\rangle
dx_t \\
&=
\int\!\!\int
q_{\mathrm{data}}(x_0)q_t(x_t\mid x_0)
\left\langle
s_\theta(x_t,t),
\nabla_{x_t}\log q_t(x_t\mid x_0)
\right\rangle
dx_0dx_t \\
&=
\mathbb{E}_{q_t(x_0,x_t)}
\left[
\left\langle
s_\theta(x_t,t),
\nabla_{x_t}\log q_t(x_t\mid x_0)
\right\rangle
\right].
\end{aligned}
$$

    <p>So the explicit objective becomes</p>

$$
\begin{aligned}
J_{\mathrm{SM}}(\theta)
&=
\mathbb{E}_{q_t(x_0,x_t)}
\left[
\frac{1}{2}\|s_\theta(x_t,t)\|_2^2
-
\left\langle
s_\theta(x_t,t),
\nabla_{x_t}\log q_t(x_t\mid x_0)
\right\rangle
\right]
+
C_1.
\end{aligned}
$$

    <p>Now define the denoising score matching objective using the conditional score:</p>

$$
J_{\mathrm{DSM}}(\theta)
=
\mathbb{E}_{q_t(x_0,x_t)}
\left[
\frac{1}{2}
\left\|
s_\theta(x_t,t)
-
\nabla_{x_t}\log q_t(x_t\mid x_0)
\right\|_2^2
\right].
$$

    <p>Expanding it gives</p>

$$
\begin{aligned}
J_{\mathrm{DSM}}(\theta)
&=
\mathbb{E}_{q_t(x_0,x_t)}
\left[
\frac{1}{2}\|s_\theta(x_t,t)\|_2^2
-
\left\langle
s_\theta(x_t,t),
\nabla_{x_t}\log q_t(x_t\mid x_0)
\right\rangle
+
\frac{1}{2}
\left\|
\nabla_{x_t}\log q_t(x_t\mid x_0)
\right\|_2^2
\right]
\\
&=
\mathbb{E}_{q_t(x_0,x_t)}
\left[
\frac{1}{2}\|s_\theta(x_t,t)\|_2^2
-
\left\langle
s_\theta(x_t,t),
\nabla_{x_t}\log q_t(x_t\mid x_0)
\right\rangle
\right]
+
C_2,
\end{aligned}
$$

    <p>where</p>

$$
C_2
=
\mathbb{E}_{q_t(x_0,x_t)}
\left[
\frac{1}{2}
\left\|
\nabla_{x_t}\log q_t(x_t\mid x_0)
\right\|_2^2
\right]
$$

    <p>does not depend on \(\theta\). Therefore</p>

$$
J_{\mathrm{SM}}(\theta)
=
J_{\mathrm{DSM}}(\theta)
+
\text{constant}.
$$

    <p>They have the same optimizer. This is the gradient-perspective reason that we can train a score model with the known conditional corruption score instead of the unknown marginal score.</p>

    <p>For the common Gaussian noising kernel</p>

$$
x_t
=
\alpha_t x_0+\sigma_t\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I),
$$

    <p>the conditional score is</p>

$$
\nabla_{x_t}\log q_t(x_t\mid x_0)
=
-\frac{x_t-\alpha_t x_0}{\sigma_t^2}
=
-\frac{\epsilon}{\sigma_t}.
$$

    <p>So the practical denoising score matching objective is</p>

$$
\mathcal{L}_{\mathrm{DSM}}
=
\mathbb{E}_{t,x_0,\epsilon}
\left[
\lambda(t)
\left\|
s_\theta(\alpha_t x_0+\sigma_t\epsilon,t)
+
\frac{\epsilon}{\sigma_t}
\right\|_2^2
\right].
$$

    <p>This is the bridge from score prediction to noise prediction: predicting \(\epsilon\) is equivalent to predicting the score up to the scale factor \(-1/\sigma_t\).</p>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>4.</strong> Deriving Tweedie's formula</span>
  </summary>
  <div class="ddim-block__content">
    <p>Use the general linear Gaussian corruption</p>

$$
x_t
=
\alpha_t x_0+\sigma_t\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

    <p>The target identity is</p>

$$
\nabla_{x_t}\log p_t(x_t)
=
\frac{\alpha_t\mathbb{E}[x_0\mid x_t]-x_t}{\sigma_t^2}.
$$

    <p>Start from the definition of the score:</p>

$$
\nabla_{x_t}\log p_t(x_t)
=
\frac{1}{p_t(x_t)}
\nabla_{x_t}p_t(x_t).
$$

    <p>The noisy marginal \(p_t(x_t)\) is obtained by integrating out \(x_0\):</p>

$$
p_t(x_t)
=
\int p_0(x_0)p_t(x_t\mid x_0)\,dx_0.
$$

    <p>Therefore</p>

$$
\begin{aligned}
\nabla_{x_t}\log p_t(x_t)
&=
\frac{1}{p_t(x_t)}
\nabla_{x_t}
\int p_0(x_0)p_t(x_t\mid x_0)\,dx_0 \\
&=
\frac{1}{p_t(x_t)}
\int p_0(x_0)\nabla_{x_t}p_t(x_t\mid x_0)\,dx_0.
\end{aligned}
$$

    <p>Rewrite the derivative of the conditional density as a conditional score:</p>

$$
\nabla_{x_t}p_t(x_t\mid x_0)
=
p_t(x_t\mid x_0)
\nabla_{x_t}\log p_t(x_t\mid x_0).
$$

    <p>Substitute this and use Bayes' rule \(p_0(x_0)p_t(x_t\mid x_0)=p_0(x_0\mid x_t)p_t(x_t)\):</p>

$$
\begin{aligned}
\nabla_{x_t}\log p_t(x_t)
&=
\frac{1}{p_t(x_t)}
\int
p_0(x_0)p_t(x_t\mid x_0)
\nabla_{x_t}\log p_t(x_t\mid x_0)\,dx_0 \\
&=
\int
p_0(x_0\mid x_t)
\nabla_{x_t}\log p_t(x_t\mid x_0)\,dx_0.
\end{aligned}
$$

    <p>For the Gaussian conditional</p>

$$
p_t(x_t\mid x_0)
=
\mathcal{N}(\alpha_t x_0,\sigma_t^2I),
$$

    <p>we have</p>

$$
\nabla_{x_t}\log p_t(x_t\mid x_0)
=
\frac{\alpha_t x_0-x_t}{\sigma_t^2}.
$$

    <p>Plug this into the posterior expectation:</p>

$$
\begin{aligned}
\nabla_{x_t}\log p_t(x_t)
&=
\int
\frac{\alpha_t x_0-x_t}{\sigma_t^2}
p_0(x_0\mid x_t)\,dx_0 \\
&=
\frac{\alpha_t\mathbb{E}[x_0\mid x_t]-x_t}{\sigma_t^2}.
\end{aligned}
$$

    <p>Rearranging gives Tweedie's formula for this linear Gaussian corruption:</p>

$$
\mathbb{E}[x_0\mid x_t]
=
\frac{x_t+\sigma_t^2\nabla_{x_t}\log p_t(x_t)}
{\alpha_t}.
$$
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>5.</strong> Why we can use denoising score matching (from Tweedie's formula)</span>
  </summary>
  <div class="ddim-block__content">
    <p>Tweedie's formula gives another route from denoising to score estimation. For the Gaussian corruption</p>

$$
x_\sigma
=
x_0+\sigma\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I),
$$

    <p>the posterior mean of the clean sample satisfies</p>

$$
\mathbb{E}[x_0\mid x_\sigma]
=
x_\sigma
+
\sigma^2\nabla_{x_\sigma}\log p_\sigma(x_\sigma).
$$

    <p>Rearranging gives the score in terms of the optimal denoiser:</p>

$$
\nabla_{x_\sigma}\log p_\sigma(x_\sigma)
=
\frac{\mathbb{E}[x_0\mid x_\sigma]-x_\sigma}{\sigma^2}.
$$

    <p>If a neural denoiser \(D_\theta(x_\sigma,\sigma)\) is trained to predict \(x_0\), then it gives a score estimator</p>

$$
s_\theta(x_\sigma,\sigma)
=
\frac{D_\theta(x_\sigma,\sigma)-x_\sigma}{\sigma^2}.
$$

    <p>Training the denoiser with reconstruction loss</p>

$$
\mathcal{L}_{\mathrm{denoise}}
=
\mathbb{E}_{x_0,\epsilon,\sigma}
\left[
\left\|
D_\theta(x_0+\sigma\epsilon,\sigma)-x_0
\right\|_2^2
\right]
$$

    <p>therefore also trains the score, because the optimal denoiser is the posterior mean \(\mathbb{E}[x_0\mid x_\sigma]\).</p>

    <p>For the VP/DDPM form \(x_t=\alpha_t x_0+\sigma_t\epsilon\), Tweedie's formula becomes</p>

$$
\mathbb{E}[x_0\mid x_t]
=
\frac{x_t+\sigma_t^2\nabla_{x_t}\log p_t(x_t)}
{\alpha_t},
$$

    <p>so the score can be recovered from a clean-sample predictor:</p>

$$
\nabla_{x_t}\log p_t(x_t)
=
\frac{\alpha_t D_\theta(x_t,t)-x_t}{\sigma_t^2}.
$$
  </div>
</details>

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

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>Q3.</strong> Why does a denoiser trained with MSE learn the conditional mean?</span>
  </summary>
  <div class="ddim-block__content">
    <p>Suppose a denoiser \(h_\theta\) observes \(x_t\) and is trained to predict the clean sample \(x_0\) with squared error:</p>

$$
J(\theta)
=
\mathbb{E}_{x_0,x_t}
\left[
\left\|
h_\theta(x_t)-x_0
\right\|_2^2
\right].
$$

    <p>Condition on \(x_t\) and expand the square:</p>

$$
J(\theta)
=
\mathbb{E}_{x_t}\mathbb{E}_{x_0\mid x_t}
\|h_\theta(x_t)-x_0\|_2^2.
$$

$$
J(\theta)
=
\mathbb{E}_{x_t}
\big[
\|h_\theta(x_t)\|_2^2
-
2h_\theta(x_t)^T\mathbb{E}[x_0\mid x_t]
+
\mathbb{E}[\|x_0\|_2^2\mid x_t]
\big].
$$

    <p>For any fixed \(x_t\), the term that depends on \(h_\theta(x_t)\) is</p>

$$
\|h_\theta(x_t)\|_2^2
-
2h_\theta(x_t)^T\mathbb{E}[x_0\mid x_t].
$$

    <p>Set \(h=h_\theta(x_t)\). Minimizing this is equivalent to</p>

$$
\arg\min_h
\left(
h^Th
-
2h^T\mathbb{E}[x_0\mid x_t]
\right).
$$

    <p>Take the derivative and set it to zero:</p>

$$
2h-2\mathbb{E}[x_0\mid x_t]=0.
$$

    <p>Therefore the optimal denoiser is</p>

$$
h_{\theta^*}(x_t)
=
\mathbb{E}[x_0\mid x_t].
$$

    <p>So under MSE, the optimal denoiser is not necessarily the original clean sample for each noisy input. It is the posterior average of all clean samples that could have produced that noisy input.</p>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>Q4.</strong> Solving a stochastic differential equation</span>
  </summary>
  <div class="ddim-block__content">
    <p>A natural way to solve the Ornstein-Uhlenbeck process is to treat it like a first-order linear ODE, except that the forcing term is stochastic:</p>

$$
dX_t=-X_t\,dt+\sqrt{2}\,dW_t,
$$

    <p>where \(W_t\) is Brownian motion. Move the drift term to the left:</p>

$$
dX_t+X_t\,dt=\sqrt{2}\,dW_t.
$$

    <p>Now multiply by the integrating factor \(e^t\):</p>

$$
e^t\,dX_t+e^tX_t\,dt=\sqrt{2}\,e^t\,dW_t.
$$

    <p>The left side is exactly</p>

$$
d(e^tX_t)
=
e^t\,dX_t+e^tX_t\,dt,
$$

    <p>so the SDE becomes</p>

$$
d(e^tX_t)=\sqrt{2}\,e^t\,dW_t.
$$

    <p>Integrating from \(0\) to \(t\),</p>

$$
e^tX_t
=
X_0
+
\sqrt{2}\int_0^t e^s\,dW_s,
$$

    <p>so</p>

$$
X_t
=
e^{-t}X_0
+
\sqrt{2}\int_0^t e^{-(t-s)}\,dW_s.
$$

    <p>Now use the fact that the stochastic integral is Gaussian with mean \(0\) and variance</p>

$$
2\int_0^t e^{-2(t-s)}\,ds
=
1-e^{-2t}.
$$

    <p>Therefore</p>

$$
X_t
=
e^{-t}X_0
+
\sqrt{1-e^{-2t}}\,Z,
\qquad
Z\sim\mathcal{N}(0,I).
$$

    <p>So the natural story is: solve the linear SDE with an integrating factor first, then compute the distribution of the stochastic integral. This is why OU-like SDEs are useful in diffusion models: the marginal has the familiar form “decayed signal plus Gaussian noise.”</p>
  </div>
</details>

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
.page__content .foundation-subblock {
  margin: 0.8rem 0 0.6rem;
  padding-left: 1rem;
  border-top: 1px solid #d9dee4;
}
.page__content .foundation-subblock .ddim-block__title {
  font-size: 0.98rem;
}
.page__content .foundation-subblock .ddim-block__content {
  border-top-color: #d9dee4;
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
        <li>Input/output scaling and time conditioning</li>
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

    <p>Here \(\beta_t\) is the step-wise noise variance, while \(\sigma_t\) is the accumulated marginal noise scale up to time \(t\):</p>

$$
\sigma_t
=
\sqrt{1-\bar{\alpha}_t}
=
\sqrt{
1-\prod_{i=1}^{t}(1-\beta_i)
}.
$$

    <p>So a schedule can be described either by the per-step noise \(\beta_t\), or by the accumulated noise level \(\sigma_t\). Small \(\beta_t\)'s can still produce a large \(\sigma_t\) after many steps because the noise accumulates.</p>

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

    <p>In terms of the marginal noise scale, this means</p>

$$
\sigma(t)
=
\sqrt{1-\bar{\alpha}_t}
=
\sqrt{
1-\prod_{i=1}^{t}(1-\beta_i)
},
\qquad
\beta_{i+1}-\beta_i=c.
$$

    <p><strong>Cosine scheduler</strong> does not make SNR itself cosine. It defines the cumulative signal \(\bar{\alpha}_t\), equivalently the marginal noise scale \(\sigma(t)\), with a cosine-shaped rule:</p>

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
\sigma(t)
=
\sqrt{1-\bar{\alpha}_t}
\approx
\sin\!\left(
\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}
\right),
$$

$$
\beta_t
=
1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}.
$$

    <p>So the names refer to different directly designed quantities: linear means \(\beta_t\) is linear in discrete time, while cosine means the cumulative signal/noise path follows a cosine/sine shape. SNR is then derived from \(\bar{\alpha}_t\).</p>

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

    <p><strong>IDEA:</strong> This could also be a way to make diffusion work better in high-dimensional latent spaces: LangFlow learns an information-uniform noise schedule for continuous language embeddings,<sup class="footnote-ref" id="fnref:langflow"><a href="#fn:langflow">10</a></sup> while RAE-style latent diffusion highlights the challenge of training diffusion transformers in semantically rich, high-dimensional representation spaces.<sup class="footnote-ref" id="fnref:rae"><a href="#fn:rae">11</a></sup></p>
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
    <span class="ddim-block__title"><strong>5.</strong> Input/output scaling and time conditioning</span>
  </summary>
  <div class="ddim-block__content">
    <p><strong>The design logic.</strong> The denoiser \(D_\theta(x;\sigma)\) predicts the clean data \(x_0\) from a noisy input \(x\) at noise level \(\sigma\).</p>

    <p>At very small noise, the input is already close to clean data, so the model should mostly pass the input through:</p>

$$
\sigma \to 0
\quad\Rightarrow\quad
x \approx x_0
\quad\Rightarrow\quad
D_\theta(x;\sigma)\approx x,
$$

    <p>At very large noise, the input is mostly noise, so the model should ignore more of the input and rely more on the neural network prediction:</p>

$$
\sigma \to \infty
\quad\Rightarrow\quad
x \text{ is mostly noise}
\quad\Rightarrow\quad
D_\theta(x;\sigma) \text{ should rely more on } F_\theta.
$$

    <p>This suggests a skip-plus-residual form:</p>

$$
D_\theta(x;\sigma)
=
c_{\mathrm{skip}}(\sigma)x
+
c_{\mathrm{out}}(\sigma)
F_\theta
\!\left(
c_{\mathrm{in}}(\sigma)x;
c_{\mathrm{noise}}(\sigma)
\right).
$$

    <p>Here \(c_{\mathrm{skip}}\) should be large at low noise and small at high noise, while \(c_{\mathrm{out}}\) controls how much residual correction comes from the network.</p>

    <p>We also want the neural network interface to be well-conditioned:</p>

$$
\mathrm{Var}\!\left[c_{\mathrm{in}}(\sigma)x\right]\approx 1,
\qquad
\mathrm{Var}\!\left[
\frac{x_0-c_{\mathrm{skip}}(\sigma)x}
{c_{\mathrm{out}}(\sigma)}
\right]\approx 1.
$$

    <p>More explicitly, write the noisy input as \(x=y+n\), where \(y\sim p_{\mathrm{data}}\), \(n\sim\mathcal{N}(0,\sigma^2I)\), and \(\mathrm{Var}(y)=\sigma_{\mathrm{data}}^2\). The input scaling should satisfy</p>

$$
\mathrm{Var}_{y,n}
\!\left[
c_{\mathrm{in}}(\sigma)(y+n)
\right]
=
1
\quad\Rightarrow\quad
c_{\mathrm{in}}(\sigma)^2
\left(
\sigma_{\mathrm{data}}^2+\sigma^2
\right)
=
1.
$$

    <p>After substituting the preconditioned denoiser into the weighted denoising loss, the objective can be rewritten as a supervised loss on \(F_\theta\):</p>

$$
\lambda(\sigma)
\left\|
D_\theta(y+n;\sigma)-y
\right\|_2^2
=
\lambda(\sigma)c_{\mathrm{out}}(\sigma)^2
\left\|
F_\theta(c_{\mathrm{in}}(\sigma)(y+n);c_{\mathrm{noise}}(\sigma))
-
F_{\mathrm{target}}(y,n;\sigma)
\right\|_2^2.
$$

    <p>So \(c_{\mathrm{out}}\) is tied to the effective loss weighting:</p>

$$
w(\sigma)
=
\lambda(\sigma)c_{\mathrm{out}}(\sigma)^2.
$$

    <p>The effective target of \(F_\theta\) is</p>

$$
F_{\mathrm{target}}(y,n;\sigma)
=
\frac{1}{c_{\mathrm{out}}(\sigma)}
\left(
y
-
c_{\mathrm{skip}}(\sigma)(y+n)
\right),
$$

    <p>EDM chooses \(c_{\mathrm{out}}\) so this target has unit variance:</p>

$$
\mathrm{Var}_{y,n}
\!\left[
F_{\mathrm{target}}(y,n;\sigma)
\right]
=
1
\quad\Rightarrow\quad
c_{\mathrm{out}}(\sigma)^2
=
\left(1-c_{\mathrm{skip}}(\sigma)\right)^2
\sigma_{\mathrm{data}}^2
+
c_{\mathrm{skip}}(\sigma)^2\sigma^2.
$$

    <p>Finally, choose the skip weight to make the required network correction as small as possible:</p>

$$
c_{\mathrm{skip}}(\sigma)
=
\arg\min_{c}
\left[
\left(1-c\right)^2\sigma_{\mathrm{data}}^2
+
c^2\sigma^2
\right].
$$

    <p><strong>What EDM uses.</strong> If the data has standard deviation \(\sigma_{\mathrm{data}}\), EDM chooses the following preconditioning coefficients:</p>

$$
c_{\mathrm{skip}}(\sigma)
=
\frac{\sigma_{\mathrm{data}}^2}
{\sigma^2+\sigma_{\mathrm{data}}^2},
\qquad
c_{\mathrm{out}}(\sigma)
=
\frac{\sigma\sigma_{\mathrm{data}}}
{\sqrt{\sigma^2+\sigma_{\mathrm{data}}^2}},
$$

$$
c_{\mathrm{in}}(\sigma)
=
\frac{1}
{\sqrt{\sigma^2+\sigma_{\mathrm{data}}^2}},
\qquad
c_{\mathrm{noise}}(\sigma)
=
\frac{1}{4}\log\sigma.
$$

    <p>Here \(c_{\mathrm{in}}\) rescales the input before it enters the network, \(c_{\mathrm{out}}\) rescales the network output, \(c_{\mathrm{skip}}\) reuses information already present in the noisy input, and \(c_{\mathrm{noise}}\) is the time/noise conditioning signal.</p>

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

    <table style="width: 100%; border-collapse: collapse; margin: 0.9rem 0; font-size: 0.95rem;">
      <thead>
        <tr>
          <th style="border-bottom: 1px solid #d6dde3; padding: 0.45rem; text-align: left;">Method</th>
          <th style="border-bottom: 1px solid #d6dde3; padding: 0.45rem; text-align: left;">Where it acts</th>
          <th style="border-bottom: 1px solid #d6dde3; padding: 0.45rem; text-align: left;">Math form</th>
          <th style="border-bottom: 1px solid #d6dde3; padding: 0.45rem; text-align: left;">Role</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="padding: 0.45rem; border-bottom: 1px solid #edf1f4;">additive bias</td>
          <td style="padding: 0.45rem; border-bottom: 1px solid #edf1f4;">residual stream</td>
          <td style="padding: 0.45rem; border-bottom: 1px solid #edf1f4;">\(h+b(e_t)\)</td>
          <td style="padding: 0.45rem; border-bottom: 1px solid #edf1f4;">shift</td>
        </tr>
        <tr>
          <td style="padding: 0.45rem; border-bottom: 1px solid #edf1f4;">adaptive norm</td>
          <td style="padding: 0.45rem; border-bottom: 1px solid #edf1f4;">normalization layer</td>
          <td style="padding: 0.45rem; border-bottom: 1px solid #edf1f4;">\(\gamma(e_t)h+\beta(e_t)\)</td>
          <td style="padding: 0.45rem; border-bottom: 1px solid #edf1f4;">scale + shift</td>
        </tr>
        <tr>
          <td style="padding: 0.45rem;">attention conditioning</td>
          <td style="padding: 0.45rem;">attention matrix</td>
          <td style="padding: 0.45rem;">modify \(Q,K,V\)</td>
          <td style="padding: 0.45rem;">interaction</td>
        </tr>
      </tbody>
    </table>

    <p><strong>IDEA:</strong> This is closely related to JiT, which also emphasizes making denoising models directly predict clean data rather than noised quantities.<sup class="footnote-ref" id="fnref:jit"><a href="#fn:jit">9</a></sup></p>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>6.</strong> Solver</span>
  </summary>
  <div class="ddim-block__content">
    <p>See sampler section.</p>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>7.</strong> Sampling-time noise schedule</span>
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
    <span class="ddim-block__title"><strong>8.</strong> Number of time steps</span>
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
          <td style="padding: 0.45rem; border-bottom: 1px solid #edf1f4;">DDIM / classic Stable Diffusion samplers</td>
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

### Training based guidance

### Training free guidance

## Distillation
> How to train one-step and few-step diffusion models.

### Distribution matching distillation

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>1.</strong> ADD and LADD</span>
  </summary>
  <div class="ddim-block__content">
    <p><strong>ADD</strong> stands for Adversarial Diffusion Distillation.<sup class="footnote-ref" id="fnref:add"><a href="#fn:add">13</a></sup> It trains a few-step student with two signals: a diffusion teacher gives a score-distillation direction, and an adversarial loss pushes the samples to look realistic in the very low-step regime.</p>

$$
\mathcal{L}_{\mathrm{ADD}}
=
\mathcal{L}_{\mathrm{score}}
+
\lambda_{\mathrm{adv}}\mathcal{L}_{\mathrm{adv}}.
$$

    <p>The score term keeps the student close to the teacher distribution. The adversarial term fixes the visual sharpness problem that often appears when a many-step diffusion model is compressed to one or a few steps.</p>

    <p><strong>LADD</strong> stands for Latent Adversarial Diffusion Distillation.<sup class="footnote-ref" id="fnref:ladd"><a href="#fn:ladd">14</a></sup> It keeps the same distribution-matching spirit, but moves the adversarial comparison into the latent or feature space of a pretrained latent diffusion model.</p>

$$
\mathcal{L}_{\mathrm{LADD}}
=
\mathcal{L}_{\mathrm{score}}
+
\lambda_{\mathrm{adv}}
\sum_{\ell}
\mathcal{L}_{\mathrm{adv}}^{(\ell)}.
$$

    <p><strong>Key idea:</strong> ADD uses adversarial feedback to make few-step samples realistic; LADD makes that feedback cheaper and more scalable by using latent/generative features instead of only pixel-space discrimination.</p>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>2.</strong> Interval KL divergence</span>
  </summary>
  <div class="ddim-block__content">
    <p>Distribution matching distillation tries to train a fast student generator so that its output distribution matches the distribution produced by a stronger teacher sampler. Instead of matching every teacher trajectory step by step, compare the distributions over a time interval. Diff-Instruct formalizes this as an Integral KL divergence.<sup class="footnote-ref" id="fnref:diff-instruct"><a href="#fn:diff-instruct">15</a></sup></p>

$$
p_\theta(x_t)
\quad\text{student distribution at time }t,
\qquad
p_{\mathrm{teach}}(x_t)
\quad\text{teacher distribution at time }t.
$$

    <p>A natural objective is the KL divergence between the student and teacher marginals:</p>

$$
\mathcal{L}_{\mathrm{KL}}(t)
=
D_{\mathrm{KL}}
\left(
p_\theta(x_t)
\;\|\;
p_{\mathrm{teach}}(x_t)
\right).
$$

    <p>For an interval \([t_a,t_b]\), the distribution matching objective can be written as</p>

$$
\mathcal{L}_{\mathrm{interval}}
=
\int_{t_a}^{t_b}
w(t)
D_{\mathrm{KL}}
\left(
p_\theta(x_t)
\;\|\;
p_{\mathrm{teach}}(x_t)
\right)
dt.
$$

    <p><strong>Key idea:</strong> the student does not need to imitate the teacher's exact path. It only needs to make its generated distribution close to the teacher distribution at the chosen times.</p>
  </div>
</details>

<details class="ddim-block">
  <summary>
    <span class="ddim-block__title"><strong>3.</strong> Score divergence</span>
  </summary>
  <div class="ddim-block__content">
    <p>The KL objective is conceptually clean, but the density \(p_\theta(x_t)\) is usually unavailable. A more practical route is to compare scores, because diffusion models already learn score or denoising fields. Score identity Distillation uses this kind of score-identity view for one-step distillation.<sup class="footnote-ref" id="fnref:sid"><a href="#fn:sid">16</a></sup></p>

$$
s_\theta(x_t,t)
=
\nabla_{x_t}\log p_\theta(x_t),
\qquad
s_{\mathrm{teach}}(x_t,t)
=
\nabla_{x_t}\log p_{\mathrm{teach}}(x_t).
$$

    <p>The score divergence compares these vector fields under samples from the student:</p>

$$
\mathcal{L}_{\mathrm{score}}(t)
=
\mathbb{E}_{x_t\sim p_\theta}
\left[
\left\|
s_\theta(x_t,t)
-
s_{\mathrm{teach}}(x_t,t)
\right\|_2^2
\right].
$$

    <p>In practice, the teacher score can be obtained from a pretrained diffusion teacher or from its denoiser through Tweedie's formula. The student is updated so its samples receive the same denoising direction as the teacher.</p>

    <p><strong>Key idea:</strong> matching scores is a local way to match distributions. If two distributions have the same score field, then they are the same distribution up to normalization.</p>
  </div>
</details>

### Trajectory distillation

## A industry level text-to-image diffusion pipeline
> Study notes on Krea 2 Technical Report<sup class="footnote-ref" id="fnref:krea2"><a href="#fn:krea2">8</a></sup>

## A industry level video diffusion pipeline

## Study Checklist

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
- Why do more sampling steps not necessarily mean better results in diffusion models?
- Why do training and sampling not require the same noise schedule?
- What is Doob's h-transform?
- Write down the closed-form formula of velocity in flow matching.
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
    <li id="fn:jit">
      Li &amp; He.
      <em>Back to Basics: Let Denoising Generative Models Denoise.</em>
      arXiv, 2025.
      <a href="https://arxiv.org/abs/2511.13720">arXiv:2511.13720</a>.
      <a href="#fnref:jit" class="footnote-back" title="back to text">↩︎</a>
    </li>
    <li id="fn:langflow">
      Chen et al.
      <em>LangFlow: Language as Continuous Interpolants for Autoregressive Generation.</em>
      arXiv, 2026.
      <a href="https://arxiv.org/abs/2604.11748">arXiv:2604.11748</a>.
      <a href="#fnref:langflow" class="footnote-back" title="back to text">↩︎</a>
    </li>
    <li id="fn:rae">
      Zheng et al.
      <em>Diffusion Transformers with Representation Autoencoders.</em>
      arXiv, 2025.
      <a href="https://arxiv.org/abs/2510.11690">arXiv:2510.11690</a>.
      <a href="#fnref:rae" class="footnote-back" title="back to text">↩︎</a>
    </li>
    <li id="fn:smdae">
      Vincent.
      <em>A Connection Between Score Matching and Denoising Autoencoders.</em>
      Technical report, 2010.
      <a href="https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf">PDF</a>.
      <a href="#fnref:smdae" class="footnote-back" title="back to text">↩︎</a>
    </li>
    <li id="fn:add">
      Sauer et al.
      <em>Adversarial Diffusion Distillation.</em>
      arXiv, 2023.
      <a href="https://arxiv.org/abs/2311.17042">arXiv:2311.17042</a>.
      <a href="#fnref:add" class="footnote-back" title="back to text">↩︎</a>
    </li>
    <li id="fn:ladd">
      Sauer et al.
      <em>Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation.</em>
      arXiv, 2024.
      <a href="https://arxiv.org/abs/2403.12015">arXiv:2403.12015</a>.
      <a href="#fnref:ladd" class="footnote-back" title="back to text">↩︎</a>
    </li>
    <li id="fn:diff-instruct">
      Luo et al.
      <em>Diff-Instruct: A Universal Approach for Transferring Knowledge From Pre-trained Diffusion Models.</em>
      arXiv, 2023.
      <a href="https://arxiv.org/abs/2305.18455">arXiv:2305.18455</a>.
      <a href="#fnref:diff-instruct" class="footnote-back" title="back to text">↩︎</a>
    </li>
    <li id="fn:sid">
      Zhou et al.
      <em>Score identity Distillation: Exponentially Fast Distillation of Pretrained Diffusion Models for One-Step Generation.</em>
      arXiv, 2024.
      <a href="https://arxiv.org/abs/2404.04057">arXiv:2404.04057</a>.
      <a href="#fnref:sid" class="footnote-back" title="back to text">↩︎</a>
    </li>
  </ol>
</section>

<script>
/* Auto-build a Contents list from h2 and h3 sections. */
(function () {
  function ready(fn) {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', fn);
    else fn();
  }
  ready(function () {
    var article = document.querySelector('.page__content');
    var toc = document.querySelector('#draft-toc');
    if (!article || !toc) return;
    var items = Array.from(article.querySelectorAll('h2, h3')).filter(function (el) {
      return el.closest('.references-section') === null && el.dataset.tocSkip !== 'true';
    });
    if (items.length === 0) {
      toc.style.display = 'none';
      return;
    }
    function slugify(s) {
      return s.toLowerCase().replace(/[^a-z0-9 \-]/g, '').trim().replace(/\s+/g, '-').slice(0, 60);
    }
    function uniqueId(base, i) {
      var id = base || ('sec-' + (i + 1));
      var candidate = id;
      var suffix = 2;
      while (document.getElementById(candidate)) {
        candidate = id + '-' + suffix;
        suffix += 1;
      }
      return candidate;
    }
    var linkByItem = new Map();
    items.forEach(function (h, i) {
      if (!h.id) h.id = uniqueId(slugify(h.textContent), i);
      var a = document.createElement('a');
      a.href = '#' + h.id;
      var title = h.cloneNode(true);
      title.querySelectorAll('.footnote-ref').forEach(function (note) { note.remove(); });
      a.textContent = title.textContent.trim();
      a.className = h.tagName.toLowerCase() === 'h3' ? 'toc-subsection' : 'toc-section';
      a.addEventListener('click', function () {
        document.querySelectorAll('#draft-toc a').forEach(function (l) { l.classList.remove('toc-active'); });
        a.classList.add('toc-active');
      });
      toc.appendChild(a);
      linkByItem.set(h, a);
    });
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        document.querySelectorAll('#draft-toc a').forEach(function (l) { l.classList.remove('toc-active'); });
        linkByItem.get(entry.target).classList.add('toc-active');
      });
    }, { rootMargin: '-25% 0px -65% 0px', threshold: 0 });
    items.forEach(function (h) { observer.observe(h); });
  });
})();
</script>
