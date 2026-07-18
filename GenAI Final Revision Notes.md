---
title: "Mathematical Foundations of Generative AI - Final Revision Notes"
author: "Exam preparation notes"
lang: en
---

# Mathematical Foundations of Generative AI

Short exam notes with English terminology, LaTeX-ready equations, Roman Punjabi intuition, and PYQ-style numericals.

## How to reuse these notes

- Inline mathematics uses `$...$`, for example `$D_f(P\|Q)$`.
- Display mathematics uses `$$...$$`.
- For a blog, keep one idea per section and explain the formula immediately below it.
- For Mathcha, copy the content inside `$...$` or `$$...$$` into a math region.
- For a LaTeX document, the companion `.tex` file is ready to compile.

---

# Week 1: $f$-Divergence and Variational Minimization

## 1. $f$-Divergence

**Concept in one line:** A convex function $f$ measures the mismatch between a real distribution $P$ and a model distribution $Q$.

$$
D_f(P\|Q)=\mathbb{E}_{x\sim Q}\left[f\left(\frac{P(x)}{Q(x)}\right)\right],
\qquad f(1)=0.
$$

Here $P$ is normally the data distribution and $Q=P_\theta$ is the generator distribution.

**Punjabi intuition:** Do pindaan di doodh supply compare karo. Je har category vich ratio $P/Q=1$ hai, distributions ikko jehiyan ne te divergence zero hai.

| Measure | Choice of $f(t)$ | Memory point |
|---|---:|---|
| Forward KL | $t\log t$ | Mode covering; missing real modes costly |
| Reverse KL | $-\log t$ | Often mode seeking |
| Pearson $\chi^2$ | $(t-1)^2$ | Squared ratio mismatch |
| Total variation | $\tfrac12|t-1|$ | Probability-mass difference |

For forward KL,

$$
\mathbb{E}_Q\left[\frac{P}{Q}\log\frac{P}{Q}\right]
=\mathbb{E}_P\left[\log\frac{P}{Q}\right]
=\mathrm{KL}(P\|Q).
$$

> **PYQ trap:** Wasserstein distance is not an $f$-divergence. It uses geometry and transport cost, not only the density ratio.

## 2. Fenchel Conjugate

$$
f^*(u)=\sup_{t>0}\{ut-f(t)\}.
$$

For $f(t)=t\log t$:

$$
\frac{d}{dt}(ut-t\log t)=u-(\log t+1)=0
\quad\Longrightarrow\quad t=e^{u-1},
$$

so

$$
f^*(u)=e^{u-1}.
$$

**Desi intuition:** Mandi vich price $u$ fixed hai. Quantity $t$ oh choose karo jo profit $ut-f(t)$ maximum kare. Best profit $f^*(u)$ hai.

## 3. Variational Lower Bound

$$
D_f(P\|Q)
\geq \mathbb{E}_P[T(x)]-\mathbb{E}_Q[f^*(T(x))].
$$

Taking the best critic gives equality:

$$
D_f(P\|Q)
=\sup_T\left\{\mathbb{E}_P[T(x)]-\mathbb{E}_Q[f^*(T(x))]\right\}.
$$

- Every candidate $T$ gives a lower bound.
- The supremum selects the tightest possible bound.
- Practical tightness depends on critic capacity and optimization.

**Numerical:** If $\mathbb{E}_P[T]=1.4$ and $\mathbb{E}_Q[f^*(T)]=0.5$, then

$$
\text{lower bound}=1.4-0.5=0.9.
$$

## 4. Optimal Critic and Adversarial Min-Max

Under regularity conditions,

$$
T^*(x)=f'\left(\frac{P(x)}{Q(x)}\right).
$$

For forward KL, $f'(t)=1+\log t$. If $P(x)/Q(x)=2$,

$$
T^*(x)=1+\log 2\approx1.693.
$$

The adversarial optimization is

$$
\min_\theta\max_\phi V(\theta,\phi).
$$

**Punjabi intuition:** Judge/critic farak labhan layi objective maximize karda; generator apni recipe sudhar ke ohi farak minimize karda.

---

# Weeks 2-3: GAN Foundations

## 5. Original GAN

$$
\min_G\max_D
\mathbb{E}_{x\sim P_{\text{data}}}[\log D(x)]
+\mathbb{E}_{z\sim P_z}[\log(1-D(G(z)))].
$$

- $D$ wants $D(x_{\text{real}})\to1$.
- $D$ wants $D(G(z))\to0$.
- $G$ wants generated samples to receive a real-like score.

For a fixed generator,

$$
D^*(x)=\frac{P_{\text{data}}(x)}{P_{\text{data}}(x)+P_g(x)}.
$$

Therefore,

$$
\frac{P_{\text{data}}(x)}{P_g(x)}=\frac{D^*(x)}{1-D^*(x)}.
$$

If $D^*(x)=0.75$, the estimated density ratio is $0.75/0.25=3$.

## 6. Saturating and Non-Saturating Generator Loss

Original minimax/saturating loss:

$$
L_G^{\text{sat}}=\mathbb{E}_z[\log(1-D(G(z)))].
$$

Common non-saturating loss:

$$
L_G^{\text{NS}}=-\mathbb{E}_z[\log D(G(z))].
$$

For $D(G(z))=[0.4,0.55,0.5]$,

$$
L_G^{\text{NS}}
=-\frac{\log0.4+\log0.55+\log0.5}{3}
\approx0.7357.
$$

**Desi memory:** Fake score ghatt hove tan non-saturating generator loss vadda hunda, is karke sudhar da strong signal milda.

## 7. Saturation and Vanishing Gradient

By the chain rule,

$$
\nabla_\theta L_G
=\frac{\partial L}{\partial D}
\frac{\partial D}{\partial x}
\frac{\partial G}{\partial\theta}.
$$

If an ideal discriminator is locally constant around all generated samples, then $\partial D/\partial x=0$, so the generator gradient is zero.

> **PYQ keyword:** Perfect discriminator + disjoint support + minimax loss $\Rightarrow$ vanishing generator gradient.

## 8. DCGAN

Generator flow:

```text
latent z -> dense/reshape -> upsampling or transposed convolution -> RGB image
```

Discriminator flow:

```text
RGB image -> strided convolution -> deep features -> logit/score
```

- Early convolution layers learn edges, colours, and textures.
- Deeper layers learn object parts and high-level structure.
- A logit is a raw score before sigmoid or softmax.
- `BCEWithLogitsLoss` combines sigmoid and BCE numerically safely.

**PYQ parameter numerical:** For linear layers $100\to128\to256\to784$, excluding biases,

$$
100(128)+128(256)+256(784)=246272.
$$

Desi rule: har adjacent layer pair layi **input units $\times$ output units**, fer sab add.

## 9. Conditional GAN

$$
\min_G\max_D
\mathbb{E}_{x,y}[\log D(x,y)]
+\mathbb{E}_{z,y}[\log(1-D(G(z,y),y))].
$$

The condition $y$ must be available to both generator and discriminator.

**Punjabi intuition:** “Mithai banao” ordinary GAN hai. “Class $y=$ jalebi de hisaab naal jalebi banao” conditional GAN hai.

---

# Week 4: Wasserstein GAN, Inversion, DANN, and Evaluation

## 10. Wasserstein-1 / Earth Mover's Distance

$$
W_1(P,Q)=\min_{T\in\Pi(P,Q)}\sum_{i,j}T_{ij}D_{ij}.
$$

$T_{ij}$ is transported mass and $D_{ij}$ is cost per unit mass.

Given

$$
T=\begin{bmatrix}
0.2&0.1&0\\
0&0.3&0.1\\
0&0&0.2
\end{bmatrix},
\qquad
D=\begin{bmatrix}
0&1&3\\
1&0&2\\
3&2&0
\end{bmatrix},
$$

the cost of the stated plan is

$$
\sum_{i,j}T_{ij}D_{ij}=0.1(1)+0.1(2)=0.3.
$$

**Important trap:** Check row sums, column sums, and total mass. The displayed $T$ transports only $0.9$, so $0.3$ is the cost of that incomplete stated plan, not necessarily the true EMD for normalized $P,Q$.

## 11. WGAN Critic and 1-Lipschitzness

Kantorovich-Rubinstein dual:

$$
W_1(P,Q)=\sup_{\|D\|_L\leq1}
\left(\mathbb{E}_P[D(x)]-\mathbb{E}_Q[D(x)]\right).
$$

The 1-Lipschitz constraint is

$$
|D(x_1)-D(x_2)|\leq\|x_1-x_2\|.
$$

- WGAN uses a real-valued critic, not a calibrated probability.
- No sigmoid is required at the critic output.
- Multiple critic steps keep the critic closer to its optimum and improve generator directions.

For real scores $[1.1,0.9,1.3]$ and fake scores $[-0.2,0,0.1]$,

$$
\widehat W=1.1-(-0.0333)=1.1333.
$$

## 12. Weight Clipping and WGAN-GP

Old exam-style clipping:

```python
for p in D.parameters():
    p.data.clamp_(-0.01, 0.01)
```

Modern safer form:

```python
with torch.no_grad():
    for p in D.parameters():
        p.clamp_(-0.01, 0.01)
```

In PyTorch, the trailing underscore in `clamp_` means **in-place**.

Gradient penalty:

$$
\lambda\left(\|\nabla_{\hat x}D(\hat x)\|_2-1\right)^2.
$$

If $\lambda=10$ and the gradient norm is $1.3$,

$$
10(1.3-1)^2=0.9.
$$

## 13. GAN Inversion

With trained generator $G$ and target image $x$,

$$
z^*=\arg\min_z\mathcal{L}(G(z),x).
$$

The generator weights stay fixed; the latent code $z$ is optimized.

```text
target x -> initialize z -> generate G(z) -> compare loss -> update z
```

Uses include image editing, reconstruction, latent analysis, and anomaly detection.

## 14. BiGAN

BiGAN learns generator $G:z\to x$ and encoder $E:x\to z$ jointly.

$$
\min_{G,E}\max_D
\mathbb{E}_{x\sim P_{\text{data}}}[\log D(x,E(x))]
+\mathbb{E}_{z\sim P_z}[\log(1-D(G(z),z))].
$$

- Real pair: $(x,E(x))$
- Fake pair: $(G(z),z)$
- At ideal equilibrium, the two joint distributions match.

## 15. Latent Regression

$$
\min_E\mathbb{E}_{z\sim P_z}\left[\|E(G(z))-z\|^2\right].
$$

**Punjabi intuition:** $z$ sanu pehlan pata hai. $G(z)$ image banaunda; encoder nu image vekh ke ohi known $z$ wapas dassna sikhaya janda.

- Optimization inversion: slower per image, flexible.
- Latent regression: fast inference, possible approximation error.

## 16. Domain-Adversarial Neural Network and GRL

```text
input x -> feature extractor F -> label classifier
                         |
                         -> GRL -> domain classifier
```

Forward pass:

$$
\operatorname{GRL}(f)=f.
$$

Backward pass:

$$
g_{\text{feature}}=-\lambda g_{\text{upstream}}.
$$

If $\lambda=0.2$ and the upstream gradient is $[-1,-5]$,

$$
-0.2[-1,-5]=[0.2,1.0].
$$

**Desi rule:** Pehlan sign ulta, fer $\lambda$ naal scale.

If the domain classifier achieves $100\%$ accuracy, source and target features are easily separable, so the domains are **misaligned**, not aligned.

## 17. FID and Generative Evaluation

Standard FID uses penultimate/global-average-pooling features from a fixed pretrained Inception-v3 network.

$$
\operatorname{FID}
=\|\mu_r-\mu_g\|^2
+\operatorname{Tr}\left(
\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}
\right).
$$

- Mean term: average/bias shift.
- Covariance term: shape and diversity mismatch.
- Lower FID is better.
- FID is not KL divergence.
- FID compares distributions, not one image with one image.

For one-dimensional features with $\mu_r=0$, $\mu_g=1$, $\sigma_r=1$, and $\sigma_g=2$,

$$
\operatorname{FID}=(0-1)^2+(1-2)^2=2.
$$

**Punjabi intuition:** Do mithai batches compare karo: average taste/shape $\mu$ te variety $\Sigma$. Sirf ik sohna ladoo kaafi nahi.

---

# PYQ Rapid-Fire Traps

1. **Which is not an $f$-divergence?** Wasserstein distance.
2. **JSD instability:** disjoint support saturation, strong discriminator, vanishing generator gradient.
3. **Several critic updates per generator step:** critic nu optimum de nere rakhna.
4. **WGAN output:** unrestricted real score, no sigmoid, 1-Lipschitz.
5. **BiGAN pairs:** $(x,E(x))$ against $(G(z),z)$.
6. **GRL:** always remember the negative sign.
7. **FID layer:** Inception-v3 penultimate/pool3 feature, not raw RGB or generator logits.
8. **Transport matrix:** element-wise cost sum; also verify marginals.

---

# Final Formula Sheet

| Topic | Formula |
|---|---|
| $f$-divergence | $D_f(P\|Q)=\mathbb{E}_Q[f(P/Q)]$ |
| Conjugate | $f^*(u)=\sup_t\{ut-f(t)\}$ |
| Variational form | $D_f(P\|Q)=\sup_T\{\mathbb{E}_P[T]-\mathbb{E}_Q[f^*(T)]\}$ |
| Optimal critic | $T^*=f'(P/Q)$ |
| GAN | $\min_G\max_D\mathbb{E}[\log D(x)]+\mathbb{E}[\log(1-D(G(z)))]$ |
| Optimal GAN discriminator | $D^*=P_{\text{data}}/(P_{\text{data}}+P_g)$ |
| Non-saturating generator | $L_G=-\mathbb{E}[\log D(G(z))]$ |
| Density ratio | $P_{\text{data}}/P_g=D^*/(1-D^*)$ |
| Transport cost | $C=\sum_{i,j}T_{ij}D_{ij}$ |
| WGAN dual | $W_1=\sup_{\|D\|_L\leq1}\mathbb{E}[D(x_r)]-\mathbb{E}[D(x_f)]$ |
| WGAN-GP | $\lambda(\|\nabla_{\hat x}D\|_2-1)^2$ |
| GAN inversion | $z^*=\arg\min_z\mathcal{L}(G(z),x)$ |
| Latent regression | $\min_E\mathbb{E}\|E(G(z))-z\|^2$ |
| GRL | $g_{\text{feature}}=-\lambda g_{\text{upstream}}$ |
| FID | $\|\mu_r-\mu_g\|^2+\operatorname{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})$ |

## Final memory line

> GAN judge nu fool karda; WGAN distance smooth karda; BiGAN inverse sikha karda; DANN domain bhula karda; FID quality plus diversity naapda.
