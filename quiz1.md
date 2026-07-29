---
title: "Mathematical Foundations of Generative AI - Final Revision Notes"
author: "Exam preparation notes"
lang: en
---

# Mathematical Foundations of Generative AI

S
# Week 1: $f$-Divergence and Variational Minimization

## 1. $f$-Divergence

**Concept in one line:** A convex function $f$ measures the mismatch between a real distribution $P$ and a model distribution $Q$.

$$
D_f(P\|Q)=\mathbb{E}_{x\sim Q}[f(\frac{P(x)}{Q(x)})],
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
\mathbb{E}_Q[\frac{P}{Q}\log\frac{P}{Q}]
=\mathbb{E}_P[\log\frac{P}{Q}]
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

**TV revision note:** For total variation, $f(t)=\tfrac12|t-1|$. Here the conjugate is bounded and piecewise:

$$
f^*(u)=
\begin{cases}
u, & -\tfrac12\le u\le \tfrac12,\\
-\tfrac12, & u<-\tfrac12,\\
+\infty, & u>\tfrac12.
\end{cases}
$$

So the key exam idea is: $u$ is the fixed argument of $f^*(u)$, while $t$ is the variable inside the supremum. For TV, the best value usually comes from a boundary, not from a smooth derivative solution.

**Desi shortcut:** KL vich “smooth log-ratio meter” chalda hai, par TV vich “bounded on/off switch” chalda hai. Je $u$ bahut vadda hove, answer infinity ban janda; je $u$ moderate hove, value clipped rehndi hai.

## 3. Variational Lower Bound

$$
D_f(P\|Q)
\geq \mathbb{E}_P[T(x)]-\mathbb{E}_Q[f^*(T(x))].
$$

Taking the best critic gives equality:

$$
D_f(P\|Q)
=\sup_T\{\mathbb{E}_P[T(x)]-\mathbb{E}_Q[f^*(T(x))]\}.
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
T^*(x)=f'(\frac{P(x)}{Q(x)}).
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
(\mathbb{E}_P[D(x)]-\mathbb{E}_Q[D(x)]).
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
\lambda(\|\nabla_{\hat x}D(\hat x)\|_2-1)^2.
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
\min_E\mathbb{E}_{z\sim P_z}[\|E(G(z))-z\|^2].
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
+\operatorname{Tr}(
\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}
).
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

# Final Formula Sheet — Last-Night Order

**How to use this page:** first identify the topic in the question; then use only that topic's formula. Do not mix $u$, $t$, $T$, $D$, and $G$.

## A. $f$-Divergence and Variational Bound (Week 1)

**1. Start with a density ratio**

$$
r(x)=\frac{p(x)}{q(x)}, \qquad D_f(P\|Q)=\mathbb{E}_{x\sim Q}[f(r(x))].
$$

- **Use when:** question says $f$-divergence, KL, total variation, or ratio $p/q$.
- **Memory:** $r$ tells how many times more common an item is in real data than model data.

**2. Fenchel conjugate**

$$
f^*(u)=\sup_{t>0}\{ut-f(t)\}.
$$

- **Roles:** $u$ is fixed input to $f^*$; $t$ is the dummy variable we maximize over.
- **Total variation:** $f(t)=\tfrac12|t-1|$. Its conjugate is finite only on a bounded interval (a piecewise answer); do not replace $t$ with $u$.

**3. Variational representation / critic objective**

$$
D_f(P\|Q)=\sup_T\{\mathbb{E}_{P}[T(x)]-\mathbb{E}_{Q}[f^*(T(x))]\}.
$$

$$
T^*(x)=f'(\frac{p(x)}{q(x)}).
$$

- **Use when:** a critic/discriminator is maximizing a lower bound.
- **Exam line:** any allowed $T$ gives a lower bound; better critic capacity gives a tighter bound.

## B. Standard GAN and Conditional GAN (Weeks 2--3)

**4. Original GAN game**

$$
\min_G\max_D\; \mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)]
+\mathbb{E}_{z\sim p(z)}[\log(1-D(G(z)))].
$$

$$
D^*(x)=\frac{p_{\text{data}}(x)}{p_{\text{data}}(x)+p_g(x)}.
$$

- **Use when:** question asks original GAN objective or optimal discriminator.
- **Memory:** $D(x)$ is a probability, so it uses sigmoid and stays in $[0,1]$.

**5. Generator losses: recognize the trap**

$$
L_G^{\text{sat}}=\mathbb{E}_{z}[\log(1-D(G(z)))],
\qquad
L_G^{\text{ns}}=-\mathbb{E}_{z}[\log D(G(z))].
$$

- **Use non-saturating loss** in practical GAN training; it gives stronger gradients when $D(G(z))\approx0$.
- **Punjabi memory:** judge fake nu turant pakad lave, ta original loss da signal sukk sakda; $-\log D$ generator nu zor naal sikhaunda.

**6. Conditional GAN**

$$
\min_G\max_D\;\mathbb{E}[\log D(x,y)]
+\mathbb{E}[\log(1-D(G(z\mid y),y))].
$$

$$
L_G^{\text{ns}}=-\operatorname{mean}(\log D(G(z\mid y),y)).
$$

- **Use when:** label/class $y$ appears. Example: “make a digit 7” or “make a red tractor,” not just any image.

## C. Wasserstein GAN / Earth Mover's Distance (Week 4)

**7. Primal transport cost (numerical matrix questions)**

$$
W_1(P,Q)=\min_{T\ge0}\sum_{i,j}T_{ij}D_{ij}.
$$

$$
\sum_jT_{ij}=P_i, \qquad \sum_iT_{ij}=Q_j.
$$

- **Solve order:** check each row sum = source $P$; check each column sum = target $Q$; then multiply each $T_{ij}$ by $D_{ij}$ and add.

**8. WGAN critic (dual form)**

$$
W_1(P_r,P_g)=\sup_{\|D\|_L\le1}
[\mathbb{E}[D(x_{\text{real}})]-\mathbb{E}[D(x_{\text{fake}})]].
$$

$$
\widehat W=\operatorname{mean}(D(x_{\text{real}}))-
\operatorname{mean}(D(x_{\text{fake}})).
$$

- **Use when:** outputs are unrestricted real critic scores, not probabilities.
- **Mini-batch:** simply take the mean of all real scores minus mean of all fake scores.

**9. 1-Lipschitz enforcement**

$$
\|D\|_L\le1,
\qquad
L_{\text{GP}}=\lambda_{\text{GP}}(\|\nabla_{\hat{x}}D(\hat{x})\|_2-1)^2.
$$

- **Weight clipping:** after critic update, clamp every critic parameter to $[-c,c]$.
- **WGAN-GP:** preferred modern alternative; it pushes gradient norm toward $1$.

## D. Inversion, BiGAN, and Domain Adaptation (Week 4)

**10. GAN inversion**

$$
z^*=\arg\min_z\mathcal{L}\big(G(z),x\big).
$$

- **Use when:** given an image $x$, find its latent code $z$ by keeping $G$ fixed and optimizing $z$.

**11. Latent regression and BiGAN pair matching**

$$
\min_E\mathbb{E}_{z\sim p(z)}[\|E(G(z))-z\|_2^2].
$$

$$
\text{BiGAN: } (x,E(x)) \quad \text{vs.} \quad (G(z),z).
$$

- **Memory:** inversion finds $z$ separately for each image; encoder $E$ learns to predict it quickly.

**12. Gradient reversal layer (GRL)**

$$
g_{\text{feature}}=-\lambda\,g_{\text{upstream}}.
$$

- **Forward pass:** identity, $\operatorname{GRL}(h)=h$.
- **Backward pass:** reverse sign and scale by $\lambda$. Example: $\lambda=0.2$, $g=[-1,-5]$ gives $[0.2,1.0]$.

## E. Evaluation (Week 4)

**13. Fréchet Inception Distance (FID)**

$$
\operatorname{FID}=\|\mu_r-\mu_g\|_2^2+
\operatorname{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}).
$$

- **Use features from:** the penultimate / pool3 layer of pretrained Inception-v3, not raw RGB, discriminator convolution features, or generator logits.
- **Interpretation:** lower FID is better; it rewards both realistic quality and diverse coverage.

## 20-Second Decision Guide

| If the question contains... | Immediately use... |
|---|---|
| $f$, $f^*$, $p/q$, $sup_t$ | $f$-divergence / Fenchel conjugate |
| $sup_T$, lower bound, critic | variational formula |
| $log D$, sigmoid probability | standard GAN loss |
| label $y$ | cGAN formula |
| transport plan $T$ and cost matrix | $sum_{ij}T_{ij}D_{ij}$ |
| real/fake critic scores, Lipschitz | WGAN estimate |
| clamp or gradient norm $1$ | WGAN clipping / GP |
| recover latent $z$ from image | GAN inversion |
| reverse gradient, coefficient $\lambda$ | $-\lambda g$ |
| mean/covariance, Inception-v3 | FID |

## Final memory line

> GAN judge nu fool karda; WGAN distance smooth karda; BiGAN inverse sikha karda; DANN domain bhula karda; FID quality plus diversity naapda.
