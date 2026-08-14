# Mathematical Foundations of Generative AI - Formula Sheet

This formula sheet is compiled based on the provided resources, starting from the EM Algorithm to advanced Generative AI concepts (VAEs, DDPMs), tailored for solving quiz questions.

## 1. Expectation-Maximization (EM) Algorithm for Gaussian Mixture Models (GMMs)
- **PDF of Normal Distribution (1D):**
  $$\mathcal{N}(x|\mu, \sigma^2) = rac{1}{\sqrt{2\pi\sigma^2}}e^{-rac{(x-\mu)^2}{2\sigma^2}}$$

- **E-Step (Responsibility / Posterior Probability):**
  For a data point $x_i$ and component $k$:
  $$\gamma(z_{ik}) = P(z=k|x_i) = rac{\pi_k \mathcal{N}(x_i|\mu_k, \sigma_k^2)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i|\mu_j, \sigma_j^2)}$$
  *(Note: $\pi_k$ is the prior probability of component $k$)*

- **M-Step (Parameter Update for Mean):**
  Updated mean for component $k$ across $N$ data points:
  $$\mu_k' = rac{\sum_{i=1}^N \gamma(z_{ik}) x_i}{\sum_{i=1}^N \gamma(z_{ik})}$$

---

## 2. Variational Autoencoders (VAEs)
- **ELBO (Evidence Lower Bound) Objective:**
  $$\mathcal{L}(q) = J_	heta(q) = \mathbb{E}_{q_\phi(z|x)}[\log p_	heta(x, z) - \log q_\phi(z|x)]$$
  $$\mathcal{L}(q) = \mathbb{E}_{q_\phi(z|x)}[\log p_	heta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))$$
  *(Note: Maximizing ELBO is equivalent to minimizing the Negative Log-Likelihood + KL Divergence)*

- **Reparameterization Trick:**
  Instead of sampling $z$ directly (which prevents backpropagation), express it as:
  $$z = \mu_\phi(x) + \sigma_\phi(x) \cdot \epsilon \quad 	ext{where} \quad \epsilon \sim \mathcal{N}(0, I)$$

- **KL Divergence Term (Standard Normal Prior):**
  If $q_\phi(z|x) = \mathcal{N}(\mu, \Sigma)$ and $p(z) = \mathcal{N}(0, I)$:
  $$D_{KL}[q_\phi(z|x) || p(z)] = -rac{1}{2} \sum_{j=1}^K \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 ight)$$

- **$eta$-VAE Objective:**
  $$F_	heta(q) = 	ext{Reconstruction Loss} + eta \cdot D_{KL}(q_\phi(z|x) || p(z))$$
  *Higher $eta$ leads to better disentanglement but can cause posterior collapse. Lower $eta$ leads to better reconstructions.*

- **VQ-VAE (Vector Quantized VAE):**
  - **Commitment Loss:** $eta ||z_e(x) - 	ext{sg}[e_k]||_2^2$ (where 'sg' is the stop-gradient operator).
  - **Size of Discrete Latent Representation (in bits):**
    For a tensor of shape $[B, H, W, D]$ and a codebook with $K$ vectors:
    $$	ext{Total Bits} = B 	imes H 	imes W 	imes \log_2(K)$$

---

## 3. Denoising Diffusion Probabilistic Models (DDPMs)
- **Forward Process (Encoding / Adding Noise):**
  The transitions are Markovian:
  $$x_t = \sqrt{lpha_t} x_{t-1} + \sqrt{1-lpha_t} \epsilon \quad 	ext{where} \quad \epsilon \sim \mathcal{N}(0, I)$$
  
  **Direct sampling of $x_t$ from $x_0$:**
  $$x_t = \sqrt{ar{lpha}_t} x_0 + \sqrt{1-ar{lpha}_t} \epsilon \quad 	ext{where} \quad ar{lpha}_t = \prod_{i=1}^t lpha_i \quad 	ext{and} \quad \epsilon \sim \mathcal{N}(0, I)$$

- **Reverse Process (Inference / Sampling):**
  To obtain a sample, the model must iteratively traverse the reverse decoding process, starting from pure Gaussian noise $x_T \sim \mathcal{N}(0, I)$.
  $$x_{t-1} = rac{1}{\sqrt{lpha_t}} \left( x_t - rac{1-lpha_t}{\sqrt{1-ar{lpha}_t}} \epsilon_	heta(x_t, t) ight) + \sigma_t z \quad 	ext{where} \quad z \sim \mathcal{N}(0, I)$$

- **DDPM ELBO terms:**
  The ELBO objective consists of three parts:
  1. **Reconstruction term:** $-\mathbb{E}_q[\log p_	heta(x_0|x_1)]$
  2. **Latent prior matching term:** $D_{KL}(q(x_T|x_0) || p_	heta(x_T))$
  3. **Denoising matching (consistency) term:** $\sum_{t=2}^T D_{KL}(q(x_{t-1}|x_t, x_0) || p_	heta(x_{t-1}|x_t))$
