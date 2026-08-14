ਬਿਲਕੁਲ ਵੀਰੇ, ਮੈਂ ਤੁਹਾਡੀ ਸ਼ੀਟ ਅਪਡੇਟ ਕਰ ਦਿੰਦਾ ਹਾਂ! ਤੈਨੂੰ ਖੁਦ ਲਿਖਣ ਦੀ ਕੋਈ ਲੋੜ ਨਹੀਂ, ਬਸ ਇੱਥੋਂ ਸਿੱਧਾ ਕਾਪੀ ਕਰ ਲੈ।

ਨਾਲੇ ਮੈਂ ਦੇਖਿਆ ਕਿ ਪੁਰਾਣੀ ਸ਼ੀਟ ਨੂੰ ਕਾਪੀ-ਪੇਸਟ ਕਰਨ ਵੇਲੇ ਕੁਝ ਲੇਟੈਕਸ (LaTeX) ਟਾਈਪਿੰਗ ਦੀਆਂ ਗਲਤੀਆਂ ਆ ਗਈਆਂ ਸਨ (ਜਿਵੇਂ `\frac` ਦੀ ਜਗ੍ਹਾ `rac` ਅਤੇ `\theta` ਦੀ ਜਗ੍ਹਾ `heta` ਲਿਖਿਆ ਹੋਇਆ ਸੀ)। ਮੈਂ ਉਹ ਸਾਰੇ ਚਿੰਨ੍ਹ ਠੀਕ ਕਰ ਦਿੱਤੇ ਹਨ ਅਤੇ ਆਪਣੀ ਅੱਜ ਦੀ ਡਿਸਕਸ਼ਨ ਵਾਲੇ ਨਵੇਂ ਫਾਰਮੂਲੇ (Marginal Likelihood, Mixing Weights, ਅਤੇ Sum to 1 trick) ਵੀ ਇਸ ਵਿੱਚ ਜੋੜ ਦਿੱਤੇ ਹਨ।

ਇੱਥੇ ਤੁਹਾਡੀ ਅਪਡੇਟ ਕੀਤੀ ਹੋਈ ਨਵੀਂ ਸ਼ੀਟ ਹੈ:

---

# Mathematical Foundations of Generative AI - Formula Sheet



This formula sheet is compiled based on the provided resources, starting from the EM Algorithm to advanced Generative AI concepts (VAEs, DDPMs), tailored for solving quiz questions.

## 1. Expectation-Maximization (EM) Algorithm for Gaussian Mixture Models (GMMs)



* **PDF of Normal Distribution (1D):**

$$\mathcal{N}(x\vert{}\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$


* **E-Step (Responsibility / Posterior Probability):**


For a data point $x_i$ and component $k$:


$$\gamma(z_{ik}) = P(z=k\vert{}x_i) = \frac{\pi_k \mathcal{N}(x_i\vert{}\mu_k, \sigma_k^2)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i\vert{}\mu_j, \sigma_j^2)}$$



*(Note: $\pi_k$ is the prior probability of component $k$)*

* **The "Sum to 1" Trick:**
The sum of responsibilities for a single data point across all components is always 1:

$$\sum_{k=1}^K \gamma(z_{ik}) = 1$$


* **M-Step (Parameter Update for Mean):**


Updated mean for component $k$ across $N$ data points:


$$\mu_k' = \frac{\sum_{i=1}^N \gamma(z_{ik}) x_i}{\sum_{i=1}^N \gamma(z_{ik})}$$


* **M-Step (Parameter Update for Mixing Weights):**
Updated prior probability for component $k$:

$$\pi_k' = \frac{1}{N} \sum_{i=1}^N \gamma(z_{ik})$$


* **Marginal Log-Likelihood (Total Data Likelihood):**
To compute the total likelihood of the dataset $X$ before/after an update:

$$P(X) = \prod_{i=1}^N \sum_{k=1}^K \pi_k \mathcal{N}(x_i\vert{}\mu_k, \sigma_k^2)$$



---

## 2. Variational Autoencoders (VAEs)



* **ELBO (Evidence Lower Bound) Objective:**

$$\mathcal{L}(q) = J_\theta(q) = \mathbb{E}_{q_\phi(z\vert{}x)}[\log p_\theta(x, z) - \log q_\phi(z\vert{}x)]$$


$$\mathcal{L}(q) = \mathbb{E}_{q_\phi(z\vert{}x)}[\log p_\theta(x\vert{}z)] - D_{KL}(q_\phi(z\vert{}x) \vert{}\vert{} p(z))$$



*(Note: Maximizing ELBO is equivalent to minimizing the Negative Log-Likelihood + KL Divergence)*

* **Reparameterization Trick:**


Instead of sampling $z$ directly (which prevents backpropagation), express it as:


$$z = \mu_\phi(x) + \sigma_\phi(x) \cdot \epsilon \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, I)$$


* **KL Divergence Term (Standard Normal Prior):**


If $q_\phi(z\vert{}x) = \mathcal{N}(\mu, \Sigma)$ and $p(z) = \mathcal{N}(0, I)$:


$$D_{KL}[q_\phi(z\vert{}x) \vert{}\vert{} p(z)] = -\frac{1}{2} \sum_{j=1}^K \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$


* **$\beta$-VAE Objective:**

$$F_\theta(q) = \text{Reconstruction Loss} + \beta \cdot D_{KL}(q_\phi(z\vert{}x) \vert{}\vert{} p(z))$$



*Higher $\beta$ leads to better disentanglement but can cause posterior collapse. Lower $\beta$ leads to better reconstructions.*

* **VQ-VAE (Vector Quantized VAE):**

* **Commitment Loss:** $\beta \vert{}\vert{}z_e(x) - \text{sg}[e_k]\vert{}\vert{}_2^2$ (where 'sg' is the stop-gradient operator).


* **Size of Discrete Latent Representation (in bits):**


For a tensor of shape $[B, H, W, D]$ and a codebook with $K$ vectors:


$$\text{Total Bits} = B \times H \times W \times \log_2(K)$$





---

## 3. Denoising Diffusion Probabilistic Models (DDPMs)



* **Forward Process (Encoding / Adding Noise):**


The transitions are Markovian:


$$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, I)$$



**Direct sampling of $x_t$ from $x_0$:**

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon \quad \text{where} \quad \bar{\alpha}_t = \prod_{i=1}^t \alpha_i \quad \text{and} \quad \epsilon \sim \mathcal{N}(0, I)$$


* **Reverse Process (Inference / Sampling):**


To obtain a sample, the model must iteratively traverse the reverse decoding process, starting from pure Gaussian noise $x_T \sim \mathcal{N}(0, I)$.


$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z \quad \text{where} \quad z \sim \mathcal{N}(0, I)$$


* **DDPM ELBO terms:**


The ELBO objective consists of three parts:


1. **Reconstruction term:** $-\mathbb{E}_q[\log p_\theta(x_0\vert{}x_1)]$

2. **Latent prior matching term:** $D_{KL}(q(x_T\vert{}x_0) \vert{}\vert{} p_\theta(x_T))$

3. **Denoising matching (consistency) term:** $\sum_{t=2}^T D_{KL}(q(x_{t-1}\vert{}x_t, x_0) \vert{}\vert{} p_\theta(x_{t-1}\vert{}x_t))$
