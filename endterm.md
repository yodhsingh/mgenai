# Guided Formula Sheet: Mathematical Foundations of Generative AI (Weeks 9-12)

This technical reference provides the mathematical synthesis and architectural insights required for advanced generative modeling, focusing on Diffusion, Transformers, RLHF, and State-Space Models.

---

## 1. Week 9: Diffusion Models (DDPM, DDIM, & Guidance)

### Formula Synthesis

**Forward Diffusion Process:**


\[
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
\]



**Reverse Transition (Tweedie’s Formula):**


\[
\mathbb{E}[x_0 | x_t] = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( x_t + (1 - \bar{\alpha}_t)\nabla_{x_t} \log p(x_t) \right)
\]



### Symbol Definitions

| Symbol | Definition |
|--------|------------|
| \(x_t\) | Latent state at timestep \(t\) |
| \(\alpha_t\) | \(1 - \beta_t\), variance schedule |
| \(\bar{\alpha}_t\) | \(\prod_{i=1}^t \alpha_i\) (cumulative noise schedule) |
| \(\epsilon_\theta(x_t, t)\) | Noise prediction network |
| \(w\) | Guidance scale |

**Punjabi Intuition:**  
"Ik saaf tasveer te mitti pauna (noise addition) te pher us mitti nu hauli-hauli saaf karke asli tasveer vapas kaddna (denoising)."

### Numerical Pattern: U-Net Broadcasting

- Reshape \(T \in B, C \to B, C, 1, 1\)  
- Broadcast temporal signal across spatial dimensions \((H, W)\) of feature map \(F \in B, C, H, W\)

**Exam Traps:**
- \(t=0\) → perfect reconstruction  
- DDPM stochastic, DDIM deterministic ODE sampler

### Classifier-Free Guidance (CFG)



\[
\tilde{\epsilon}_\theta(x_t, c) = (1+w)\epsilon_\theta(x_t, c) - w\epsilon_\theta(x_t, \emptyset)
\]



---

## 2. Week 10: Transformer Architecture & Attention Mechanics

### Attention Formula



\[
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]



### Multi-Head Attention (MHA) Counting

Example: \(d_{model}=512, h=8\)

- Parameters per \(W_Q, W_K, W_V\): \(512 \times 512 = 262,144\)  
- Total for Q,K,V: \(786,432\)  
- Output projection \(W_O\): \(262,144\)  
- **Total Attention Sub-layer:** \(1,048,576\)

**Punjabi Intuition:**  
"Transformer ik Library Index varga hai. Query oh hai jo tuhanu chahida hai, Key kitaab de baahar laggia label hai, te Value oh jaankari hai jo kitaab de andar likhi hoyi hai."

### Architectural Components

- **Causal Masking:** \(M_{ij} = -\infty\) for \(j>i\)  
- **LayerNorm:** Normalizes across \(d_{model}\)  
- **Positional Encoding:**


\[
PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad
PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]



---

## 3. Week 11: RLHF & Policy Optimization

### Policy Gradient & PPO

Objective:


\[
J(\theta) = \mathbb{E}\sum \gamma^t r_t
\]



Gradient:


\[
\nabla_\theta J(\theta) = \mathbb{E}\nabla_\theta \log \pi_\theta(a|s) A^\pi(s,a)
\]



**PPO Clipped Surrogate Objective:**


\[
L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\right)\right]
\]



Where:


\[
r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{old}(a|s)}
\]



### Direct Preference Optimization (DPO)



\[
\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x,y_w,y_l)} \log \sigma \Big( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \Big)
\]



**Punjabi Intuition:**  
"RLHF eda hai jivein ik Ustad (Human) apne bache (Model) nu changey kamm te 'shabaash' (reward) denda hai taaki oh apna vyavhaar sudhaar sake."

---

## 4. Week 12: State-Space Models (SSMs) & Mamba

### Discretization & Duality



\[
\bar{A} = \exp(\Delta A), \quad
\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I)\cdot \Delta B
\]



Hidden State Update:


\[
h_k = \bar{A}h_{k-1} + \bar{B}x_k, \quad y_k = \bar{C}h_k
\]



**Punjabi Intuition:**  
"Mamba ik 'Channi' (sieve/filter) varga hai jo sirf zaroori jaankari nu yaad rakhda hai te baaki mitti nu baahar kadd denda hai."

### Symbol Definitions

| Symbol | Definition |
|--------|------------|
| \(\Delta\) | Step size |
| \(h_t\) | Hidden state (memory) |
| \(A,B,C\) | System matrices |

**Exam Traps:**
- Duality: Parallel training \(O(L \log L)\), sequential inference \(O(L)\)  
- Hardware efficiency: SRAM recurrent state avoids HBM bottleneck

---

## 5. Comprehensive Exam Readiness Checklist

| Model Type | Key Strength | Complexity | Formula |
|------------|--------------|------------|---------|
| Diffusion | Visual Quality | \(O(T)\) | Tweedie’s Denoiser |
| Transformer | Global Context | \(O(L^2)\) | Scaled Dot-Product Attention |
| RLHF/DPO | Human Alignment | Fine-tuning Overhead | PPO / DPO Loss |
| SSMs (Mamba) | Long Context | \(O(L)\) | ZOH Discretization (\(\bar{A}, \bar{B}\)) |

---
