### Guided Formula Sheet: Mathematical Foundations of Generative AI (Weeks 9-12)

This technical reference provides the mathematical synthesis and architectural insights required for advanced generative modeling, focusing on Diffusion, Transformers, RLHF, and State-Space Models.

#### 1\. Week 9: Diffusion Models (DDPM, DDIM, & Guidance)

##### Formula Synthesis

Diffusion models define a forward manifold transition through noise injection and a learned reverse transition to recover the data distribution.**Forward Diffusion Process:**   $$x\_t \= \\sqrt{\\bar{\\alpha}*t}x\_0 \+ \\sqrt{1 \- \\bar{\\alpha}t}\\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0, \\mathbf{I})$$**Reverse Transition**  \*\*$p\\theta(x*{t-1}|x\_t)\*\*$  **:**  The model learns to approximate the score function of the data distribution. Using  **Tweedie’s Formula** , we can estimate the clean image  $x\_0$  directly from the noisy observation  $x\_t$  using the predicted noise (or score):  $$\\mathbb{E}x\_0 | x\_t \= \\frac{1}{\\sqrt{\\bar{\\alpha}\_t}}\\left(x\_t \+ (1 \- \\bar{\\alpha}*t)\\nabla*{x\_t} \\log p(x\_t)\\right)$$

##### Symbol Definitions

Symbol,Definition

$x\_t$,Latent state at timestep  $t$

$\\alpha\_t$,"$1 \- \\beta\_t$ , where  $\\beta\_t$  is the variance schedule at step  $t$"

$\\bar{\\alpha}\_t$,$\\prod\_{i=1}^t \\alpha\_i$  (Cumulative noise schedule)

"$\\epsilon\_\\theta(x\_t, t)$",The noise prediction network (approximating  $\\epsilon$ )

$w$,Guidance scale for Classifier-Free Guidance

**Punjabi Intuition:**  "Ik saaf tasveer te mitti pauna (noise addition) te pher us mitti nu hauli-hauli saaf karke asli tasveer vapas kaddna (denoising)."

##### Numerical Pattern: U-Net Broadcasting

As per  **SOURCE\_CONTEXT Q135** , when integrating the time embedding  $T$  into spatial feature maps:

* **Reshape Operation:**  The embedding  $T \\in B, C$  must be reshaped to  $B, C, 1, 1$ .  
* **Application:**  This allows the temporal signal to be broadcasted across the spatial dimensions  $(H, W)$  of the feature map  $F \\in B, C, H, W$  for additive or multiplicative fusion.**Exam Traps:**  
* **The**  **$t=0**$  **Boundary:**  As noted in  **SOURCE\_CONTEXT Q132** , at  $t=0$ , we assume  **perfect reconstruction**  because the posterior becomes deterministic.  
* **Stochasticity:**  Remember that  **DDPM**  uses stochastic Langevin-like sampling, whereas  **DDIM**  is a deterministic ODE-based sampler that allows for faster traversal of the latent space.

##### Classifier-Free Guidance (CFG)

To enhance sample quality by pushing the model away from the unconditioned density:  $$\\tilde{\\epsilon}*\\theta(x\_t, c) \= (1+w)\\epsilon*\\theta(x\_t, c) \- w\\epsilon\_\\theta(x\_t, \\emptyset)$$

* **Scientist Note:**  If  $w \> 1$ , the model "over-emphasizes" the condition  $c$ , increasing fidelity at the expense of sample diversity.

#### 2\. Week 10: Transformer Architecture & Attention Mechanics

##### Attention Formula

$$\\text{Attention}(Q, K, V) \= \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d\_k}}\\right)V$$

##### Multi-Head Attention (MHA) Counting

Based on the linear layer parameter logic in  **SOURCE\_CONTEXT Q189** , we calculate the trainable parameters for the attention mechanism of a Transformer Block.**Numerical Example:**  Let  $d\_{model} \= 512$  and heads  $h \= 8$ .

* **Projecting**  **$Q, K, V**$  **:**  Each projection matrix ( $W\_Q, W\_K, W\_V$ ) maps  $d\_{model} \\to d\_{model}$ .  
* Parameters per matrix:  $d\_{model} \\times d\_{model} \= 512 \\times 512 \= 262,144$ .  
* Total for  $Q, K, V \= 3 \\times d\_{model}^2 \= 786,432$ .  
* **Output Projection (**  **$W\_O**$  **):**  After concatenating head outputs, we project back to  $d\_{model}$ .  
* Parameters for  $W\_O \= d\_{model} \\times d\_{model} \= 262,144$ .  
* **Total Attention Sub-layer:**   $4 \\times d\_{model}^2 \= 1,048,576$  (excluding biases).**Punjabi Intuition:**  "Transformer ik Library Index varga hai.  **Query**  oh hai jo tuhanu chahida hai,  **Key**  kitaab de baahar laggia label hai, te  **Value**  oh jaankari hai jo kitaab de andar likhi hoyi hai."

##### Architectural Components

* **Causal Masking:**  Ensuring  $M\_{ij} \= \-\\infty$  for  $j \> i$  to prevent look-ahead during auto-regression.  
* **LayerNorm:**  Preferred over BatchNorm because it normalizes across the  $d\_{model}$  dimension for each token, providing stability against variable sequence lengths.  
* **Sinusoidal Positional Encoding:**   $$PE\_{(pos, 2i)} \= \\sin\\left(\\frac{pos}{10000^{\\frac{2i}{d\_{model}}}}\\right)$$   $$PE\_{(pos, 2i+1)} \= \\cos\\left(\\frac{pos}{10000^{\\frac{2i}{d\_{model}}}}\\right)$$**Exam Traps:**  After MHA concatenation, the dimensionality  **must**  return to  $d\_{model}$  via  $W\_O$ . If  $d\_k$  is the dimension per head, then  $h \\times d\_k \= d\_{model}$ .

#### 3\. Week 11: RLHF & Policy Optimization

##### Policy Gradient & PPO

The objective is to maximize  $J(\\theta) \= \\mathbb{E}\\sum \\gamma^t r\_t$ . The gradient is estimated as:  $$\\nabla\_\\theta J(\\theta) \= \\mathbb{E}\\nabla\_\\theta \\log \\pi\_\\theta(a|s) A^\\pi(s,a)$$**PPO Clipped Surrogate Objective:**   $$L^{CLIP}(\\theta) \= \\mathbb{E}\\left\\min\\left(r\_t(\\theta) \\hat{A}\_t, \\text{clip}(r\_t(\\theta), 1-\\epsilon, 1+\\epsilon)\\hat{A}\_t\\right)\\right$$  Where  $r\_t(\\theta) \= \\frac{\\pi\_\\theta(a|s)}{\\pi\_{old}(a|s)}$ . The  $\\epsilon$  hyperparameter prevents the "collapse" of the policy by limiting the update magnitude.

##### Direct Preference Optimization (DPO)

DPO replaces the Reward Model and PPO loop with a single maximum likelihood objective.  **Implicit Preference Loss:**   $$\\mathcal{L}*{DPO}(\\pi*\\theta; \\pi\_{ref}) \= \-\\mathbb{E}\_{(x, y\_w, y\_l)} \\left\\log \\sigma \\left(\\beta \\log \\frac{\\pi\_\\theta(y\_w|x)}{\\pi\_{ref}(y\_w|x)} \- \\beta \\log \\frac{\\pi\_\\theta(y\_l|x)}{\\pi\_{ref}(y\_l|x)}\\right)\\right$$

* **Pedagogical Note:**  DPO leverages the fact that the optimal reward can be expressed analytically in terms of the log-ratio of the optimal policy to the reference policy.**Punjabi Intuition:**  "RLHF eda hai jivein ik Ustad (Human) apne bache (Model) nu changey kamm te 'shabaash' (reward) denda hai taaki oh apna vyavhaar sudhaar sake."

#### 4\. Week 12: State-Space Models (SSMs) & Mamba

##### Discretization & Duality

SSMs transform continuous differential equations into discrete-time recurrences using the Zero-Order Hold (ZOH) method.**Discretization Formula:**   $$\\mathbf{\\bar{A}} \= \\exp(\\Delta \\mathbf{A})$$   $$\\mathbf{\\bar{B}} \= (\\Delta \\mathbf{A})^{-1}(\\exp(\\Delta \\mathbf{A}) \- I) \\cdot \\Delta \\mathbf{B}$$**Hidden State Update (Inference Mode):**   $$h\_k \= \\bar{A}h\_{k-1} \+ \\bar{B}x\_k$$   $$y\_k \= \\bar{C}h\_k$$**Punjabi Intuition:**  "Mamba ik 'Channi' (sieve/filter) varga hai jo sirf zaroori jaankari nu yaad rakhda hai te baaki mitti nu baahar kadd denda hai (Selective SSM)."

##### Symbol Definitions

Symbol,Definition

$\\Delta$,Step size (determines the resolution of discretization)

$h\_t$,"Hidden state (The ""memory"" of the sequence)"

"$\\mathbf{A, B, C}$","System matrices; in Mamba, these are input-dependent (Selective)"

**Exam Traps:**

* **The Duality:**  SSMs allow for  **Parallel/Convolutional Training**  (efficient  $O(L \\log L)$  on GPUs) but enable  **Sequential/Recurrent Inference**  (memory-efficient  $O(L)$ ).  
* **Hardware Efficiency:**  Mamba’s performance gain comes from keeping the recurrent state in  **SRAM**  to avoid the "Memory Wall" (HBM bottlenecks) associated with Transformer KV-caches.

#### 5\. Comprehensive Exam Readiness Checklist

Model Type,Key Strength,Computational Complexity,Formula to Recall

Diffusion,Visual Quality,$O(T)$  Sampling Steps,Tweedie's Denoiser

Transformer,Global Context,$O(L^2)$  Sequence Scaling,Scaled Dot-Product Attention

RLHF/DPO,Human Alignment,Fine-tuning Overhead,Clipped Surrogate / DPO Loss

SSMs (Mamba),Long Context,$O(L)$  Sequence Scaling,"ZOH Discretization  $(\\bar{A}, \\bar{B})$"

&nbsp;
