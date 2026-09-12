# Reinforcement Learning in LLM Context

ਭFinal Notes – NLP ਤੋਂ RLHF ਤੱਕ
🟢 ਭਾਗ 1: NLP ਵਿੱਚ ਟੋਕਨਸ ਕਿਵੇਂ ਬਦਲਦੇ ਨੇ?
Embedding Matrix: ਹਰ ਟੋਕਨ ਨੂੰ high‑dimensional vector ਵਿੱਚ convert ਕੀਤਾ ਜਾਂਦਾ ਹੈ।

Causal Attention: Query (Q), Key (K), Value (V) ਮੈਟ੍ਰਿਕਸ ਬਣਦੇ ਹਨ। Masking ਨਾਲ model future words ਨਹੀਂ ਵੇਖਦਾ।

Output Layer: Attention + FCL ਤੋਂ ਬਾਅਦ logits ਬਣਦੇ ਹਨ → vocabulary distribution.

Softmax: Logits → probability distribution.
👉 ਇਹੀ distribution NLP ਵਿੱਚ “next token probability” ਹੈ, ਅਤੇ RL ਵਿੱਚ “policy action distribution” ਬਣ ਜਾਂਦੀ ਹੈ।

🔵 ਭਾਗ 2: RLHF (Reinforcement Learning from Human Feedback)
Reference Policy (
𝜋
𝑟
𝑒
𝑓
): Pre‑trained/SFT model.

Reward Model (
𝑟
𝜙
): Human feedback ਤੋਂ train ਕੀਤਾ neural net, ਜੋ response ਨੂੰ score ਦਿੰਦਾ ਹੈ।

Active Policy (
𝜋
𝜃
): ਉਹ LLM ਜਿਸਨੂੰ RL ਨਾਲ fine‑tune ਕੀਤਾ ਜਾ ਰਿਹਾ ਹੈ।

KL Penalty: Model ਨੂੰ reward hacking ਤੋਂ ਬਚਾਉਣ ਲਈ divergence penalty ਲਗਾਈ ਜਾਂਦੀ ਹੈ।
👉 Reward function: 
𝑅
(
𝑥
,
𝑦
)
=
𝑟
𝜙
(
𝑥
,
𝑦
)
−
𝛽
log
⁡
𝜋
𝜃
𝜋
𝑟
𝑒
𝑓

🟠 ਭਾਗ 3: PPO (Proximal Policy Optimization)
Probability Ratio (
𝑟
𝑡
(
𝜃
)
): ਨਵੀਂ policy vs ਪੁਰਾਣੀ policy ਦੀ probability change.

Advantage (
𝐴
𝑡
): Token expected value ਨਾਲੋਂ better ਜਾਂ worse ਸੀ?

Clipping (
𝜖
): Ratio ਨੂੰ limit ਕਰਦਾ ਹੈ ਤਾਂ ਜੋ model unstable ਨਾ ਹੋਵੇ।
👉 PPO surrogate objective:

𝐿
𝑃
𝑃
𝑂
(
𝜃
)
=
𝐸
𝑡
[
min
⁡
(
𝑟
𝑡
(
𝜃
)
𝐴
𝑡
,
clip
(
𝑟
𝑡
(
𝜃
)
,
1
−
𝜖
,
1
+
𝜖
)
𝐴
𝑡
)
]
🔄 ਪੂਰੀ ਕੜੀ (Full Loop)
User prompt → state (
𝑠
𝑡
)

Transformer maths (attention + softmax) → policy distribution (
𝜋
𝜃
)

Model → actions (tokens)

Reward model → score + KL penalty

PPO algorithm → weights update (
𝜃
)
👉 Next time model ਹੋਰ better token choose ਕਰੇਗਾ।

📌 Key Takeaway
NLP ਦੇ logits → RL ਵਿੱਚ policy distribution

RLHF ਨਾਲ model helpful, honest, harmless ਬਣਦਾ ਹੈ



