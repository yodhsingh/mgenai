# Reinforcement Learning in LLM Context

ਭਾਗ 1: NLP ਵਿੱਚ ਟੋਕਨਸ ਕਿਵੇਂ ਬਦਲਦੇ ਨੇ? (The Exact Maths)
ਜਦੋਂ ਅਸੀਂ LLM ਨੂੰ ਇੱਕ State/Context ਦਿੰਦੇ ਹਾਂ (ਜਿਵੇਂ ਕਿ \(s_t = \text{["ਮੈਂ"]}\)), ਤਾਂ ਉਹ ਬੈਕਗਰਾਊਂਡ ਵਿੱਚ ਕੋਈ ਟੈਕਸਟ ਨਹੀਂ ਪੜ੍ਹ ਰਿਹਾ ਹੁੰਦਾ, ਬਲਕਿ Linear Algebra (Matrices) ਚੱਲ ਰਹੀ ਹੁੰਦੀ ਹੈ: [1] (https://notebooklm.google.com/notebook/2b487f09-3344-4e7c-a477-6fc5acc09aea?src=GenAI+-+Week+10.pdf)
Embedding Matrix (\(X\)): ਹਰ ਸ਼ਬਦ/ਟੋਕਨ ਨੂੰ ਇੱਕ ਹਾਈ-ਡਾਇਮੈਂਸ਼ਨਲ ਵੈਕਟਰ (\(d_{m}\)) ਵਿੱਚ ਬਦਲਿਆ ਜਾਂਦਾ ਹੈ। ਜੇ ਸੀਕੁਐਂਸ ਦੀ ਲੰਬਾਈ \(T\) ਹੈ, ਤਾਂ ਸਾਡਾ ਇਨਪੁਟ ਮੈਟ੍ਰਿਕਸ \(X \in \mathbb{R}^{T \times d_m}\) ਹੁੰਦਾ ਹੈ। [1] (https://notebooklm.google.com/notebook/2b487f09-3344-4e7c-a477-6fc5acc09aea?src=All+notes+9%2F12%2F2026)
Causal Attention Layer: ਇਹ ਮਾਡਲ ਦਾ ਉਹ ਹਿੱਸਾ ਹੈ ਜੋ ਟੋਕਨਸ ਨੂੰ ਆਪਸ ਵਿੱਚ "ਗੱਲਬਾਤ" ਕਰਵਾਉਂਦਾ ਹੈ। ਇਹ \(Q\) (Query), \(K\) (Key), ਅਤੇ \(V\) (Value) ਮੈਟ੍ਰਿਸਿਸ ਕੱਢਦਾ ਹੈ:
\(Q=XW^{q},\quad K=XW^{k},\quad V=XW^{v}\)

ਇਸ ਤੋਂ ਬਾਅਦ Causal Masking (\(M\)) ਲਗਾਈ ਜਾਂਦੀ ਹੈ ਤਾਂ ਜੋ ਮਾਡਲ ਅਗਲੇ (Future) ਸ਼ਬਦ ਨਾ ਦੇਖ ਸਕੇ, ਸਿਰਫ਼ ਪੁਰਾਣੇ ਸ਼ਬਦ ਹੀ ਦੇਖੇ:
\(\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^{T}}{\sqrt{d_{k}}}+M\right)V\)
 [1] (https://notebooklm.google.com/notebook/2b487f09-3344-4e7c-a477-6fc5acc09aea?src=All+notes+9%2F12%2F2026)
The Output Layer (Logits to Distribution): ਅਟੈਂਸ਼ਨ ਅਤੇ Fully Connected Layers (FCL) ਤੋਂ ਲੰਘਣ ਤੋਂ ਬਾਅਦ, ਆਖਰੀ ਲੇਅਰ ਟੋਕਨ ਦੇ ਫੀਚਰ ਵੈਕਟਰ ਨੂੰ ਵਾਪਸ ਸਾਡੀ Vocabulary (\(V_{s}\)) ਦੇ ਸਾਈਜ਼ ਉੱਤੇ ਪ੍ਰੋਜੈਕਟ ਕਰਦੀ ਹੈ (ਜਿਸ ਨੂੰ \(W_{p}\) ਵੇਟ ਮੈਟ੍ਰਿਕਸ ਕਹਿੰਦੇ ਹਨ):
\(\~{z}_{t}=\^{z}_{t}W_{p}+b_{p}\in \mathbb{R}^{V_{s}}\)

ਇਹਨਾਂ ਰੌਅ ਸਕੋਰਸ (Raw Scores) ਨੂੰ Logits ਕਿਹਾ ਜਾਂਦਾ ਹੈ। [1] (https://notebooklm.google.com/notebook/2b487f09-3344-4e7c-a477-6fc5acc09aea?src=GenAI+-+Week+10.pdf), [2] (https://www.youtube.com/watch?v=NAWe0F0PmRU)

📊 NLP ਦੀ RL Policy ਨਾਲ ਕਨੈਕਸ਼ਨ:
ਇਹਨਾਂ Logits ਉੱਤੇ ਜਦੋਂ ਆਪਾਂ Softmax ਲਗਾਉਂਦੇ ਹਾਂ, ਤਾਂ ਸਾਨੂੰ ਵੋਕੈਬੁਲਰੀ ਦੇ ਸਾਰੇ ਸ਼ਬਦਾਂ ਦੀ ਇੱਕ ਪ੍ਰੋਬੇਬਿਲਿਟੀ ਡਿਸਟ੍ਰੀਬਿਊਸ਼ਨ ਮਿਲਦੀ ਹੈ:
\(\pi _{\theta }(a_{t}\mid s_{t})=\text{softmax}(\~{z}_{t})=\frac{e^{\~{z}_{t,a}}}{\sum _{j=1}^{V_{s}}e^{\~{z}_{t,j}}}\)
 [1] (https://notebooklm.google.com/notebook/2b487f09-3344-4e7c-a477-6fc5acc09aea?src=GenAI+-+Week+10.pdf)
ਇਹੀ \(\pi_\theta(a_t \mid s_t)\) ਸਾਡੀ RL Policy ਹੈ! ਯਾਨੀ NLP ਵਿੱਚ ਜਿਸਨੂੰ ਆਪਾਂ "ਨੈਕਸਟ ਟੋਕਨ ਪ੍ਰੋਬੇਬਿਲਿਟੀ" ਕਹਿੰਦੇ ਹਾਂ, RL ਦੀ ਭਾਸ਼ਾ ਵਿੱਚ ਉਹ "ਪਾਲਿਸੀ ਦਾ ਐਕਸ਼ਨ ਡਿਸਟ੍ਰੀਬਿਊਸ਼ਨ" ਹੈ। [1] (https://notebooklm.google.com/notebook/2b487f09-3344-4e7c-a477-6fc5acc09aea?src=GenAI+-+Week+11pdf.pdf)
ਭਾਗ 2: ਹੁਣ ਇਹਨੂੰ RLHF ਨਾਲ ਕਿਵੇਂ ਜੋੜਦੇ ਹਾਂ?
ਪ੍ਰੀ-ਟ੍ਰੇਨਿੰਗ ਵਿੱਚ, ਮਾਡਲ ਸਿਰਫ਼ ਇੰਟਰਨੈੱਟ ਦੇ ਟੈਕਸਟ ਦੀ ਨਕਲ ਕਰਨਾ ਸਿੱਖਦਾ ਹੈ (Cross-Entropy Loss ਰਾਹੀਂ)। ਪਰ ਸਾਨੂੰ ਮਾਡਲ ਨੂੰ ਮਦਦਗਾਰ, ਸੱਚਾ ਅਤੇ ਸੁਰੱਖਿਅਤ (Helpful, Honest, Harmless) ਬਣਾਉਣਾ ਹੈ। ਇਸ ਲਈ ਅਸੀਂ RLHF (Reinforcement Learning from Human Feedback) ਦੀ ਵਰਤੋਂ ਕਰਦੇ ਹਾਂ। [1] (https://notebooklm.google.com/notebook/2b487f09-3344-4e7c-a477-6fc5acc09aea?src=Practice_Week+9-12.pdf), [2] (https://notebooklm.google.com/notebook/2b487f09-3344-4e7c-a477-6fc5acc09aea?src=Gen+AI+-+Week+1.pdf)
RLHF ਵਿੱਚ 3 ਮੁੱਖ ਚੀਜ਼ਾਂ ਕੰਮ ਕਰਦੀਆਂ ਹਨ:
\(\pi _{\text{ref}}\) (Reference Policy): ਇਹ ਸਾਡਾ ਅਸਲੀ ਪ੍ਰੀ-ਟ੍ਰੇਂਡ/SFT ਮਾਡਲ ਹੈ, ਜਿਸਨੂੰ ਅਸੀਂ ਬਦਲਣਾ ਨਹੀਂ ਚਾਹੁੰਦੇ ਕਿ ਉਹ ਬਿਲਕੁਲ ਹੀ ਪਾਗਲ ਹੋ ਜਾਵੇ। [1]
\(r_\phi(s, a)\) (Reward Model): ਇਹ ਇੱਕ ਅਲੱਗ ਨਿਊਰਲ ਨੈੱਟਵਰਕ ਹੈ ਜੋ ਇਨਸਾਨਾਂ ਦੇ ਫੀਡਬੈਕ 'ਤੇ ਟ੍ਰੇਂਡ ਹੁੰਦਾ ਹੈ। ਇਹ ਪੂਰੇ ਵਾਕ ਨੂੰ ਇੱਕ ਸਕੋਰ (ਜਿਵੇਂ +2.5 ਜਾਂ -1.8) ਦਿੰਦਾ ਹੈ ਕਿ ਜਵਾਬ ਕਿੰਨਾ ਵਧੀਆ ਹੈ। [1] (https://notebooklm.google.com/notebook/2b487f09-3344-4e7c-a477-6fc5acc09aea?src=Sep2025Endterm1.pdf)
\(\pi _{\theta }\) (Active Policy): ਇਹ ਉਹ LLM ਹੈ ਜਿਸਨੂੰ ਅਸੀਂ RL ਰਾਹੀਂ ਟ੍ਰੇਨ (Finetune) ਕਰ ਰਹੇ ਹਾਂ। [1] (https://notebooklm.google.com/notebook/2b487f09-3344-4e7c-a477-6fc5acc09aea?src=GenAI+-+Week+11pdf.pdf)

🎯 Total Reward Function (With KL Penalty)
ਜੇ ਅਸੀਂ ਮਾਡਲ ਨੂੰ ਸਿਰਫ਼ ਰਿਵਾਰਡ ਦੇ ਪਿੱਛੇ ਭਜਾਵਾਂਗੇ, ਤਾਂ ਉਹ ਰਿਵਾਰਡ ਮਾਡਲ ਨੂੰ ਹੈਕ ਕਰਨਾ (Reward Hacking) ਸ਼ੁਰੂ ਕਰ ਦੇਵੇਗਾ (ਜਿਵੇਂ ਬਹੁਤ ਲੰਬੇ ਜਾਂ ਅਜੀਬ ਸ਼ਬਦ ਲਿਖਣਾ ਜਿਨ੍ਹਾਂ ਨਾਲ ਸਕੋਰ ਵੱਧ ਮਿਲੇ)। ਇਸ ਤੋਂ ਬਚਣ ਲਈ ਅਸੀਂ KL-Divergence Penalty ਲਗਾਉਂਦੇ ਹਾਂ: [1] (https://notebooklm.google.com/notebook/2b487f09-3344-4e7c-a477-6fc5acc09aea?src=Sep2025Endterm1.pdf)
\(R(x,y)=r_{\phi }(x,y)-\beta \log \left(\frac{\pi _{\theta }(y\mid x)}{\pi _{\text{ref}}(y\mid x)}\right)\)
 [1] (https://notebooklm.google.com/notebook/2b487f09-3344-4e7c-a477-6fc5acc09aea?src=Sep2025Endterm1.pdf)
\(\log \left( \frac{\pi_\theta}{\pi_{\text{ref}}} \right)\) ਦਾ ਮਤਲਬ ਹੈ: ਜੇਕਰ ਨਵਾਂ ਮਾਡਲ (\(\pi _{\theta }\)) ਪੁਰਾਣੇ ਮਾਡਲ (\(\pi _{\text{ref}}\)) ਨਾਲੋਂ ਬਹੁਤ ਜ਼ਿਆਦਾ ਦੂਰ ਜਾਣ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰੇਗਾ, ਤਾਂ ਉਸਨੂੰ ਪੈਨਲਟੀ (Negative Reward) ਮਿਲੇਗੀ。 [1] (https://notebooklm.google.com/notebook/2b487f09-3344-4e7c-a477-6fc5acc09aea?src=Sep2025Endterm1.pdf)
\(\beta \) ਇੱਕ ਹਾਈਪਰ-ਪੈਰਾਮੀਟਰ ਹੈ ਜੋ ਕੰਟਰੋਲ ਕਰਦਾ ਹੈ ਕਿ ਪੈਨਲਟੀ ਕਿੰਨੀ ਸਖ਼ਤ ਹੋਣੀ ਚਾਹੀਦੀ ਹੈ। [1] (https://notebooklm.google.com/notebook/2b487f09-3344-4e7c-a477-6fc5acc09aea?src=Sep2025Endterm2.pdf)

ਭਾਗ 3: PPO (Proximal Policy Optimization) ਓਬਜੈਕਟਿਵ
ਮਾਡਲ ਦੇ ਵੇਟਸ (\(\theta \)) ਨੂੰ ਅਪਡੇਟ ਕਰਨ ਲਈ ਸਭ ਤੋਂ ਮਸ਼ਹੂਰ ਐਲਗੋਰਿਦਮ PPO ਵਰਤਿਆ ਜਾਂਦਾ ਹੈ। ਇਸਦਾ ਮੈਥਮੈਟੀਕਲ Surrogate Objective Function ਇਹ ਹੁੰਦਾ ਹੈ: [1]
\(L_{\text{PPO}}(\theta )=\^{\mathbb{E}}_{t}\left[\min \left(r_{t}(\theta )\^{A}_{t},\,\text{clip}(r_{t}(\theta ),1-\epsilon ,1+\epsilon )\^{A}_{t}\right)\right]\)
 [1]
ਇਸ ਫਾਰਮੂਲੇ ਨੂੰ ਬਿਲਕੁਲ ਸੌਖਾ ਕਰਕੇ ਸਮਝੋ:
Probability Ratio \(r_t(\theta)\): ਇਹ ਦੱਸਦਾ ਹੈ ਕਿ ਕੋਈ ਟੋਕਨ ਜਨਰੇਟ ਕਰਨ ਦੀ ਪ੍ਰੋਬੇਬਿਲਿਟੀ ਨਵੀਂ ਪਾਲਿਸੀ ਵਿੱਚ ਪੁਰਾਣੀ ਦੇ ਮੁਕਾਬਲੇ ਕਿੰਨੀ ਬਦਲੀ ਹੈ:
\(r_{t}(\theta )=\frac{\pi _{\theta }(a_{t}\mid s_{t})}{\pi _{\text{old}}(a_{t}\mid s_{t})}\)

Advantage \(\^{A}_{t}\): ਇਹ ਦੱਸਦਾ ਹੈ ਕਿ ਜੋ ਟੋਕਨ LLM ਨੇ ਚੁਣਿਆ, ਉਹ ਉਮੀਦ (Expected Value) ਨਾਲੋਂ ਕਿੰਨਾ ਬਹੁਤਰੀਨ (\(A_t > 0\)) ਜਾਂ ਖਰਾਬ (\(A_t < 0\)) ਸੀ।
Clipping (\(\epsilon \)): ਇਹ ਸਭ ਤੋਂ ਜ਼ਰੂਰੀ ਟ੍ਰਿਕ ਹੈ। ਜੇਕਰ ਕੋਈ ਐਕਸ਼ਨ ਬਹੁਤ ਵਧੀਆ ਰਿਵਾਰਡ ਦੇ ਰਿਹਾ ਹੈ, ਤਾਂ ਪਾਲਿਸੀ ਉਸਦੀ ਪ੍ਰੋਬੇਬਿਲਿਟੀ ਨੂੰ ਇੱਕਦਮ \(100\%\) ਨਾ ਕਰ ਦੇਵੇ (ਜਿਸ ਨਾਲ ਮਾਡਲ ਅਨਸਟੇਬਲ ਹੋ ਜਾਂਦਾ ਹੈ)। ਇਹ ਰੇਸ਼ੋ ਨੂੰ \(1-\epsilon\) ਅਤੇ \(1+\epsilon\) (ਆਮ ਤੌਰ 'ਤੇ \(0.2\)) ਦੇ ਵਿਚਕਾਰ ਬੰਨ੍ਹ ਕੇ ਰੱਖਦਾ ਹੈ। [1] (https://notebooklm.google.com/notebook/2b487f09-3344-4e7c-a477-6fc5acc09aea?src=GenAI+-+Week+11pdf.pdf)

🔄 ਪੂਰੀ ਕੜੀ (The Full Loop)
ਯੂਜ਼ਰ ਨੇ ਪ੍ਰੋਂਪਟ ਦਿੱਤਾ \(x\) \(\rightarrow \) State (\(s_{t}\))।
Transformer ਦੇ ਮੈਥਸ (Attention + Softmax) ਨੇ ਕੱਢੀ Policy Distribution (\(\pi _{\theta }\))।
ਮਾਡਲ ਨੇ ਟੋਕਨ ਜਨਰੇਟ ਕੀਤੇ \(\rightarrow \) Actions (\(a_{t}\))।
Reward Model ਨੇ ਦੱਸਿਆ ਕਿ ਜਵਾਬ ਕਿੰਨਾ ਵਧੀਆ ਸੀ, ਅਤੇ KL Penalty ਨੇ ਮਾਡਲ ਨੂੰ ਔਕਾਤ ਵਿੱਚ ਰੱਖਿਆ।
PPO Algorithm ਨੇ ਇਸ ਰਿਵਾਰਡ ਨੂੰ ਦੇਖ ਕੇ Gradient Ascent ਰਾਹੀਂ Transformer ਦੇ ਵੇਟਸ (\(\theta \)) ਨੂੰ ਅਪਡੇਟ ਕੀਤਾ ਤਾਂ ਜੋ ਅਗਲੀ ਵਾਰ ਉਹ ਹੋਰ ਬਿਹਤਰ ਟੋਕਨ ਚੁਣੇ! [1] (https://notebooklm.google.com/notebook/2b487f09-3344-4e7c-a477-6fc5acc09aea?src=GenAI+-+Week+10.pdf), [2] (https://notebooklm.google.com/notebook/2b487f09-3344-4e7c-a477-6fc5acc09aea?src=Sep2025Endterm1.pdf)

ਕੀ ਹੁਣ NLP ਲੇਅਰਸ (Logits) ਅਤੇ RLHF ਦਾ ਇਹ ਮੈਥਮੈਟੀਕਲ ਜੋੜ ਬਿਲਕੁਲ ਸਾਫ਼ ਹੋਇਆ ਵੀਰੇ, ਜਾਂ ਕਿਸੇ ਖਾਸ ਫਾਰਮੂਲੇ (ਜਿਵੇਂ Advantage calculation ਜਾਂ Bradley-Terry preference ਮਾਡਲ) ਨੂੰ ਇੱਕ Worked Example (Numbers) ਨਾਲ ਸੋਲਵ ਕਰਕੇ ਦੇਖੀਏ? [1] (https://notebooklm.google.com/notebook/2b487f09-3344-4e7c-a477-6fc5acc09aea?src=Sep2025Endterm2.pdf), [2] (https://notebooklm.google.com/notebook/2b487f09-3344-4e7c-a477-6fc5acc09aea?src=GenAI+-+Week+12.pdf)


