Every reasoning model is quietly paying a tax. Test-time scaling — longer chains of thought, more tool calls, hundred-step agent trajectories — runs straight into attention's quadratic cost, and the bill comes due exactly where the new capabilities live: in very long contexts. **DeepSeek-V4** is a paper about paying less. Its two models, **V4-Pro** (1.6T total, 49B activated) and **V4-Flash** (284B total, 13B activated), both natively support one-million-token contexts, and at that length V4-Pro needs **27% of the single-token inference FLOPs and 10% of the KV cache** of DeepSeek-V3.2 — while activating *more* parameters per token.

![Left: benchmark performance of DeepSeek-V4-Pro-Max against Claude Opus 4.6, GPT-5.4, and Gemini-3.1-Pro. Right: single-token inference FLOPs and accumulated KV cache versus sequence length, showing V4-Pro at 3.7× lower FLOPs and 9.5× smaller KV cache than V3.2 at 1M tokens, and V4-Flash at 9.8× and 13.7× respectively.](/blog/deepseek-v4/performance.png)

## Two ways to shrink attention, interleaved

The architectural centerpiece is a hybrid of two new attention mechanisms that make opposite trade-offs and are stacked in alternating layers.

![Overall architecture of the DeepSeek-V4 series: interleaved CSA and HCA attention layers, DeepSeekMoE feed-forward layers, and Manifold-Constrained Hyper-Connections replacing conventional residuals.](/blog/deepseek-v4/architecture.png)

**Compressed Sparse Attention (CSA)** does compression *and* sparsity. It first pools every *m* tokens' KV entries into one, using learned per-token compression weights and positional biases — with a deliberate overlap, so each compressed entry draws from 2*m* source entries while the sequence still shrinks by *m*×. Then it applies DeepSeek Sparse Attention on top: a lightning indexer scores each query against the compressed entries and a top-*k* selector keeps only the best ones for core attention. In V4-Pro, *m* = 4 and top-*k* = 1024.

![Core architecture of CSA. KV entries are compressed to 1/m, then DeepSeek Sparse Attention selects the top-k compressed entries; a small sliding-window branch is added back to preserve local fine-grained dependencies.](/blog/deepseek-v4/csa.png)

**Heavily Compressed Attention (HCA)** goes the other way: crush the KV cache far harder — *m*′ = 128, thirty-two times CSA's ratio — but keep attention fully dense over what remains. No indexer, no top-*k*, no selection overhead.

![Core architecture of HCA. Every m′ (≫ m) tokens are consolidated into a single KV entry, with dense attention over the result plus the same sliding-window branch.](/blog/deepseek-v4/hca.png)

Interleaving them gives the model both a fine-grained-but-selective view and a coarse-but-complete one at every few layers. Several smaller details make the scheme work: queries and compressed entries get an extra RMSNorm before core attention to keep logits from exploding; RoPE is applied to only the last 64 dimensions, and — because a compressed entry serves as both key *and* value, which would smuggle absolute positions into the output — RoPE at position −*i* is applied back to the attention output so the result carries relative positions; a 128-token sliding-window branch restores the local dependencies compression destroys; and learnable attention sinks let a head route its attention mass to nothing at all.

The efficiency accounting is blunt. Against BF16 GQA-8 with head dimension 128 — an ordinary configuration — V4's KV cache at 1M tokens is roughly **2%** the size. Mixed storage (BF16 for RoPE dimensions, FP8 for the rest) roughly halves it again, and the indexer's attention runs in FP4.

Two more changes round out the architecture. **Manifold-Constrained Hyper-Connections (*m*HC)** widen the residual stream by a factor of `n_hc` = 4, but constrain the residual mixing matrix to the Birkhoff polytope of doubly stochastic matrices via 20 Sinkhorn-Knopp iterations. That bounds its spectral norm by 1, making the residual transformation non-expansive — and since doubly stochastic matrices are closed under multiplication, the guarantee survives stacking, which is precisely where plain Hyper-Connections became numerically unstable. And **Muon** replaces AdamW for most parameters, with hybrid Newton-Schulz iterations: eight aggressive steps to drive singular values near 1, then two conservative steps to settle them exactly there.

## Loss spikes, and two things that fixed them

The most useful section for anyone training large MoE models is the one on instability, and it's admirably honest — the authors say plainly that they don't fully understand why their fixes work, and publish them anyway.

Loss spikes traced consistently back to outliers in the MoE layers, with routing appearing to amplify them. The first fix, **Anticipatory Routing**, decouples the backbone and the router in time: at step *t*, features are computed with current parameters `θ_t`, but routing indices come from historical parameters `θ_(t−Δt)`, computed and cached in advance during the earlier step. Careful pipeline overlap holds the overhead to ~20%, and an automatic detector switches the mode on only after a spike, then reverts — so the average cost is negligible. The second fix is simply clamping SwiGLU's linear component to [−10, 10] and capping the gate at 10.

Elsewhere the pre-training recipe is conservative: 32T+ tokens, 128K vocabulary, sequence length grown 4K → 16K → 64K → 1M, and sparse attention introduced only after a dense warm-up (1T tokens for Flash, longer for Pro) with a separate warm-up stage for the lightning indexer itself.

## Distillation instead of a final RL stage

Post-training makes one substitution that I think is the paper's quietest big idea: the mixed-RL stage that produced V3.2 is **replaced entirely by on-policy distillation**. Domain specialists — math, coding, agent, instruction following — are each trained with SFT then GRPO, and more than ten teachers are then merged into one student by having the student generate its own trajectories and minimizing reverse KL against whichever teacher is relevant.

Crucially, they do *not* use the cheap trick of reusing the RL framework with a per-token log-ratio as an advantage estimate. That estimator is high-variance and destabilizes training; V4 computes **full-vocabulary logit distillation** instead. Making that affordable at 100K+ vocabulary and trillion-parameter teachers takes real engineering: teacher weights live in distributed storage and stream in on demand, only last-layer hidden states are cached (logits are reconstructed on the fly through the prediction head), and training samples are ordered by teacher index so at most one teacher head sits in device memory at a time.

Reward modeling gets a similar consolidation. Rather than a separate scalar reward model, the actor itself acts as a **generative reward model** and is RL-optimized in that role, so judging and generating improve together from a small set of human annotations.

Three reasoning modes — Non-think, Think High, Think Max — are trained with different length penalties and context windows (8K / 128K / 384K at evaluation). Max mode additionally gets an instruction prepended to the system prompt telling it to use "absolute maximum" effort with no shortcuts permitted, which is a delightfully direct solution to a subtle problem.

![HLE and Terminal-Bench 2.0 performance by reasoning effort across DeepSeek-V4-Pro, V4-Flash, and V3.2.](/blog/deepseek-v4/effort.png)

The 1M context also changes context management. V3.2 discarded reasoning traces whenever a new user message arrived; V4 keeps the complete reasoning history across all rounds *including* user-turn boundaries in tool-calling scenarios, so an agent no longer reconstructs its problem-solving state from scratch every turn. General chat keeps the old discard-on-new-turn behavior, where persistent traces buy little.

## Where it lands

V4-Pro-Max sets a new bar for open models on world knowledge — 57.9% on SimpleQA-Verified, about 20 points clear of every other open model, and 84.4% on Chinese-SimpleQA — while still trailing Gemini-3.1-Pro (75.6%). On code and competition math it is at the frontier: LiveCodeBench 93.5% (best in the comparison), an internal Codeforces rating of 3206 that ranks 23rd among human competitors, Apex Shortlist 90.2%. On formal mathematics with Lean it reaches state of the art in the agentic setting and a proof-perfect 120/120 on Putnam-2025 under a compute-intensive hybrid informal-then-formal pipeline.

![DeepSeek-V4 series performance on the MRCR long-context retrieval task. Retrieval is stable within 128K and degrades gracefully beyond it, remaining strong at 1M tokens.](/blog/deepseek-v4/mrcr.png)

On long context, V4-Pro beats Gemini-3.1-Pro on both MRCR 1M (83.5 vs 76.3) and CorpusQA, but sits clearly behind Claude Opus 4.6 (92.9). Retrieval holds nearly flat to 128K and then degrades visibly — the honest reading is that a million tokens is now *usable*, not uniform.

Agentic performance is the soft spot the paper doesn't hide: comparable to K2.6 and GLM-5.1, but all three trail the closed models. The report's own framing of reasoning is worth quoting: V4 sits behind GPT-5.4 and Gemini-3.1-Pro by "approximately 3 to 6 months."

![Win-rate comparison against Opus-4.6-Max on Chinese white-collar tasks across analysis, generation, and editing.](/blog/deepseek-v4/winrate.png)

The real-world evaluations are where V4 looks strongest, and they're the ones benchmarks miss. On Chinese functional writing it beats Gemini-3.1-Pro 62.7% to 34.1%, and on creative writing quality 77.5% — the paper attributes Gemini's losses to its stylistic preferences overriding explicit user requirements. On 30 Chinese professional white-collar tasks judged blind by humans, V4-Pro-Max reaches a 63% non-loss rate against Opus-4.6-Max, strongest on task completion and content quality, weakest on formatting aesthetics and on condensing long inputs. And in a survey of 85 DeepSeek engineers who use it daily for agentic coding, 52% said it's ready to be their default coding model and 39% leaned yes.

## Why it matters

The conclusion contains an unusual admission: to minimize risk, the team kept many separately validated components, and the resulting architecture is "relatively complex." Reading it, that's hard to disagree with — CSA and HCA and sliding windows and attention sinks and *m*HC and hash-routed early layers and hybrid Newton-Schulz is a lot of moving parts, several of which exist to patch problems the others introduced. They commit to distilling it down in future work.

But complexity is what an efficiency-first bet looks like before anyone knows which pieces were load-bearing. The core claim — that a million-token context can cost a fraction of what it does today, natively, in an open model — is demonstrated, and that reframes what test-time scaling can afford. It's instructive to read alongside [Kimi K3](/blog/kimi-k3), which reaches for the same million-token target by replacing softmax attention with a fixed-state recurrence rather than compressing it. Two open labs, two opposite architectural bets, same year. We'll know within a generation of models which one was right — or, more likely, that both were.

---

**Links:** [arXiv:2606.19348](https://arxiv.org/abs/2606.19348) · [Model weights](https://huggingface.co/collections/deepseek-ai/deepseek-v4) · [MegaMoE kernel](https://github.com/deepseek-ai/DeepGEMM/pull/304)
