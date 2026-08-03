For most of the past two years, open-weight models have been scaling along one axis at a time. Test-time compute got most of the attention — longer reasoning chains, more tool calls, agent swarms — while the pre-trained foundations underneath stayed parked around the 1T-parameter mark. **Kimi K3** is Moonshot's argument that you have to push both axes at once. It is a 2.8-trillion-parameter Mixture-of-Experts model with 104B activated parameters, native vision, and a 1-million-token context window, and the technical report is unusually candid about what it costs to build one.

![Kimi K3 main results across coding, general, and visual agent benchmarks. K3 leads ProgramBench (77.8), SWE-Marathon (42.0), BrowseComp (91.2), and AutomationBench (30.8), while trailing Claude Fable 5 and GPT-5.6 Sol on DeepSWE, GDPval-AA v2, and JobBench.](/blog/kimi-k3/results.png)

## Three directions of information flow

The architecture is organized around a single framing: information has to flow along three axes — sequence length, network depth, and model width — and each gets its own mechanism.

![The Kimi K3 architecture. Each block contains three Kimi Delta Attention layers followed by one Gated MLA layer, each paired with a Stable LatentMoE feed-forward network. Attention Residuals use learned pseudo-queries to weight the embedding and all preceding block outputs. Bottom right: the native vision pathway through MoonViT-V2.](/blog/kimi-k3/architecture.png)

**Along the sequence**, K3 uses a 3:1 hybrid of **Kimi Delta Attention (KDA)** and **Gated MLA**. KDA is a linear-attention variant: instead of a KV cache that grows with context, it carries a fixed-size recurrent state and updates it with a delta rule gated channel-wise. Three KDA layers handle efficient long-range mixing, then one Gated MLA layer restores unrestricted global attention. Both use **no positional encoding at all** — position is carried implicitly by KDA's decay, which is why K3 extrapolates to 1M tokens with no RoPE rescaling or YaRN interpolation.

The engineering detail I found most satisfying is the decay reparameterization. Kimi Linear used an unbounded negative-Softplus map for the log-decay, which forces the chunkwise algorithm to rescale keys by a reciprocal that can overflow BF16 — so diagonal tiles needed slow explicit position-pair math. K3 replaces it with a scaled sigmoid bounded below at `g_min` = −5. Every retention factor now exceeds *e*⁻⁵, the cumulative log-decay over a 16-token tile stays in (−80, 0), and the rescaling factor stays inside BF16 range. A bound on a gate turns into every tile running as a dense Tensor Core matmul.

**Along depth**, **Attention Residuals (AttnRes)** replace the standard residual stream. The observation is that a plain residual connection compresses everything before layer *l* into one state — a bottleneck structurally identical to an RNN over time. So K3 does to depth what the Transformer did to time: each layer gets a learned pseudo-query and attends over the outputs of *all* preceding layers. The full version is *O*(*L*²*d*) arithmetic, affordable at *L* < 100, but the *O*(*Ld*) memory is not, so K3 partitions its 93 layers into 8 blocks and attends over block-level summaries instead.

**Along width**, **Stable LatentMoE** pushes to 896 routed experts with 16 active per token — a sparsity of 56. The trick that makes this affordable is separating model width from routed-expert width: shared experts keep the full-width path, routed experts work in a compact latent space. At this sparsity, two things break, and K3 patches both. Activations explode through the near-four-matmul routed chain, so K3 adds an RMSNorm before the up-projection and swaps SwiGLU for **SiTU-GLU**, which soft-caps both multiplicative factors with scaled tanh (β₁ = 4 on the gate, β₂ = 25 on the up branch) — approximately linear near the origin, bounded far from it. And balancing ~10³ experts breaks the usual fixed-step bias update, so K3 introduces **Quantile Balancing**: set each expert's bias directly to the router-score quantile matching its target load, read from a histogram that a single all-reduce makes global.

![Fitted scaling-law curves for Kimi K2 and K3. The architectural, data, and training changes together deliver roughly a 2.5× gain in overall scaling efficiency.](/blog/kimi-k3/scaling-law.png)

Together these changes are worth about **2.5× in scaling efficiency** over K2 — the same loss for 2.5× less compute. K3 also reports that a clean scaling-law search consistently favors cosine decay over WSD, with the caveat that the two schedules have substantially different optimal peak learning rates and batch sizes, so any comparison sharing hyperparameters is rigged from the start.

## Nine experts, one model

Post-training is where the 1M context stops being a spec sheet number. K3 runs RL over three domains — general tasks, general agents, coding agents — crossed with three reasoning-effort levels (low/high/max), giving nine expert models, then consolidates them into one via **multi-teacher on-policy distillation**: the student generates, and a per-token log-ratio against the matching teacher becomes a dense reward inside the same RL framework.

![Scores and average assistant steps across evaluations during RL. As RL FLOPs scale, tool-call steps grow consistently, alongside broad capability improvement.](/blog/kimi-k3/rl-scaling.png)

Effort levels are trained, not prompted: each problem gets a token budget estimated from the cold-start model, and any trajectory exceeding τ × that budget gets its reward overridden to −1. Train τ large first for the max-effort expert, then anneal for high and low. The same budget trick reappears in the reward model to stop verbosity hacking — a candidate whose output exceeds σ × the reference length automatically loses its pairwise comparison.

The environments are the expensive part. K3 trains on a **unified white-box harness** that treats an agent scaffold as composable modules — tools, system prompts, context management, skills, memories, subagents — so it can instantiate Kimi Code, Claude Code, Codex, or OpenClaw from configuration and randomize across them during training, rather than overfitting to one tool schema. Personal-assistant tasks run in mock Gmail/Notion/Slack workspaces over multiple simulated days, where a single rollout can span thousands of tool calls and millions of context tokens. Kernel-optimization tasks reward both correctness and measured speed against an expert implementation, with a dedicated hacking-detection system that penalizes CUDA graph replay, input caching, and precision reduction.

## The infrastructure is the paper

If you read one section, make it §5. Million-token agentic RL is not a training-loop problem, it's a state-management problem, and the report is refreshingly concrete about it.

**MoonEP** makes expert parallelism perfectly balanced through dynamically planned redundant experts, with a proven bound: a balanced plan always exists using at most *E*/*R* redundant experts per rank, so reserving that many slots guarantees training never stalls on a routing skew. Because every rank then receives exactly *S*×*K* tokens, communication buffers shrink from DeepEP's worst-case *S*×*K*×*R* to a fixed *S*×*K*, and static shapes eliminate per-layer host–device syncs.

**Sandboxes** turn out to dominate the RL wall clock: a sandbox spends **up to 98% of its lifetime** waiting on model inference. So K3 built AgentENV on Firecracker with incremental checkpointing — 133 ms to checkpoint, 49 ms to resume — and a paused sandbox consumes no memory or CPU. With copy-on-write memory and a custom image format they reach a **6.5× memory overcommit ratio**. Total across K3's training and evaluation: **51,219,741 sandboxes across 1,505,678 images**.

**Serving** hits a problem specific to hybrid attention. KDA's recurrent state can only be snapshotted at sparse boundaries, so naive block-hash prefix caching would force a shared block size of 1024–6144 tokens. K3 decouples the granularities: prefix hashing runs on 512-token hash blocks inside larger 6144-token physical blocks, with KDA checkpoints saved at a sparse subset of endpoints, so any shared prefix is reusable at any 512-token boundary regardless of how requests were chunked or interleaved. It matters because at 1M context a typical coding request carries a 400K-token prefix and only a 4K-token increment — a cache hit is orders of magnitude cheaper than a miss.

## Where it lands

K3 trails Claude Fable 5 and GPT-5.6 Sol overall and beats everything else in the suite. The specifics are more interesting than the headline: best-in-suite on ProgramBench (77.8%), BrowseComp (91.2%), MCPMark-Verified (94.5%), and Harvey Lab-AA (94.6%); second on FrontierSWE (81.2% vs Fable 5's 86.6%); 42.0% on the GPU-kernel-heavy SWE-Marathon, seven points clear of Fable 5. Research-level reasoning is the honest weak spot — 23.4% on CritPt, behind three proprietary models.

![Score versus per-task inference cost on Kimi Code Bench 2.0. K3 sits on or near the cost-efficiency frontier.](/blog/kimi-k3/cost-kcb.png)

The cost story is the one that will matter to most readers. On Kimi Code Bench 2.0, K3 is 4 points behind Fable 5 at **38% of the cost**, and its high-effort mode already matches Claude Opus 4.8's max-effort score at roughly a third of the price. On BrowseComp it takes the top score at $2.03 per task — half of GPT-5.6 Sol's cost and an order of magnitude below the Claude models at max effort. It's also the first open model to top LMArena's WebDev Arena (1,678 Elo).

The case studies are worth a look for their own sake. Given 48 hours and Kimi Code, K3 autonomously designed, optimized, and verified an inference-chip prototype with open-source EDA tools — 1.46M standard cells, timing closed at 100 MHz in a 4 mm² budget, over 8,700 tokens/s in RTL simulation. On its own kernels, it cut AttnRes latency from 283.6 ms to 114.4 ms and KDA runtime by 73.6%. An early K3 checkpoint was already doing most of Moonshot's kernel-optimization work while K3 was still being trained, which is either a nice result or a slightly vertiginous one depending on your mood.

The report also runs a cybersecurity evaluation that most labs would omit, finding 16 previously unknown vulnerabilities across six projects (roughly 70% of human-reviewed findings confirmed genuine) and solving 14 of 36 end-to-end exploit tasks, while noting the model remains well short of human experts on hardened kernel targets.

## Why it matters

The framing that stuck with me is that K3 is less a modeling paper than a systems paper wearing a modeling paper's clothes. The architectural ideas are elegant — bounding a gate to unlock Tensor Cores, attending over depth, quantile-matching expert loads — but the reason a 2.8T model with 1M context can be trained with RL on a few hundred GPUs is MoonEP, resumable microVMs, and a prefix cache that understands recurrent state. It's a useful counterpoint to [DeepSeek-V4](/blog/deepseek-v4), which chases the same million-token goal from the opposite direction: compress and sparsify softmax attention rather than replace it. Both shipped open weights in 2026; the field gets to run the experiment.

---

**Links:** [arXiv:2607.24653](https://arxiv.org/abs/2607.24653) · [Model weights](https://huggingface.co/moonshotai/Kimi-K3) · [MoonEP](https://github.com/MoonshotAI/MoonEP) · [AgentENV](https://github.com/kvcache-ai/AgentENV) · [MiniTriton](https://github.com/MoonshotAI/minitriton)
