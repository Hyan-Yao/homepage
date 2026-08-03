For most of the past two years, open-weight models have been scaling along one axis at a time. Test-time compute got most of the attention — longer reasoning chains, more tool calls, agent swarms — while the pre-trained foundations underneath stayed parked around the 1T-parameter mark. **Kimi K3** is Moonshot's argument that you have to push both axes at once. It is a 2.8-trillion-parameter Mixture-of-Experts model with 104B activated parameters, native vision, and a 1-million-token context window, and the technical report is unusually candid about what it costs to build one.

过去两年里，开源权重模型基本上一次只沿着一个轴扩展。测试时计算拿走了大部分注意力——更长的推理链、更多的工具调用、智能体集群——而底下的预训练基座却一直停在 1T 参数量级附近。**Kimi K3** 是 Moonshot 给出的回应：这两个轴必须同时推进。它是一个 2.8 万亿参数的 MoE 模型，激活参数 1040 亿，原生支持视觉，上下文窗口达到 100 万 token；而这份技术报告对"建造这样一个模型的代价"的坦诚程度相当罕见。

![Kimi K3 main results across coding, general, and visual agent benchmarks. K3 leads ProgramBench (77.8), SWE-Marathon (42.0), BrowseComp (91.2), and AutomationBench (30.8), while trailing Claude Fable 5 and GPT-5.6 Sol on DeepSWE, GDPval-AA v2, and JobBench. ／ Kimi K3 在编程、通用与视觉智能体基准上的主要结果。K3 在 ProgramBench（77.8）、SWE-Marathon（42.0）、BrowseComp（91.2）和 AutomationBench（30.8）上领先，而在 DeepSWE、GDPval-AA v2 和 JobBench 上落后于 Claude Fable 5 与 GPT-5.6 Sol。](/blog/kimi-k3/results.png)

## Three directions of information flow ／ 信息流动的三个方向

The architecture is organized around a single framing: information has to flow along three axes — sequence length, network depth, and model width — and each gets its own mechanism.

整个架构围绕一个统一的表述展开：信息必须沿三个轴流动——序列长度、网络深度、模型宽度——而每个轴都有专门的机制。

![The Kimi K3 architecture. Each block contains three Kimi Delta Attention layers followed by one Gated MLA layer, each paired with a Stable LatentMoE feed-forward network. Attention Residuals use learned pseudo-queries to weight the embedding and all preceding block outputs. Bottom right: the native vision pathway through MoonViT-V2. ／ Kimi K3 架构。每个 block 包含三层 Kimi Delta Attention 加一层 Gated MLA，每层注意力都配一个 Stable LatentMoE 前馈网络。Attention Residuals 用可学习的伪查询对 embedding 和此前所有 block 的输出加权。右下：经由 MoonViT-V2 的原生视觉通路。](/blog/kimi-k3/architecture.png)

**Along the sequence**, K3 uses a 3:1 hybrid of **Kimi Delta Attention (KDA)** and **Gated MLA**. KDA is a linear-attention variant: instead of a KV cache that grows with context, it carries a fixed-size recurrent state and updates it with a delta rule gated channel-wise. Three KDA layers handle efficient long-range mixing, then one Gated MLA layer restores unrestricted global attention. Both use **no positional encoding at all** — position is carried implicitly by KDA's decay, which is why K3 extrapolates to 1M tokens with no RoPE rescaling or YaRN interpolation.

**沿序列方向**，K3 采用 **Kimi Delta Attention（KDA）** 与 **Gated MLA** 的 3:1 混合。KDA 是线性注意力的一个变体：它不维护随上下文增长的 KV cache，而是携带一个固定大小的循环状态，并用带逐通道门控的 delta 规则更新它。三层 KDA 负责高效的长程混合，随后一层 Gated MLA 恢复不受限的全局注意力。两者都**完全不使用位置编码**——位置信息由 KDA 的衰减隐式承载，这也是 K3 能够在不做任何 RoPE 缩放或 YaRN 插值的情况下直接外推到 100 万 token 的原因。

The engineering detail I found most satisfying is the decay reparameterization. Kimi Linear used an unbounded negative-Softplus map for the log-decay, which forces the chunkwise algorithm to rescale keys by a reciprocal that can overflow BF16 — so diagonal tiles needed slow explicit position-pair math. K3 replaces it with a scaled sigmoid bounded below at `g_min` = −5. Every retention factor now exceeds *e*⁻⁵, the cumulative log-decay over a 16-token tile stays in (−80, 0), and the rescaling factor stays inside BF16 range. A bound on a gate turns into every tile running as a dense Tensor Core matmul.

其中我觉得最妙的工程细节是衰减项的重参数化。Kimi Linear 用的是无界的负 Softplus 映射来产生对数衰减，这迫使分块算法用一个可能在 BF16 下溢出的倒数去重缩放 key——于是对角块只能走缓慢的显式逐位置对计算。K3 换成了一个下界为 `g_min` = −5 的缩放 sigmoid。此时每个保留因子都大于 *e*⁻⁵，16-token 小块上的累积对数衰减落在 (−80, 0) 区间内，重缩放因子始终留在 BF16 的动态范围里。给一个门加上界，换来的是所有小块都能走稠密的 Tensor Core 矩阵乘。

**Along depth**, **Attention Residuals (AttnRes)** replace the standard residual stream. The observation is that a plain residual connection compresses everything before layer *l* into one state — a bottleneck structurally identical to an RNN over time. So K3 does to depth what the Transformer did to time: each layer gets a learned pseudo-query and attends over the outputs of *all* preceding layers. The full version is *O*(*L*²*d*) arithmetic, affordable at *L* < 100, but the *O*(*Ld*) memory is not, so K3 partitions its 93 layers into 8 blocks and attends over block-level summaries instead.

**沿深度方向**，**Attention Residuals（AttnRes）** 取代了标准的残差流。出发点是：普通残差连接把第 *l* 层之前的一切压缩进单一状态——这个瓶颈在结构上与 RNN 在时间维上的瓶颈完全一致。于是 K3 对深度做了 Transformer 当年对时间做的事：每层配一个可学习的伪查询，对*所有*前序层的输出做注意力。完整版本的计算量是 *O*(*L*²*d*)，在 *L* < 100 时尚可承受，但 *O*(*Ld*) 的显存开销不行，因此 K3 把 93 层切成 8 个 block，改为在 block 级摘要上做注意力。

**Along width**, **Stable LatentMoE** pushes to 896 routed experts with 16 active per token — a sparsity of 56. The trick that makes this affordable is separating model width from routed-expert width: shared experts keep the full-width path, routed experts work in a compact latent space. At this sparsity, two things break, and K3 patches both. Activations explode through the near-four-matmul routed chain, so K3 adds an RMSNorm before the up-projection and swaps SwiGLU for **SiTU-GLU**, which soft-caps both multiplicative factors with scaled tanh (β₁ = 4 on the gate, β₂ = 25 on the up branch) — approximately linear near the origin, bounded far from it. And balancing ~10³ experts breaks the usual fixed-step bias update, so K3 introduces **Quantile Balancing**: set each expert's bias directly to the router-score quantile matching its target load, read from a histogram that a single all-reduce makes global.

**沿宽度方向**，**Stable LatentMoE** 把路由专家推到 896 个、每 token 激活 16 个——稀疏度 56。让这变得可负担的关键是把模型宽度与路由专家宽度解耦：共享专家保留全宽通路，路由专家在一个紧凑的隐空间里工作。在这个稀疏度下有两件事会崩，K3 都打了补丁。其一，激活值会在近乎四连乘的路由链路上爆炸，于是 K3 在上投影前加了一层 RMSNorm，并把 SwiGLU 换成 **SiTU-GLU**——用缩放 tanh 对两个乘性因子分别做软截断（门分支 β₁ = 4，上分支 β₂ = 25），原点附近近似线性、远处有界。其二，平衡约 10³ 个专家的负载超出了固定步长偏置更新的适用范围，于是 K3 提出 **Quantile Balancing**：直接把每个专家的偏置设为与其目标负载相匹配的路由分数分位点，而这个分位点从一个直方图里读出——一次 all-reduce 就能让它变成全局的。

![Fitted scaling-law curves for Kimi K2 and K3. The architectural, data, and training changes together deliver roughly a 2.5× gain in overall scaling efficiency. ／ Kimi K2 与 K3 的拟合 scaling law 曲线。架构、数据与训练上的改动合计带来约 2.5 倍的整体扩展效率提升。](/blog/kimi-k3/scaling-law.png)

Together these changes are worth about **2.5× in scaling efficiency** over K2 — the same loss for 2.5× less compute. K3 also reports that a clean scaling-law search consistently favors cosine decay over WSD, with the caveat that the two schedules have substantially different optimal peak learning rates and batch sizes, so any comparison sharing hyperparameters is rigged from the start.

这些改动加在一起，相对 K2 带来约 **2.5 倍的扩展效率**——同样的 loss，少用 2.5 倍算力。K3 还报告了一个干净的 scaling law 搜索结果：cosine decay 一致优于 WSD，但前提是两种调度的最优峰值学习率和 batch size 差异很大，所以任何共用超参数的对比从一开始就是不公平的。

## Nine experts, one model ／ 九个专家，一个模型

Post-training is where the 1M context stops being a spec sheet number. K3 runs RL over three domains — general tasks, general agents, coding agents — crossed with three reasoning-effort levels (low/high/max), giving nine expert models, then consolidates them into one via **multi-teacher on-policy distillation**: the student generates, and a per-token log-ratio against the matching teacher becomes a dense reward inside the same RL framework.

后训练才是 100 万上下文从规格表变成实际能力的地方。K3 在三个域上做 RL——通用任务、通用智能体、编程智能体——再与三档推理努力（low/high/max）相乘，得到九个专家模型，然后用**多教师在线策略蒸馏（MOPD）** 把它们合并成一个：由学生自己生成轨迹，与对应教师的逐 token 对数比值作为稠密奖励，直接接入同一套 RL 框架。

![Scores and average assistant steps across evaluations during RL. As RL FLOPs scale, tool-call steps grow consistently, alongside broad capability improvement. ／ RL 过程中各项评测的分数与平均助手步数。随着 RL FLOPs 扩大，工具调用步数持续增长，模型整体能力同步全面提升。](/blog/kimi-k3/rl-scaling.png)

Effort levels are trained, not prompted: each problem gets a token budget estimated from the cold-start model, and any trajectory exceeding τ × that budget gets its reward overridden to −1. Train τ large first for the max-effort expert, then anneal for high and low. The same budget trick reappears in the reward model to stop verbosity hacking — a candidate whose output exceeds σ × the reference length automatically loses its pairwise comparison.

推理努力档位是训出来的，不是提示出来的：每道题都有一个由冷启动模型估计的 token 预算，任何超过 τ × 预算的轨迹奖励被直接改写为 −1。先用较大的 τ 训出 max 档专家，再退火得到 high 和 low。同样的预算技巧也出现在奖励模型里，用来阻止"啰嗦"式的奖励攻击——输出长度超过 σ × 参考长度的候选自动判负。

The environments are the expensive part. K3 trains on a **unified white-box harness** that treats an agent scaffold as composable modules — tools, system prompts, context management, skills, memories, subagents — so it can instantiate Kimi Code, Claude Code, Codex, or OpenClaw from configuration and randomize across them during training, rather than overfitting to one tool schema. Personal-assistant tasks run in mock Gmail/Notion/Slack workspaces over multiple simulated days, where a single rollout can span thousands of tool calls and millions of context tokens. Kernel-optimization tasks reward both correctness and measured speed against an expert implementation, with a dedicated hacking-detection system that penalizes CUDA graph replay, input caching, and precision reduction.

真正昂贵的是环境。K3 在一个**统一白盒 harness** 上训练，把智能体脚手架当作可组合的模块——工具、system prompt、上下文管理、技能、记忆、子智能体——因此可以通过配置实例化 Kimi Code、Claude Code、Codex 或 OpenClaw，并在训练中随机切换，而不是过拟合到某一套工具 schema。个人助理任务运行在模拟的 Gmail/Notion/Slack 工作区里，跨越多个模拟日，单次 rollout 可以涉及数千次工具调用和数百万上下文 token。内核优化任务同时对正确性和实测速度给奖励（以专家实现为基准），并配有专门的作弊检测系统，惩罚 CUDA graph 重放、输入缓存和降精度等策略。

## The infrastructure is the paper ／ 这篇论文的重点其实是基础设施

If you read one section, make it §5. Million-token agentic RL is not a training-loop problem, it's a state-management problem, and the report is refreshingly concrete about it.

如果只读一节，就读第 5 节。百万 token 的智能体 RL 不是训练循环的问题，而是状态管理的问题，而这份报告在这方面具体得令人耳目一新。

**MoonEP** makes expert parallelism perfectly balanced through dynamically planned redundant experts, with a proven bound: a balanced plan always exists using at most *E*/*R* redundant experts per rank, so reserving that many slots guarantees training never stalls on a routing skew. Because every rank then receives exactly *S*×*K* tokens, communication buffers shrink from DeepEP's worst-case *S*×*K*×*R* to a fixed *S*×*K*, and static shapes eliminate per-layer host–device syncs.

**MoonEP** 通过动态规划的冗余专家实现了完美均衡的专家并行，并给出了一个可证明的界：每个 rank 至多 *E*/*R* 个冗余专家就一定存在可行的均衡方案，因此预留这么多槽位就能保证训练永远不会因路由倾斜而停顿。由于此时每个 rank 恰好收到 *S*×*K* 个 token，通信缓冲区从 DeepEP 最坏情况下的 *S*×*K*×*R* 缩小到固定的 *S*×*K*，而静态形状也消除了每层 MoE 的主机—设备同步。

**Sandboxes** turn out to dominate the RL wall clock: a sandbox spends **up to 98% of its lifetime** waiting on model inference. So K3 built AgentENV on Firecracker with incremental checkpointing — 133 ms to checkpoint, 49 ms to resume — and a paused sandbox consumes no memory or CPU. With copy-on-write memory and a custom image format they reach a **6.5× memory overcommit ratio**. Total across K3's training and evaluation: **51,219,741 sandboxes across 1,505,678 images**.

**沙箱**出人意料地主导了 RL 的墙钟时间：一个沙箱生命周期中**高达 98%** 的时间是在等待模型推理。于是 K3 基于 Firecracker 构建了 AgentENV，支持增量检查点——检查点 133 毫秒、恢复 49 毫秒——而暂停中的沙箱不占用任何内存和 CPU。配合写时复制内存和自定义镜像格式，内存超分比达到 **6.5×**。K3 训练与评测期间的总量是：**51,219,741 个沙箱，跨越 1,505,678 个镜像**。

**Serving** hits a problem specific to hybrid attention. KDA's recurrent state can only be snapshotted at sparse boundaries, so naive block-hash prefix caching would force a shared block size of 1024–6144 tokens. K3 decouples the granularities: prefix hashing runs on 512-token hash blocks inside larger 6144-token physical blocks, with KDA checkpoints saved at a sparse subset of endpoints, so any shared prefix is reusable at any 512-token boundary regardless of how requests were chunked or interleaved. It matters because at 1M context a typical coding request carries a 400K-token prefix and only a 4K-token increment — a cache hit is orders of magnitude cheaper than a miss.

**推理服务**遇到了混合注意力特有的难题。KDA 的循环状态只能在稀疏的边界上做快照，因此朴素的块哈希前缀缓存会被迫采用 1024–6144 token 的共享块大小。K3 把两种粒度解耦：前缀哈希跑在更大的 6144-token 物理块内部的 512-token 哈希块上，KDA 检查点只在其中一个稀疏子集的端点保存，于是任何共享前缀都能在任意 512-token 边界上复用，与请求如何分块或如何交错调度无关。这件事之所以重要：在 100 万上下文下，一个典型的编程请求携带 40 万 token 的前缀，而增量只有 4K token——缓存命中比未命中便宜几个数量级。

## Where it lands ／ 它最终落在哪里

K3 trails Claude Fable 5 and GPT-5.6 Sol overall and beats everything else in the suite. The specifics are more interesting than the headline: best-in-suite on ProgramBench (77.8%), BrowseComp (91.2%), MCPMark-Verified (94.5%), and Harvey Lab-AA (94.6%); second on FrontierSWE (81.2% vs Fable 5's 86.6%); 42.0% on the GPU-kernel-heavy SWE-Marathon, seven points clear of Fable 5. Research-level reasoning is the honest weak spot — 23.4% on CritPt, behind three proprietary models.

K3 整体上落后于 Claude Fable 5 与 GPT-5.6 Sol，领先于评测套件里的其他所有模型。具体项比结论更有意思：在 ProgramBench（77.8%）、BrowseComp（91.2%）、MCPMark-Verified（94.5%）和 Harvey Lab-AA（94.6%）上全场最佳；FrontierSWE 第二（81.2% vs Fable 5 的 86.6%）；在偏重 GPU 内核的 SWE-Marathon 上拿到 42.0%，领先 Fable 5 七个百分点。研究级推理是诚实承认的弱项——CritPt 只有 23.4%，落后于三个闭源模型。

![Score versus per-task inference cost on Kimi Code Bench 2.0. K3 sits on or near the cost-efficiency frontier. ／ Kimi Code Bench 2.0 上的分数与单任务推理成本。K3 位于或接近成本效率前沿。](/blog/kimi-k3/cost-kcb.png)

The cost story is the one that will matter to most readers. On Kimi Code Bench 2.0, K3 is 4 points behind Fable 5 at **38% of the cost**, and its high-effort mode already matches Claude Opus 4.8's max-effort score at roughly a third of the price. On BrowseComp it takes the top score at $2.03 per task — half of GPT-5.6 Sol's cost and an order of magnitude below the Claude models at max effort. It's also the first open model to top LMArena's WebDev Arena (1,678 Elo).

对大多数读者真正重要的是成本这条线。在 Kimi Code Bench 2.0 上，K3 比 Fable 5 低 4 分，但成本只有其 **38%**；而它的 high 档已经能以约三分之一的价格匹平 Claude Opus 4.8 的 max 档成绩。在 BrowseComp 上它拿到最高分，单任务 2.03 美元——是 GPT-5.6 Sol 的一半，比 max 档的 Claude 系列便宜一个数量级。它也是第一个登顶 LMArena WebDev Arena 的开源模型（1,678 Elo）。

The case studies are worth a look for their own sake. Given 48 hours and Kimi Code, K3 autonomously designed, optimized, and verified an inference-chip prototype with open-source EDA tools — 1.46M standard cells, timing closed at 100 MHz in a 4 mm² budget, over 8,700 tokens/s in RTL simulation. On its own kernels, it cut AttnRes latency from 283.6 ms to 114.4 ms and KDA runtime by 73.6%. An early K3 checkpoint was already doing most of Moonshot's kernel-optimization work while K3 was still being trained, which is either a nice result or a slightly vertiginous one depending on your mood.

案例研究本身也值得一读。给定 48 小时和 Kimi Code，K3 自主完成了一款推理芯片原型的设计、优化与验证，全程使用开源 EDA 工具——146 万个标准单元，在 4 mm² 面积预算下于 100 MHz 收敛时序，RTL 仿真解码吞吐超过 8,700 tokens/s。在自家内核上，它把 AttnRes 延迟从 283.6 毫秒降到 114.4 毫秒，KDA 运行时间削减 73.6%。K3 还在训练期间，一个早期检查点就已经承担了 Moonshot 大部分的内核优化工作——这个事实是令人欣喜还是略感眩晕，取决于你当天的心情。

The report also runs a cybersecurity evaluation that most labs would omit, finding 16 previously unknown vulnerabilities across six projects (roughly 70% of human-reviewed findings confirmed genuine) and solving 14 of 36 end-to-end exploit tasks, while noting the model remains well short of human experts on hardened kernel targets.

报告还做了一项多数实验室会略过的网络安全评测：在六个项目中发现 16 个此前未知的漏洞（人工复核的发现中约 70% 被确认为真实），并在 36 个端到端漏洞利用任务中解出 14 个，同时明确指出模型在加固过的内核目标上仍远不及人类专家。

## Why it matters ／ 它为什么重要

The framing that stuck with me is that K3 is less a modeling paper than a systems paper wearing a modeling paper's clothes. The architectural ideas are elegant — bounding a gate to unlock Tensor Cores, attending over depth, quantile-matching expert loads — but the reason a 2.8T model with 1M context can be trained with RL on a few hundred GPUs is MoonEP, resumable microVMs, and a prefix cache that understands recurrent state. It's a useful counterpoint to [DeepSeek-V4](/blog/deepseek-v4), which chases the same million-token goal from the opposite direction: compress and sparsify softmax attention rather than replace it. Both shipped open weights in 2026; the field gets to run the experiment.

让我印象最深的一点是：K3 与其说是一篇建模论文，不如说是一篇穿着建模论文外衣的系统论文。架构上的想法确实优雅——给一个门加下界从而解锁 Tensor Core、在深度上做注意力、用分位点匹配专家负载——但一个 2.8T、100 万上下文的模型之所以能用几百张 GPU 跑 RL，靠的是 MoonEP、可恢复的 microVM，以及一个理解循环状态的前缀缓存。它也构成了对 [DeepSeek-V4](/blog/deepseek-v4) 的一个有趣对照：后者奔向同一个百万 token 的目标，路线却完全相反——压缩并稀疏化 softmax 注意力，而不是替换它。两者都在 2026 年开放了权重；这场实验，整个领域都可以亲自跑一遍。

---

**Links:** [arXiv:2607.24653](https://arxiv.org/abs/2607.24653) · [Model weights](https://huggingface.co/moonshotai/Kimi-K3) · [MoonEP](https://github.com/MoonshotAI/MoonEP) · [AgentENV](https://github.com/kvcache-ai/AgentENV) · [MiniTriton](https://github.com/MoonshotAI/minitriton)

**链接：** [arXiv:2607.24653](https://arxiv.org/abs/2607.24653) · [模型权重](https://huggingface.co/moonshotai/Kimi-K3) · [MoonEP](https://github.com/MoonshotAI/MoonEP) · [AgentENV](https://github.com/kvcache-ai/AgentENV) · [MiniTriton](https://github.com/MoonshotAI/minitriton)
