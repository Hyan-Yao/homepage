Every reasoning model is quietly paying a tax. Test-time scaling — longer chains of thought, more tool calls, hundred-step agent trajectories — runs straight into attention's quadratic cost, and the bill comes due exactly where the new capabilities live: in very long contexts. **DeepSeek-V4** is a paper about paying less. Its two models, **V4-Pro** (1.6T total, 49B activated) and **V4-Flash** (284B total, 13B activated), both natively support one-million-token contexts, and at that length V4-Pro needs **27% of the single-token inference FLOPs and 10% of the KV cache** of DeepSeek-V3.2 — while activating *more* parameters per token.

每一个推理模型都在悄悄交一笔税。测试时扩展——更长的思维链、更多的工具调用、上百步的智能体轨迹——会直接撞上注意力的二次方开销，而账单恰好落在新能力所在的地方：超长上下文。**DeepSeek-V4** 是一篇关于"少交点税"的论文。它的两个模型——**V4-Pro**（总参数 1.6T，激活 49B）和 **V4-Flash**（总参数 284B，激活 13B）——都原生支持百万 token 上下文；而在这个长度下，V4-Pro 只需要 DeepSeek-V3.2 的 **27% 单 token 推理 FLOPs 和 10% 的 KV cache**——尽管它每个 token 激活的参数*更多*。

![Left: benchmark performance of DeepSeek-V4-Pro-Max against Claude Opus 4.6, GPT-5.4, and Gemini-3.1-Pro. Right: single-token inference FLOPs and accumulated KV cache versus sequence length, showing V4-Pro at 3.7× lower FLOPs and 9.5× smaller KV cache than V3.2 at 1M tokens, and V4-Flash at 9.8× and 13.7× respectively. ／ 左：DeepSeek-V4-Pro-Max 与 Claude Opus 4.6、GPT-5.4、Gemini-3.1-Pro 的基准对比。右：单 token 推理 FLOPs 与累积 KV cache 随序列长度的变化——在 100 万 token 时，V4-Pro 的 FLOPs 低 3.7 倍、KV cache 小 9.5 倍，V4-Flash 分别为 9.8 倍与 13.7 倍。](/blog/deepseek-v4/performance.png)

## Two ways to shrink attention, interleaved ／ 两种压缩注意力的方式，交替堆叠

The architectural centerpiece is a hybrid of two new attention mechanisms that make opposite trade-offs and are stacked in alternating layers.

架构的核心是两种新注意力机制的混合——它们做出了相反的取舍，并以交替的方式逐层堆叠。

![Overall architecture of the DeepSeek-V4 series: interleaved CSA and HCA attention layers, DeepSeekMoE feed-forward layers, and Manifold-Constrained Hyper-Connections replacing conventional residuals. ／ DeepSeek-V4 系列的整体架构：交替的 CSA 与 HCA 注意力层、DeepSeekMoE 前馈层，以及取代传统残差的 Manifold-Constrained Hyper-Connections。](/blog/deepseek-v4/architecture.png)

**Compressed Sparse Attention (CSA)** does compression *and* sparsity. It first pools every *m* tokens' KV entries into one, using learned per-token compression weights and positional biases — with a deliberate overlap, so each compressed entry draws from 2*m* source entries while the sequence still shrinks by *m*×. Then it applies DeepSeek Sparse Attention on top: a lightning indexer scores each query against the compressed entries and a top-*k* selector keeps only the best ones for core attention. In V4-Pro, *m* = 4 and top-*k* = 1024.

**压缩稀疏注意力（CSA）** 同时做压缩*和*稀疏。它先用可学习的逐 token 压缩权重和位置偏置，把每 *m* 个 token 的 KV 条目池化成一个——并且刻意保留重叠，使每个压缩条目实际来自 2*m* 个源条目，而序列长度仍然缩短 *m* 倍。然后在其上应用 DeepSeek Sparse Attention：闪电索引器为每个 query 对压缩条目打分，top-*k* 选择器只保留最相关的若干条送入核心注意力。在 V4-Pro 中，*m* = 4，top-*k* = 1024。

![Core architecture of CSA. KV entries are compressed to 1/m, then DeepSeek Sparse Attention selects the top-k compressed entries; a small sliding-window branch is added back to preserve local fine-grained dependencies. ／ CSA 的核心架构。KV 条目被压缩到 1/m，再由 DeepSeek Sparse Attention 选出 top-k 压缩条目；另外加回一个小的滑动窗口分支以保留局部细粒度依赖。](/blog/deepseek-v4/csa.png)

**Heavily Compressed Attention (HCA)** goes the other way: crush the KV cache far harder — *m*′ = 128, thirty-two times CSA's ratio — but keep attention fully dense over what remains. No indexer, no top-*k*, no selection overhead.

**重压缩注意力（HCA）** 则走向另一端：把 KV cache 压得狠得多——*m*′ = 128，是 CSA 压缩率的 32 倍——但对剩下的部分保持完全稠密的注意力。没有索引器，没有 top-*k*，没有选择开销。

![Core architecture of HCA. Every m′ (≫ m) tokens are consolidated into a single KV entry, with dense attention over the result plus the same sliding-window branch. ／ HCA 的核心架构。每 m′（≫ m）个 token 被合并为单个 KV 条目，在其上做稠密注意力，并配同样的滑动窗口分支。](/blog/deepseek-v4/hca.png)

Interleaving them gives the model both a fine-grained-but-selective view and a coarse-but-complete one at every few layers. Several smaller details make the scheme work: queries and compressed entries get an extra RMSNorm before core attention to keep logits from exploding; RoPE is applied to only the last 64 dimensions, and — because a compressed entry serves as both key *and* value, which would smuggle absolute positions into the output — RoPE at position −*i* is applied back to the attention output so the result carries relative positions; a 128-token sliding-window branch restores the local dependencies compression destroys; and learnable attention sinks let a head route its attention mass to nothing at all.

交替堆叠让模型每隔几层就同时拥有一个细粒度但有选择的视角，和一个粗粒度但完整的视角。若干细节让这套方案真正跑得通：query 和压缩条目在进入核心注意力前额外做一次 RMSNorm，防止 logits 爆炸；RoPE 只施加在最后 64 维上——而由于压缩条目同时充当 key *和* value，会把绝对位置偷偷带进输出，因此还要对注意力输出施加位置为 −*i* 的 RoPE，让结果重新携带相对位置；一个 128 token 的滑动窗口分支补回被压缩摧毁的局部依赖；可学习的 attention sink 则允许某个头把注意力质量分配给"什么都不看"。

The efficiency accounting is blunt. Against BF16 GQA-8 with head dimension 128 — an ordinary configuration — V4's KV cache at 1M tokens is roughly **2%** the size. Mixed storage (BF16 for RoPE dimensions, FP8 for the rest) roughly halves it again, and the indexer's attention runs in FP4.

效率账算得很直白。以头维度 128 的 BF16 GQA-8 这种常规配置为基准，V4 在 100 万 token 时的 KV cache 大约只有其 **2%**。混合存储（RoPE 维度用 BF16、其余用 FP8）又几乎再减半，而索引器的注意力计算跑在 FP4 上。

Two more changes round out the architecture. **Manifold-Constrained Hyper-Connections (*m*HC)** widen the residual stream by a factor of `n_hc` = 4, but constrain the residual mixing matrix to the Birkhoff polytope of doubly stochastic matrices via 20 Sinkhorn-Knopp iterations. That bounds its spectral norm by 1, making the residual transformation non-expansive — and since doubly stochastic matrices are closed under multiplication, the guarantee survives stacking, which is precisely where plain Hyper-Connections became numerically unstable. And **Muon** replaces AdamW for most parameters, with hybrid Newton-Schulz iterations: eight aggressive steps to drive singular values near 1, then two conservative steps to settle them exactly there.

还有两处改动补全了架构。**流形约束超连接（*m*HC）** 把残差流的宽度扩展 `n_hc` = 4 倍，但通过 20 次 Sinkhorn-Knopp 迭代把残差混合矩阵约束在双随机矩阵构成的 Birkhoff 多面体上。这使其谱范数不超过 1，残差变换成为非扩张映射——而由于双随机矩阵在乘法下封闭，这个保证在多层堆叠后依然成立，而这正是朴素 Hyper-Connections 出现数值不稳定的地方。另外，大多数参数改用 **Muon** 优化器，配合混合 Newton-Schulz 迭代：前 8 步用激进系数把奇异值迅速逼近 1，最后 2 步换成保守系数把它们精确稳定在 1。

## Loss spikes, and two things that fixed them ／ Loss 尖峰，以及两个奏效的办法

The most useful section for anyone training large MoE models is the one on instability, and it's admirably honest — the authors say plainly that they don't fully understand why their fixes work, and publish them anyway.

对任何训练大型 MoE 的人来说，最有用的一节是关于训练不稳定的部分，而且写得相当坦诚——作者直言他们并不完全理解这些办法为什么有效，但还是选择公开出来。

Loss spikes traced consistently back to outliers in the MoE layers, with routing appearing to amplify them. The first fix, **Anticipatory Routing**, decouples the backbone and the router in time: at step *t*, features are computed with current parameters `θ_t`, but routing indices come from historical parameters `θ_(t−Δt)`, computed and cached in advance during the earlier step. Careful pipeline overlap holds the overhead to ~20%, and an automatic detector switches the mode on only after a spike, then reverts — so the average cost is negligible. The second fix is simply clamping SwiGLU's linear component to [−10, 10] and capping the gate at 10.

Loss 尖峰始终能追溯到 MoE 层中的异常值，而路由机制似乎在放大它们。第一个办法 **Anticipatory Routing（前瞻路由）** 在时间上解耦主干网络与路由网络：在第 *t* 步，特征用当前参数 `θ_t` 计算，但路由索引来自历史参数 `θ_(t−Δt)`——在更早的那一步就提前算好并缓存。精心设计的流水线重叠把额外开销控制在约 20%，而一个自动检测机制只在出现尖峰后才切换到该模式、运行一段时间后再切回——因此平均代价可以忽略。第二个办法更简单：把 SwiGLU 的线性分量截断到 [−10, 10]，并把门分支上界设为 10。

Elsewhere the pre-training recipe is conservative: 32T+ tokens, 128K vocabulary, sequence length grown 4K → 16K → 64K → 1M, and sparse attention introduced only after a dense warm-up (1T tokens for Flash, longer for Pro) with a separate warm-up stage for the lightning indexer itself.

其余的预训练配方相当保守：32T+ token，12.8 万词表，序列长度按 4K → 16K → 64K → 1M 逐步扩展，稀疏注意力只在稠密预热之后引入（Flash 是前 1T token，Pro 更长），而闪电索引器本身还有单独的预热阶段。

## Distillation instead of a final RL stage ／ 用蒸馏取代最后的 RL 阶段

Post-training makes one substitution that I think is the paper's quietest big idea: the mixed-RL stage that produced V3.2 is **replaced entirely by on-policy distillation**. Domain specialists — math, coding, agent, instruction following — are each trained with SFT then GRPO, and more than ten teachers are then merged into one student by having the student generate its own trajectories and minimizing reverse KL against whichever teacher is relevant.

后训练做了一处替换，我认为是这篇论文最低调的大想法：产生 V3.2 的混合 RL 阶段被**完全替换为在线策略蒸馏**。各领域专家——数学、编程、智能体、指令遵循——分别经过 SFT 再加 GRPO 训练，随后十几个教师被合并进一个学生：由学生自己生成轨迹，并针对当前任务对应的教师最小化反向 KL。

Crucially, they do *not* use the cheap trick of reusing the RL framework with a per-token log-ratio as an advantage estimate. That estimator is high-variance and destabilizes training; V4 computes **full-vocabulary logit distillation** instead. Making that affordable at 100K+ vocabulary and trillion-parameter teachers takes real engineering: teacher weights live in distributed storage and stream in on demand, only last-layer hidden states are cached (logits are reconstructed on the fly through the prediction head), and training samples are ordered by teacher index so at most one teacher head sits in device memory at a time.

关键在于，他们**没有**采用那个便宜的技巧——复用 RL 框架、把逐 token 对数比值当作优势估计。那个估计器方差很高，会破坏训练稳定性；V4 改为计算**全词表 logit 蒸馏**。要在 10 万+ 词表和万亿参数教师的规模下负担得起，需要真刀真枪的工程：教师权重存放在分布式存储中按需流式加载，只缓存最后一层隐状态（logits 在使用时经预测头即时重建），训练样本按教师索引排序，使得任一时刻设备内存中至多驻留一个教师的预测头。

Reward modeling gets a similar consolidation. Rather than a separate scalar reward model, the actor itself acts as a **generative reward model** and is RL-optimized in that role, so judging and generating improve together from a small set of human annotations.

奖励建模也做了类似的合并。V4 不再使用独立的标量奖励模型，而是让 actor 本身充当**生成式奖励模型**并在该角色上做 RL 优化，于是"评判"与"生成"能力一同提升，且只需要少量多样化的人工标注。

Three reasoning modes — Non-think, Think High, Think Max — are trained with different length penalties and context windows (8K / 128K / 384K at evaluation). Max mode additionally gets an instruction prepended to the system prompt telling it to use "absolute maximum" effort with no shortcuts permitted, which is a delightfully direct solution to a subtle problem.

三档推理模式——Non-think、Think High、Think Max——用不同的长度惩罚和上下文窗口训练（评测时分别为 8K / 128K / 384K）。Max 模式还会在 system prompt 开头插入一条指令，要求模型使用"绝对最大"的推理努力、不允许走任何捷径——对一个微妙问题的一个爽快直接的解法。

![HLE and Terminal-Bench 2.0 performance by reasoning effort across DeepSeek-V4-Pro, V4-Flash, and V3.2. ／ DeepSeek-V4-Pro、V4-Flash 与 V3.2 在不同推理努力下的 HLE 与 Terminal-Bench 2.0 表现。](/blog/deepseek-v4/effort.png)

The 1M context also changes context management. V3.2 discarded reasoning traces whenever a new user message arrived; V4 keeps the complete reasoning history across all rounds *including* user-turn boundaries in tool-calling scenarios, so an agent no longer reconstructs its problem-solving state from scratch every turn. General chat keeps the old discard-on-new-turn behavior, where persistent traces buy little.

百万上下文也改变了上下文管理策略。V3.2 每当新的用户消息到来就丢弃推理轨迹；在工具调用场景下，V4 则保留跨所有轮次（*包括*用户轮边界）的完整推理历史，智能体不必每一轮都从零重建自己的问题求解状态。普通对话场景保留原来的"新轮即丢弃"行为——在那里保留轨迹带来的收益有限。

## Where it lands ／ 它最终落在哪里

V4-Pro-Max sets a new bar for open models on world knowledge — 57.9% on SimpleQA-Verified, about 20 points clear of every other open model, and 84.4% on Chinese-SimpleQA — while still trailing Gemini-3.1-Pro (75.6%). On code and competition math it is at the frontier: LiveCodeBench 93.5% (best in the comparison), an internal Codeforces rating of 3206 that ranks 23rd among human competitors, Apex Shortlist 90.2%. On formal mathematics with Lean it reaches state of the art in the agentic setting and a proof-perfect 120/120 on Putnam-2025 under a compute-intensive hybrid informal-then-formal pipeline.

V4-Pro-Max 在世界知识上为开源模型立了新标杆——SimpleQA-Verified 57.9%，领先其他所有开源模型约 20 个百分点，Chinese-SimpleQA 84.4%——但仍落后于 Gemini-3.1-Pro（75.6%）。在代码与竞赛数学上它已进入前沿：LiveCodeBench 93.5%（对比中最佳），内部 Codeforces 评分 3206，在人类选手中排第 23 位，Apex Shortlist 90.2%。在基于 Lean 的形式化数学上，它在智能体设定下达到 SOTA，并在算力更密集的"先非形式化再形式化"混合流程下于 Putnam-2025 上取得 120/120 的完美证明。

![DeepSeek-V4 series performance on the MRCR long-context retrieval task. Retrieval is stable within 128K and degrades gracefully beyond it, remaining strong at 1M tokens. ／ DeepSeek-V4 系列在 MRCR 长上下文检索任务上的表现。128K 以内检索能力稳定，超出后平缓下降，在 100 万 token 时仍然很强。](/blog/deepseek-v4/mrcr.png)

On long context, V4-Pro beats Gemini-3.1-Pro on both MRCR 1M (83.5 vs 76.3) and CorpusQA, but sits clearly behind Claude Opus 4.6 (92.9). Retrieval holds nearly flat to 128K and then degrades visibly — the honest reading is that a million tokens is now *usable*, not uniform.

在长上下文上，V4-Pro 在 MRCR 1M（83.5 vs 76.3）和 CorpusQA 上都胜过 Gemini-3.1-Pro，但明显落后于 Claude Opus 4.6（92.9）。检索性能在 128K 内几乎持平，之后可见下滑——诚实的读法是：一百万 token 现在是*可用*的，但不是均质的。

Agentic performance is the soft spot the paper doesn't hide: comparable to K2.6 and GLM-5.1, but all three trail the closed models. The report's own framing of reasoning is worth quoting: V4 sits behind GPT-5.4 and Gemini-3.1-Pro by "approximately 3 to 6 months."

智能体能力是论文没有回避的弱项：与 K2.6 和 GLM-5.1 相当，但这三者都落后于闭源模型。报告对推理能力的自我定位值得原样引用：V4 落后 GPT-5.4 和 Gemini-3.1-Pro "大约 3 到 6 个月"。

![Win-rate comparison against Opus-4.6-Max on Chinese white-collar tasks across analysis, generation, and editing. ／ 在中文白领任务的分析、生成、编辑三类上与 Opus-4.6-Max 的胜率对比。](/blog/deepseek-v4/winrate.png)

The real-world evaluations are where V4 looks strongest, and they're the ones benchmarks miss. On Chinese functional writing it beats Gemini-3.1-Pro 62.7% to 34.1%, and on creative writing quality 77.5% — the paper attributes Gemini's losses to its stylistic preferences overriding explicit user requirements. On 30 Chinese professional white-collar tasks judged blind by humans, V4-Pro-Max reaches a 63% non-loss rate against Opus-4.6-Max, strongest on task completion and content quality, weakest on formatting aesthetics and on condensing long inputs. And in a survey of 85 DeepSeek engineers who use it daily for agentic coding, 52% said it's ready to be their default coding model and 39% leaned yes.

真实场景评测才是 V4 最亮眼的地方，而这些恰恰是基准测试测不到的。中文功能性写作上它以 62.7% 对 34.1% 胜过 Gemini-3.1-Pro，创意写作质量维度上胜率 77.5%——论文把 Gemini 的失分归因于其固有文体偏好压过了用户的明确要求。在 30 个中文专业白领任务上，由人工盲评，V4-Pro-Max 对 Opus-4.6-Max 取得 63% 的不败率，强在任务完成度与内容质量，弱在排版美观和把长输入压缩成简洁摘要。而在一份对 85 位每天用它做智能体编程的 DeepSeek 工程师的调查中，52% 认为它已经可以作为默认编程模型，39% 倾向于认为可以。

## Why it matters ／ 它为什么重要

The conclusion contains an unusual admission: to minimize risk, the team kept many separately validated components, and the resulting architecture is "relatively complex." Reading it, that's hard to disagree with — CSA and HCA and sliding windows and attention sinks and *m*HC and hash-routed early layers and hybrid Newton-Schulz is a lot of moving parts, several of which exist to patch problems the others introduced. They commit to distilling it down in future work.

结论部分有一段不常见的自陈：为了降低风险，团队保留了许多经过单独验证的组件，导致最终架构"相对复杂"。读完全文很难不同意——CSA 加 HCA 加滑动窗口加 attention sink 加 *m*HC 加哈希路由的前几层加混合 Newton-Schulz，确实有很多活动部件，其中好几个的存在是为了给另一些引入的问题打补丁。他们承诺在后续工作中把它精简下来。

But complexity is what an efficiency-first bet looks like before anyone knows which pieces were load-bearing. The core claim — that a million-token context can cost a fraction of what it does today, natively, in an open model — is demonstrated, and that reframes what test-time scaling can afford. It's instructive to read alongside [Kimi K3](/blog/kimi-k3), which reaches for the same million-token target by replacing softmax attention with a fixed-state recurrence rather than compressing it. Two open labs, two opposite architectural bets, same year. We'll know within a generation of models which one was right — or, more likely, that both were.

但在没人知道哪些部件是承重结构之前，一个"效率优先"的赌注看起来本来就是这样。核心主张——百万 token 上下文的代价可以只是今天的一小部分，原生支持，且在一个开源模型里——已经被证明了，而这重新定义了测试时扩展能负担得起什么。把它与 [Kimi K3](/blog/kimi-k3) 对照阅读会很有启发：后者奔向同一个百万 token 目标，选择的却是用固定状态的循环机制*替换* softmax 注意力，而不是压缩它。两个开源实验室，两个相反的架构赌注，同一年。再过一代模型，我们就会知道哪一个是对的——或者更可能的是，两个都对。

---

**Links:** [arXiv:2606.19348](https://arxiv.org/abs/2606.19348) · [Model weights](https://huggingface.co/collections/deepseek-ai/deepseek-v4) · [MegaMoE kernel](https://github.com/deepseek-ai/DeepGEMM/pull/304)

**链接：** [arXiv:2606.19348](https://arxiv.org/abs/2606.19348) · [模型权重](https://huggingface.co/collections/deepseek-ai/deepseek-v4) · [MegaMoE 内核](https://github.com/deepseek-ai/DeepGEMM/pull/304)
