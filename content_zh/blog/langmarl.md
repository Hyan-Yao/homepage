Put a team of LLM agents on a cooperative task and you hit a familiar management problem: the team failed, but *who* should change what they're doing? A single "the order failed, customer complained" tells every agent it did badly — including the one who played its part perfectly. Without a way to trace team outcomes back to individual contributions, the usual fixes are blunt: a global reflection, or a wholesale rewrite of everyone's prompt. **LangMARL** takes a different route, borrowing a tool that classical multi-agent reinforcement learning (MARL) spent years perfecting: **credit assignment**.

把一队 LLM 智能体放到一个协作任务上，你很快就会撞上一个熟悉的管理难题：团队失败了，可*究竟是谁*该改变自己的做法？一句笼统的"订单失败了，客户投诉了"会告诉每一个智能体它做得不好——包括那个把自己那部分完成得完美无缺的智能体。如果没有办法把团队层面的结果追溯回个体的贡献，常见的补救手段就只能很粗暴：来一次全局反思，或者把所有人的 prompt 推倒重写。**LangMARL** 走了另一条路，它借用了经典多智能体强化学习（MARL）打磨多年的一件工具：**信用分配（credit assignment）**。

## The core problem: coarse feedback ／ 核心问题：过于粗糙的反馈

In a cooperative game like Overcooked, two cooks share one reward. When dinner doesn't go out, a global evaluation can't tell the cook who prepped correctly from the cook who stalled. Both get the same vague "what went wrong?" — and both flail.

在像 Overcooked 这样的协作游戏里，两位厨师共享同一份奖励。当这道菜没能端出去时，一个全局评价无法区分：谁把配菜备得妥妥当当，谁又在原地空转。两人拿到的都是同样含糊的"哪里出问题了？"——于是两人都只能瞎折腾。

![Why global feedback fails. With only a team-level "order failed" signal (middle), each agent gets an ambiguous reflection and can't tell what to fix. LangMARL (bottom) decomposes the outcome into agent-specific credit: one agent is told it failed to provide requested items, the other that it correctly identified the missing ingredients. ／ 为什么全局反馈会失效。当只有团队级的"订单失败"信号时（中），每个智能体得到的反思都是含混的，无从判断该修正什么。LangMARL（下）把结果分解为针对每个智能体的信用：一个智能体被告知它没能提供被请求的物品，另一个则被告知它正确识别出了缺失的配料。](/blog/langmarl/challenge.png)

This is exactly the **credit assignment** problem — *how do you derive a useful learning signal for each agent from one shared team reward?* MARL has mature answers (counterfactual baselines, value decomposition), but they're built for numeric gradients and don't transfer to black-box LLMs that "think" in language. LangMARL's bet is that you can do credit assignment **in language space** instead.

这正是**信用分配**问题——*如何从一份共享的团队奖励中，为每个智能体推导出有用的学习信号？* MARL 对此已有成熟的答案（反事实基线、值分解），但它们都是为数值梯度设计的，无法迁移到用语言"思考"的黑盒 LLM 上。LangMARL 押的赌注是：信用分配完全可以**在语言空间中**完成。

## MARL, rewritten in natural language ／ 用自然语言重写的 MARL

The key design move is to take the standard actor–critic machinery of cooperative MARL and re-express each component as a natural-language operation. The whole thing follows the classic **centralized-training, decentralized-execution (CTDE)** paradigm: agents act on local observations alone, but a centralized critic that sees the full trajectory does the teaching.

关键的设计动作，是把协作式 MARL 中标准的 actor–critic 机制，逐个组件地重新表述为自然语言操作。整套框架遵循经典的**集中式训练、分散式执行（centralized-training, decentralized-execution, CTDE）**范式：智能体仅依据局部观测行动，而由一个能看到完整轨迹的中心化 critic 来负责"教学"。

![LangMARL as a toolkit. Each classical MARL abstraction has a natural-language counterpart: TensorDict states become language State, the probabilistic actor becomes an LLMActor with a text policy, the value operator becomes an LLMCritic, and the numeric optimizer becomes a LanguagePolicyOptimizer — keeping the same syntax for the MARL principle while operating on policy text. ／ 作为工具箱的 LangMARL。每一个经典 MARL 抽象都有其自然语言对应物：TensorDict 状态变成语言化的 State，概率式 actor 变成带有文本策略的 LLMActor，值算子变成 LLMCritic，数值优化器变成 LanguagePolicyOptimizer——在操作策略文本的同时，保持与 MARL 原理相同的语法。](/blog/langmarl/toolkit.png)

Four components carry the framework:

支撑起整个框架的是四个组件：

1. **Language Policy Actors** — each agent's "policy" is a block of natural-language instructions, and it acts by querying the LLM conditioned on that policy text plus its observation. This plays the role of the stochastic policy in classical MARL, but its "parameters" live in language, not numeric vectors.
1. **语言策略 Actor（Language Policy Actors）**——每个智能体的"策略"就是一段自然语言指令，它以这段策略文本加上自身观测为条件去查询 LLM 来行动。它扮演的是经典 MARL 中随机策略的角色，只不过它的"参数"活在语言里，而不是数值向量里。
2. **Centralized Language Critic** — reads the complete episode and writes agent-specific feedback describing *how each agent's actions helped or hurt the outcome*. This is the credit assignment step, and it drives everything downstream.
2. **中心化语言 Critic（Centralized Language Critic）**——读取完整的 episode，并为每个智能体写出针对性的反馈，说明*该智能体的动作如何促成或损害了最终结果*。这就是信用分配环节，下游的一切都由它驱动。
3. **Language Policy Gradient Estimator** — turns each agent's language credit into a language-form "update direction": concrete suggestions for how that agent's policy text should change. A language-space analogue of the policy gradient, with no numbers computed.
3. **语言策略梯度估计器（Language Policy Gradient Estimator）**——把每个智能体拿到的语言信用转化为语言形式的"更新方向"：关于该智能体的策略文本应当如何修改的具体建议。这是策略梯度在语言空间中的类比，全程不计算任何数字。
4. **Language Policy Optimizer** — aggregates the per-trajectory language gradients across a batch (semantically merging consistent suggestions, resolving conflicts, suppressing noise) and applies the result to rewrite the policy.
4. **语言策略优化器（Language Policy Optimizer）**——在一个 batch 内聚合各条轨迹的语言梯度（在语义层面合并一致的建议、化解冲突、抑制噪声），再把结果应用到策略文本的重写上。

![The LangMARL pipeline. (a) Language Policy Actors produce decentralized actions from text policies; (b) a Centralized Language Critic assigns trajectory-level causal credit to each agent; (c) a Language Policy Gradient Estimator converts credit into language update directions; (d) a Language Policy Optimizer applies the updates — all in natural language, under a CTDE structure (e). ／ LangMARL 的流程。(a) 语言策略 Actor 依据文本策略产生分散式动作；(b) 中心化语言 Critic 为每个智能体分配轨迹级的因果信用；(c) 语言策略梯度估计器把信用转换为语言更新方向；(d) 语言策略优化器施加这些更新——全部以自然语言进行，并置于 CTDE 结构之下 (e)。](/blog/langmarl/pipeline.png)

The payoff of this framing is practical: LangMARL is packaged as a toolkit that mirrors the syntax of standard RL libraries like TorchRL, so building an LLM multi-agent optimizer looks much like wiring up an ordinary deep-RL pipeline.

这套表述带来的好处很实际：LangMARL 被封装成一个工具箱，其语法与 TorchRL 等标准 RL 库保持一致，因此搭建一个 LLM 多智能体优化器，看起来就和接一条普通的深度 RL 流水线差不多。

## What we found ／ 我们的发现

We evaluated LangMARL across two kinds of environments — open-ended **language tasks** (HotPotQA, MATH, HumanEval, reframed as multi-agent problems) and **strategic games** (Overcooked-AI, Pistonball) — against static-prompting and self-evolving baselines like Reflexion, TextGrad, DSPy, and Symbolic Learning.

我们在两类环境中评测了 LangMARL——开放式的**语言任务**（HotPotQA、MATH、HumanEval，被重新表述为多智能体问题）和**策略博弈**（Overcooked-AI、Pistonball）——对手是静态 prompt 方法，以及 Reflexion、TextGrad、DSPy、Symbolic Learning 等自演化基线。

- **It wins across the board.** LangMARL takes the top score on every benchmark: MATH 56.0 (vs. 53.8 for the best baseline), HotPotQA 60.2, HumanEval 73.2, and the highest mean reward on all five Overcooked layouts.
- **它全面胜出。** LangMARL 在每一个基准上都拿到最高分：MATH 56.0（最佳基线为 53.8）、HotPotQA 60.2、HumanEval 73.2，并在全部五个 Overcooked 布局上取得最高平均奖励。
- **It scales where others collapse.** In Pistonball, as the team grows from 10 to 20 pistons, baselines like TextGrad degrade into negative returns; LangMARL holds stable positive reward (+22.9 at N=20 vs. +14.3). Per-agent credit is exactly what keeps coordination from falling apart as the team grows.
- **别人崩溃的地方它还能扩展。** 在 Pistonball 中，当团队从 10 个 piston 增长到 20 个时，TextGrad 这类基线退化到负回报；LangMARL 则保持稳定的正奖励（N=20 时为 +22.9，对比 +14.3）。正是逐智能体的信用，使协作不会随团队规模变大而瓦解。
- **Credit assignment is the active ingredient.** Strip it out — train every agent on the shared global reward — and convergence slows and destabilizes across all five tasks. The orange (with credit) curves consistently beat the blue (without).
- **信用分配是真正起作用的成分。** 把它抽掉——让每个智能体都用共享的全局奖励来训练——全部五个任务上的收敛都会变慢、变得不稳定。带信用分配的橙色曲线始终优于不带的蓝色曲线。

![Learning curves across five tasks. With agent-specific credit assignment (orange), LangMARL converges faster and to higher final performance; without it (blue), learning is slower and less stable — most visibly on the reasoning and multi-agent coordination tasks. ／ 五个任务上的学习曲线。有了针对单个智能体的信用分配（橙色），LangMARL 收敛更快、最终性能更高；没有它时（蓝色），学习更慢也更不稳定——在推理任务和多智能体协作任务上尤为明显。](/blog/langmarl/training-curve.png)

## Roles nobody assigned ／ 没人指派过的角色

The most striking result is qualitative. Start two coding agents from *identical*, symmetric prompts — both are just "generic solvers." After a few iterations of trajectory-level credit and language-policy updates, they spontaneously **specialize**: one agent drifts toward structured implementation, the other toward critical review and edge-case validation. Nobody wrote those roles into the prompt. They emerge because the critic hands out *asymmetric* feedback based on each agent's actual causal contribution, and that pressure breaks the initial symmetry into a complementary division of labor.

最引人注目的结果是定性的。让两个编程智能体从*完全相同*的对称 prompt 出发——两者一开始都只是"通用求解器"。经过几轮轨迹级信用分配与语言策略更新之后，它们自发地**分化**了：一个智能体逐渐偏向结构化的实现，另一个则偏向批判性审查与边界情况验证。没有人把这些角色写进 prompt。它们之所以涌现，是因为 critic 依据每个智能体真实的因果贡献给出了*不对称*的反馈，而这种压力把最初的对称性打破成了一种互补的分工。

![Emergent role specialization. Top: from identical prompts, both coding agents start as generic solvers. Bottom: after iterative credit assignment and language-policy optimization, Agent 1 specializes in structured implementation and Agent 2 in critical evaluation and refinement — a division of labor that was never specified. ／ 涌现出的角色分化。上：从完全相同的 prompt 出发，两个编程智能体最初都是通用求解器。下：经过迭代的信用分配与语言策略优化后，Agent 1 专精于结构化实现，Agent 2 专精于批判性评估与改进——一种从未被指定过的分工。](/blog/langmarl/patterns.png)

The effect is robust across backbones (Gemini, GPT, and open-weight LLaMA-3.3-70B all benefit), and the gains track rollout budget — though in large teams there's a sweet spot beyond which extra rollouts start to amplify policy bias rather than help.

这一效应在不同底座模型上都很稳健（Gemini、GPT 以及开放权重的 LLaMA-3.3-70B 都能受益），收益也随 rollout 预算增加而提升——不过在大规模团队中存在一个甜蜜点，越过之后，额外的 rollout 反而会放大策略偏差，而不是带来帮助。

## Why it matters ／ 它为什么重要

LangMARL treats natural language as a first-class optimization space: a place where you can do credit assignment, estimate "gradients," and run a batch actor–critic loop — all the structure of MARL, none of the numeric machinery that black-box LLMs can't expose. The result is multi-agent systems that adapt *and* stay interpretable: you can literally read why an agent was credited or blamed, and watch a coordination strategy evolve. The open challenges are honest ones — long-horizon tasks where causal links stretch thin, and fixed agent topologies that can't yet spawn new specialists on the fly — and they point straight at where this line of work goes next.

LangMARL 把自然语言当作一等的优化空间：在这里可以做信用分配、估计"梯度"、跑一个成批的 actor–critic 循环——保留了 MARL 的全部结构，却不需要黑盒 LLM 根本无法暴露的那套数值机制。其结果是：多智能体系统既能自适应*又*保持可解释——你可以逐字读到某个智能体为何被记功或被追责，并亲眼看着一套协作策略一步步演化。尚未解决的挑战也很诚实——因果链条被拉得很稀薄的长时程任务，以及无法即时衍生出新专家角色的固定智能体拓扑——它们恰恰指明了这条研究路线接下来该往哪里走。

Code and documentation are available at [langmarl-tutorial.readthedocs.io](https://langmarl-tutorial.readthedocs.io/).

代码与文档见 [langmarl-tutorial.readthedocs.io](https://langmarl-tutorial.readthedocs.io/)。

---

**Links:** [Docs & code](https://langmarl-tutorial.readthedocs.io/) · [All publications](/publications)

**链接：** [文档与代码](https://langmarl-tutorial.readthedocs.io/) · [全部论文](/publications)
