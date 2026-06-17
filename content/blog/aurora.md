An LLM agent solving a long task is a bit like a person navigating an unfamiliar building: dozens of small decisions in a row, and one wrong turn early on can quietly doom everything that follows. When a step goes wrong, the cleanest fix is **rollback** — restore the agent to an earlier checkpoint, keep the good progress, and redo only the faulty part. The hard question isn't *how* to roll back. It's *when*. **AURORA (Attention-intended Dual Verification with Adaptive Rollback)** is our answer.

## The trouble with a second opinion

Agents are bad judges of their own work — a generator model tends to trust its own outputs and won't reliably flag its own mistakes. The popular fix is to bolt on a separate **assistant model** that reviews the trajectory after every step and decides whether to roll back. It helps, but it creates a new failure mode: the more eagerly the assistant intervenes, the more noise it injects. Sometimes it "corrects" steps that were perfectly fine, derailing a trajectory that was on track.

![Three intervention strategies on the same ALFWorld task. ReAct never intervenes and skips a required step, failing irreversibly. GA-Rollback intervenes unconditionally and misjudges valid early progress as an error, exhausting its step budget. AURORA verifies selectively with two independent heads and intervenes only when needed.](/blog/aurora/problem.png)

So gating intervention on *whether the current action is valid* seems sensible — but a single check isn't enough, because trajectories fail in two fundamentally different ways:

- **Local errors** — the current action is physically invalid (trying to place an object you're not holding, opening something already open). Detectable by looking at the single step.
- **Global deviations** — every individual action is valid, but the trajectory as a whole has drifted off-goal: repeating completed steps, ignoring a target in plain sight. Only visible when you reason over the *whole* history.

A single scalar judgment can't separate these. That's the gap AURORA fills.

## Two views from one model

The core idea is elegant: trajectory text is just a token sequence, and the **attention mask** controls which tokens a head is allowed to look at. So instead of training two separate verifier models, AURORA runs one shared encoder with two attention masks.

![AURORA's execution framework. The verifier applies two masking strategies over trajectory tokens — the local head attends to the current step only (physical validity), the global head attends to the full trajectory (overall coherence) — producing two scores. The assistant is invoked only when a score crosses its threshold, with the triggering source passed along as a hint.](/blog/aurora/framework.png)

- The **local head** masks out all history, attending only to the current step. It asks: *is this action valid right now?*
- The **global head** uses full attention over the entire trajectory. It asks: *is this path still heading toward the goal?*

Together they form a two-dimensional verification vector. The system uses simple OR logic — if *either* score crosses its threshold, the assistant is called, and it's told *which* head fired so it can focus its correction on the right kind of error. Both heads share one DeBERTa encoder and differ only in their mask.

![The dual-head verifier architecture. A shared DeBERTa encoder reads the trajectory (task, history, current action, environment feedback) under two attention masks. The local head pools current-step tokens for a physical-validity score; the global head pools the full trajectory for a coherence score. A rollback fires when either score exceeds its threshold.](/blog/aurora/verifier.png)

There's a subtlety. When both heads share the same attention computation, the local head's representation inevitably soaks up trajectory history, and the two scores start to converge instead of staying distinct. AURORA ships two variants to handle this: **Integration** masks at the attention-weight level (heads share context), while **Isolation** masks at the *input* level, cutting history out before attention so the local head sees only the current step. As we'll see, which one wins depends on model size.

Training needs no manual labels. A strong closed-source LLM (DeepSeek V3) annotates each step with local and global labels, which are distilled into the lightweight verifier — and a manual audit found 94% agreement with human judgments.

## Does it work?

We tested AURORA on two complementary benchmarks: **Game of 24** (symbolic reasoning) and **ALFWorld** (embodied, multi-step household tasks), across four generators from 3B to 14B parameters.

- **It beats unconditional rollback everywhere.** AURORA-Integration outperforms the GA-Rollback baseline on all four models on both benchmarks — up to **+19.2 percentage points** in success rate. Gating the assistant with a learned signal is simply more precise than triggering it every step.
- **Two heads beat one.** Adding the global head on top of a local-only verifier lifts ALFWorld success on LLaMA-3.1-8B from 50.0% to 55.2%, and on GLM4-9B from 74.6% to 80.6%. The coherence signal catches drift that step-level checks miss.
- **It's more selective.** AURORA-Integration achieves the lowest (or near-lowest) rollback count in 7 of 8 model-dataset combinations — fewer interventions, better results.
- **It's also faster.** On LLaMA-3.1-8B it cuts average steps-to-success from 11.63 to 10.68 while raising success from 36.0% to 55.2%: not just more accurate rollback decisions, but more targeted recovery paths.

The case study makes the behavior concrete. On a cooling task, ReAct gets lost and runs out of steps; GA-Rollback's constant meddling tells the agent to put the mug down *before* cooling it. AURORA intervenes exactly twice — once when the local score collapses (v_local = 0.023) on a bad placement, once when the global score drops (v_global = 0.142) on a loss of direction — and finishes the task.

![Case study on an ALFWorld cooling task. ReAct (top) drifts into aimless exploration. GA-Rollback (middle) is misled by constant assistant intervention and drops the mug prematurely. AURORA (bottom) fires only on dual-head alarms, applying two precisely timed corrections to complete the task.](/blog/aurora/case-study.png)

## A capacity-dependent twist

One finding is worth dwelling on: the Integration-vs-Isolation choice flips with model scale. For 3B–9B generators, **Integration** (shared context) wins — smaller models can't produce reliable independent signals, so strict separation backfires and even over-triggers. But at 14B, **Isolation** (strict separation) takes the top results on both benchmarks. Larger models have the capacity to exploit truly independent validity and coherence signals; smaller ones are better off sharing. Verification quality, in other words, scales with the generator: when the model is weak, the trajectory itself is noisy, and multi-trial methods like Reflexion remain the safer bet.

## Why it matters

AURORA reframes trajectory verification from a single yes/no judgment into a structured, two-dimensional signal — physical validity *and* overall coherence — derived cheaply from attention masks over a single shared encoder. As a plug-and-play module, it requires no changes to the generator or assistant and drops into existing rollback frameworks with minimal overhead. For robotic and embodied planning, where long horizons make error propagation expensive, knowing *when* to hit undo is half the battle.

Code is available at [github.com/ziqiwang0908/AURORA](https://github.com/ziqiwang0908/AURORA).

---

**Links:** [Code](https://github.com/ziqiwang0908/AURORA) · [All publications](/publications) — accepted at IROS 2026.

---

# 中文版

一个 LLM 智能体在完成长任务时，有点像一个人在陌生大楼里找路：要连续做出几十个小决定，而早期一个错误的转弯，就可能悄无声息地毁掉之后的一切。当某一步走错时，最干净的修复办法是**回滚（rollback）**——把智能体恢复到更早的一个检查点，保留有效的进展，只重做出错的那部分。难点不在于*怎么*回滚，而在于*什么时候*回滚。**AURORA**（Attention-intended Dual Verification with Adaptive Rollback，基于注意力的双重验证与自适应回滚）就是我们给出的答案。

## "第二意见"带来的麻烦

智能体不擅长评判自己的工作——生成器模型往往过于相信自己的输出，无法可靠地发现自己的错误。常见的解决办法是再外挂一个独立的**助手模型（assistant model）**，在每一步之后审查轨迹，决定是否回滚。这确实有帮助，但也带来了新的失效模式：助手干预得越积极，注入的噪声反而越多。有时它会去"纠正"那些本来完全正确的步骤，把一条本来走在正轨上的轨迹带偏。

![在同一个 ALFWorld 任务上的三种干预策略。ReAct 从不干预，漏掉了一个必需的步骤，导致不可逆的失败。GA-Rollback 无条件干预，把早期有效的进展误判为错误，耗尽了步数预算。AURORA 用两个独立的验证头进行选择性验证，只在必要时才干预。](/blog/aurora/problem.png)

因此，根据*当前动作是否有效*来决定要不要干预，看起来很合理——但单一的检查并不够，因为轨迹的失败有两种本质上不同的方式：

- **局部错误（Local errors）**——当前动作在物理上无效（试图放下一个并没有拿在手里的物体，去打开一个已经打开的东西）。只看当前这一步就能发现。
- **全局偏离（Global deviations）**——每个单独的动作都有效，但整条轨迹作为一个整体已经偏离了目标：重复已经完成的步骤、对眼前的目标视而不见。只有在对*整段*历史进行推理时才能看出来。

单一的标量判断无法区分这两者。这正是 AURORA 要填补的空白。

## 一个模型，两种视角

核心思想很优雅：轨迹文本本质上就是一段 token 序列，而**注意力掩码**（attention mask）控制着每个验证头能看到哪些 token。于是，AURORA 不去训练两个独立的验证器模型，而是用一个共享编码器配上两套注意力掩码。

![AURORA 的执行框架。验证器对轨迹 token 施加两种掩码策略——局部头只关注当前步（物理有效性），全局头关注整条轨迹（整体一致性）——产生两个分数。只有当某个分数越过其阈值时才会调用助手，并把触发来源作为提示一并传入。](/blog/aurora/framework.png)

- **局部头**（local head）屏蔽掉全部历史，只关注当前这一步。它问的是：*这个动作此刻是否有效？*
- **全局头**（global head）对整条轨迹使用完整注意力。它问的是：*这条路径是否仍然朝着目标前进？*

二者共同构成一个二维的验证向量。系统采用简单的"或（OR）"逻辑——只要*任一*分数越过其阈值，就调用助手，并告诉它是*哪个*头被触发，好让它把纠正聚焦在相应的错误类型上。两个头共享同一个 DeBERTa 编码器，唯一的区别就在于掩码。

![双头验证器的架构。一个共享的 DeBERTa 编码器在两套注意力掩码下读取轨迹（任务、历史、当前动作、环境反馈）。局部头汇聚当前步 token 得到物理有效性分数；全局头汇聚整条轨迹得到一致性分数。当任一分数超过阈值时即触发回滚。](/blog/aurora/verifier.png)

这里有个微妙之处。当两个头共享同一套注意力计算时，局部头的表示不可避免地会吸收进轨迹历史，导致两个分数趋于趋同，而不是保持各自独立。AURORA 为此提供了两个变体：**Integration**（融合）在注意力权重层面施加掩码（两个头共享上下文），而 **Isolation**（隔离）在*输入*层面施加掩码，在进入注意力之前就把历史切掉，使局部头只能看到当前步。正如后面会看到的，哪一个更优，取决于模型规模。

训练无需任何人工标注。一个强大的闭源 LLM（DeepSeek V3）为每一步标注局部和全局标签，再蒸馏进轻量级的验证器——人工抽检显示其与人类判断的一致性达到 94%。

## 它真的有效吗？

我们在两个互补的基准上测试了 AURORA：**Game of 24**（符号推理）和 **ALFWorld**（具身的、多步家务任务），覆盖从 3B 到 14B 参数的四个生成器。

- **它在各处都胜过无条件回滚。** AURORA-Integration 在两个基准、全部四个模型上都优于 GA-Rollback 基线——成功率最高提升 **+19.2 个百分点**。用一个学习到的信号来把关助手，远比每一步都触发更精准。
- **两个头胜过一个。** 在只有局部头的验证器之上加入全局头，使 LLaMA-3.1-8B 在 ALFWorld 上的成功率从 50.0% 升至 55.2%，GLM4-9B 从 74.6% 升至 80.6%。一致性信号能抓住步级检查会漏掉的偏离。
- **它更具选择性。** 在 8 个"模型-数据集"组合中，AURORA-Integration 有 7 个取得了最低（或接近最低）的回滚次数——干预更少，结果更好。
- **它也更快。** 在 LLaMA-3.1-8B 上，它把平均成功步数从 11.63 降到 10.68，同时把成功率从 36.0% 提到 55.2%：这不只是更准确的回滚决策，也是更有针对性的恢复路径。

案例研究让这一行为变得具体。在一个"冷却"任务上，ReAct 迷失方向、耗尽步数；GA-Rollback 不停插手，竟让智能体在*冷却之前*就把杯子放下了。AURORA 恰好只干预两次——一次是在局部分数骤降（v_local = 0.023）时纠正一个错误的放置，一次是在全局分数下降（v_global = 0.142）时纠正方向的丢失——最终完成了任务。

![在 ALFWorld 冷却任务上的案例研究。ReAct（上）陷入漫无目的的探索。GA-Rollback（中）被助手的持续干预误导，过早地放下了杯子。AURORA（下）只在双头报警时才触发，用两次恰到好处的纠正完成任务。](/blog/aurora/case-study.png)

## 一个与模型容量相关的反转

有一个发现值得细说：Integration 与 Isolation 之间的优劣会随模型规模而反转。对于 3B–9B 的生成器，**Integration**（共享上下文）胜出——较小的模型无法产生可靠的独立信号，因此严格的隔离反而会适得其反，甚至导致过度触发。但在 14B 上，**Isolation**（严格隔离）在两个基准上都取得了最佳结果。更大的模型有能力利用真正独立的有效性信号和一致性信号；较小的模型则更适合共享。换句话说，验证质量会随生成器一起提升：当模型本身较弱时，轨迹本身就充满噪声，此时像 Reflexion 这样的多次重试方法仍是更稳妥的选择。

## 它为什么重要

AURORA 把轨迹验证从单一的是/否判断，重新构造为一个结构化的二维信号——物理有效性*与*整体一致性——并且只需在一个共享编码器上施加注意力掩码就能廉价地得到。作为一个即插即用的模块，它无需改动生成器或助手，能以极小的开销融入现有的回滚框架。在机器人与具身规划中，长时间跨度让错误传播代价高昂，而知道*什么时候*该按下"撤销"，已经是成功的一半。

代码已开源：[github.com/ziqiwang0908/AURORA](https://github.com/ziqiwang0908/AURORA)。

---

**链接：** [代码](https://github.com/ziqiwang0908/AURORA) · [全部论文](/publications) —— 已被 IROS 2026 接收。
