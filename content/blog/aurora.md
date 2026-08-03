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
