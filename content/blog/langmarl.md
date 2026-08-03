Put a team of LLM agents on a cooperative task and you hit a familiar management problem: the team failed, but *who* should change what they're doing? A single "the order failed, customer complained" tells every agent it did badly — including the one who played its part perfectly. Without a way to trace team outcomes back to individual contributions, the usual fixes are blunt: a global reflection, or a wholesale rewrite of everyone's prompt. **LangMARL** takes a different route, borrowing a tool that classical multi-agent reinforcement learning (MARL) spent years perfecting: **credit assignment**.

## The core problem: coarse feedback

In a cooperative game like Overcooked, two cooks share one reward. When dinner doesn't go out, a global evaluation can't tell the cook who prepped correctly from the cook who stalled. Both get the same vague "what went wrong?" — and both flail.

![Why global feedback fails. With only a team-level "order failed" signal (middle), each agent gets an ambiguous reflection and can't tell what to fix. LangMARL (bottom) decomposes the outcome into agent-specific credit: one agent is told it failed to provide requested items, the other that it correctly identified the missing ingredients.](/blog/langmarl/challenge.png)

This is exactly the **credit assignment** problem — *how do you derive a useful learning signal for each agent from one shared team reward?* MARL has mature answers (counterfactual baselines, value decomposition), but they're built for numeric gradients and don't transfer to black-box LLMs that "think" in language. LangMARL's bet is that you can do credit assignment **in language space** instead.

## MARL, rewritten in natural language

The key design move is to take the standard actor–critic machinery of cooperative MARL and re-express each component as a natural-language operation. The whole thing follows the classic **centralized-training, decentralized-execution (CTDE)** paradigm: agents act on local observations alone, but a centralized critic that sees the full trajectory does the teaching.

![LangMARL as a toolkit. Each classical MARL abstraction has a natural-language counterpart: TensorDict states become language State, the probabilistic actor becomes an LLMActor with a text policy, the value operator becomes an LLMCritic, and the numeric optimizer becomes a LanguagePolicyOptimizer — keeping the same syntax for the MARL principle while operating on policy text.](/blog/langmarl/toolkit.png)

Four components carry the framework:

1. **Language Policy Actors** — each agent's "policy" is a block of natural-language instructions, and it acts by querying the LLM conditioned on that policy text plus its observation. This plays the role of the stochastic policy in classical MARL, but its "parameters" live in language, not numeric vectors.
2. **Centralized Language Critic** — reads the complete episode and writes agent-specific feedback describing *how each agent's actions helped or hurt the outcome*. This is the credit assignment step, and it drives everything downstream.
3. **Language Policy Gradient Estimator** — turns each agent's language credit into a language-form "update direction": concrete suggestions for how that agent's policy text should change. A language-space analogue of the policy gradient, with no numbers computed.
4. **Language Policy Optimizer** — aggregates the per-trajectory language gradients across a batch (semantically merging consistent suggestions, resolving conflicts, suppressing noise) and applies the result to rewrite the policy.

![The LangMARL pipeline. (a) Language Policy Actors produce decentralized actions from text policies; (b) a Centralized Language Critic assigns trajectory-level causal credit to each agent; (c) a Language Policy Gradient Estimator converts credit into language update directions; (d) a Language Policy Optimizer applies the updates — all in natural language, under a CTDE structure (e).](/blog/langmarl/pipeline.png)

The payoff of this framing is practical: LangMARL is packaged as a toolkit that mirrors the syntax of standard RL libraries like TorchRL, so building an LLM multi-agent optimizer looks much like wiring up an ordinary deep-RL pipeline.

## What we found

We evaluated LangMARL across two kinds of environments — open-ended **language tasks** (HotPotQA, MATH, HumanEval, reframed as multi-agent problems) and **strategic games** (Overcooked-AI, Pistonball) — against static-prompting and self-evolving baselines like Reflexion, TextGrad, DSPy, and Symbolic Learning.

- **It wins across the board.** LangMARL takes the top score on every benchmark: MATH 56.0 (vs. 53.8 for the best baseline), HotPotQA 60.2, HumanEval 73.2, and the highest mean reward on all five Overcooked layouts.
- **It scales where others collapse.** In Pistonball, as the team grows from 10 to 20 pistons, baselines like TextGrad degrade into negative returns; LangMARL holds stable positive reward (+22.9 at N=20 vs. +14.3). Per-agent credit is exactly what keeps coordination from falling apart as the team grows.
- **Credit assignment is the active ingredient.** Strip it out — train every agent on the shared global reward — and convergence slows and destabilizes across all five tasks. The orange (with credit) curves consistently beat the blue (without).

![Learning curves across five tasks. With agent-specific credit assignment (orange), LangMARL converges faster and to higher final performance; without it (blue), learning is slower and less stable — most visibly on the reasoning and multi-agent coordination tasks.](/blog/langmarl/training-curve.png)

## Roles nobody assigned

The most striking result is qualitative. Start two coding agents from *identical*, symmetric prompts — both are just "generic solvers." After a few iterations of trajectory-level credit and language-policy updates, they spontaneously **specialize**: one agent drifts toward structured implementation, the other toward critical review and edge-case validation. Nobody wrote those roles into the prompt. They emerge because the critic hands out *asymmetric* feedback based on each agent's actual causal contribution, and that pressure breaks the initial symmetry into a complementary division of labor.

![Emergent role specialization. Top: from identical prompts, both coding agents start as generic solvers. Bottom: after iterative credit assignment and language-policy optimization, Agent 1 specializes in structured implementation and Agent 2 in critical evaluation and refinement — a division of labor that was never specified.](/blog/langmarl/patterns.png)

The effect is robust across backbones (Gemini, GPT, and open-weight LLaMA-3.3-70B all benefit), and the gains track rollout budget — though in large teams there's a sweet spot beyond which extra rollouts start to amplify policy bias rather than help.

## Why it matters

LangMARL treats natural language as a first-class optimization space: a place where you can do credit assignment, estimate "gradients," and run a batch actor–critic loop — all the structure of MARL, none of the numeric machinery that black-box LLMs can't expose. The result is multi-agent systems that adapt *and* stay interpretable: you can literally read why an agent was credited or blamed, and watch a coordination strategy evolve. The open challenges are honest ones — long-horizon tasks where causal links stretch thin, and fixed agent topologies that can't yet spawn new specialists on the fly — and they point straight at where this line of work goes next.

Code and documentation are available at [langmarl-tutorial.readthedocs.io](https://langmarl-tutorial.readthedocs.io/).

---

**Links:** [Docs & code](https://langmarl-tutorial.readthedocs.io/) · [All publications](/publications)
