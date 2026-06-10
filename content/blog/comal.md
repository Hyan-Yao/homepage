Stop-and-go waves are the bane of any commute. A single driver tapping the brakes can send a ripple backward through traffic that turns a free-flowing road into a crawling parking lot — no accident, no bottleneck, just the inherent instability of human driving. The promising news from traffic research is that you don't need to automate *every* car to fix this. Even a handful of well-behaved autonomous vehicles mixed into human traffic can smooth the whole flow. This is the **mixed-autonomy traffic** problem, and it's where **CoMAL (Collaborative Multi-Agent LLMs)** comes in.

## Why not just use reinforcement learning?

The standard tool for mixed-autonomy control is multi-agent reinforcement learning (RL). It works well — in the specific scenario it was trained on. But RL has two stubborn weaknesses. First, it struggles to **generalize**: change the road layout or the mix of human drivers and you often need to retrain from scratch on mountains of data. Second, it's a **black box** — it's hard to understand *why* an RL policy decided to brake or accelerate.

Humans don't learn to drive that way. A teenager can become a competent driver in about 20 hours and handle situations they've never seen, because human driving is **knowledge-driven**: we rely on common sense, we communicate, and we reason. CoMAL asks a simple question — what if the autonomous vehicles could do the same, by talking to each other?

## How CoMAL works

CoMAL builds each connected autonomous vehicle (CAV) as an LLM agent, and the agents coordinate through a structured workflow.

![The CoMAL framework: a single-agent pipeline feeds perception, memory, and shared messages into an LLM, which drives a multi-agent workflow of Collaboration, Reasoning, and Execution across the Ring, Figure-Eight, and Merge benchmarks.](/blog/comal/framework.png)

The pipeline has a few moving parts:

- **Perception Module** — turns the raw simulation state into a natural-language description of the road (a static map) and the surrounding vehicles (their motion), so the LLM can actually "see" the scene.
- **Memory Module** — stores past driving experiences and hand-written tips that the agent recalls (and updates) when reasoning about new situations.
- **Collaboration Module** — the heart of the system. Agents take turns posting to a shared message pool, brainstorming a strategy and **allocating roles** among themselves.
- **Reason Engine** — once an agent knows its role, a hierarchical chain-of-thought walks it through role clarification → scene understanding → motion instruction → planner generation.
- **Execution Module** — here's the clever bit. The LLM doesn't directly steer. Instead it outputs parameters for the rule-based **Intelligent Driver Model (IDM)** — desired speed, maximum acceleration, minimum spacing. The LLM acts as a high-level commander; the proven car-following model handles the low-level control. This sidesteps the LLM's well-known weakness at precise, real-time control.

The collaboration really is a conversation. In the Figure-Eight scenario, for example, the agents recognize they need to form a queue through the intersection, then negotiate who is the **leader** and who are the **followers**.

![A collaboration trace in the Figure-Eight scenario: agents agree to form a queue, then allocate the leader and follower roles among themselves.](/blog/comal/collaboration.png)

## Does it actually work?

We tested CoMAL on the **Flow** benchmark across three classic scenarios — Ring, Figure-Eight, and Merge — and measured two things: average vehicle speed (higher is better) and the standard deviation of speed (lower means smoother, more stable traffic).

The clearest picture is the Ring road. With only human drivers, you get the textbook stop-and-go shockwave. Drop in three CoMAL-controlled vehicles, and the wave is damped out.

![Space-time trajectories on the 230 m Ring road with 22 vehicles. Left: human-only traffic forms stop-and-go shockwaves. Right: three CoMAL-controlled CAVs stabilize the flow.](/blog/comal/trajectory.png)

A few headline findings:

- **It beats human drivers**, and the advantage grows as more CAVs join the network — strong evidence that the agents are genuinely cooperating, not just driving well individually.
- **Collaboration is the key ingredient.** In ablations, removing the Collaboration Module hurt performance the most — without it, every agent converges on the *same* strategy, leading to conflicts and duplicated effort. Sometimes that was worse than the plain rule-based baseline.
- **vs. RL:** CoMAL beats RL methods on Figure-Eight, where global role differentiation matters and RL struggles to assign distinct roles. On Merge, RL still wins — CoMAL's cooperation there is more local than global, which is an honest limitation.
- **Model size matters for *cooperation*, not just reasoning.** GPT-4o-mini led the pack; among open models Qwen-72B was competitive, while the 7B model fell off sharply — and it degraded *fastest* in scenarios demanding heavy collaboration. The takeaway: collaboration is a harder skill for an LLM than individual reasoning.

## The bigger picture

CoMAL is, to our knowledge, the first framework to put the *collaborative* capability of multi-agent LLMs to work in autonomous driving. It trades the data-hunger and opacity of RL for the generalization and interpretability of language-based reasoning — you can literally read the agents' negotiation. The natural next steps: scale to many more agents to see whether emergent cooperative behaviors appear, and combine LLMs with RL to get the best of exploration and common-sense reasoning.

Code is available at [github.com/Hyan-Yao/CoMAL](https://github.com/Hyan-Yao/CoMAL).

---

**Links:** [Paper (SDM 2025)](https://epubs.siam.org/doi/10.1137/1.9781611978520.43) · [Code](https://github.com/Hyan-Yao/CoMAL) · [All publications](/publications)
