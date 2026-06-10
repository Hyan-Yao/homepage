Two write-ups on recent work from our group on multi-agent large language models — one for smarter traffic, one for scalable course design.

---

# CoMAL: Can LLM Agents Talk Their Way Out of a Traffic Jam?

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

# Instructional Agents: A Faculty Meeting Inside an LLM

Ask any instructor what eats their time, and "preparing course materials" is near the top. A single course needs learning objectives, a syllabus, slides, lecture scripts, and assessments — and they all need to be *coherent* with one another. In a well-resourced department, that work is shared among faculty, instructional designers, and teaching assistants. In an under-resourced one, it lands on one overstretched person. **Instructional Agents** is our attempt to package that whole collaborative process into a multi-agent LLM system.

## Beyond one-task tools

Plenty of AI tools handle *isolated* slices of teaching — a quiz generator here, a grading assistant there. What's been missing is an **end-to-end** workflow that produces a complete, internally consistent course package. The risk with piecemeal tools is exactly the fragmentation good course design tries to avoid: objectives that don't match the assessments, slides that drift from the syllabus.

Our key design choice is to **simulate the team, not just the task.** Instructional Agents spins up role-specialized agents — a **Teaching Faculty** agent (the primary authority), an **Instructional Designer**, a **Teaching Assistant**, a **Course Coordinator**, and a **Program Chair** — and has them collaborate the way a real curriculum-development team would.

![Overview of Instructional Agents. Left: educator input and human feedback drive the generation of learning objectives, syllabi, slides, slide scripts, and assessments. Right: role-specialized agents collaborate across the Analyze, Design, and Develop phases of the ADDIE framework.](/blog/instructional-agents/framework.png)

## Structured by ADDIE

Rather than letting the agents improvise, we ground the workflow in **ADDIE** — a standard instructional-design framework — and implement its first three phases:

1. **Analyze** — the Teaching Faculty agent sets the instructional objectives while the Course Coordinator supplies context (student background, constraints). The output is an *Instructional Foundation Report* that grounds everything downstream.
2. **Design** — Faculty and Instructional Designer agents jointly build the syllabus, plan the slide flow, and design assessments (including a multi-stage capstone in place of a traditional exam), keeping objectives, content, and assessment aligned.
3. **Develop** — the Teaching Assistant agent turns the plans into concrete **LaTeX slides**, slide scripts, and finalized assessments; the Program Chair reviews at the program level; and a simulated Test Student surfaces confusing or mispaced material for refinement.

![From key points and drafts to final slides, slide scripts, and assessments — the generation workflow bridging the Design and Develop phases.](/blog/instructional-agents/slide-workflow.png)

We deliberately stop before ADDIE's *Implement* and *Evaluate* phases. Those require deploying materials to real students, and we believe AI-generated content should pass human review first.

## Four modes, one dial for human involvement

Different instructors want different amounts of control, so the system runs in four modes that slide from fully automated to fully collaborative:

- **Autonomous** — give it a course name and walk away. Best for rapid prototyping and benchmarking.
- **Catalog-Guided** — seed the agents with an `Educator_Catalog` of institutional policies, existing course structures, and prior feedback, so outputs stay consistent with departmental standards.
- **Feedback-Guided** — let the system run, then review and request targeted regeneration of specific pieces without restarting the pipeline.
- **Full Co-Pilot** — the system pauses after each subtask for the instructor to approve, edit, or redirect. The most hands-on, and — as the results show — the highest quality.

Across all human-in-the-loop modes, the Teaching Faculty keeps final approval. The system is built to *support* faculty, not replace them.

## What we found

We evaluated Instructional Agents on **five university-level courses** (Data Mining, Foundations of ML, Data Processing at Scale, Intro to AI, and Topics in RL), using an adapted **Quality Matters** rubric, with both human expert reviewers and LLM reviewers scoring six outputs on a 1–5 scale.

**Cheaper models hold their own.** Comparing GPT-4o, GPT-4o-mini, and o1-preview as backends, a Friedman test found *no significant difference* in quality (Q = 0.473, p = 0.789). Since GPT-4o-mini is by far the cheapest, it became our default backend.

![Quality, cost, and success rate across three LLM backends. GPT-4o-mini matches GPT-4o and o1-preview on quality while costing the least.](/blog/instructional-agents/model-comparison.png)

**More human involvement means better materials.** Full Co-Pilot mode consistently scored highest — about **0.5 to 0.9 points** above Autonomous mode — with the biggest gains in Learning Objectives, Slide Scripts, and the overall Package. Each mode trades refinement against effort: Catalog-Guided shines on structural pieces (objectives, syllabi), while Feedback-Guided does well on content-heavy ones (assessments, slides).

![Radar chart of human-rated quality across material types for each operational mode. Full Co-Pilot covers the most area.](/blog/instructional-agents/radar.png)

**Two findings worth flagging:**

- **Roles aren't decoration.** Ablating any single agent hurt quality, and the single-agent baseline was the worst of all (avg 2.33 vs. 3.74 for full Co-Pilot). Drop the Instructional Designer and learning-objective clarity collapses; drop the Teaching Assistant and slide/LaTeX consistency suffers. Role specialization is a genuine design requirement, not a flourish.
- **LLMs make poor judges.** As automated reviewers, LLMs clustered their scores tightly around 3.0, while human reviewers spread their judgments across the full range. LLM evaluators struggle to tell great work from mediocre — so we treat human assessment as the ground truth.

The cost side is striking: a full course package runs on the order of **$0.22–$0.36** in compute, with human time ranging from zero (Autonomous) to 30–45 minutes (Co-Pilot).

## Why it matters

The point isn't to take instructors out of the loop — it's to lower the barrier to *good* course design for the institutions that need it most. Community colleges, international programs, and under-resourced departments rarely have dedicated instructional-design staff. A system that drafts a coherent, rubric-aligned course package for under a dollar, then hands it to faculty for review and refinement, is a step toward more equitable access to high-quality education.

The project website and source code are available at [darl-genai.github.io/instructional_agents_homepage](https://darl-genai.github.io/instructional_agents_homepage/).
