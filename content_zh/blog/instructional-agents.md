Ask any instructor what eats their time, and "preparing course materials" is near the top. A single course needs learning objectives, a syllabus, slides, lecture scripts, and assessments — and they all need to be *coherent* with one another. In a well-resourced department, that work is shared among faculty, instructional designers, and teaching assistants. In an under-resourced one, it lands on one overstretched person. **Instructional Agents** is our attempt to package that whole collaborative process into a multi-agent LLM system.

随便问一位任课教师，什么最消耗他们的时间，"准备课程材料"一定名列前茅。一门课需要教学目标、教学大纲、幻灯片、讲稿和测评——而且它们彼此之间还必须*相互一致*。在资源充足的院系里，这些工作由教师、教学设计师和助教共同分担；而在资源匮乏的院系里，它们全都压在一个本已不堪重负的人身上。**Instructional Agents**（教学智能体）就是我们把这一整套协作流程封装进多智能体 LLM 系统的尝试。

## Beyond one-task tools ／ 超越单任务工具

Plenty of AI tools handle *isolated* slices of teaching — a quiz generator here, a grading assistant there. What's been missing is an **end-to-end** workflow that produces a complete, internally consistent course package. The risk with piecemeal tools is exactly the fragmentation good course design tries to avoid: objectives that don't match the assessments, slides that drift from the syllabus.

已经有不少 AI 工具能处理教学中*孤立*的片段——这边一个测验生成器，那边一个批改助手。一直缺失的，是一条能产出完整、内部自洽的课程包的**端到端**工作流。零散工具的风险，恰恰是好的课程设计极力想避免的碎片化：教学目标与测评对不上，幻灯片与教学大纲渐行渐远。

Our key design choice is to **simulate the team, not just the task.** Instructional Agents spins up role-specialized agents — a **Teaching Faculty** agent (the primary authority), an **Instructional Designer**, a **Teaching Assistant**, a **Course Coordinator**, and a **Program Chair** — and has them collaborate the way a real curriculum-development team would.

我们的关键设计选择是**模拟整个团队，而不只是模拟任务**。Instructional Agents 会启动一组角色专门化的智能体——**任课教师（Teaching Faculty）**智能体（主要权威）、**教学设计师（Instructional Designer）**、**助教（Teaching Assistant）**、**课程协调员（Course Coordinator）**和**项目负责人（Program Chair）**——让它们像真实的课程开发团队那样协作。

![Overview of Instructional Agents. Left: educator input and human feedback drive the generation of learning objectives, syllabi, slides, slide scripts, and assessments. Right: role-specialized agents collaborate across the Analyze, Design, and Develop phases of the ADDIE framework. ／ Instructional Agents 总览。左侧：教师输入与人类反馈驱动教学目标、教学大纲、幻灯片、讲稿和测评的生成。右侧：角色专门化的智能体在 ADDIE 框架的分析（Analyze）、设计（Design）与开发（Develop）阶段中协同工作。](/blog/instructional-agents/framework.png)

## Structured by ADDIE ／ 以 ADDIE 为骨架

Rather than letting the agents improvise, we ground the workflow in **ADDIE** — a standard instructional-design framework — and implement its first three phases:

我们没有让智能体自由发挥，而是把整个工作流建立在**ADDIE**——一套标准的教学设计框架——之上，并实现了它的前三个阶段：

1. **Analyze** — the Teaching Faculty agent sets the instructional objectives while the Course Coordinator supplies context (student background, constraints). The output is an *Instructional Foundation Report* that grounds everything downstream.
1. **分析（Analyze）**——任课教师智能体设定教学目标，课程协调员提供背景信息（学生基础、各类约束）。产出是一份*教学基础报告（Instructional Foundation Report）*，为后续所有环节奠定基础。
2. **Design** — Faculty and Instructional Designer agents jointly build the syllabus, plan the slide flow, and design assessments (including a multi-stage capstone in place of a traditional exam), keeping objectives, content, and assessment aligned.
2. **设计（Design）**——任课教师与教学设计师智能体共同搭建教学大纲、规划幻灯片脉络、设计测评（包括用一个多阶段的结课项目替代传统考试），并保持教学目标、内容与测评三者对齐。
3. **Develop** — the Teaching Assistant agent turns the plans into concrete **LaTeX slides**, slide scripts, and finalized assessments; the Program Chair reviews at the program level; and a simulated Test Student surfaces confusing or mispaced material for refinement.
3. **开发（Develop）**——助教智能体把这些方案落实为具体的 **LaTeX 幻灯片**、讲稿和最终版测评；项目负责人从整个专业项目的层面进行审阅；同时由一个模拟的"测试学生"找出令人困惑或节奏失当的材料，供进一步打磨。

![From key points and drafts to final slides, slide scripts, and assessments — the generation workflow bridging the Design and Develop phases. ／ 从要点和草稿到最终的幻灯片、讲稿与测评——衔接设计阶段与开发阶段的生成工作流。](/blog/instructional-agents/slide-workflow.png)

We deliberately stop before ADDIE's *Implement* and *Evaluate* phases. Those require deploying materials to real students, and we believe AI-generated content should pass human review first.

我们有意在 ADDIE 的*实施（Implement）*与*评估（Evaluate）*阶段之前停下。这两个阶段需要把材料真正投放给学生，而我们认为，AI 生成的内容应当先通过人类审核。

## Four modes, one dial for human involvement ／ 四种模式，一个调节人类参与度的旋钮

Different instructors want different amounts of control, so the system runs in four modes that slide from fully automated to fully collaborative:

不同的教师想要的控制程度并不相同，因此系统提供了四种模式，从完全自动一路滑向完全协作：

- **Autonomous** — give it a course name and walk away. Best for rapid prototyping and benchmarking.
- **自主模式（Autonomous）**——给它一个课程名称，然后就可以走开了。最适合快速原型和基准测试。
- **Catalog-Guided** — seed the agents with an `Educator_Catalog` of institutional policies, existing course structures, and prior feedback, so outputs stay consistent with departmental standards.
- **目录引导模式（Catalog-Guided）**——用一份包含院校政策、既有课程结构和以往反馈的 `Educator_Catalog` 为智能体提供起点，使产出与院系标准保持一致。
- **Feedback-Guided** — let the system run, then review and request targeted regeneration of specific pieces without restarting the pipeline.
- **反馈引导模式（Feedback-Guided）**——先让系统跑完，再进行审阅，并针对特定部分请求重新生成，无需重启整条流水线。
- **Full Co-Pilot** — the system pauses after each subtask for the instructor to approve, edit, or redirect. The most hands-on, and — as the results show — the highest quality.
- **全程协作模式（Full Co-Pilot）**——系统在每个子任务之后暂停，等待教师确认、修改或调整方向。这是参与度最高的模式，而且——正如实验结果所示——质量也最高。

Across all human-in-the-loop modes, the Teaching Faculty keeps final approval. The system is built to *support* faculty, not replace them.

在所有人在环（human-in-the-loop）模式中，最终审批权始终掌握在任课教师手里。这个系统的设计初衷是*支持*教师，而不是取代他们。

## What we found ／ 我们的发现

We evaluated Instructional Agents on **five university-level courses** (Data Mining, Foundations of ML, Data Processing at Scale, Intro to AI, and Topics in RL), using an adapted **Quality Matters** rubric, with both human expert reviewers and LLM reviewers scoring six outputs on a 1–5 scale.

我们在**五门大学课程**（数据挖掘、机器学习基础、大规模数据处理、人工智能导论、强化学习专题）上评估了 Instructional Agents，采用改编自 **Quality Matters** 的评分量表，由人类专家评审和 LLM 评审共同对六类产出按 1–5 分打分。

**Cheaper models hold their own.** Comparing GPT-4o, GPT-4o-mini, and o1-preview as backends, a Friedman test found *no significant difference* in quality (Q = 0.473, p = 0.789). Since GPT-4o-mini is by far the cheapest, it became our default backend.

**更便宜的模型毫不逊色。** 在对比 GPT-4o、GPT-4o-mini 和 o1-preview 作为后端时，Friedman 检验发现质量上*没有显著差异*（Q = 0.473，p = 0.789）。由于 GPT-4o-mini 的成本远低于其他两者，它成了我们的默认后端。

![Quality, cost, and success rate across three LLM backends. GPT-4o-mini matches GPT-4o and o1-preview on quality while costing the least. ／ 三种 LLM 后端在质量、成本和成功率上的对比。GPT-4o-mini 在质量上与 GPT-4o 和 o1-preview 持平，成本却最低。](/blog/instructional-agents/model-comparison.png)

**More human involvement means better materials.** Full Co-Pilot mode consistently scored highest — about **0.5 to 0.9 points** above Autonomous mode — with the biggest gains in Learning Objectives, Slide Scripts, and the overall Package. Each mode trades refinement against effort: Catalog-Guided shines on structural pieces (objectives, syllabi), while Feedback-Guided does well on content-heavy ones (assessments, slides).

**人类参与越多，材料质量越好。** 全程协作模式的得分始终最高——比自主模式高出大约 **0.5 到 0.9 分**——其中提升最明显的是教学目标、讲稿和整体课程包。每种模式都是在打磨程度与投入精力之间做权衡：目录引导模式在结构性产出（教学目标、教学大纲）上表现出色，而反馈引导模式则在内容密集的产出（测评、幻灯片）上更胜一筹。

![Radar chart of human-rated quality across material types for each operational mode. Full Co-Pilot covers the most area. ／ 各运行模式下不同材料类型的人工评分雷达图。全程协作模式覆盖的面积最大。](/blog/instructional-agents/radar.png)

**Two findings worth flagging:**

**有两个发现值得特别指出：**

- **Roles aren't decoration.** Ablating any single agent hurt quality, and the single-agent baseline was the worst of all (avg 2.33 vs. 3.74 for full Co-Pilot). Drop the Instructional Designer and learning-objective clarity collapses; drop the Teaching Assistant and slide/LaTeX consistency suffers. Role specialization is a genuine design requirement, not a flourish.
- **角色分工不是摆设。** 去掉任何一个智能体都会损害质量，而单智能体基线是所有配置中最差的（平均 2.33，全程协作模式为 3.74）。去掉教学设计师，教学目标的清晰度就会崩塌；去掉助教，幻灯片与 LaTeX 的一致性就会变差。角色专门化是一项真正的设计需求，而非锦上添花。
- **LLMs make poor judges.** As automated reviewers, LLMs clustered their scores tightly around 3.0, while human reviewers spread their judgments across the full range. LLM evaluators struggle to tell great work from mediocre — so we treat human assessment as the ground truth.
- **LLM 不适合当评委。** 作为自动评审，LLM 给出的分数紧紧聚集在 3.0 附近，而人类评审的判断则分布在整个分数区间。LLM 评审很难区分优秀与平庸的作品——因此我们把人类评估视为真实标准。

The cost side is striking: a full course package runs on the order of **$0.22–$0.36** in compute, with human time ranging from zero (Autonomous) to 30–45 minutes (Co-Pilot).

成本这一面尤其惊人：生成一整份课程包的算力开销约为 **$0.22–$0.36**，人类投入的时间则从零（自主模式）到 30–45 分钟（全程协作模式）不等。

## Why it matters ／ 它为什么重要

The point isn't to take instructors out of the loop — it's to lower the barrier to *good* course design for the institutions that need it most. Community colleges, international programs, and under-resourced departments rarely have dedicated instructional-design staff. A system that drafts a coherent, rubric-aligned course package for under a dollar, then hands it to faculty for review and refinement, is a step toward more equitable access to high-quality education.

我们的目的不是把教师排除在外，而是为最需要帮助的院校降低做出*好的*课程设计的门槛。社区学院、国际项目和资源匮乏的院系，很少配备专职的教学设计人员。一个能以不到一美元的成本起草出连贯、契合评分量表的课程包，再交给教师审阅和打磨的系统，正是迈向更公平地获取优质教育的一步。

The project website and source code are available at [darl-genai.github.io/instructional_agents_homepage](https://darl-genai.github.io/instructional_agents_homepage/).

项目主页与源代码见 [darl-genai.github.io/instructional_agents_homepage](https://darl-genai.github.io/instructional_agents_homepage/)。

---

**Links:** [Project page](https://darl-genai.github.io/instructional_agents_homepage/) · [All publications](/publications) — accepted at EACL 2026.

**链接：** [项目主页](https://darl-genai.github.io/instructional_agents_homepage/) · [全部论文](/publications) —— 已被 EACL 2026 接收。
