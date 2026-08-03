Stop-and-go waves are the bane of any commute. A single driver tapping the brakes can send a ripple backward through traffic that turns a free-flowing road into a crawling parking lot — no accident, no bottleneck, just the inherent instability of human driving. The promising news from traffic research is that you don't need to automate *every* car to fix this. Even a handful of well-behaved autonomous vehicles mixed into human traffic can smooth the whole flow. This is the **mixed-autonomy traffic** problem, and it's where **CoMAL (Collaborative Multi-Agent LLMs)** comes in.

走走停停的交通波是每一次通勤的噩梦。一位司机轻点一下刹车，就可能在车流中激起一道向后传播的涟漪，把原本畅通的道路变成缓慢挪动的停车场——没有事故，也没有瓶颈，纯粹是人类驾驶本身固有的不稳定性。交通研究带来的好消息是：要解决这个问题，并不需要让*每一辆*车都实现自动驾驶。只要在人类车流中混入少数几辆行为得当的自动驾驶车辆，就足以让整体交通流变得平顺。这就是**混合自动驾驶交通（mixed-autonomy traffic）**问题，也正是 **CoMAL（Collaborative Multi-Agent LLMs，协作式多智能体大模型）**登场的地方。

## Why not just use reinforcement learning? ／ 为什么不直接用强化学习？

The standard tool for mixed-autonomy control is multi-agent reinforcement learning (RL). It works well — in the specific scenario it was trained on. But RL has two stubborn weaknesses. First, it struggles to **generalize**: change the road layout or the mix of human drivers and you often need to retrain from scratch on mountains of data. Second, it's a **black box** — it's hard to understand *why* an RL policy decided to brake or accelerate.

混合自动驾驶控制的标准工具是多智能体强化学习（RL）。它确实好用——前提是在它被训练的那个特定场景里。但 RL 有两个顽固的弱点。第一，它难以**泛化**：换一种道路布局，或者换一种人类驾驶员的构成，往往就得用海量数据从头再训一遍。第二，它是个**黑箱**——很难理解一个 RL 策略*为什么*决定刹车或加速。

Humans don't learn to drive that way. A teenager can become a competent driver in about 20 hours and handle situations they've never seen, because human driving is **knowledge-driven**: we rely on common sense, we communicate, and we reason. CoMAL asks a simple question — what if the autonomous vehicles could do the same, by talking to each other?

人类学开车可不是这样的。一个青少年大约 20 小时就能成为称职的司机，还能应付从未见过的状况，因为人类驾驶是**知识驱动的（knowledge-driven）**：我们依靠常识，我们彼此沟通，我们进行推理。CoMAL 提出了一个朴素的问题——如果自动驾驶车辆之间也能互相交谈，是不是就能做到同样的事？

## How CoMAL works ／ CoMAL 是怎么工作的

CoMAL builds each connected autonomous vehicle (CAV) as an LLM agent, and the agents coordinate through a structured workflow.

CoMAL 把每一辆网联自动驾驶车辆（CAV）都构建成一个 LLM 智能体，并让这些智能体通过一套结构化的工作流彼此协调。

![The CoMAL framework: a single-agent pipeline feeds perception, memory, and shared messages into an LLM, which drives a multi-agent workflow of Collaboration, Reasoning, and Execution across the Ring, Figure-Eight, and Merge benchmarks. ／ CoMAL 框架：单智能体流程把感知、记忆与共享消息送入 LLM，再由 LLM 驱动"协作—推理—执行"的多智能体工作流，运行在 Ring、Figure-Eight 和 Merge 三个基准场景上。](/blog/comal/framework.png)

The pipeline has a few moving parts:

整条流程由几个部件组成：

- **Perception Module** — turns the raw simulation state into a natural-language description of the road (a static map) and the surrounding vehicles (their motion), so the LLM can actually "see" the scene.
- **感知模块（Perception Module）**——把原始的仿真状态转写成对道路（静态地图）和周边车辆（运动状态）的自然语言描述，让 LLM 真正能够"看见"场景。
- **Memory Module** — stores past driving experiences and hand-written tips that the agent recalls (and updates) when reasoning about new situations.
- **记忆模块（Memory Module）**——存储过往的驾驶经验以及人工撰写的经验提示，智能体在推理新情况时会调取（并更新）它们。
- **Collaboration Module** — the heart of the system. Agents take turns posting to a shared message pool, brainstorming a strategy and **allocating roles** among themselves.
- **协作模块（Collaboration Module）**——整个系统的核心。智能体轮流向一个共享消息池发言，共同头脑风暴出策略，并在彼此之间**分配角色**。
- **Reason Engine** — once an agent knows its role, a hierarchical chain-of-thought walks it through role clarification → scene understanding → motion instruction → planner generation.
- **推理引擎（Reason Engine）**——一旦智能体明确了自己的角色，一条分层的思维链就会带着它依次走完：角色澄清 → 场景理解 → 运动指令 → 规划器生成。
- **Execution Module** — here's the clever bit. The LLM doesn't directly steer. Instead it outputs parameters for the rule-based **Intelligent Driver Model (IDM)** — desired speed, maximum acceleration, minimum spacing. The LLM acts as a high-level commander; the proven car-following model handles the low-level control. This sidesteps the LLM's well-known weakness at precise, real-time control.
- **执行模块（Execution Module）**——这里是精巧之处。LLM 并不直接操控车辆，而是输出基于规则的**智能驾驶员模型（Intelligent Driver Model, IDM）**的参数——期望速度、最大加速度、最小车距。LLM 扮演高层指挥官的角色，底层控制则交给久经验证的跟驰模型。这样就绕开了 LLM 众所周知的短板：精确的实时控制。

The collaboration really is a conversation. In the Figure-Eight scenario, for example, the agents recognize they need to form a queue through the intersection, then negotiate who is the **leader** and who are the **followers**.

这里的协作是名副其实的对话。比如在 Figure-Eight 场景中，智能体们会意识到需要排成队列通过交叉口，然后协商出谁来当**领队（leader）**、谁来当**跟随者（follower）**。

![A collaboration trace in the Figure-Eight scenario: agents agree to form a queue, then allocate the leader and follower roles among themselves. ／ Figure-Eight 场景中的一段协作记录：智能体们先达成共识要排成队列，随后在彼此之间分配领队与跟随者角色。](/blog/comal/collaboration.png)

## Does it actually work? ／ 它真的有效吗？

We tested CoMAL on the **Flow** benchmark across three classic scenarios — Ring, Figure-Eight, and Merge — and measured two things: average vehicle speed (higher is better) and the standard deviation of speed (lower means smoother, more stable traffic).

我们在 **Flow** 基准上、跨 Ring、Figure-Eight 和 Merge 三个经典场景测试了 CoMAL，并度量两项指标：车辆平均速度（越高越好）和速度的标准差（越低说明交通越平顺、越稳定）。

The clearest picture is the Ring road. With only human drivers, you get the textbook stop-and-go shockwave. Drop in three CoMAL-controlled vehicles, and the wave is damped out.

最清晰的画面来自环形道路（Ring）。只有人类司机时，你会得到教科书式的走走停停冲击波。放入三辆由 CoMAL 控制的车辆，这道波就被抑制住了。

![Space-time trajectories on the 230 m Ring road with 22 vehicles. Left: human-only traffic forms stop-and-go shockwaves. Right: three CoMAL-controlled CAVs stabilize the flow. ／ 230 米环形道路上 22 辆车的时空轨迹图。左：纯人类车流形成走走停停的冲击波。右：三辆 CoMAL 控制的 CAV 稳定了整个车流。](/blog/comal/trajectory.png)

A few headline findings:

几项主要发现：

- **It beats human drivers**, and the advantage grows as more CAVs join the network — strong evidence that the agents are genuinely cooperating, not just driving well individually.
- **它胜过人类司机**，而且随着更多 CAV 加入路网，优势还会继续扩大——这有力地说明智能体是在真正地互相协作，而不只是各自开得好。
- **Collaboration is the key ingredient.** In ablations, removing the Collaboration Module hurt performance the most — without it, every agent converges on the *same* strategy, leading to conflicts and duplicated effort. Sometimes that was worse than the plain rule-based baseline.
- **协作是关键成分。** 在消融实验中，去掉协作模块对性能的损害最大——没有它，每个智能体都会收敛到*同一套*策略，导致相互冲突和重复劳动。有时甚至比朴素的基于规则的基线还要差。
- **vs. RL:** CoMAL beats RL methods on Figure-Eight, where global role differentiation matters and RL struggles to assign distinct roles. On Merge, RL still wins — CoMAL's cooperation there is more local than global, which is an honest limitation.
- **对比 RL：** 在 Figure-Eight 上 CoMAL 胜过 RL 方法，这个场景需要全局层面的角色分化，而 RL 很难分配出互不相同的角色。在 Merge 上 RL 仍然占优——CoMAL 在那里的协作更偏局部而非全局，这是一个应当坦诚承认的局限。
- **Model size matters for *cooperation*, not just reasoning.** GPT-4o-mini led the pack; among open models Qwen-72B was competitive, while the 7B model fell off sharply — and it degraded *fastest* in scenarios demanding heavy collaboration. The takeaway: collaboration is a harder skill for an LLM than individual reasoning.
- **模型规模影响的是*协作*，而不只是推理。** GPT-4o-mini 表现领先；开源模型中 Qwen-72B 颇具竞争力，而 7B 模型则明显掉队——并且在需要大量协作的场景中，它退化得*最快*。由此可见：对 LLM 来说，协作是一项比个体推理更难掌握的技能。

## The bigger picture ／ 更大的图景

CoMAL is, to our knowledge, the first framework to put the *collaborative* capability of multi-agent LLMs to work in autonomous driving. It trades the data-hunger and opacity of RL for the generalization and interpretability of language-based reasoning — you can literally read the agents' negotiation. The natural next steps: scale to many more agents to see whether emergent cooperative behaviors appear, and combine LLMs with RL to get the best of exploration and common-sense reasoning.

据我们所知，CoMAL 是第一个把多智能体 LLM 的*协作*能力真正用于自动驾驶的框架。它以基于语言的推理所带来的泛化性与可解释性，换掉了 RL 的数据饥渴与不透明——你可以逐句读到智能体之间的协商过程。接下来自然的方向是：把规模扩展到更多的智能体，看看是否会涌现出协作行为；以及把 LLM 与 RL 结合起来，兼得探索能力与常识推理。

Code is available at [github.com/Hyan-Yao/CoMAL](https://github.com/Hyan-Yao/CoMAL).

代码已开源：[github.com/Hyan-Yao/CoMAL](https://github.com/Hyan-Yao/CoMAL)。

---

**Links:** [Paper (SDM 2025)](https://epubs.siam.org/doi/10.1137/1.9781611978520.43) · [Code](https://github.com/Hyan-Yao/CoMAL) · [All publications](/publications)

**链接：** [论文（SDM 2025）](https://epubs.siam.org/doi/10.1137/1.9781611978520.43) · [代码](https://github.com/Hyan-Yao/CoMAL) · [全部论文](/publications)
