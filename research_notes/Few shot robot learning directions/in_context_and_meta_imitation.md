# In-Context Learning and Meta-Imitation Learning for Few-Shot (1-2 Demo) Robot Task Acquisition

Scope note: these notes come from about 15 tool calls (arXiv abstract and HTML pages, plus searches). Numbers marked **[verified]** were read from the paper's arXiv HTML or abstract page during this session. Items marked **[from memory, verify]** come from background knowledge of well-known papers. The citation links point to the real arXiv entries, but I did not re-read those pages this session, so check any exact number before quoting it.

---

## 1. One-shot / meta-imitation: what has been shown, and what generalizes to genuinely held-out tasks versus held-out instances?

### Takeaway
Almost all classic one-shot imitation work (2017-2020) generalizes to **held-out object instances or configurations within a training task family**, not to new task types. Since 2024, methods have reported real held-out-*task* transfer from 1-2 demos: Instant Policy, ICRT, SimpleICL, and VIMA's L4 level. Each does it either with a strong structural prior (graphs over point clouds, retrieval, alignment-then-replay) or with a held-out set that shares motion primitives with training. Two things remain unshown: replication across labs, and transfer to new *skills*, as opposed to new object/primitive combinations.

### Cited Findings

**Classic meta-imitation (2017-2020)**
- **One-Shot Imitation Learning** (Duan et al., NeurIPS 2017). A soft-attention network is conditioned on one demo and outputs actions for a new initial state. Experiments use state-based block stacking, and the test "tasks" are held-out *stacking configurations* (orderings) of the same block-stacking family, not new skills. [from memory, verify] — [arXiv 1703.07326](https://arxiv.org/abs/1703.07326)
- **MIL: One-Shot Visual Imitation Learning via Meta-Learning** (Finn, Yu, Zhang, Abbeel, Levine; CoRL 2017). MAML is applied to demonstrations: one inner gradient step on one demo, from pixels. Tests cover simulated reaching and pushing and real-robot placing, all with *novel objects* in the same task family. [from memory, verify] — [arXiv 1709.04905](https://arxiv.org/abs/1709.04905)
- **DAML: One-Shot Imitation from Observing Humans** (Yu et al., RSS 2018). Meta-learns adaptation from a single *human video* with no robot actions, using a learned adaptation loss. Generalizes to new objects in pick-place and push. [from memory, verify] — [arXiv 1802.01557](https://arxiv.org/abs/1802.01557)
- **TecNets: Task-Embedded Control Networks** (James, Bloesch, Davison; CoRL 2018). Learns a metric task embedding from demos, which conditions a policy without gradient steps at test time. Evaluated on reaching and pushing with novel objects, and on sim-to-real placing. [from memory, verify] — [arXiv 1810.03237](https://arxiv.org/abs/1810.03237)
- **Watch-Try-Learn** (Zhou et al., ICLR 2020). Meta-learns to use one demo plus one trial episode with its reward. Gains over pure one-shot imitation on held-out tasks *within* a family of similar gripper tasks. [from memory, verify] — [arXiv 1906.03352](https://arxiv.org/abs/1906.03352)
- **Transformers for One-Shot Visual Imitation** (Dasari & Gupta, CoRL 2020). A transformer attends over demo video features plus inverse-dynamics/self-supervised losses. Tested on held-out object configurations of pick-place in simulation. [from memory, verify] — [arXiv 2011.05970](https://arxiv.org/abs/2011.05970)
- **MOSAIC: Towards More Generalizable One-Shot Visual Imitation** (Mandi, Liu, Lee, Abbeel; ICRA 2022). This paper is notable because it explicitly tests multi-task training and held-out tasks, and reports that held-out-task one-shot imitation remains much harder than held-out variations. [from memory, verify] — [arXiv 2110.13423](https://arxiv.org/abs/2110.13423)
- **OSVI-WM** (2025) makes the same point: existing one-shot visual imitation methods "typically train and evaluate on the same set of tasks." It proposes world-model-guided trajectory generation for semantically or structurally different unseen tasks. — [arXiv 2505.20425](https://arxiv.org/abs/2505.20425)

**Language / multi-context conditioning (closely related, but not demo-conditioned)**
- **MCIL: Language Conditioned Imitation Learning over Unstructured Data** (Lynch & Sermanet; RSS 2021). A single policy is conditioned on either goal images or language and trained from play data. It is zero-shot by instruction, not few-shot by demo. [from memory, verify] — [arXiv 2005.07648](https://arxiv.org/abs/2005.07648)
- **BC-Z** (Jang et al., CoRL 2021). Conditions on language or human video, with about 100 training tasks, and reports zero-shot success on 24 held-out task compositions (verb-object recombinations). The reported average is modest, about 44% by my recollection; verify. — [arXiv 2202.02005](https://arxiv.org/abs/2202.02005)

**Prompted generalist models (2023)**
- **VIMA** (Jiang et al., ICML 2023). Multimodal prompts interleave text, images, and *demonstration frames* (for example "follow this motion") and feed an object-centric transformer. VIMA-Bench has 17 meta-task templates and about 650K expert trajectories, with a 4-level generalization protocol: L1 placement, L2 combinatorial objects, L3 novel task (held-out combinations), L4 novel *task templates*. Performance falls sharply at L4. The paper reports up to about 2.9x improvement over baselines in the hardest zero-shot setting with the same data. [from memory, verify] — [arXiv 2210.03094](https://arxiv.org/abs/2210.03094)
- **RoboCat** (Bousmalis et al., 2023; TMLR). A goal-image-conditioned Gato-style VQ transformer that adapts to new tasks and embodiments from **100-1000 demos via fine-tuning**. It is *not* in-context one-shot. A self-improvement loop generates more data. [from memory, verify] — [arXiv 2306.11706](https://arxiv.org/abs/2306.11706)
- **Gato** (Reed et al., 2022). Prompt conditioning on a demo, but the reported new-task adaptation is through fine-tuning. [from memory, verify] — [arXiv 2205.06175](https://arxiv.org/abs/2205.06175)
- **Prompt-DT** (Xu et al., ICML 2022). Trajectory-segment prompts condition a Decision Transformer. Few-shot generalization is only to *parametric* variants (Cheetah-vel, Ant-dir, Meta-World reach goal positions). [from memory, verify] — [arXiv 2206.13499](https://arxiv.org/abs/2206.13499)

**2024-2026 in-context imitation (the most relevant direction)**
- **Instant Policy** (Vosylius & Johns; arXiv Nov 2024, v2 Apr 2025; I believe it was accepted at ICLR 2025, but the arXiv page states no venue). In-context imitation is posed as **graph generation with a learned diffusion process** over point-cloud/gripper graphs of demos, current observation, and future actions. The policy is trained only on **procedurally generated "pseudo-demonstrations"** (arbitrary trajectories on ShapeNet objects in simulation), and learns new tasks from 1-2 demos with no further training. — [arXiv 2411.12633](https://arxiv.org/abs/2411.12633)
  - RLBench, 24 tasks, 100 rollouts each, N=2 demos. **Pseudo-demos only: 0.71 average success**, versus BC-Z 0.36, Vid2Robot 0.38, GPT2-style 0.34. Adding 20 real demos for 12 tasks (PD++) gives 0.97 on seen tasks and **0.82 on the 12 unseen tasks**, versus 0.31-0.43 for baselines. **[verified]** — [arXiv HTML](https://arxiv.org/html/2411.12633)
  - Real world: 16 tasks, 10 rollouts each, **88.75%** versus 34-41% for baselines. Novel-object generalization rises with demo count: **40% with 1 demo, 52.5% with 2, 85% with 3, 90% with 4.** **[verified]** — [arXiv HTML](https://arxiv.org/html/2411.12633)
  - Training took about 2.5M steps, roughly 700K unique pseudo-trajectories generated on the fly, in **about 5 days on a single RTX 3080-Ti**. Going from 69M to 117M parameters helped significantly, and 178M added little. **[verified]** — [arXiv HTML](https://arxiv.org/html/2411.12633)
- **ICRT: In-Context Imitation Learning via Next-Token Prediction** (Fu, Huang, Datta, Chen, Panitch, Liu, Li, Goldberg; arXiv Aug 2024, reported as ICRA 2025). A causal transformer runs over (image, proprio, action) trajectories, and the prompt is a teleoperated demo of the new task. No language or reward. — [arXiv 2408.15980](https://arxiv.org/abs/2408.15980); venue per [search snippet](https://arxiv.org/html/2408.15980)
  - Data: about 2k DROID trajectories for pretraining, plus ICRT-MT with 1,098 trajectories over 29 tasks and 6 primitives on a Franka. "Unseen tasks" means **new objects with training primitives** (pick-place, poke). **[verified]**
  - Unseen-task success averages **79.2%** (pick-place 65.0%, poke 93.3%), versus goal-conditioned 20.0%, Octo 9.2%, OpenVLA 7.5%. Ablations: DROID-only pretraining gives **0.0%**, MT-only 73.3%, a pretrained Llama2-7B with LoRA 58.3%, and adding a prompt loss 22.5%. **[verified]** — [arXiv HTML](https://arxiv.org/html/2408.15980)
  - The ablations show that in-domain multitask data does most of the work. Large generic pretraining (DROID) alone does not produce ICL.
- **Action Tokenizer Matters in In-Context Imitation Learning** (IROS 2025). Shows that existing action tokenizers break temporal smoothness, which hurts ICL execution, and proposes a better tokenizer. — [arXiv 2503.01206](https://arxiv.org/abs/2503.01206)
- **SimpleICL: In-context Robot Learning Made Simple** (Li et al.; arXiv 2609.38173, Sep/Oct 2026). Uses a visual prompt encoder on top of a **Wan2.2-5B video model backbone** with about a 1B action DiT, plus a low-cost "ICL-oriented" data collection protocol. **[verified]** — [arXiv HTML](https://arxiv.org/html/2609.38173v1)
  - RoboTwin 2.0 simulation, 18K ICL-oriented demos. Seven **novel OOD tasks** reach **65.9%**, versus Fast-WAM 24.6%; removing video conditioning gives 26.3%. On a visual-ambiguity test, ICL-oriented data scores 72.1% versus 40.3% for native RoboTwin data. **[verified]**
  - Real Franka unseen tasks (easy/medium/hard): 81/70/68%, versus Fast-WAM 80/65/42% and π0.5 72/47/31%. **[verified]**
  - Main lesson: *how demos are paired* (prompt and query from the same task in different scenes, with distractors) matters more than raw data volume. This is a single-paper claim.
- **MT3: Learning a Thousand Tasks in a Day** (Dreczkowski, Vitiello, Vosylius, Johns; *Science Robotics* 2025). Splits each trajectory into an alignment phase and an interaction phase and uses retrieval plus replay. **1,000 real tasks taught with 1 demo each in under 24 h** of demonstrator time, across 3,450 + 2,200 real rollouts. Decomposition gives about an **order-of-magnitude data-efficiency gain** below 10 demos per task, and retrieval beats BC in both phases. It generalizes to novel object *instances*. The abstract does not claim novel task *types*. **[verified]** — [arXiv 2511.10110](https://arxiv.org/abs/2511.10110)
- **Annotation-Free One-Shot Imitation for Multi-Step Manipulation** (Sep 2025). A trajectory-replay method that uses one demo and no further training. — [arXiv 2509.24972](https://arxiv.org/abs/2509.24972)
- **MimicDroid** (2025). In-context learning for humanoid manipulation from human play videos. — [arXiv 2509.09769](https://arxiv.org/pdf/2509.09769)

### Inferences
- There are three families of approach that work:
  - (a) **Structure-heavy in-context policies**: Instant Policy's graph diffusion, and Johns-lab alignment-and-replay (MT3, DINOBot, DOME).
  - (b) **Sequence-model ICL** on multitask teleop data (ICRT, SimpleICL, VIMA).
  - (c) **Retrieval / nearest-neighbor replay**.
  For 1-2 demos on tasks that are basically "reach this object, then replay this interaction," (a) and (c) are the strongest and cheapest.
- "Held-out task" usually means a new object/primitive combination, or a new RLBench task that uses the same pick/place/push motor vocabulary. No paper I found shows one-shot acquisition of a new *motor skill* (for example, peg-upright reorientation when the training set has only pick/push).
- Instant Policy matters most for AutoManip. It runs on one consumer GPU in about 5 days, trains on *procedurally generated* arbitrary trajectories rather than curated tasks, and is the clearest evidence that task breadth can be synthesized cheaply in simulation.

### Gaps
- No independent replication of the Instant Policy or ICRT numbers was found. Both come from the authors' own labs.
- I could not verify exact numbers for MIL, TecNets, DAML, Watch-Try-Learn, VIMA, or BC-Z this session, because I did not open those pages. The descriptions are from memory and should be checked before quoting.
- I did not check newer VLA-plus-ICL work such as RICL (Sridhar et al., 2025; adds in-context retrieval to a pretrained VLA). It is not verified.

---

## 2. In-context RL: Algorithm Distillation, DPT, AdA, RL², VariBAD. What pretraining scale is needed, and what is the evidence in continuous control and manipulation?

### Takeaway
In-context RL is well established in discrete or low-dimensional settings: bandits, gridworlds, Dark Room, Watermaze, and XLand. Evidence in manipulation is limited to Meta-World and is weak at the ML45 held-out-task level. The best 2026 result (Vintix II) uses 209 training tasks and about 700M transitions, and still treats ML45 as hard. ICL appears to need **hundreds of tasks** or very large procedural task spaces. A handful of tasks gives only memorization or task identification.

### Cited Findings
- **RL²** (Duan et al., 2016). An RNN policy with reward and done signals in its input is meta-trained across MDPs, so learning happens in the hidden state. Shown on bandits, tabular MDPs, and visual navigation. [from memory, verify] — [arXiv 1611.02779](https://arxiv.org/abs/1611.02779)
- **VariBAD** (Zintgraf et al., ICLR 2020). A VAE-based belief over the task with a Bayes-adaptive policy. Strong on MuJoCo parametric variations (direction and velocity), which are **not** distinct manipulation tasks. [from memory, verify] — [arXiv 1910.08348](https://arxiv.org/abs/1910.08348)
- **Algorithm Distillation** (Laskin et al., ICLR 2023). A causal transformer is trained on *multi-episode learning histories* of RL algorithms and learns in context, improving across episodes with frozen weights. Shown on Dark Room, Dark Key-to-Door, and DMLab Watermaze, all with discrete actions and simple goal-location task families. It needs long across-episode contexts and the source algorithm's learning histories. [from memory, verify] — [arXiv 2210.14215](https://arxiv.org/abs/2210.14215)
- **DPT: Decision-Pretrained Transformer** (Lee et al., NeurIPS 2023). Supervised pretraining to predict the *optimal* action given a query state and an in-context dataset. Shown to perform posterior-sampling-like exploration and offline conservatism on bandits, Dark Room, and Miniworld. The search did not surface manipulation results in the original paper. — [Liner review of DPT](https://liner.com/review/supervised-pretraining-can-learn-incontext-reinforcement-learning); [arXiv 2306.14892](https://arxiv.org/abs/2306.14892)
- **AdA: Human-Timescale Adaptation in an Open-Ended Task Space** (Adaptive Agent Team, DeepMind; ICML 2023). RL² at scale with a transformer memory, automatic curriculum, and distillation, in the procedurally generated 3D XLand 2.0 task space. Reported to adapt within a few episodes on held-out tasks, with performance scaling with model size, memory length, and task-pool richness. [from memory, verify] — [arXiv 2301.07608](https://arxiv.org/abs/2301.07608)
- **Meta-World ML45** (Yu et al., CoRL 2019) has 45 training tasks and 5 held-out test tasks, with no task IDs. Original baselines struggled even on multi-task learning with about 10 tasks. One secondary source reports 39.9% held-out success for MAML on ML45, which needs checking against the paper. — [Meta-World PMLR](https://proceedings.mlr.press/v100/yu20a); [benchmark descriptions](https://metaworld.farama.org/benchmark/benchmark_descriptions/); MAML figure via [alphaxiv summary](https://alphaxiv.org/overview/1910.10897v2)
- **Vintix II** (Polubarov et al.; ICLR 2026 poster). A DPT-style agent with a flow-matching action head, about 928M parameters, trained on **209 tasks across 10 domains** (Meta-World 45, Bi-DexHands 15, MuJoCo 11, and others), **3.8M episodes, about 709.7M transitions**. There are 46 held-out tasks. **[verified]** — [arXiv HTML 2604.05112](https://arxiv.org/html/2604.05112)
  - Against the AD-based Vintix on held-out tasks: **+63% scaled return on Meta-World ML45**, +17% on Bi-DexHands, +4% on MuJoCo. It beats REGENT by 8.2% normalized return on 5 unseen ML45 tasks. It reaches 85% normalized score on ML1 goal variations. **[verified]**
  - Cold-start online (no prompt) matches the prompted version in 8 of 10 domains but **lags on Meta-World ML45 and Bi-DexHands**, the manipulation domains. The prompt is 500-4,000 *transitions* (dozens of episodes), not 1-2 demos. The context of 4096 tokens holds about 40 Meta-World episodes. **[verified]**
- **REGENT** (Sridhar et al., ICLR 2025). A retrieval-augmented transformer of 138.6M parameters, pretrained on about 14.5M transitions (45 Meta-World, 9 MuJoCo, 52 Atari, 39 BabyAI environments). It generalizes in context to 5 unseen Meta-World environments using retrieval over 25-100 demos. **The parameter-free "Retrieve-and-Play" 1-nearest-neighbor baseline is described as "surprisingly strong"** on unseen Meta-World. JAT/Gato fails to generalize to most unseen environments even after fine-tuning. **[verified]** — [arXiv HTML 2412.04759](https://arxiv.org/html/2412.04759); [ICLR poster](https://iclr.cc/virtual/2025/poster/29847)
- **Related scaling evidence on ICL emergence** (supervised and toy settings; from memory, verify):
  - Kirsch et al. (2022) find a phase transition from memorization to general in-context learning as the number of meta-training tasks grows, on the order of thousands of tasks in their setup. — [arXiv 2212.04458](https://arxiv.org/abs/2212.04458)
  - Raventós et al. (NeurIPS 2023) find a task-diversity threshold for ICL in linear regression. — [arXiv 2306.15063](https://arxiv.org/abs/2306.15063)
  - Raparthy et al. (2023) find that trajectory burstiness and environment diversity drive ICL emergence in sequential decision-making. — [arXiv 2312.03801](https://arxiv.org/abs/2312.03801)
- **Related infrastructure** (from memory, verify):
  - XLand-MiniGrid (Nikulin et al., 2023) is a JAX meta-RL benchmark built for ICRL scaling studies. — [arXiv 2312.12044](https://arxiv.org/abs/2312.12044)
  - XLand-100B (2024) is a large in-context RL dataset. — [arXiv 2406.08973](https://arxiv.org/abs/2406.08973)
  - AMAGO and AMAGO-2 (Grigsby et al., ICLR 2024 / NeurIPS 2024) cover long-context transformer meta-RL, and AMAGO-2 reports that Meta-World ML45 multi-task meta-RL is limited by return-scale imbalance. — [arXiv 2411.11188](https://arxiv.org/abs/2411.11188)

### Inferences
- Classic in-context RL (AD, DPT) adapts from *many* episodes of interaction or large transition prompts, which is not the "show it once or twice" regime. For 1-2 demos, demo-conditioned imitation (Section 1) is a better fit than ICRL. A DPT-style objective (predict the expert action from a context dataset) applied to a context of 1-2 expert demos is essentially in-context imitation.
- Evidence suggests that held-out-task ICL in manipulation needs at least hundreds of procedurally distinct tasks. With 4-5 ManiSkill tasks, AutoManip can only test held-out *variations*, unless it builds a procedural task generator.
- REGENT's strong nearest-neighbor result means any AutoManip ICL experiment must include a nearest-neighbor or replay baseline. Otherwise "ICL" gains may just be retrieval.

### Gaps
- I found no in-context RL result on ManiSkill, and no published ICRL result on contact-rich state-based manipulation from 1-2 demos.
- I did not extract exact absolute ML45 held-out success rates for Vintix II or REGENT, because those sit in appendices that were not read.

---

## 3. LLM/VLM-as-policy in context

### Takeaway
Off-the-shelf text LLMs can do few-shot imitation once observations and actions are tokenized as numbers or keypoints (KAT, RoboPrompt, Mirchandani). Results are low-data and real-world but small-scale, slow (API calls), and limited to short-horizon, keypoint-describable tasks. These are single-lab results.

### Cited Findings
- **Keypoint Action Tokens** (Di Palo & Johns; RSS 2024). Visual keypoints from DINO features and end-effector action trajectories are serialized as text tokens. **GPT-4 Turbo** then maps a new observation to an action sequence in context, with no training. It performs "on par or better than" Diffusion Policy in the low-data regime on real everyday tasks (about 10 demos in the paper, from memory). **[verified except demo count]** — [arXiv 2403.19578](https://arxiv.org/abs/2403.19578)
- **Large Language Models as General Pattern Machines** (Mirchandani et al.; CoRL 2023). LLMs complete abstract token sequences (ARC-style), extrapolate simple motion sequences, and do crude in-context reward-improvement ("sequence improvement") on CartPole-like control and a real robot wiping task. The authors frame this as a proof of concept, not a practical controller. [from memory, verify] — [arXiv 2307.04721](https://arxiv.org/abs/2307.04721)
- **RoboPrompt: In-Context Learning Enables Robot Action Prediction in LLMs** (Yin et al., 2024; ICRA 2025 per my memory). Textual keyframe states and actions are used as ICL examples for an off-the-shelf LLM, and it beats zero-shot and ICL baselines in simulation and the real world. [from memory, verify] — [arXiv 2410.12782](https://arxiv.org/abs/2410.12782)

### Inferences
- For a state-based ManiSkill loop, an LLM-as-policy baseline is cheap to build: serialize object poses and the tool center point (TCP) pose, plus a demo trajectory. But latency makes closed-loop control at 20+ Hz impractical. It is best used as a keyframe/waypoint generator, which is a cheap baseline but not a core research direction.

### Gaps
- I found no rigorous held-out-task benchmark comparison of LLM-as-policy against trained ICL policies on a shared benchmark.

---

## 4. Task-distribution breadth, data volume, and benchmarks used for held-out-task evaluation

### Takeaway
The standard held-out-task protocols are:
- **Meta-World ML10/ML45**: state-based, 5 held-out tasks.
- **RLBench**: Instant Policy uses 24 tasks.
- **VIMA-Bench L1-L4**: L4 is novel task templates.
- **RoboTwin 2.0**: SimpleICL uses 7 OOD tasks.
- **LIBERO**: mainly lifelong and multitask learning, and sometimes used for held-out suites.

ManiSkill has no standard held-out-task ICL protocol. Data volumes in the successful 1-2-demo methods range from about 1k real trajectories (ICRT-MT) to about 700K pseudo-trajectories (Instant Policy), 650K (VIMA), and about 700M transitions (Vintix II).

### Cited Findings
- **Breadth and volume reported:**
  - Instant Policy: about 700K procedurally generated pseudo-trajectories, no curated tasks. **[verified]** — [arXiv HTML](https://arxiv.org/html/2411.12633)
  - ICRT: 29 tasks, 1,098 trajectories, and DROID-only pretraining gives 0% on unseen tasks. **[verified]** — [arXiv HTML](https://arxiv.org/html/2408.15980)
  - SimpleICL: 18K ICL-oriented demos on RoboTwin. **[verified]** — [arXiv HTML](https://arxiv.org/html/2609.38173v1)
  - MT3: 1,000 tasks with 1 demo each. **[verified]** — [arXiv](https://arxiv.org/abs/2511.10110)
  - Vintix II: 209 tasks, about 709.7M transitions. **[verified]** — [arXiv HTML](https://arxiv.org/html/2604.05112)
  - REGENT: 145 environments, 14.5M transitions. **[verified]** — [arXiv HTML](https://arxiv.org/html/2412.04759)
- **Meta-World ML45**: 45 train and 5 held-out tasks with no task IDs, and ML1 covers within-task goal variation. — [Meta-World docs](https://metaworld.farama.org/benchmark/benchmark_descriptions/)
- **RLBench** (James et al., RA-L 2020) has 100+ tasks and is used by Instant Policy (24 tasks, 12 unseen in PD++). — [arXiv 1909.12271](https://arxiv.org/abs/1909.12271); [Instant Policy HTML](https://arxiv.org/html/2411.12633)
- **LIBERO** (Liu et al., NeurIPS 2023 Datasets & Benchmarks) has 130 tasks in suites (Spatial, Object, Goal, Long/LIBERO-90) and is mostly used for multitask and lifelong learning, not ICL. [from memory, verify] — [arXiv 2306.03310](https://arxiv.org/abs/2306.03310)
- **ManiSkill3** (Tao et al., 2024) offers GPU-parallelized simulation with many tasks and is used for RL/IL baselines. No meta or ICL split is defined. [from memory, verify] — [arXiv 2410.00425](https://arxiv.org/abs/2410.00425)

### Inferences
- Reported numbers are not comparable across papers, because each defines "unseen task" differently: new object, new object-primitive combination, new RLBench task with shared primitives, or a new template. AutoManip should state its held-out level explicitly, for example with a VIMA-like L1-L4 taxonomy.

### Gaps
- I did not obtain per-benchmark leaderboard tables (for example, every method's ML45 held-out success) from a single trusted source.

---

## 5. Cheapest credible first experiment in ManiSkill (state / low-dim observations)

### Takeaway
The recommendation is a state-based **demo-conditioned transformer, in the style of DPT, ICRT, or Instant Policy**, trained on a *procedurally generated family* of ManiSkill pick/push/place variants, using scripted or motion-planned experts so no RL is needed per task. Evaluate it on held-out task *compositions*, and always include a Retrieve-and-Play / replay baseline and an oracle task-ID policy. This fits on one L40S in hours to about a day. The reasoning for this rests on the evidence above (an inference, not a published recipe).

### Cited Findings (supporting evidence)
- Procedural pseudo-demos alone gave 0.71 on 24 RLBench tasks, trained on one consumer GPU in about 5 days. **[verified]** — [Instant Policy HTML](https://arxiv.org/html/2411.12633)
- Small in-domain multitask data (29 tasks, about 1k trajectories) gives held-out-object/primitive ICL, and a 12-layer model trained from scratch outperforms a LoRA-tuned 7B. **[verified]** — [ICRT HTML](https://arxiv.org/html/2408.15980)
- A nearest-neighbor Retrieve-and-Play baseline is "surprisingly strong" on unseen Meta-World. **[verified]** — [REGENT HTML](https://arxiv.org/html/2412.04759)
- Prompt and query pairing (same task, different scene, with distractors) is critical: ICL-oriented data scores 72.1% versus 40.3% for native data. **[verified]** — [SimpleICL HTML](https://arxiv.org/html/2609.38173v1)

### Inferences (proposed design, speculative)
1. **Task generator.** Parameterize ManiSkill scenes with 2-4 cubes of different colors plus target regions or spheres. A "task" is a program such as pick(X) → place-on(Y) or push(X → region R) or stack(X on Y), with ordering. The demo reveals *which* object and *which* target. Without the demo, the task is ambiguous given the observation, because the observation includes all objects. This ambiguity is essential: without it, a goal-free policy can solve the task and the demo carries no information.
2. **Experts.** Use ManiSkill motion-planning solutions or scripted end-effector controllers to generate demos cheaply on CPU or GPU. This keeps PPO/TD3 out of the inner loop and makes the experiment cheap.
3. **Model.** A roughly 10-50M-parameter transformer takes 1-2 demo trajectories (state, action, possibly subsampled) plus the current state, and predicts an action chunk. Train it with BC/MSE or a diffusion/flow head.
4. **Splits.** Use L1 (new poses), L2 (new object/target identities), L3 (held-out *compositions*, for example "stack" with color pairs never seen together), and L4 (a held-out primitive). Expect L4 to fail, since the literature shows none succeeding.
5. **Baselines.**
   - (a) Nearest-neighbor action replay from the demo (R&P).
   - (b) Demo-pose-aligned trajectory replay (MT3/DINOBot-style).
   - (c) Oracle task-ID or goal-conditioned multitask policy (upper bound).
   - (d) Demo-free policy (lower bound, which measures the ambiguity).
6. **Metrics.** Strict pick/place/release success (distinct from stock ManiSkill success, per AGENTS.md), with paired seeds and held-out evaluation seeds.
7. **Scaling probe.** Plot held-out success against the number of training task programs (for example 8, 32, 128, 512) to look for the task-diversity threshold reported in Kirsch 2022 and Raventós 2023.
8. **Data and compute guess (unverified estimate).** About 100K-1M state-based demos from GPU-parallel ManiSkill, with training feasible in a few L40S-hours. These figures are scaled down from ICRT and Instant Policy, not measured.

### Gaps
- No published ManiSkill ICL/meta-imitation protocol exists to compare against, so all baselines must be built in-house.
- Whether state-based observations make ICL *easier* or *trivially solvable by retrieval* is unknown. The R&P baseline is the guard against this.
