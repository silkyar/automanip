# Robot Foundation Models (VLAs / Generalist Policies) and World Models: Few-Shot Adaptation to New Manipulation Tasks

Scope note: research conducted 2026-11-01. Items marked **[verified]** were checked this session against the paper's full-text HTML or arXiv abstract page (via fetch). Items marked **[abstract/recall]** are cited to the paper's arXiv page and come from the paper's abstract as recalled by the researcher, but the specific number was not re-fetched this session. The report writer should treat [abstract/recall] numbers as needing a spot-check before quoting exactly. Nearly all headline results below are **author-reported (often company-internal) evaluations**, not independent replications, unless stated otherwise.

AutoManip context: ManiSkill3 sim, state-based single-task policies trained from scratch (TD3: 80% StackCube in about 3.6M transitions), single L40S-class GPU, campaigns costing a few hundred dollars.

---

## Q1. VLAs and generalist policies: demos needed to fine-tune to a new task, success rates, fine-tuning cost, zero-shot new-task results

### Takeaway
No current generalist policy reliably does "show a new task once or twice and it works" in the general case. Across RT-2, Octo, OpenVLA, pi0, GR00T N1 and Gemini Robotics, the practical regime for a genuinely new task is still **about 10 to 100+ demonstrations** (Gemini Robotics: 7 of 8 tasks over 70% with at most 100 demos; OpenVLA: 10 to 150 demos per task; pi0: 1 to 5+ hours of data for simple tasks, 100+ hours for complex ones). Pretraining mainly buys **data efficiency** (often around 2x, or the same success with about 10x less data) and **robustness to distractors and language variation**, not true one-shot learning. Zero-shot "new task" success happens mostly when the task is a recombination of skills already in pretraining (new objects or instructions, short-horizon). Real one-/few-shot results (RDT 1 to 5 demos; ViVLA one video; in-context 10 demos for Gemini Robotics-ER) are early, author-reported, and narrow.

### Cited Findings

**RT-1 / RT-2 / Open X-Embodiment (Google, 2022–2023)**
- RT-1 (Brohan et al., 2022): Transformer trained on about 130k real episodes covering 700+ instructions from 13 robots over 17 months. Reports 97% success on seen tasks and 76% on unseen instructions. [abstract/recall] — [RT-1, arXiv:2212.06817](https://arxiv.org/abs/2212.06817)
- RT-2 (Brohan et al., 2023): co-fine-tunes 12B/55B VLMs (PaLI-X, PaLM-E) on robot actions as text tokens. In about 6k eval trials, generalization to unseen objects, backgrounds and instructions rose from about 32% (RT-1) to about 62%. It also showed emergent semantic reasoning, e.g. "pick up the extinct animal." This is zero-shot generalization to new *instructions and objects*, not to new motor skills. [abstract/recall] — [RT-2, arXiv:2307.15818](https://arxiv.org/abs/2307.15818)
- Open X-Embodiment / RT-X (Open X-Embodiment Collaboration, 2023): pooled 1M+ trajectories from 22 robot embodiments. The cross-embodiment RT-1-X beat the original per-lab methods by about 50% on average across 5 labs. [abstract/recall] — [OXE, arXiv:2310.08864](https://arxiv.org/abs/2310.08864)

**Octo (Octo Model Team: Ghosh, Walke, Pertsch, Black, Mees, ..., Finn, Levine; 2024)**
- Pretrained on 800k trajectories from Open X-Embodiment. Fine-tunes to new observation and action spaces "within a few hours on standard consumer GPUs," evaluated on 9 robot platforms. [verified] — [Octo, arXiv:2405.12213](https://arxiv.org/abs/2405.12213)
- Model sizes are 27M (Small) and 93M (Base). Fine-tuning used about 100 target-domain demos and reportedly beat training from scratch by about 52% on average. [abstract/recall; not re-verified this session] — [Octo project/paper](https://arxiv.org/abs/2405.12213)
- Independent signal: in Meta's V-JEPA 2 zero-shot Franka evaluation, Octo scored 15% on cup grasp, 0% on box grasp, and 10–15% on pick-and-place (vs. 100% on reach). Off-the-shelf generalist policies fail on simple new-lab pick-and-place without fine-tuning. [verified] — [V-JEPA 2, arXiv:2506.09985](https://arxiv.org/html/2506.09985)

**OpenVLA (Kim, Pertsch, Karamcheti, et al.; 2024)**
- 7B parameters (Prismatic VLM, Llama 2 backbone) trained on 970k OXE trajectories. Pretraining used 64 A100s for 14 days (about 21,500 A100-hours). [verified] — [OpenVLA, arXiv:2406.09246](https://arxiv.org/html/2406.09246)
- Beats RT-2-X (55B) by 16.5% absolute success across 29 tasks on WidowX and Google Robot. [verified] — [OpenVLA](https://arxiv.org/html/2406.09246)
- Fine-tuning: new Franka tasks with **10–150 demos each**. OpenVLA had the best average and was the only method at 50% or more on every task. Diffusion Policy from scratch matched or beat the generalists on narrow single-instruction tasks; the generalists did better on diverse multi-instruction tasks. [verified] — [OpenVLA](https://arxiv.org/html/2406.09246)
- Fine-tuning cost: LoRA (rank 32, about 1.4% of params) matched full fine-tuning (68.2% vs. 69.7%). It takes **10–15 hours on a single A100**, about 8x cheaper than full fine-tuning. [verified] — [OpenVLA](https://arxiv.org/html/2406.09246)

**OpenVLA-OFT (Kim, Finn, Liang; RSS 2025)**
- Its recipe (parallel decoding, action chunking, continuous actions, L1 regression) raised the LIBERO 4-suite average from 76.5% (fine-tuned OpenVLA) to **97.1%**, with 26x faster action generation. Per-suite (secondary summary): Spatial 97.6, Object 98.4, Goal 97.9, Long 94.5. [abstract verified; per-suite numbers from a secondary summary] — [OpenVLA-OFT, arXiv:2502.19645](https://arxiv.org/abs/2502.19645); [project page](https://openvla-oft.github.io)
- Caveat (inference): LIBERO fine-tuning uses about 50 human demos per task (the LIBERO default, see Q3), so "97%" is a ~50-demo in-distribution result, not few-shot. LIBERO is close to saturated.

**pi0 / pi0-FAST / pi0.5 / pi*0.6 (Physical Intelligence, 2024–2025)**
- pi0 (Black et al., 2024): about 3.3B params (3B PaliGemma plus a ~300M flow-matching action expert). Pretrained on **over 10,000 hours** of robot data (903M timesteps of in-house data; 9.1% of the mixture is open-source), for 700k steps. No GPU-hour figure is reported. Runs at 73 ms per inference on an RTX 4090. [verified] — [pi0, arXiv:2410.24164](https://arxiv.org/html/2410.24164)
- pi0 fine-tuning: post-training sets range from **about 5 hours for the simplest tasks to 100+ hours for complex ones**. A Tupperware-in-microwave task was tested with 5-hour and 1-hour datasets. Pretraining helps "by as much as 2x" over from-scratch. The strongest baselines were from-scratch ACT and Diffusion Policy, not Octo or OpenVLA. [verified] — [pi0](https://arxiv.org/html/2410.24164)
- pi0 zero-shot (out of the box, on tasks in the pretraining distribution): "near perfect" on shirt folding and easier bussing. [verified] — [pi0](https://arxiv.org/html/2410.24164)
- pi0-FAST (Pertsch et al., 2025): DCT-based "FAST" action tokenization lets autoregressive VLAs match diffusion pi0, with up to 5x less training time. [abstract/recall] — [FAST, arXiv:2501.09747](https://arxiv.org/abs/2501.09747)
- pi0.5 (Physical Intelligence, 2025): co-training on heterogeneous data (web, multi-robot, high-level subtask prediction) enables long-horizon cleaning in entirely unseen homes. Generalization is to new *environments* for trained task families. [abstract/recall] — [pi0.5, arXiv:2504.16054](https://arxiv.org/abs/2504.16054)
- pi*0.6 with RECAP (Physical Intelligence, Nov 2025): fine-tune on demos, then run rounds of on-robot data collection with offline RL (advantage-conditioned via a learned value function). On the hardest tasks it more than doubles throughput and roughly halves the failure rate. Reported deployments include 13 h of continuous espresso making, laundry folding in a new home, and factory box assembly. Company claims. [verified, abstract-level] — [pi*0.6, arXiv:2511.14759](https://arxiv.org/abs/2511.14759)
- A secondary source claims base pi0.6 handles "new short-horizon tasks out of the box." This was not verified against a primary source. — [secondary page](https://web-3kk.pages.dev/en/knowledge/models/pi0-6-star)

**RDT-1B (Liu et al., Tsinghua, 2024)**
- 1.2B-param diffusion transformer for bimanual manipulation, pretrained on about 46 datasets (1M+ episodes) and fine-tuned on a 6K+ episode ALOHA dataset. It claims zero-shot generalization to unseen objects and scenes and **learning new skills from 1–5 demonstrations**. Author-reported, real-robot, small trial counts. [abstract/recall] — [RDT-1B, arXiv:2410.07864](https://arxiv.org/abs/2410.07864)

**GR00T N1 (NVIDIA: Bjorck, Castañeda, ..., Fan, Zhu; Mar 2025)**
- 2.2B params (1.34B in the VLM), with a System-2 VLM plus a System-1 diffusion transformer. Pretraining took about 50,000 H100 GPU-hours (up to 1,024 GPUs). [verified] — [GR00T N1, arXiv:2503.14734](https://arxiv.org/html/2503.14734v2)
- **Success vs. demos per task (sim, Table 4; author-run):** [verified] — [GR00T N1](https://arxiv.org/html/2503.14734v2)

| Benchmark | Model | 30 demos | 100 demos | 300 demos |
|---|---|---|---|---|
| RoboCasa (24 tasks) | Diffusion Policy | 14.7% | 25.6% | 43.2% |
| RoboCasa (24 tasks) | GR00T N1-2B | 17.4% | 32.1% | 49.6% |
| DexMimicGen (9 tasks) | Diffusion Policy | 23.7% | 46.9% | 68.4% |
| DexMimicGen (9 tasks) | GR00T N1-2B | 29.6% | 58.5% | 74.2% |
| GR-1 Tabletop (24 tasks) | Diffusion Policy | 21.3% | 32.7% | 40.4% |
| GR-1 Tabletop (24 tasks) | GR00T N1-2B | 43.2% | 50.0% | 49.3% |

- Real GR-1 humanoid: with 10% of the data, N1 scored 42.6% vs. Diffusion Policy 10.2%. With full data, 76.8% vs. 46.4%. [verified] — [GR00T N1](https://arxiv.org/html/2503.14734v2)
- Synthetic data: about 827 h of "neural trajectories" generated from 88 h of teleop (about 105k L40 GPU-hours), plus 780k DexMimicGen sim trajectories (about 6,500 h equivalent) generated in 11 hours. [verified] — [GR00T N1](https://arxiv.org/html/2503.14734v2)
- GR00T N1.6 (later 2025) is a 3B base model with embodiment-specific fine-tuned variants (e.g., Unitree G1). No benchmark numbers were found this session. — [GR00T repo (community fork listing)](https://github.com/SLCZ687/GR00T4me)

**Gemini Robotics (Google DeepMind, Gemini Robotics Team; Mar 2025), Gemini Robotics 1.5 (Oct 2025)**
- Abstract claim: learns new short-horizon tasks from **as few as 100 demonstrations**. [verified] — [Gemini Robotics, arXiv:2503.20020](https://arxiv.org/abs/2503.20020)
- Fast adaptation (Sec. 4.3): **7 of 8 sub-tasks exceeded 70% success with at most 100 demos**, and two reached 100%. Baselines (a pi0 re-implementation and diffusion) were competitive on easier tasks. Internal evaluation. [verified] — [Gemini Robotics HTML](https://arxiv.org/html/2503.20020)
- Specialization (Sec. 4.1): 6 long-horizon dexterous tasks, each fine-tuned on **2,000–5,000 episodes**, averaged 79% success. A from-scratch specialist scored 0% on all of them. [verified] — [Gemini Robotics HTML](https://arxiv.org/html/2503.20020)
- New embodiment: bi-arm Franka at 63% average on in-distribution tasks, about on par with a single-task diffusion policy. Apollo humanoid results are qualitative only. [verified] — [Gemini Robotics HTML](https://arxiv.org/html/2503.20020)
- **In-context few-shot (closest to "show it once or twice"):** Gemini Robotics-ER (the reasoning VLM generating trajectories, *not* the end-to-end VLA), given 10 demonstrations in context on ALOHA 2, reached 65% average in sim and 65% in real (banana handover 70%, fold dress 56%, wiping 67%). Zero-shot code generation reached 53% in sim and 25% in real. [verified] — [Gemini Robotics HTML](https://arxiv.org/html/2503.20020)
- Gemini Robotics 1.5: a "Motion Transfer" mechanism claimed to enable zero-shot skill transfer across embodiments. A secondary report says ER 1.5 plus the VLA reaches about 80% long-horizon progress, roughly 2x the VLA alone (unverified). — [Gemini Robotics 1.5, arXiv:2510.03342](https://arxiv.org/abs/2510.03342)

**Small and open VLAs / one-shot work (2025–2026)**
- SmolVLA (Shukor et al., Hugging Face, 2025): about 450M params, trained on community LeRobot datasets, trainable on a single GPU and deployable on consumer hardware. [abstract/recall] — [SmolVLA, arXiv:2506.01844](https://arxiv.org/abs/2506.01844)
- ViVLA (Dec 2025): learns a new task from **one expert demo video** at test time without gradient updates. Reports over 30% gains on unseen LIBERO tasks and over 38% on real Franka with human videos; removing the latent-action module drops unseen-task success from 0.71 to 0.48. Numbers are from secondary summaries. — [ViVLA, arXiv:2512.07582](https://arxiv.org/html/2512.07582)
- MoS-VLA (Oct 2025): one-shot skill adaptation via convex optimization over learned basis policies. Reports lower action error on unseen datasets (offline metric). — [MoS-VLA summary](https://newx.sg/paper/detail/004cee3c-f824-11f0-b84c-00163e10baa7)
- VLBiMan (ICLR 2026): one-demo bimanual manipulation via primitive decomposition and VLM anchoring. Structured, not end-to-end VLA. — [VLBiMan, arXiv:2509.21723](https://arxiv.org/pdf/2509.21723v3)

### Inferences
- The honest 2026 frontier: **about 50–100 demos for robust (>70%) success on a new short-horizon task** with a large pretrained VLA (Gemini Robotics, GR00T, OpenVLA). One- to five-shot results exist (RDT, ViVLA, Gemini-ER in-context) but are narrow, author-reported, and often rely on the task being close to pretraining.
- Pretraining gains are largest at low data and shrink at high data. At 300 demos the GR00T vs. DP gap is about 6 points on RoboCasa, and on GR-1 Tabletop GR00T does not improve from 100 to 300 demos. A well-tuned from-scratch specialist stays a strong baseline for narrow tasks (OpenVLA's own finding).
- RL post-training (pi*0.6 RECAP; RLinf/RL4VLA in ManiSkill, see Q4) is a growing second axis: demos give the starting point, autonomous RL closes the gap. That fits AutoManip's existing RL strengths.

### Gaps
- No independent replication of Gemini Robotics, pi0.5/pi0.6, or GR00T N1.6 few-shot numbers was found. They are closed or partially open and evaluated in-house.
- Exact success-vs-demos curves at 5 and 20 demos for Gemini Robotics are in Fig. 26, which was not retrievable as text.
- Octo's "52% over scratch, ~100 demos" and RDT's "1–5 demos" figures were not re-fetched this session.
- No reliable GPU-hour cost was found for pi0 fine-tuning; the paper reports only steps.
- No official GR00T N1.6 or Gemini Robotics 1.5 quantitative few-shot benchmarks were found.

---

## Q2. World models: DreamerV3, TD-MPC2 multi-task, video world models (UniSim, Genie, Cosmos, V-JEPA 2). Do multi-task world models adapt to new tasks with few samples?

### Takeaway
Latent world-model RL agents (DreamerV3, TD-MPC2) are sample-efficient and fixed-hyperparameter, but "few-shot" means **about 20k environment steps of online RL**, not a handful of demos. TD-MPC2's multi-task pretraining gives about 2x the performance of from-scratch at 20k steps on held-out tasks. Large video world models (UniSim, Genie, Cosmos) are mostly data generators or evaluators so far. V-JEPA 2-AC is the strongest evidence for **zero-shot, goal-image planning** of simple pick-and-place with no task data (65–80%), at a cost of 16 s per action.

### Cited Findings
- DreamerV3 (Hafner, Pasukonis, Ba, Lillicrap; 2023, later in Nature 2025): one fixed hyperparameter configuration across 150+ tasks in diverse domains. It was the first to collect diamonds in Minecraft from scratch without human data. A single-task method, with no multi-task few-shot transfer claim. [abstract/recall] — [DreamerV3, arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
- TD-MPC2 (Hansen, Su, Wang; ICLR 2024): evaluated on 104 online RL tasks across 4 domains with one hyperparameter set. A single **317M-param agent covers 80 tasks** across embodiments and action spaces. [verified] — [TD-MPC2, arXiv:2310.16828](https://arxiv.org/abs/2310.16828)
- The TD-MPC2 317M multitask model trained in about **33 GPU-days on a single RTX 3090**, on about 545M transitions taken from the replay buffers of 240 single-task agents. [verified] — [TD-MPC2 HTML](https://arxiv.org/html/2310.16828)
- **TD-MPC2 few-shot transfer:** a 19M-param agent pretrained on 70 DMControl and Meta-World tasks was fine-tuned with online RL on 10 held-out tasks. With an empty replay buffer and unchanged hyperparameters, it reached **about 2x from-scratch performance at 20k env steps** (20 episodes in DMControl, 100 in Meta-World). [verified] — [TD-MPC2 HTML](https://arxiv.org/html/2310.16828)
- Related: Hansen et al. maintain ManiSkill2 tasks in the TD-MPC2 benchmark. TD-MPC2 is from the same group (Hao Su) as ManiSkill. [inference from shared authorship; ManiSkill3 authors include Hao Su — [ManiSkill3, arXiv:2410.00425](https://arxiv.org/abs/2410.00425)]
- UniSim (Yang, Du, Ghasemipour, Tompson, Kaelbling, Schuurmans, Abbeel; 2023, ICLR 2024): an action-conditioned video generative simulator. Policies trained purely inside it transferred zero-shot to real robots. [abstract/recall] — [UniSim, arXiv:2310.06114](https://arxiv.org/abs/2310.06114)
- Genie (Bruce et al., DeepMind 2024): an 11B "foundation world model" learned from unlabeled internet gameplay video with a latent action model. It demonstrates that latent actions learned from video can be used to train policies (robotics demonstrations are limited). [abstract/recall] — [Genie, arXiv:2402.15391](https://arxiv.org/abs/2402.15391)
- Cosmos (NVIDIA 2025): a "world foundation model" platform (diffusion and autoregressive video models) for physical AI, released with open weights. The underlying video corpus is reportedly about 20M hours. [abstract/recall; video-hour figure not re-verified] — [Cosmos, arXiv:2501.03575](https://arxiv.org/abs/2501.03575)
- As a robot planner, Cosmos was weak in Meta's comparison (Lab 2): 80% reach, 0% cup grasp, 20% box grasp, 0% pick-and-place, at **about 4 min per action**. [verified] — [V-JEPA 2 HTML](https://arxiv.org/html/2506.09985)
- GR00T N1 used video world models to generate "neural trajectories": 827 h from 88 h of teleop, about 105k L40 GPU-hours. This is world models as a **data multiplier**. [verified] — [GR00T N1](https://arxiv.org/html/2503.14734v2)
- **V-JEPA 2-AC** (Assran et al., Meta, Jun 2025): a ~300M action-conditioned predictor on a frozen ~1B ViT-g encoder, post-trained on **less than 62 h of unlabeled DROID video**. It was deployed zero-shot on Franka arms in 2 labs with image-goal MPC, with no data from those labs and no rewards. [verified] — [V-JEPA 2, arXiv:2506.09985](https://arxiv.org/abs/2506.09985)
  - Average success: reach 100%, grasp cup 65%, grasp box 25%, reach-with-object 75/75%, **pick-and-place cup 80%, box 65%**. Octo scored 15%/10% on pick-and-place. Planning takes **about 16 s per action**. 10 trials per task (small n). [verified] — [V-JEPA 2 HTML](https://arxiv.org/html/2506.09985)

### Inferences
- For AutoManip, the most transferable world-model idea is **TD-MPC2-style multi-task pretraining on state, then online fine-tuning on a held-out ManiSkill task**. It is open, single-GPU (a 19M model is trivial; 317M took 33 GPU-days on a 3090), and its few-shot unit (environment steps) matches AutoManip's existing RL measurement. Direct test: does multi-task pretraining cut AutoManip's 3.6M-transition StackCube budget by 2x or more?
- Video world models (Cosmos, Genie-scale) are out of budget to train. Using them as frozen data generators or planners on low-res ManiSkill renders is possible but expensive per action and likely suffers a sim-vs-real-video domain gap.
- V-JEPA 2-AC's zero-shot success is on goal-image-specified, short-horizon tasks. The "task specification" is a goal image, which sidesteps the demo question entirely and could be an alternative task-spec interface.

### Gaps
- No published results were found of TD-MPC2 multi-task pretraining transferring to *held-out ManiSkill3* tasks specifically.
- No quantitative few-shot (k-demo) adaptation results were found for DreamerV3 or for Cosmos-based policies.
- V-JEPA 2-AC trial counts are small (10 per task per lab), and no independent replication was found.

---

## Q3. Simulation side: ManiSkill3, RoboCasa, LIBERO, MimicGen-family automatic demo generation (how 10 demos become 1000s)

### Takeaway
The best-validated way to get from "a handful of demos" to "a working policy" in simulation is **demo multiplication**: MimicGen-family methods turn about 10 human demos per task into about 1,000 or more, using object-centric segment transformation plus sim replay and success filtering. GR00T and RoboCasa use this at scale. ManiSkill3 provides GPU-parallel sim and motion-planning demo generation that make this cheap. LIBERO (50 demos per task) is near saturated for VLAs.

### Cited Findings
- ManiSkill3 (Tao, Xiang, Shukla, et al., Hao Su; 2024): GPU-parallelized simulation *and rendering*. Claims 30,000+ FPS including rendering on many tasks and 10–1000x faster, with 2–3x less GPU memory than other platforms. It provides 12 diverse environment domains, and its demo datasets and baselines cover RL and imitation learning (state and RGB). [abstract/recall] — [ManiSkill3, arXiv:2410.00425](https://arxiv.org/abs/2410.00425)
- ManiSkill docs: unified learning-from-demos API with "ready, already tested" IL baselines, dataset setup docs, and public W&B baseline results. The index page does not list the models; see sub-pages. [verified] — [ManiSkill LfD docs](https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/index.html); [baselines](https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/baselines.html); [W&B results](https://wandb.ai/stonet2000/ManiSkill)
- LIBERO (Liu, Zhu, et al., 2023, NeurIPS D&B): 130 language-conditioned tasks in suites (Spatial, Object, Goal, LIBERO-100 = LIBERO-90 + LIBERO-10/Long), with **50 human demos per task**, designed for lifelong learning. [abstract/recall] — [LIBERO, arXiv:2306.03310](https://arxiv.org/abs/2306.03310)
- LIBERO saturation: OpenVLA-OFT reaches 97.1% average. [verified] — [OpenVLA-OFT](https://arxiv.org/abs/2502.19645). SimpleVLA-RL reports matching SoTA on LIBERO and 80% relative gain on RoboTwin. — [SimpleVLA-RL, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cbfbcb4da14235bd69b134070898ae9d-Abstract-Conference.html)
- RoboCasa (Nasiriany, Maddukuri, Zhang, et al., Zhu; RSS 2024): large-scale kitchen simulation with 100 tasks, generative-AI assets, and atomic tasks seeded by 50 human demos each, augmented via MimicGen. Shows clear policy scaling with synthetic data. [abstract/recall] — [RoboCasa, arXiv:2406.02523](https://arxiv.org/abs/2406.02523)
- RoboCasa low-data numbers (from GR00T N1): at 30/100/300 demos per task, Diffusion Policy gets 14.7/25.6/43.2% and GR00T N1 gets 17.4/32.1/49.6%. RoboCasa is hard at low demo counts even for a 2B VLA. [verified] — [GR00T N1](https://arxiv.org/html/2503.14734v2)
- MimicGen (Mandlekar, Nasiriany, Wen, et al., Fox; CoRL 2023): generated **50K+ demos from about 200 human demos across 18 tasks**, by transforming object-centric subtask segments to new object poses and keeping only successful rollouts. Agents trained on the generated data match those trained on more human data. [abstract/recall] — [MimicGen, arXiv:2310.17596](https://arxiv.org/abs/2310.17596)
- SkillMimicGen (Garrett, Mandlekar, et al.; CoRL 2024): segments demos into skills and uses motion planning for the transit between skills. Generated **24K+ demos from 60 human demos across 18 tasks**, about 24% better on average than MimicGen. [abstract/recall] — [SkillMimicGen, arXiv:2410.18907](https://arxiv.org/abs/2410.18907)
- DexMimicGen (Jiang, Xie, et al., Zhu, Mandlekar; ICRA 2025): bimanual dexterous extension. Generated **21K demos from 60 source demos across 9 tasks**. [abstract/recall] — [DexMimicGen, arXiv:2410.24185](https://arxiv.org/abs/2410.24185)
- At industrial scale: GR00T N1 generated 780k DexMimicGen trajectories (about 6,500 h) in 11 hours; its pretraining set includes 540k (10,000 per each of 54 receptacle pairs). [verified] — [GR00T N1](https://arxiv.org/html/2503.14734v2)
- SimplerEnv (Li, Hsu, et al., 2024): simulated eval environments mirroring Google Robot and WidowX/Bridge setups, showing sim-real correlation for RT-1, RT-1-X, RT-2-X and Octo. It was later ported to ManiSkill3 GPU sim. [abstract/recall; port claim based on ManiSkill3 paper and RL4VLA usage] — [SimplerEnv, arXiv:2405.05941](https://arxiv.org/abs/2405.05941)

### Inferences
- For AutoManip, "show the task 1–10 times" is most realistically served by **MimicGen-style augmentation inside ManiSkill**: take 1–10 demos (scripted, teleop, or from a trained TD3 policy), annotate object-centric subtask segments, replay under randomized poses on GPU-parallel envs, filter by a *trusted* success check, then train BC or diffusion policy or RL-from-demos. This is cheap on one L40S and matches the existing frozen-evaluator design (success filtering must use the trusted evaluator or a separate trusted check, never an agent-editable one).
- The augmentation success filter is a natural point of evaluator leakage. With stock PickCube success (not strict place/release), generated data may teach "lift but don't release." The strict/stock distinction from AGENTS.md matters here.

### Gaps
- No published MimicGen port specific to ManiSkill3 was confirmed this session (MimicGen and RoboCasa are robosuite/MuJoCo-based).
- Exact ManiSkill3 demo counts per task (motion-planning datasets) were not retrieved this session.

---

## Q4. Small-scale feasibility: what is reproducible on a modest GPU in ManiSkill (state or low-res vision)? Open checkpoints and datasets that support ManiSkill?

### Takeaway
Feasible on one L40S: (a) state-based multi-task RL or TD-MPC2-style pretraining with held-out-task fine-tuning, (b) MimicGen-style demo multiplication plus Diffusion Policy/BC, (c) LoRA fine-tuning or inference of small VLAs (Octo 27–93M, SmolVLA ~450M) on low-res ManiSkill RGB, and (d) evaluating open VLAs via SimplerEnv-in-ManiSkill. Infeasible: pretraining any VLA or video world model (tens of thousands of GPU-hours). The ManiSkill+VLA RL ecosystem (RL4VLA, RLinf-VLA) exists but uses 8xH100 for 7B models.

### Cited Findings
- Pretraining costs are far out of budget: OpenVLA about 21,500 A100-hours [verified — [OpenVLA](https://arxiv.org/html/2406.09246)]; GR00T N1 about 50,000 H100-hours [verified — [GR00T N1](https://arxiv.org/html/2503.14734v2)]; GR00T neural-trajectory generation about 105k L40-hours [verified — same].
- Fine-tuning costs are within range: OpenVLA LoRA takes 10–15 h on 1 A100 [verified — [OpenVLA](https://arxiv.org/html/2406.09246)]; Octo fine-tunes in "a few hours on consumer GPUs" [verified — [Octo](https://arxiv.org/abs/2405.12213)]; TD-MPC2 317M multitask takes 33 GPU-days on an RTX 3090 (a 19M variant is much cheaper) [verified — [TD-MPC2](https://arxiv.org/html/2310.16828)].
- **ManiSkill3 VLA RL ecosystem:** RLinf-VLA fine-tunes OpenVLA and OpenVLA-OFT with PPO in ManiSkill3 (224x224 third-person RGB), with both exceeding 90% on the `PutOnPlateInScene25Main-v3` task on one 8xH100 node. The base OpenVLA checkpoint comes from RL4VLA, and OpenVLA-OFT base was LoRA-SFT'd on motion-planning demos from that task. OOD averages: RL4VLA PPO 79.15%, RLinf-PPO 85.42%; PPO was more stable than GRPO. The table numbers vary between doc versions, so check the paper. — [RLinf ManiSkill docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/maniskill.html); [RLinf-VLA, arXiv:2510.06710](https://arxiv.org/pdf/2510.06710)
- RL4VLA (Liu et al., 2025, "What Can RL Bring to VLA Generalization?") uses ManiSkill-based pick-and-place variants and finds that PPO fine-tuning improves semantic and execution OOD generalization over SFT. [abstract/recall] — [RL4VLA, arXiv:2505.19789](https://arxiv.org/abs/2505.19789)
- SimpleVLA-RL (ICLR 2026) does RL for OpenVLA-OFT on LIBERO and RoboTwin. No ManiSkill result was found. — [SimpleVLA-RL](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cbfbcb4da14235bd69b134070898ae9d-Abstract-Conference.html)
- Open checkpoints and datasets: Octo (open weights) [verified — [Octo](https://arxiv.org/abs/2405.12213)]; OpenVLA and OpenVLA-OFT (open code and checkpoints) [verified — [OpenVLA-OFT site](https://openvla-oft.github.io)]; GR00T N1/N1.6 (open, "open foundation model") [verified — [GR00T N1](https://arxiv.org/abs/2503.14734); [N1.6 repo listing](https://github.com/SLCZ687/GR00T4me)]; pi0 (openpi weights) [abstract/recall — [pi0](https://arxiv.org/abs/2410.24164)]; SmolVLA [abstract/recall — [SmolVLA](https://arxiv.org/abs/2506.01844)]; Cosmos (open weights) [abstract/recall — [Cosmos](https://arxiv.org/abs/2501.03575)]; V-JEPA 2 (open) [abstract/recall — [V-JEPA 2](https://arxiv.org/abs/2506.09985)]; Open X-Embodiment and DROID datasets [abstract/recall — [OXE](https://arxiv.org/abs/2310.08864)].
- Gemini Robotics models are closed (API or trusted-tester only); pi0.5/pi0.6 availability varies. [inference; no primary source checked this session]

### Inferences
- **Cheapest meaningful experiment (state-based, fits current code):** build a ManiSkill multi-task state suite (PickCube, StackCube, PushCube, PegInsertion, PlaceSphere, etc.), pretrain a single multitask TD3 or TD-MPC2-style agent on N-1 tasks, then measure transitions-to-80% on the held-out task vs. from scratch (the existing 3.6M StackCube baseline). This directly tests "does multitask pretraining buy few-shot adaptation" without vision or language. A single L40S is likely enough for small models (TD-MPC2's 19M-param transfer experiment is the relevant precedent).
- **Cheapest demo-based experiment:** k-demo (k = 1, 2, 5, 10) source demos, then MimicGen-style augmentation, then BC or Diffusion Policy or demo-seeded RL (e.g., RFCL-style reverse curriculum). Report a success-vs-k curve under the frozen strict-success evaluator.
- **VLA experiment (stretch):** evaluate frozen Octo-Small or SmolVLA zero-shot on ManiSkill/SimplerEnv RGB tasks, then LoRA fine-tune with k demos. Expect near-zero zero-shot (cf. Octo's 10–15% on real pick-and-place in V-JEPA 2's eval) and a large sim-to-OXE visual domain gap. 7B OpenVLA LoRA (10–15 A100-hours per run) is affordable only for a handful of runs, not an autoresearch sweep.
- Pretrained VLAs require **RGB plus language**, which AutoManip's version-0 fixed state observation and action interfaces do not provide. Any VLA path is a new interface version, not a v0 edit.

### Gaps
- No ManiSkill-specific few-shot (k ≤ 10 demos) benchmark results for Octo, OpenVLA or SmolVLA were found.
- RLinf/RL4VLA compute on a single L40S (rather than 8xH100) is not documented.
- It was not confirmed which ManiSkill3 tasks have official SimplerEnv-ported VLA eval support beyond the Bridge/WidowX plate tasks.

---

## Q5. What infrastructure would AutoManip need (vision obs, language conditioning, multi-task suites, held-out-task evaluation protocol)?

### Takeaway
To study few-shot adaptation credibly, AutoManip needs: (1) a **multi-task suite with a frozen train/held-out task split** (held-out tasks never visible to the research agent during pretraining), (2) a **k-shot protocol** fixing demo budget, environment-step budget and seeds, (3) a trusted demo-generation/augmentation pipeline whose success filter is a trusted component, (4) optional versioned RGB and language observation interfaces for VLA comparisons, and (5) reporting of success-vs-k and success-vs-env-steps curves with both stock and strict success metrics.

### Cited Findings
- Field-standard protocol elements: success-vs-demos curves at 30/100/300 demos per task (GR00T N1) [verified — [GR00T N1](https://arxiv.org/html/2503.14734v2)]; held-out-task fine-tuning at a fixed env-step budget (20k steps, 10 held-out of 80 tasks, TD-MPC2) [verified — [TD-MPC2](https://arxiv.org/html/2310.16828)]; fixed-demo suites (LIBERO 50 demos per task) [abstract/recall — [LIBERO](https://arxiv.org/abs/2306.03310)]; OOD splits (RL4VLA/RLinf semantic, visual and execution OOD on ManiSkill) [— [RLinf docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/maniskill.html)].
- Real-world evaluations are small-n (V-JEPA 2: 10 trials per task; Gemini Robotics: 20 trials per task), which limits statistical confidence. [verified — [V-JEPA 2](https://arxiv.org/html/2506.09985); [Gemini Robotics](https://arxiv.org/html/2503.20020)]
- From-scratch specialists can match generalists on narrow tasks, so a from-scratch baseline at equal demo and compute budget is mandatory. [verified — [OpenVLA](https://arxiv.org/html/2406.09246)]
- Compute-matched baselines matter: pi0 includes a "compute parity" 160k-step variant against baselines. [verified — [pi0](https://arxiv.org/html/2410.24164)]

### Inferences (proposed infrastructure; not sourced, design recommendations)
- **Task suite manifest (trusted, frozen):** list of ManiSkill task IDs, split into `pretrain_tasks` and `heldout_tasks`, plus a private holdout split, all hashed in the manifest. The research agent may not read held-out task configs or demos before the adaptation phase.
- **k-shot protocol (trusted):** k ∈ {0, 1, 2, 5, 10, 50} source demos from a fixed, seeded demo pool; a fixed adaptation budget (env steps and GPU-minutes); multiple training seeds; paired dev episodes, confirmation seeds, and final holdout per the existing AGENTS.md rules. Always log demos used, augmented-demo count, env steps and wall clock.
- **Demo source and augmentation (trusted generator, editable recipe within a narrow interface):** ManiSkill motion-planning or scripted demos as the "human" demos; a MimicGen-style augmenter whose success filter calls the trusted strict-success checker.
- **Observation interface v1 (versioned, separate from v0 state):** low-res RGB (e.g., 128x128 or 224x224 to match VLA pretraining), proprioception, an optional language instruction string per task, and a fixed action space (EE delta pose plus gripper, matching OXE/Bridge conventions if VLA transfer is planned).
- **Artifact formats:** safetensors for policy weights; restricted architecture metadata. Large VLA checkpoints (7B) need a separate trusted loader path. Loading HF/LeRobot checkpoints must avoid pickle.
- **Metrics:** report stock ManiSkill success and strict place/release success separately; success-vs-k and success-vs-env-steps curves; "transitions-to-80%" relative to the from-scratch baseline (3.6M for StackCube TD3).
- **Sequencing suggestion:** state-only multitask held-out RL (cheapest, closest to the existing code), then k-demo plus augmentation plus BC/RL, then RGB/language and small-VLA fine-tuning.

### Gaps
- No standard community benchmark was found for "k-shot held-out-task adaptation in ManiSkill3." AutoManip would likely need to define its own, which is a contribution opportunity and a risk to comparability.
- No source quantifies how much evaluator or filter leakage affects MimicGen-style pipelines.
