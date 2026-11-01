# World models, off-policy learning and data reuse: primary literature survey

2026-10-28. Literature only: no implementation, probes, training, uploads or
experiment queue changes. All existing proposal payloads remain frozen.

This selected map covers twelve substantive papers, rather than claiming an
exhaustive October 2026 ranking. Results below are the authors' benchmark results,
not reproductions in AutoManip. **F** means online learning from random
initialization without external demonstrations; **A** means prior-data-assisted
learning, even when actor and critic start randomly. A model trained from an
agent's own ordinary interactions is not an external teacher, but its training,
synthetic transitions and inference still cost compute.

The current comparison contract is a state48/action8 restricted feed-forward
actor, 50 decisions/full-reset episode, original task reward and protected final
scoring. Relevance statements below are our inferences, not paper results on that
contract. No paper here establishes a guaranteed reduction in steps to reliable
96% or higher final StackCube success under our exact interface.

## Evidence table: learning and using world models

| Primary paper and track | Mechanism | Actual domain, interaction evidence and baseline | Compute/data assumptions | StackCube relevance and critical negative conditions |
|---|---|---|---|---|
| **DreamerV3 — Mastering Diverse Domains through World Models** (arXiv v2, 2024; Nature successor, 2025). **F**. [Full preprint](https://arxiv.org/html/2301.04104v2); [Nature](https://www.nature.com/articles/s41586-025-08744-2). | Replayed sequences train a recurrent stochastic latent model; actor/critic learn from imagined trajectories. Symlog/two-hot targets, balanced/free-bit KL and return normalization stabilize different signal scales. | Over 150 tasks across eight domains; proprioceptive control at 500k environment steps and visual control at 1M. Compares specialist methods and a tuned, fixed-configuration PPO. | Single A100 per agent; larger models and replay ratios buy interaction efficiency with more computation. No external demos in the from-scratch experiments. | Useful evidence for training signal normalization and learned imagination. Actual actor consumes recurrent model state, so the full agent is **not** our deployable MLP. Accurate-looking predictions and broad benchmark averages do not certify release/contact dynamics or our final-success metric. |
| **TD-MPC2: Scalable, Robust World Models for Continuous Control** (ICLR 2024). **F** single-task; separate multitask-data track. [Paper](https://arxiv.org/html/2310.16828v2). | Decoder-free latent consistency, reward/TD prediction, SimNorm and five value heads; MPPI plans with a learned policy prior and terminal value. | 104 tasks across DMControl, Meta-World, ManiSkill2 and MyoSuite. Three-seed manipulation curves compare SAC, TD-MPC and the paper's DreamerV3 baseline. Appendix explicitly includes Stack Cube. | Planning at every action; model and replay training count. Separate 317M/80-task scaling result costs approximately 33 RTX3090 GPU-days; that is not the single-task requirement. | Closest task evidence here, but Stack Cube is **state55/action4**, 200 environment steps, repeat 2/100 decisions. Final-step success is used. Neither this interface nor its planner matches ours. A distilled restricted actor would be a different, unverified method. |
| **When to Trust Your Model: Model-Based Policy Optimization (MBPO)** (NeurIPS 2019). **F**. [Paper](https://arxiv.org/html/1906.08253). | Probabilistic dynamics/reward ensemble generates short branches from real replay states; SAC learns from real/synthetic data. | Full-length MuJoCo continuous-control tasks; compares SAC/PPO and PETS/STEVE/SLBO. Ant at 300k interactions reaches the SAC score at 3M in the reported comparison. One-step branches retain much of the gain. | Roughly 20–40 policy-learning updates per real sample plus model fitting/generation. Reported simulated minutes of experience are **not** training wall time. No oracle reward/dynamics required. | Training-only imagination can in principle leave an ordinary actor at deployment. Contact-state coverage and model bias remain decisive; even a good average next-state loss can miss release instability. Long imagined trajectories are not justified by the short-branch evidence. |
| **EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data** (2024). **F**. [Paper](https://arxiv.org/html/2403.00564v2). | Learned latent dynamics plus continuous-action Gumbel search, search-target policy distillation and search-based value reanalysis of stale replay. | Atari100k and DMControl proprioceptive/visual low-data settings; beats DreamerV3 on 50/66 tasks. Table 2 is more qualified: proprioceptive aggregate mean **723.2 vs TD-MPC2 740.9**, so it does not dominate every baseline. | Search/reanalysis and distributed collection add compute. Walker-run appendix: 2.7h/100k for EZ-V2 versus 3.3h TD-MPC2 on an **eight-RTX3090 server**; not a single-GPU guarantee. | Relevant to stale-target correction, not demonstrated contact-rich stacking. Deployed search violates our actor contract. Policy-improvement arguments depend on value estimates/asymptotic sampling, not a guarantee under learned-model error. |

## Evidence table: stronger online replay controls

| Primary paper and track | Mechanism | Actual domain, interaction evidence and baseline | Compute/data assumptions | StackCube relevance and critical negative conditions |
|---|---|---|---|---|
| **Randomized Ensembled Double Q-Learning: Learning Fast Without a Model (REDQ)** (ICLR 2021). **F**. [Paper](https://arxiv.org/html/2101.05982). | Ten critics, minimum over a randomly selected pair for targets, average-critic actor objective; typically 20 critic updates per real transition and a delayed actor update. | Hopper, Walker2d, Ant and Humanoid: matches/exceeds reproduced MBPO with matched high update-to-data settings; also analyzes SAC at UTD 20 and ensemble alternatives. | More critic computation than SAC at UTD 1. Authors measured MBPO taking roughly 75% longer on a 2080Ti; this is their implementation/hardware result. No world model or prior demos. | Essential competent control before attributing gains to synthetic dynamics. The evidence is locomotion rather than stacking. Simply increasing SAC updates was insufficient; bias/variance control matters, and equal interaction counts do not mean equal learner cost. |
| **CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity** (ICLR 2024 version). **F**. [Paper](https://arxiv.org/html/1902.05605v4). | Target-free critic; concatenate current/next inputs for a shared normalization pass; wide critics and Batch Renormalization. Critic UTD 1, delayed actor. | State-based Gym MuJoCo plus DMControl; matches/exceeds REDQ/DroQ interaction efficiency. RTX3090 timing table: Walker2d 1.1h versus REDQ 4.0h, over the paper's run/protocol. | Wide critics cost more than small SAC. Hardware/compiler/batch choices determine actual savings. No demos or learned dynamics. | Plausible replay baseline with train-only richer critics. Its combined normalization/target treatment is crucial: naively adding normalization while keeping targets fails; ordinary BN can later collapse. Lower measured Q bias does not reliably predict higher return in its analysis. |
| **Bigger, Regularized, Optimistic (BRO)** (NeurIPS 2024). **F**. [Paper](https://arxiv.org/html/2405.16158v2). | Approximately 5M-parameter regularized residual critics, replay ratio 10 (Fast: 2), LayerNorm, decay/resets and optimistic exploration; non-pessimistic value treatment. | Forty state-based DMControl/MetaWorld/MyoSuite tasks at 1M interactions; compares SAC/TD3/SR-SAC/CrossQ and published TD-MPC2 results. Most reproduced baselines use ten seeds; TD-MPC2 is an exception. | Larger critics and repeated learning; A100/CPU timing study. Fast variant trades reuse for time. Resets are part of the recipe, not free pretrained capacity. | Supports testing representation/regularization rather than assuming extra epochs suffice. Full actor/dual-policy recipe is not our restricted actor. Actor scaling, n-step targets and naive categorical targets did not uniformly help; stronger fitting can still reinforce erroneous targets. |
| **FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control** (2025 v1 report). **F**. [Paper](https://arxiv.org/html/2505.22642v1). | Parallel rollouts, batch 32,768, distributional critics, GPU replay and tuned deterministic-policy training. | HumanoidBench, IsaacLab and MuJoCo Playground; selected tasks train under three hours. Three FastTD3 seeds; some baseline curves are repository results with interpolated timestamps, and SimbaV2 comparison is single-seed. | A100 plus 16 CPU cores; mixed precision/compiler gains are throughput claims. Typically 2/4/8 updates **per 128–4096 parallel transitions**, not per individual sample. Actor 512/256/128 exceeds our width cap. | Useful vector-replay/data-coverage lessons, not established stacking sample efficiency. Some suites use privileged critic inputs. Real deployment required algorithm-specific reward changes; those advantages cannot be silently imported into our frozen task. |
| **Mastering Visual Continuous Control: Improved Data-Augmented RL (DrQ-v2)** (ICLR 2022). **F**. [Paper](https://arxiv.org/html/2107.09645). | Pixel random shifts plus off-policy deterministic actor-critic, three-step targets, larger replay and scheduled exploration noise. | Twenty-four DMControl visual tasks; outperforms DrQ/CURL, and rivals DreamerV2 on many, but not all, tasks. Easy/medium/hard budgets are 1M/3M/30M steps; do not read humanoid success as low-data learning. | Same V100 comparisons; measured96 FPS versus DreamerV2 24 FPS. Frames, action-repeat 2 and actor decisions are distinguished. No demos. | State48 does not need pixel augmentation. Replay and exploration ablations are relevant hypotheses, not transferable constants. Model-based comparison covers only 12 tasks; n-step/exploration results come from a limited ablation subset. |

## Evidence table: assisted learning and offline-to-online transitions

| Primary paper and track | Mechanism | Actual domain, interaction evidence and baseline | Compute/data assumptions | StackCube relevance and critical negative conditions |
|---|---|---|---|---|
| **Efficient Online Reinforcement Learning with Offline Data (RLPD)** (ICML 2023). **A**, random actor/critic, no offline pretraining required. [Paper](https://arxiv.org/html/2302.02948). | Half-online/half-prior replay sampling, LayerNorm critics, ensembles/high UTD; optional entropy/min-target/architecture choices are domain-sensitive. | Twenty-one state tasks: sparse Adroit, AntMaze and D4RL locomotion; ten seeds, comparing SACfD and suite-specific IQL-finetuning/Off2On. Measures Adroit completion speed, not just success. | Default ten critics and20 critic updates/real transition; prior dataset generation is outside online-step curves. Pixel extension adds augmentation. | Direct alternative to cloning a mean then discarding demo replay during PPO. But its prior data include behavior-policy trajectories, not only successful demos. Normalization constrains extrapolation, not an absolute proof of correct Q; entropy/target choices vary by domain. |
| **Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning** (NeurIPS 2023). **A**. [Paper](https://arxiv.org/html/2303.05479). | CQL-style conservative values are calibrated against behavior return-to-go, preventing excessive downward pressure; mix prior/new replay during fine-tuning. | AntMaze, FrankaKitchen, sparse Adroit and visual ball-to-bowl manipulation. Best fine-tuned result on 9/11 tasks and smallest regret on 8/11 versus CQL/IQL/AWAC and others; six-seed main curves. | Offline RL pretraining plus online learning; UTD 20 variant also imports RLPD normalization/ensembles. Offline/teacher costs are not online samples. | Good evidence that value-scale mismatch can accompany unlearning. It studies **over-conservative pretrained Q**, not our fresh PPO V-critic. Reliable reference returns and narrow-data coverage matter; it is not a generic critic-warmup proof or a prescription to copy teacher values. |
| **AWAC: Accelerating Online Reinforcement Learning with Offline Datasets** (2021 final preprint). **A**. [Paper](https://arxiv.org/html/2006.09359v6). | Off-policy TD critic plus advantage-weighted log-likelihood on replay actions, avoiding explicit behavior-density estimation; continue with newly collected data. | Simulated dexterous pen/door/relocate and physical claw/drawer/hand tasks. Pen solved at 120k online steps, approximately 20 min interaction, versus DAPG/other assisted controls. | Simulated Adroit uses 25 human demos **plus 500 BC trajectories**. Physical tasks also use demos and BC rollouts, with automatic resets. Interaction minutes exclude offline work/learner time. | Most directly motivates retaining a data-supported actor objective during assisted improvement. Critic-derived weights can misrank examples, a unimodal actor can still average incompatible actions, and unsupported recovery states remain unseen. No matched AutoManip stacking evidence. |

## Five connections and unresolved questions

These are our synthesis and open questions, not established explanations of the
recent AutoManip failures or a new experiment queue.

1. **Extra optimization is not a single mechanism.** REDQ reuses transitions with
   action-conditional Bellman targets; MBPO adds model branches; Dreamer changes
   the training state/target distribution. Continuing PPO's fixed value targets
   after actor stopping tests none of those complete methods. Which useful
   transitions or targets are missing, rather than merely how many updates?
   [REDQ](https://arxiv.org/html/2101.05982),
   [MBPO](https://arxiv.org/html/1906.08253).

2. **BC-to-RL collapse has several distinct precedents.** AWAC retains weighted
   replay action support; RLPD retains prior transitions; Cal-QL repairs an
   excessively conservative pretrained value scale. None identifies the cause of
   our fresh-critic PPO collapse. Is the bottleneck actor displacement, erroneous
   value ranking, or insufficient recovery coverage? Their results do not
   distinguish those alternatives for us.
   [AWAC](https://arxiv.org/html/2006.09359v6),
   [RLPD](https://arxiv.org/html/2302.02948),
   [Cal-QL](https://arxiv.org/html/2303.05479).

3. **Contact prediction and safe deployment are separate questions.** TD-MPC2's
   relevant stacking evidence uses a different task interface and online planning.
   EfficientZero's distilled policy is trained with search targets but its reported
   control also uses search. What transfers to a restricted actor without giving
   it planning privileges or changing episode semantics remains unresolved.
   [TD-MPC2](https://arxiv.org/html/2310.16828v2),
   [EZ-V2](https://arxiv.org/html/2403.00564v2).

4. **Reuse needs denominators and a compute frontier.** A vector-environment
   round, individual transition, gradient call, sampled training row, model step
   and repeated action are different quantities. FastTD3's low per-transition
   reuse and DrQ-v2's FPS results cannot be compared with Dreamer replay ratios
   or PPO epochs by label alone. Would a claimed step saving survive equal
   learner-time accounting and the same deployable actor?
   [FastTD3](https://arxiv.org/html/2505.22642v1),
   [DrQ-v2](https://arxiv.org/html/2107.09645),
   [DreamerV3](https://arxiv.org/html/2301.04104v2).

5. **The papers disagree usefully about universal fixes.** CrossQ removes target
   networks with carefully coupled normalization; BRO retains benefits from target
   networks and strong regularization. Neither makes value bias alone a reliable
   causal diagnostic. Which interaction between data distribution, normalization
   and target lag matters in manipulation cannot be answered by copying an
   isolated component.
   [CrossQ](https://arxiv.org/html/1902.05605v4),
   [BRO](https://arxiv.org/html/2405.16158v2).

## Reading depth and source limits

Primary full-text algorithm and empirical sections were inspected, not only
abstracts. This was a targeted reading, not proof verification or reproduction.
HTML conversion sometimes omits equations/tables; quantitative entries above
were checked against the displayed relevant text/table. No ratios were estimated
from plot pixels. Author project pages were navigation aids, not independent
empirical evidence. Secondary summaries were not used as authorities.

| Paper | Beyond-abstract material inspected |
|---|---|
| DreamerV3 | World-model/critic/actor methods; benchmarks, ablations/scaling; A100 and PPO comparison description. |
| TD-MPC2 | Sections 3–4; MPPI/SimNorm/value ensemble; Appendix C task/interface table; multitask cost table. |
| MBPO | Model/branching algorithm; comparative evaluation; no-model, rollout-length and value-expansion ablations. |
| EfficientZeroV2 | Continuous Gumbel search/policy targets and value reanalysis; Table 2; Appendix J.3 compute table. |
| REDQ | Algorithm 1; comparative evaluation and bias analysis; architecture/runtime and SAC at UTD 20 controls. |
| CrossQ | Concatenated normalization/target-free algorithm; bias and normalization ablations; RTX3090 timing table. |
| BRO | Design/scaling study; unsuccessful components; hyperparameters/resets and Appendix E.1 timing method. |
| FastTD3 | Design choices and replay implementation; experiment setup; baseline provenance and reward-transfer caveat. |
| DrQ-v2 | Algorithm; experimental axes/hardware/budgets; n-step, replay and exploration ablations. |
| RLPD | Sections 4–5, Algorithm 1; dataset/metric definitions; Appendix B hyperparameter table. |
| Cal-QL | Calibration/unlearning mechanism and objective; experiments; high-UTD/data-composition analyses. |
| AWAC | Advantage-weighted actor objective/algorithm; simulation and real-robot comparisons; prior-data/reset descriptions. |

Versions are explicit where materially relevant. Dreamer detailed methods were
read from the accessible 2024 v2 preprint; the 2025 Nature paper's indexed methods
and author project confirm the successor, but direct full-text Nature retrieval
was blocked by its cookie redirect. Do not silently equate implementations across
those releases. CrossQ's arXiv identifier begins in 2019, but the inspected v4 is
dated March 2024. FastTD3 here means the May 2025 v1 report, not a claim about every
later revision. Several timing studies use different hardware, software,
evaluation frequency or published baseline curves; there is no common hardware
speed ranking across these twelve papers.

No new substantial compute is authorized by this survey. Choosing a next method
or decisive experiment remains a separate review step.
