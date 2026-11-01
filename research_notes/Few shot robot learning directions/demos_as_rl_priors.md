# Demonstrations as Priors for Sample-Efficient Manipulation RL (1–10 demos, video, or goal)

Scope note: this covers methods where 1–10 demonstrations (or one video, or one goal image/state) bootstrap RL, written for AutoManip. AutoManip runs GPU-parallel ManiSkill with state-based PPO/TD3. Its best result so far is TD3 at UTD 16 reaching 80% on StackCube in about 3.6M transitions. A known failure is a BC-initialized policy at 38% that collapsed to 0% under PPO fine-tuning.

Verification legend:
- **[fetched]**: numbers read from the paper's arXiv HTML or abstract in this session.
- **[search]**: taken from search-result snippets.
- **[prior knowledge, not re-fetched]**: from my background knowledge of a well-known paper. The link is the canonical arXiv page, but this session did not re-check the numbers. The report writer should treat these as lower confidence.

---

## 1. Demo-augmented off-policy RL (DDPGfD, DAPG, RLPD, IBRL, Cal-QL, offline-to-online)

### Takeaway
The strongest few-demo results for manipulation pair off-policy actor-critic with three things: high UTD, critic ensembles or LayerNorm, and a demo buffer sampled about 50/50. IBRL goes further and uses a frozen BC policy as an action proposer for both exploration and TD targets. IBRL (TD3-based) solves Meta-World tasks with 3 demos in under 40K samples. On a real robot it reaches 85–100% with 10–30 demos in 8K–30K steps. That makes it the most direct fit for AutoManip's existing TD3 stack. Plain offline-to-online methods (Cal-QL and similar) do not have a clear edge with very few demos and sometimes fail outright.

### Cited Findings

**IBRL, Imitation Bootstrapped RL (Hu, Mirchandani, Sadigh; arXiv Nov 2023, RSS 2024)** [fetched]
- Method: train a BC policy on demos. At each step, both the IL policy and the RL actor propose an action, and the one with the higher Q is executed. The same selection is used in TD-target bootstrapping. The paper contrasts this with prior work that oversamples demos or adds an imitation loss. — [arXiv 2311.02198](https://arxiv.org/abs/2311.02198)
- The backbone is **TD3**. State-based runs use a Q-ensemble of 5 and 5 critic updates per env step, higher than the default UTD. — [arXiv HTML](https://arxiv.org/html/2311.02198)
- Meta-World: **3 scripted demos per task**. IBRL and RFT solve all tasks "within 40K samples". — [arXiv HTML](https://arxiv.org/html/2311.02198)
- Robomimic Can (pixels): 10 demos, and all methods eventually solve it, IBRL fastest. Robomimic Square (pixels): 50 demos. IBRL is "the only method able to solve it within 0.5M samples", while baselines stay under 60%. — [arXiv HTML](https://arxiv.org/html/2311.02198)
- Real robot, 20 evaluations each:

  | Task | Demos | Env steps | Time | IBRL | RLPD | RFT | BC |
  |---|---|---|---|---|---|---|---|
  | Lift | 10 | 8K | 32 min | 100% | 95% | 90% | 50% |
  | Drawer | 30 | 16K | 64 min | 95% | 85% | 50% | 55% |
  | Hang | 30 | 30K | 120 min | 85% | 15% | 35% | 65% |

  — [arXiv HTML](https://arxiv.org/html/2311.02198)
- The paper says that on state-based Robomimic, existing methods "fail to learn completely". — [arXiv HTML](https://arxiv.org/html/2311.02198)

**RLPD (Ball, Smith, Kostrikov, Levine; ICML 2023)** [search]
- Method: SAC plus "symmetric sampling" (50% of each batch from online replay, 50% from offline/demo data), LayerNorm in the critics, and large critic ensembles. There is no offline pre-training. — [arXiv 2302.02948](https://arxiv.org/pdf/2302.02948)
- On sparse Adroit Door it beats the best previously reported result (IQL + fine-tuning) by 2.5×. It reaches or beats SACfD using about 10K online samples. — [arXiv 2302.02948](https://arxiv.org/pdf/2302.02948)
- Code: [github.com/ikostrikov/rlpd](https://github.com/ikostrikov/rlpd)
- Independent evidence on RLPD with few demos is mixed:
  - In IBRL's real-robot table, RLPD drops to 15% on Hang and 0% on Drawer with early stopping. — [IBRL HTML](https://arxiv.org/html/2311.02198)
  - In DSRL's real-world test (10 demos, about 40 episodes), RLPD scores 0/10. — [DSRL HTML](https://arxiv.org/html/2506.15799v2)
  - DSRL also reports that RLPD and Cal-QL "are not able to learn at all" on Robomimic Can/Square in its offline-to-online setting. — [DSRL HTML](https://arxiv.org/html/2506.15799v2)

**Cal-QL (Nakamoto et al., NeurIPS 2023)** [prior knowledge, not re-fetched]
- Calibrated conservative Q-learning, designed so that values do not "unlearn" at the start of online fine-tuning. It needs a sizeable offline dataset, so it is not a 1–10-demo method. — [arXiv 2303.05479](https://arxiv.org/abs/2303.05479)
- It appears as a baseline in RFCL, and RFCL with 5 demos beats it. — [RFCL HTML](https://arxiv.org/html/2405.03379)

**WSRL, Warm-Start RL (Zhou, Peng, Li, Levine, Kumar; ICLR 2025)** [search]
- Finding: keeping offline data during fine-tuning mainly serves to prevent value divergence at the start of fine-tuning, caused by distribution mismatch. WSRL instead runs a short warmup of rollouts from the pretrained policy, which "recalibrates" the Q-function. Offline data can then be discarded entirely, and the method learns faster than baselines that retain it. — [arXiv 2412.07762](https://arxiv.org/html/2412.07762v3)
- I did not retrieve exact benchmark numbers (see Gaps).

**DDPGfD (Večerík et al., 2017)** [prior knowledge, not re-fetched]
- DDPG with demos kept permanently in a prioritized replay buffer, plus n-step returns and an L2 penalty. Shown on sparse-reward insertion tasks in simulation and on a real robot. This is the template for "demo buffer plus off-policy RL". — [arXiv 1707.08817](https://arxiv.org/abs/1707.08817)

**DAPG (Rajeswaran et al., RSS 2018)** [prior knowledge, not re-fetched]
- Natural policy gradient with BC pretraining and a decaying BC-style augmentation term. About 25 human VR demos per Adroit task. This is the on-policy demo-augmentation baseline. — [arXiv 1709.10087](https://arxiv.org/abs/1709.10087)
- RFCL uses DAPG as a baseline and outperforms it with 5 demos. — [RFCL HTML](https://arxiv.org/html/2405.03379)

**MoDem (Hansen et al., ICLR 2023; arXiv Dec 2022)** [search]
- Model-based RL (TD-MPC) with demos, in three phases: BC pretraining, seeding with BC rollouts, and demo oversampling.
- With **5 demos and 100K interactions**, from pixels, it reaches about 53% average success on Adroit, where baselines achieve nothing nontrivial. Across 21 tasks it is 160–250% more successful than prior methods. — [arXiv 2212.05698](https://arxiv.org/pdf/2212.05698); [project page](https://nicklashansen.github.io/modemrl)

**DEMO³ (López Escoriza, Hansen, Tao, Mu, Su; ICML 2025)** [fetched abstract]
- Demonstration-augmented reward, policy and world-model learning for multi-stage tasks. Works with "as few as five demonstrations". Averages 40% better data-efficiency, and 70% on hard tasks, across 16 sparse-reward tasks in 4 domains. Co-authors include ManiSkill's lead (Tao) and Hao Su's group, which suggests ManiSkill tasks are included, but I did not confirm this. — [arXiv 2503.01837](https://arxiv.org/abs/2503.01837)

### Inferences
- AutoManip already runs high-UTD TD3, so IBRL is a near drop-in experiment. The additions are a BC actor trained on 1–5 demos, a Q-based max over {BC action, RL action} for both acting and TD targets, and a demo buffer. RLPD-style symmetric sampling plus LayerNorm critics is an even smaller change.
- With very few demos (1–5), per-demo state resets (Section 3) appear to matter more than buffer tricks. Buffer-only methods get little coverage from 1–2 trajectories.
- Independent replications show RLPD failing in some few-demo real-world settings. Its single-paper Adroit success should not be assumed to transfer.

### Gaps
- I did not retrieve a published RLPD or IBRL number on a ManiSkill task with 1–2 demos.
- WSRL's and Cal-QL's exact numbers were not fetched.

---

## 2. Residual RL and BC-to-RL fine-tuning collapse (and fixes)

### Takeaway
Fine-tuning a BC policy directly with PPO is widely reported as unstable. The most robust fixes keep the BC policy frozen:
- Learn a residual on top of it (Residual RL, ResiP, Policy Decorator).
- Steer its latent noise (DSRL).
- Select among its proposals with a critic (IBRL, PA-RL).

When fine-tuning the full policy, the common recipe is critic warmup with the actor frozen, plus BC/KL anchoring and entropy control. These match AutoManip's 38% → 0% collapse.

### Cited Findings

**ResiP, residual RL for precise assembly (Ankile, Simeonov, Shenfeld, Torne, Agrawal; arXiv Jul 2024)** [fetched]
- Setup: a frozen, action-chunked BC model (diffusion policy, 50 demos) with a closed-loop per-step residual policy trained by PPO.
- Results on FurnitureBench, low-randomness setting:

  | Task | BC alone | ResiP |
  |---|---|---|
  | one_leg | 54% | 98% |
  | round_table | 12% | 94% |
  | lamp | 7% | 97% |
  | peg-in-hole | 5% | 99% |

- Direct PPO on a chunked MLP (PPO-C) only reached 70/38/32/4%. — [arXiv HTML](https://arxiv.org/html/2407.16677v4)
- The paper says PPO-C "exhibits training instability and requires careful KL regularization to avoid collapse." It also says the base diffusion policy has "insufficient variance for effective exploration." — [arXiv HTML](https://arxiv.org/html/2407.16677v4)
- Scaling BC alone saturates: 10,000 demos only reach 56% on round_table. — [arXiv HTML](https://arxiv.org/html/2407.16677v4)
- Compute: 1,024 parallel envs and up to 500M env steps. This method is not sample-efficient. Its strength is reliability at GPU-sim scale. — [arXiv HTML](https://arxiv.org/html/2407.16677v4)

**Residual RL (Johannink et al., ICRA 2019; Silver et al., 2018)** [prior knowledge, not re-fetched]
- Both learn an additive RL correction on top of a fixed hand-designed or model-based controller. Johannink et al. do real-robot block insertion. — [arXiv 1812.03201](https://arxiv.org/abs/1812.03201); [arXiv 1812.06298](https://arxiv.org/abs/1812.06298)

**Policy Decorator (Yuan et al., 2024)** [prior knowledge, not re-fetched]
- A model-agnostic residual over large BC policies with a bounded residual magnitude and progressive exploration. From Hao Su's group and evaluated on ManiSkill tasks. — [arXiv 2412.13630](https://arxiv.org/abs/2412.13630)

**DSRL, Diffusion Steering via RL (Wagenmaker et al., CoRL 2025)** [fetched]
- Method: run RL over the input noise of a frozen diffusion policy instead of its actions. It needs only black-box access to the policy. — [arXiv HTML](https://arxiv.org/html/2506.15799v2); [PMLR](https://proceedings.mlr.press/v305/wagenmaker25a.html)
- Real-world single task with 10 human demos and 3,500 online steps (about 40 episodes): base 2/10 → DSRL 9/10. RLPD scores 0/10. — [arXiv HTML](https://arxiv.org/html/2506.15799v2)
- π0 on LIBERO goes from about 20% to about 100% in about 10K online samples. The paper claims 5–10× better efficiency than DPPO, IDQL and DQL on Robomimic. — [arXiv HTML](https://arxiv.org/html/2506.15799v2)

**Other evidence on collapse and stabilizers** [search]
- PIRLNav (Ramrakhya et al., CVPR 2023): BC→RL fine-tuning. It first trains the critic alone on rollouts from the frozen BC policy, then ramps up the actor learning rate. The authors note that a critic trained only on demos would be over-optimistic. — [arXiv 2301.07302](https://arxiv.org/pdf/2301.07302)
- A 2025 VLA post-training paper pairs action-chunked PPO with an adaptive auxiliary BC loss to stabilize training. — [arXiv 2509.25718](https://www.arxiv.org/pdf/2509.25718)
- A 2026 study (non-robotics) shows a BC student at 100% collapsing to 0% within about 1,000 actor-critic gradient steps. Neither entropy tuning nor demo anchoring prevented it, and the authors attribute the collapse to the critic. — [arXiv 2609.06667](https://arxiv.org/pdf/2609.06667)
- Evidence is mixed, though: a 2026 critic-optimization study found no benefit from critic warmup in a small-data setting. — [arXiv 2608.23566](https://arxiv.org/pdf/2608.23566)
- OGPO (2026) fine-tunes poorly initialized generative BC policies by using off-policy critics inside a PPO-style objective. — [arXiv 2605.03065](https://arxiv.org/html/2605.03065v1)

**Additional fixes** [prior knowledge, not re-fetched]
- "Fine-tuning RL models is secretly a forgetting mitigation problem" (Wołczyk et al., ICML 2024): pretrained capabilities on states not visited early in fine-tuning get forgotten. Remedies are BC/KL-to-pretrained regularization, EWC, and replaying pretraining data. — [arXiv 2402.02868](https://arxiv.org/abs/2402.02868)
- DPPO (Ren et al., 2024): PPO through the denoising steps of a diffusion policy. Reported as stable fine-tuning on Robomimic and FurnitureBench. — [arXiv 2409.00588](https://arxiv.org/abs/2409.00588)
- PA-RL, Policy-Agnostic RL (Mark et al., 2024): sample candidate actions from the base policy, rerank them with a critic, then distill the result into the policy. Reported as sample-efficient fine-tuning of diffusion and transformer policies, including on a real robot. — [arXiv 2412.06685](https://arxiv.org/abs/2412.06685)
- JSRL, Jump-Start RL (Uchendu et al., ICML 2023): roll in with the guide (BC) policy for the first h steps, let the RL policy act afterwards, and shrink h over time. This avoids initializing the RL policy from BC at all. — [arXiv 2204.02372](https://arxiv.org/abs/2204.02372)

### Inferences
Likely mechanisms for AutoManip's 38% → 0% collapse:
1. A randomly initialized value function gives noisy advantages that wipe out the BC actor in the first updates. Critic warmup with the actor frozen addresses this (PIRLNav, WSRL).
2. The BC policy's log-std is mismatched to what PPO needs. If it is too low, there is no exploration (ResiP's point about low variance). If it is too high, success drops to 0 before learning starts.
3. With no BC/KL anchor, the actor forgets behavior on states it has not yet revisited (Wołczyk et al.).

Cheap PPO-side fixes, in order of cost:
- Freeze the actor for N iterations and fit the critic first.
- Reset or clamp log-std.
- Add a KL-to-BC term with decay.
- Switch to a zero-initialized residual head on top of the frozen BC policy (ResiP or Policy Decorator style). This is the most robust option.

### Gaps
- No source fetched here quantifies how often the 38%→0% type of collapse happens in ManiSkill specifically.
- PA-RL's and DPPO's numbers were not re-verified.

---

## 3. Curriculum and resets from demos (Backplay, reverse curriculum, DemoStart, Nair 2018, RFCL)

### Takeaway
Resetting the simulator to states along a demo, starting near the goal and moving backward, is the most demo-efficient lever in simulation, and the best fit for ManiSkill. RFCL (built by the ManiSkill team) is the only method in its benchmark that solved every main task from **5 or fewer demos** within about 1M interactions. That includes PegInsertionSide and PlugCharger, which had never been solved before under sparse reward and few demos. DemoStart (2024) reaches about 99% in simulation from 2–60 demos and transfers zero-shot to a real robot.

### Cited Findings

**RFCL, Reverse Forward Curriculum Learning (Tao, Shukla, Chan, Su; ICLR 2024)** [fetched]
- Method: a per-demo reverse curriculum via state resets, which solves the task from states near each demo, followed by a forward curriculum that extends the policy to the full initial-state distribution. — [arXiv 2405.03379](https://arxiv.org/abs/2405.03379)
- Demos: **5 per run** (motion-planned for ManiSkill2, scripted for Meta-World, human for Adroit). A ManiSkill2 ablation varies the count from **1 to 20** at a 2M-step budget. The main benchmark budget is **1M interactions**. — [arXiv HTML](https://arxiv.org/html/2405.03379)
- RFCL is "the only method that can solve every main task from just 5 demonstrations or less." PegInsertionSide and PlugCharger "have never been solved before under sparse rewards and few demonstrations." — [arXiv HTML](https://arxiv.org/html/2405.03379)
- Reverse-stage ablation, 5 demos, steps (×10⁴) to about 100% from demo states:

  | Reset scheme | Steps (×10⁴) |
  |---|---|
  | Uniform demo-state reset (Nair 2018 style) | >100, did not finish within 1M |
  | Global curriculum | 41.5 ± 10.8 |
  | Per-demo curriculum | 33.5 ± 3.0 |
  | Per-demo + dynamic time limit | **24.1 ± 5.9** |

  — [arXiv HTML](https://arxiv.org/html/2405.03379)
- Base algorithm: SAC with a 10-critic ensemble (2 sampled per update), UTD 10 in the sample-efficient configuration, and 50:50 online/demo buffer sampling. — [arXiv HTML](https://arxiv.org/html/2405.03379)
- Wall clock to 90% on Meta-World tasks on an RTX 2080: 38–47 GPU-minutes with the wall-time-efficient hyperparameters, 104–180 minutes with the sample-efficient ones. — [arXiv HTML](https://arxiv.org/html/2405.03379)
- Stack-3: RFCL used 20 demos, compared with 100 for Nair et al. 2018. — [arXiv HTML](https://arxiv.org/html/2405.03379)
- Baselines it beats: RLPD, DAPG and Cal-QL, plus JSRL using numbers from the JSRL paper. — [arXiv HTML](https://arxiv.org/html/2405.03379)
- Limitation, in the authors' words: "the use of state reset… is generally restricted to training in simulation." — [arXiv HTML](https://arxiv.org/html/2405.03379)
- Code: project page [reverseforward-cl.github.io](https://reverseforward-cl.github.io), listed on arXiv.

**DemoStart (Bauza et al., Google DeepMind, 2024)** [fetched]
- Auto-curriculum: start states are sampled from temporal chunks of a demo, and Zero-Variance Filtering keeps only start states where the policy sometimes succeeds and sometimes fails. Reward is sparse and binary. Demo actions are *not* added to replay. — [arXiv HTML](https://arxiv.org/html/2409.06613)
- Demos per task: cube reorientation 2, plug lift 5, plug insertion 20, screwdriver-in-cup 20, nut-and-bolt 60.
- Simulation success: 98.6–99.9% on all five tasks, with typically under 10M learner updates (32M for the screwdriver task). — [arXiv HTML](https://arxiv.org/html/2409.06613)
- Ablation on plug insertion: vanilla RL scores 0%. Adding demo-state resets alone also scores **0%**. Resets plus the success filter score 0%. Both mechanisms together reach 97.2%, and the full method 99.6%. — [arXiv HTML](https://arxiv.org/html/2409.06613)
- Zero-shot sim-to-real after distillation:

  | Task | DemoStart | BC on real teleoperation data |
  |---|---|---|
  | Plug lift | 97% | 64% |
  | Plug insertion | 64% | 2% |
  | Cube reorientation | 97% | not given |

  The abstract claims 100× fewer demos than learning from real-robot demos. — [arXiv HTML](https://arxiv.org/html/2409.06613); [arXiv abs](https://arxiv.org/abs/2409.06613)
- The paper is internally inconsistent on the number of real teleoperation demos (2116 in the text vs 556 in a table). — [arXiv HTML](https://arxiv.org/html/2409.06613)

**Earlier reset and curriculum methods** [prior knowledge, not re-fetched]
- Nair et al., "Overcoming Exploration in RL with Demonstrations" (ICRA 2018): DDPG+HER with a BC loss gated by a Q-filter, plus resets to demo states, using about 100 demos for block stacking. — [arXiv 1709.10089](https://arxiv.org/abs/1709.10089)
- Florensa et al., "Reverse Curriculum Generation" (CoRL 2017): needs only a single goal state. It expands start states backward from the goal via random-walk perturbations, keeping starts of intermediate difficulty. — [arXiv 1707.05300](https://arxiv.org/abs/1707.05300)
- Backplay (Resnick et al., 2018): starts episodes near the end of a single demo and moves the start point backward over training. — [arXiv 1807.06919](https://arxiv.org/abs/1807.06919)
- Salimans & Chen (2018): the same idea solved Montezuma's Revenge from one demo. — [arXiv 1812.03381](https://arxiv.org/abs/1812.03381)

### Inferences
- ManiSkill3 supports setting env state, so per-demo reverse curricula are directly implementable. They are also GPU-parallelizable: different envs can start from different demo timesteps. The RFCL authors maintain ManiSkill, so porting should be low-friction.
- The DemoStart ablation is a warning. Naïve demo-state resets alone scored 0% there, and filtering start states toward intermediate difficulty was essential. RFCL's uniform-reset baseline also failed within 1M steps. The curriculum schedule matters, not just the resets.
- Resets need privileged sim state, which conflicts with the "show the robot once in reality" end goal. They remain a valid lever for sim-trained policies that are later distilled, as DemoStart shows.

### Gaps
- I did not extract RFCL's per-task curves (e.g., StackCube steps to 80%) or the per-count numbers from its 1-demo vs 2-demo ablation.
- I found no published ManiSkill3 GPU-parallel RFCL port with timing numbers.

---

## 4. Reward from one demo (OT rewards, trajectory matching, VIP/LIV, IRL, GAIL/AMP)

### Takeaway
ROT (2022) is the clearest single-demo success: with **one demo and about one hour of online training**, it averages 90.1% across 14 real manipulation tasks. It combines optimal-transport trajectory-matching rewards with adaptively weighted BC. For an embodiment gap, object-centric rewards (an object 6D-pose trajectory extracted from one human video) are emerging as the transferable signal. Pretrained visual reward models (VIP/LIV) are less reliable as standalone dense rewards.

### Cited Findings
- **ROT, Regularized Optimal Transport (Haldar, Mathur, Yarats, Pinto; CoRL 2022)** [fetched abstract] — [arXiv 2206.15469](https://arxiv.org/abs/2206.15469)
  - Simulation, 20 visual tasks (DMC, OpenAI Robotics, Meta-World): on average 7.8× faster to reach 90% of expert performance than prior state of the art.
  - Real world: **1 demo plus about 1 hour of online training gives a 90.1% average success over 14 tasks**.
- **Human2Sim2Robot (Lum, Lee, Liu, Bohg; CoRL 2025)** [search] — [arXiv 2504.12609](https://arxiv.org/html/2504.12609v3); [CoRL 2025](https://mlanthology.org/corl/2025/lum2025corl-crossing)
  - Uses **one RGB-D human video**. It extracts the object's 6D-pose trajectory as an embodiment-agnostic object-centric reward and uses the demo for guidance rather than for retargeting.
  - Trains RL in a reconstructed simulation and transfers zero-shot to a Kuka arm with an Allegro hand.
  - Beats object-aware open-loop replay by 55% and imitation learning with augmentation by 68%.
- **X-Sim** [search]: similar real-to-sim-to-real approach that uses object motion from an RGB-D human video as a dense reward. — [found via search for 2504.12609](https://arxiv.org/abs/2504.12609v3) (no dedicated X-Sim URL fetched)
- **Prior knowledge, not re-fetched:**
  - VIP (Ma et al., ICLR 2023): a goal-conditioned value function pretrained on human video (Ego4D), used as a zero-shot embedding-distance reward toward a goal image. — [arXiv 2210.00030](https://arxiv.org/abs/2210.00030)
  - LIV (Ma et al., ICML 2023): a language-image value representation used the same way. — [arXiv 2306.00958](https://arxiv.org/abs/2306.00958)
  - GAIL (Ho & Ermon, 2016): adversarial IRL, typically sample-hungry. — [arXiv 1606.03476](https://arxiv.org/abs/1606.03476)
  - AMP (Peng et al., 2021): an adversarial style prior, used mostly for locomotion and character control. — [arXiv 2104.02180](https://arxiv.org/abs/2104.02180)
  - DEMO³ (above) learns dense stage-wise rewards from about 5 demos. — [arXiv 2503.01837](https://arxiv.org/abs/2503.01837)

### Inferences
- In state-based ManiSkill, a one-demo OT or trajectory-matching reward over *object state* (cube pose relative to the goal) is cheap to compute. It can replace or augment the sparse reward without touching evaluation code.
- For AutoManip this fits the "narrow, explicit reward interface" principle. Evaluation must stay on the frozen strict-success metric so that reward hacking against the OT reward is caught.

### Gaps
- I did not fetch ROT's per-task simulation numbers, or any ManiSkill-specific OT-reward results.
- I did not fetch quantitative VIP/LIV reward-shaping results on manipulation RL. Their effectiveness as dense RL rewards is reported as mixed in the community, but I have no fetched citation for that, so it is unverified.

---

## 5. Learning from video and human demos across embodiment gaps

### Takeaway
The working pattern in 2024–2025 is to convert a human video into an object-centric target: an object pose trajectory, object flow, or point tracks. That target becomes a reward or plan for RL in simulation, rather than a source of retargeted hand actions. Human2Sim2Robot shows that one video is enough for dexterous sim-to-real RL when an object tracker and a reconstructed simulator are available.

### Cited Findings
- Human2Sim2Robot: per-frame hand retargeting is fragile because hand-pose noise and embodiment differences make IK solutions infeasible or suboptimal. An object-pose-trajectory reward avoids this. Stated limitations: single-object tasks and dependence on high-quality trackers and simulators. — [arXiv 2504.12609](https://arxiv.org/html/2504.12609v3)
- **Prior knowledge, not re-fetched:**
  - ATM, Any-point Trajectory Modeling (Wen et al., RSS 2024): predicts point tracks from video and uses them to guide policies with few action-labeled demos. — [arXiv 2401.00025](https://arxiv.org/abs/2401.00025)
  - Track2Act (Bharadhwaj et al., 2024): predicts point tracks from web video plus a residual correction policy. — [arXiv 2405.01527](https://arxiv.org/abs/2405.01527)
  - Im2Flow2Act (Xu et al., CoRL 2024): object flow from human video, with the policy trained on simulated robot data. — [arXiv 2407.15208](https://arxiv.org/abs/2407.15208)

### Inferences
- In state-based ManiSkill, the "video" case reduces to "an object-state trajectory with no robot actions". That is a cheap proxy experiment for the long-term "show it once" goal: give the agent only the cube-pose trajectory from one demo, with no actions, and use it for the reward and for resets.

### Gaps
- Quantitative sample-efficiency numbers (env steps) for ATM, Track2Act and Im2Flow2Act were not fetched.

---

## 6. What could reach 80% on a held-out ManiSkill task with 1–2 demos and under 1M interactions? Cheapest first experiment?

### Takeaway
The most plausible route is RFCL-style per-demo reverse curriculum resets plus high-UTD off-policy RL with a 50/50 demo buffer. This fits AutoManip's existing TD3 at UTD 16, optionally with IBRL-style BC action proposals. RFCL solved all its main ManiSkill2 tasks (PickCube/StackCube class through PegInsertionSide and PlugCharger) with 5 or fewer demos within about 1M interactions. 1–2 demos is in the range of its ablation, but I did not verify per-count numbers. The cheapest first experiment is to add demo-state reverse-curriculum resets to the current TD3 on StackCube with 1, 2 and 5 demos. Compare steps-to-80% against the 3.6M-transition baseline under the frozen evaluator.

### Cited Findings
- RFCL: 5 demos and 1M interactions solve every main ManiSkill2 task. Per-demo reverse curriculum with a dynamic time limit reaches about 100% from demo states in about 240K steps. The demo-count ablation covers 1–20. — [RFCL HTML](https://arxiv.org/html/2405.03379)
- IBRL: TD3 backbone with 3 demos solves Meta-World tasks in under 40K samples. — [IBRL HTML](https://arxiv.org/html/2311.02198)
- DemoStart: 2 demos are enough for about 99.9% on simulated cube reorientation, but the curriculum filtering is essential. — [DemoStart HTML](https://arxiv.org/html/2409.06613)
- ROT: 1 demo plus about 1 hour gives a 90.1% real-world average. — [ROT](https://arxiv.org/abs/2206.15469)

### Inferences

Candidate ranking for "80% under 1M, 1–2 demos, held-out ManiSkill task":
1. **RFCL / reverse-curriculum resets plus a demo buffer (highest plausibility).** It was demonstrated on ManiSkill itself with ≤5 demos at 1M steps. The main open risk is whether 1–2 demos cover the initial-state distribution well enough for the forward stage.
2. **IBRL on TD3.** It is a natural fit for AutoManip's code. With 1–2 demos the BC policy will be weak, but IBRL only uses it when Q prefers its actions, so the downside is bounded.
3. **Residual RL over a frozen BC policy.** It fixes the PPO collapse failure mode, but published ResiP runs used about 500M steps with 50 demos, so it is unlikely to meet a 1M budget on its own.
4. **OT or trajectory-matching reward from 1 demo.** A complement to 1–3, not a standalone route.

**Cheapest first experiment (about one GPU-day or less):**
- Setup: on StackCube, use 1, 2 and 5 demos from ManiSkill's motion-planned demo set. For each demo, reset a fraction of the parallel envs to state t = T−k. Lower k whenever per-demo success exceeds a threshold, as RFCL does with its dynamic time limit. Keep the remaining envs on the true initial distribution.
- Training: keep TD3 at UTD 16 and add 50/50 symmetric sampling from a demo buffer (RLPD-style). Use LayerNorm critics.
- Measurement: env steps to 80% *on the frozen evaluator's true initial distribution* (never on reset states). Report it against the 3.6M baseline across ≥3 training seeds.
- Second arm: IBRL action selection with a BC policy trained on the same demos.
- Follow-up: run the same recipe on a *held-out* task the agent never tuned on.

**PPO-collapse fix, if PPO stays in scope:**
- Freeze the actor and warm up the critic for N iterations.
- Fix the log-std at the start of fine-tuning.
- Add a decaying KL-to-BC term.
- Alternatively, use a zero-initialized residual head on the frozen BC policy.

All of this is cheap to test on the existing 38% BC checkpoint.

Integrity note for AutoManip: reset-from-demo states and demo-derived rewards must live in the editable training interface. Evaluation must keep the frozen benchmark initial distribution and seeds, and must report stock vs strict PickCube/StackCube success separately.

### Gaps
- I found no published result that meets exactly "held-out ManiSkill task, 1–2 demos, under 1M steps, 80%". This is an extrapolation from RFCL's 5-demo/1M results and its 1–20 demo ablation, whose exact per-count numbers were not extracted.
- I did not find a ManiSkill3 GPU-parallel benchmark of RFCL vs RLPD vs IBRL. Whether massively parallel envs keep RFCL's sample efficiency, or trade it for wall-clock, is unverified.
