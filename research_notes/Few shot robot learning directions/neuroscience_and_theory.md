# Neuroscience, Cognitive Science, and Theory of Fast (Few-Shot) Motor and Task Learning, Mapped to Robot-Learning Algorithms for AutoManip

Scope note: these notes cover (a) biological mechanisms for learning new motor tasks from one or a few observations or trials, and (b) theoretical results on what makes few-shot learning possible. Each is mapped to a cheap, testable ManiSkill experiment. AutoManip context: PPO stalls at the "release" step because the policy grows unboundedly certain that the gripper should stay closed; TD3 is about 3x more sample-efficient. Goal: show the robot a new task once or twice and it works the task out.

Sourcing convention: links marked (verified) were confirmed by web search in this session. Links marked (biblio) are standard DOI or arXiv identifiers given from bibliographic knowledge and were not re-fetched here. Treat specific numbers in (biblio) items as needing a check before quoting.

Evidence-strength labels: ESTABLISHED = replicated, broadly accepted; SUPPORTED = good evidence, details debated; CONTESTED = actively disputed; SPECULATIVE = mainly a computational proposal.

---

## Q1. Complementary learning systems, hippocampal one-shot episodic memory, episodic control, and replay/consolidation

### Takeaway
The brain pairs a fast, pattern-separated, one-shot episodic store (hippocampus) with a slow, interleaved, generalizing learner (neocortex), linked by replay. This is the most directly usable biological template for "learn from one demo". Its ML counterparts (episodic control, non-parametric memory, prioritized/demo replay) give large early-learning gains. The sharpest AutoManip version: store the few successful release transitions verbatim and let a fast episodic value or action estimate override the slow, saturated parametric policy at the release step.

### Cited Findings
- CLS theory (McClelland, McNaughton & O'Reilly 1995) argues that one-shot hippocampal storage plus slow, interleaved neocortical learning avoids catastrophic interference in distributed networks (ESTABLISHED as a framework). — [McClelland et al. 1995, Psych. Review, doi:10.1037/0033-295X.102.3.419](https://doi.org/10.1037/0033-295X.102.3.419) (biblio)
- Kumaran, Hassabis & McClelland (2016, Trends in Cognitive Sciences) update CLS. The hippocampus also supports generalization via recurrent similarity computations and goal-dependent replay that can weight experiences by reward and salience, not only by frequency. They explicitly connect this to experience replay in deep RL. — [Kumaran, Hassabis & McClelland 2016, TiCS](https://web.stanford.edu/~jlmcc/papers/KumaranHassabisMcClelland16FinalMS.pdf) (verified)
- Schema-dependent rapid consolidation: once rats had a neocortical "schema" (6 flavor-place pairs in a familiar arena), new associations learned in one trial became hippocampus-independent within about 48 h. In an inconsistent arena the animals failed at one-trial learning. Prior structure is what makes one-shot learning and fast consolidation possible (SUPPORTED, influential single-lab paradigm). — [Tse et al. 2007, Science 316:76-82](https://pubmed.ncbi.nlm.nih.gov/17412951/) (verified)
- Model-Free Episodic Control (Blundell et al. 2016) is a tabular, kNN-based "hippocampal" controller that stores the highest return seen per (state embedding, action) and acts greedily on it. It learned faster than DQN-style agents early in training on Atari and Labyrinth. — [Blundell et al. 2016, arXiv:1606.04460](https://alphaxiv.org/abs/1606.04460) (verified)
- Neural Episodic Control (Pritzel et al. 2017, ICML) uses a differentiable per-action key-value memory (Differentiable Neural Dictionary) with n-step returns. It is designed to "rapidly assimilate new experiences and act upon them" and is far more data-efficient than DQN/A3C early on. — [Pritzel et al. 2017, PMLR 70:2827-2836](https://proceedings.mlr.press/v70/pritzel17a.html) (verified)
- Hippocampal "replay" of place-cell sequences during rest and sleep, including reverse replay after reward, is well documented in rodents (ESTABLISHED phenomenon; its causal role in planning is SUPPORTED but debated). — [Foster & Wilson 2006, Nature, doi:10.1038/nature04587](https://doi.org/10.1038/nature04587) (biblio)
- Normative account: replay prioritized by expected value of backup ("gain x need") explains many replay statistics. This maps onto prioritized experience replay. — [Mattar & Daw 2018, Nat. Neurosci., doi:10.1038/s41593-018-0232-z](https://doi.org/10.1038/s41593-018-0232-z) (biblio)

### Inferences
- The AutoManip finding that off-policy TD3 beats PPO by about 3x already fits CLS: replay buffers reuse rare events (a successful release) many times, while on-policy PPO discards them after one update. A demo-seeded, prioritized replay buffer is the cheapest CLS step.
- Experiment E1 (episodic release override): keep a small kNN memory over (low-dimensional state such as TCP-to-goal distance, grasp flag, gripper width) holding the best observed return per action bin. At decision time, blend `a = π(s)` with the memory's best action when the memory is confident (k nearest neighbours within radius r). Seed it with 1-2 scripted or teleoperated demos of PickCube or PlaceCube. Metric: environment steps to reach the strict place+release success threshold vs. TD3 baseline, across 5 seeds. Cheap: CPU-feasible on state observations.
- Experiment E2 (schema transfer à la Tse): pretrain on a family of place-target layouts (the "schema"), then present a new target with a single demo. Compare steps to success against a from-scratch learner. The prediction is a large gain only when the new task is consistent with the schema, and no gain for an "inconsistent arena" variant (e.g., an inverted gripper-sign convention). That gives a built-in negative control.
- Experiment E3 (gain x need replay): prioritize replay of transitions near the release decision by TD error times state-visitation. Compare against uniform replay and standard PER.

### Gaps
- Direct quantitative comparisons of MFEC/NEC on continuous-control manipulation (as opposed to Atari/Labyrinth) were not found in this session. Their benefit in continuous action spaces is not established.
- How much "schema" pretraining is needed before one-shot assimilation helps in robot tasks has no theory or known benchmark answer.

---

## Q2. Prefrontal cortex as meta-RL, learning-to-learn (Harlow), task schemas and structure learning, Bayesian program learning

### Takeaway
Learning-to-learn is one of the most robust phenomena in comparative psychology (Harlow 1949). A leading computational account says slow, dopamine-driven RL trains recurrent prefrontal dynamics to implement a fast, activity-based learning algorithm (Wang et al. 2018). Bayesian program induction (Lake et al. 2015) shows that one-shot generalization becomes possible when the learner has a compositional, causal generative prior. Both suggest building a task-family prior and doing fast inference over tasks, not fast gradient descent on weights.

### Cited Findings
- Harlow (1949) gave rhesus monkeys hundreds of different two-object discrimination problems. Across problems they came to solve new ones in essentially one trial ("learning sets", win-stay/lose-shift). He defined this as "learning how to learn efficiently in the situations the animal frequently encounters" (ESTABLISHED, widely replicated). — [Harlow 1949, Psych. Review 56:51-65](https://psyc.franklin.uga.edu/sites/default/files/inline-files/Harlow.pdf) (verified); historical review in [Medin 1972](https://groups.psych.northwestern.edu/medin/documents/Medin1972PsychBull.pdf) (verified)
- Wang et al. (2018, Nature Neuroscience 21:860-868): an LSTM "PFC" trained by standard deep RL, with dopamine-like reward-prediction-error signals adjusting weights, comes to implement its own learning algorithm in its activations. With frozen weights it adapts to new tasks. The model reproduced six meta-learning findings from neuroscience, including Harlow-style one-shot behavior (SUPPORTED as a computational account; the PFC mapping itself is a hypothesis). — [Wang et al. 2018, Nat. Neurosci.](https://www.nature.com/articles/s41593-018-0147-8.pdf) (verified); [DeepMind summary](https://deepmind.google/discover/blog/prefrontal-cortex-as-a-meta-reinforcement-learning-system/) (verified)
- The same "learning in activations" idea in ML: RL^2 (Duan et al. 2016, arXiv:1611.02779) and learning to reinforcement learn (Wang et al. 2016, arXiv:1611.05763). Probabilistic task-inference meta-RL: PEARL (Rakelly et al. 2019, arXiv:1903.08254). — [RL^2](https://arxiv.org/abs/1611.02779), [L2RL](https://arxiv.org/abs/1611.05763), [PEARL](https://arxiv.org/abs/1903.08254) (biblio)
- Gradient-based meta-learning (MAML, Finn, Abbeel & Levine 2017) and its one-shot imitation variant (Finn et al. 2017, "One-Shot Visual Imitation Learning via Meta-Learning", arXiv:1709.04905); also Duan et al. 2017 One-Shot Imitation Learning (arXiv:1703.07326). — [MAML arXiv:1703.03400](https://arxiv.org/abs/1703.03400), [One-shot visual imitation](https://arxiv.org/abs/1709.04905), [One-Shot Imitation](https://arxiv.org/abs/1703.07326) (biblio)
- Lake, Salakhutdinov & Tenenbaum (2015, Science 350:1332-1338), Bayesian Program Learning: concepts represented as compositional stroke programs and inferred by Bayesian criteria. It reached human-level one-shot classification of novel handwritten characters, beat deep nets of that era, and passed "visual Turing tests". The authors credit compositionality, causality and learning-to-learn. — [Lake et al. 2015 (CBMM record)](https://cbmm.mit.edu/publications/human-level-concept-learning-through-probabilistic-program-induction) (verified)
- Schemas and prior knowledge enable one-trial learning in rodents (Tse et al. 2007, above). — [PubMed 17412951](https://pubmed.ncbi.nlm.nih.gov/17412951/) (verified)

### Inferences
- Harlow and Wang 2018 together imply that few-shot ability comes from training on a distribution of related tasks. AutoManip trains on a single task, so it should not expect few-shot behavior to emerge. The minimal change is a procedurally varied ManiSkill task family (goal pose, cube size and mass, release height, place-vs-stack) with a task-conditioned or recurrent policy.
- Experiment E4 (Harlow test for robots): build N=20-50 PickPlace variants differing in a latent parameter (target region, required release height, "drop" vs "place gently"). Meta-train a recurrent TD3 or SAC (RL^2-style), or PEARL with a task-latent posterior. On held-out variants, measure success after episode 1, 2 and 3. The signature of learning-to-learn is a steep within-task curve that sharpens across meta-training, as in Harlow's curves. Cheap on state observations; the trusted evaluator can own the held-out variant distribution.
- Experiment E5 (program-like task prior): represent tasks as short sequences of parameterized primitives (reach(obj), grasp, move(goal), release, retreat). Infer the program and parameters from 1 demo by likelihood over the demo's end-effector trajectory and gripper events, then execute or refine with RL. Compare against BC from 1 demo. This is the robot analogue of BPL and directly gives "release" an explicit symbol.

### Gaps
- Wang et al. 2018's claim that PFC literally implements meta-RL is a model-to-data fit, not a causal demonstration. I found no direct causal test in this session.
- No source found quantifying how many meta-training tasks are needed for Harlow-like one-shot performance in continuous manipulation.

---

## Q3. Motor control mechanisms: primitives and synergies, internal and forward models (MOSAIC), adaptation timescales, contextual inference

### Takeaway
Motor learning is fast when the system (i) chooses among, or blends, existing modules (MOSAIC; COIN contextual inference) instead of relearning, (ii) runs a fast-learning, fast-forgetting process alongside a slow, retentive one (Smith 2006), and (iii) acts through a low-dimensional basis of primitives or synergies (debated as neural fact, useful as engineering prior). For AutoManip's release stall, the most pertinent idea is contextual inference: treat "holding" and "releasing" as discrete contexts with separate controllers and an inferred switch. One saturated monolithic gripper output should not have to represent both.

### Cited Findings
- Two-state model (Smith, Ghazizadeh & Shadmehr 2006, PLoS Biology 4:e179): adaptation comes from a fast process (high error sensitivity, poor retention) and a slow process (low sensitivity, good retention). Their interaction explains savings, anterograde interference, spontaneous recovery and rapid unlearning (ESTABLISHED and widely replicated in reaching and saccade adaptation). — [Smith et al. 2006 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1463025) (verified)
- COIN model (Heald, Lengyel & Wolpert 2021, Nature 600:489-493): one computation, contextual inference in a nonparametric Bayesian model, governs creating, updating and expressing motor memories. It reinterprets savings and spontaneous recovery as changes in context belief rather than in memory strength, and the authors tested new predictions (SUPPORTED; recent, from a strong group, and still being evaluated against two-state accounts). — [Heald et al. 2021 (Lengyel lab page)](https://cbl.eng.cam.ac.uk/lengyel/publications/heald-nature-2021) (verified); [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC8809113/) (verified)
- Internal models in the cerebellum: Wolpert & Kawato (1998, Neural Networks 11:1317-1329) proposed multiple paired forward (predictor) and inverse (controller) models, with "responsibility" signals from forward-model prediction accuracy. — [Wolpert & Kawato 1998, doi:10.1016/S0893-6080(98)00066-5](https://doi.org/10.1016/S0893-6080(98)00066-5) (biblio)
- MOSAIC (Haruno, Wolpert & Kawato 2001, Neural Computation 13:2201-2220) selects controllers before and during movement, using both feedforward and feedback information. It learns more efficiently than mixture-of-experts and generalizes to novel objects whose dynamics lie within the span of learned ones (SPECULATIVE as brain mechanism; SUPPORTED as computational model). — [Haruno et al. 2001 PDF](https://www.inf.ed.ac.uk/teaching/courses/mlsc/HW2papers/HarWolKaw.pdf) (verified)
- That the cerebellum supports error-driven adaptation and forward-model-like prediction is ESTABLISHED in broad outline (e.g., cerebellar patients fail to adapt to prism and force fields). Detailed module-switching architectures like MOSAIC are not established. — [Wolpert, Miall & Kawato 1998, TiCS, doi:10.1016/S1364-6613(98)01221-2](https://doi.org/10.1016/S1364-6613(98)01221-2) (biblio)
- Muscle synergies: d'Avella, Saltiel & Bizzi (2003, Nat. Neurosci. 6:300-308) showed frog behaviors reconstructed from combinations of a few time-varying muscle synergies. — reference listing in [Frontiers 2013 review](https://www.frontiersin.org/articles/10.3389/fncom.2013.00051) (verified)
- CONTESTED: Tresch & Jarc (2009, Curr. Opin. Neurobiol. 19:601-607) argue that synergies may reflect task and biomechanical constraints (few ways to do the task) rather than a neural control strategy. Bizzi & Cheung (2013) defend a neural origin from spinal microstimulation and NMDA iontophoresis evidence. — [Tresch & Jarc 2009 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2818278) (verified); [Bizzi & Cheung 2013, Front. Comput. Neurosci.](https://www.frontiersin.org/articles/10.3389/fncom.2013.00051) (verified)
- Engineering counterpart: Dynamic Movement Primitives (Ijspeert et al. 2013, Neural Computation 25:328-373) are learnable attractor systems fit from a single demonstration and generalized to new goals by changing a goal parameter. — [Ijspeert et al. 2013, doi:10.1162/NECO_a_00393](https://doi.org/10.1162/NECO_a_00393) (biblio)
- Error-sensitivity itself is learned: savings arise because the nervous system increases sensitivity to errors it has experienced before (Herzfeld et al. 2014, Science 345:1349-1353). — [doi:10.1126/science.1253138](https://doi.org/10.1126/science.1253138) (biblio)

### Inferences
- The release stall reads like a failure of contextual switching. A single Gaussian gripper head whose mean is driven to "closed" with ever-shrinking variance cannot express "closed until context = at-goal, then open". MOSAIC and COIN suggest a discrete latent (hold vs. release) with its own controller and a learned responsibility signal.
- Experiment E6 (context-gated gripper): replace the continuous gripper action with (a) a Bernoulli or categorical "phase" head with an entropy floor, or (b) two gripper sub-policies selected by a learned classifier on (object-at-goal, grasped) features. Compare the release success rate and steps to strict success against the baseline PPO and TD3 under identical seeds. It is cheap, state-based, and directly tests the diagnosed failure. It must respect AutoManip's v0 fixed action interface: implement it as policy-internal parameterization that still emits the same action vector.
- Experiment E7 (two-timescale learner): maintain fast weights (high learning rate, decay toward slow weights each update) plus slow weights, and act with their sum (Smith 2006 structure). Test for quick within-task adaptation when a task parameter shifts (e.g., goal moves) and for "spontaneous recovery" on switching back. Also test whether it reduces the need for many episodes after a single demo.
- Experiment E8 (primitive basis from 1-2 demos): fit DMPs or a small set of end-effector primitives from one teleop demo, let RL learn only the primitive parameters and switch times (residual RL on top), and measure sample efficiency. This also makes the release time an explicit, learnable scalar.

### Gaps
- No study found applying COIN-style nonparametric context inference to RL manipulation policies. This looks like an open direction.
- Whether synergies are neural or task-imposed remains unresolved. For robots it matters only as a prior, not a fact.

---

## Q4. Observational learning: mirror neurons, affordances, infant imitation

### Takeaway
Primates and humans clearly learn from observation, but the popular "mirror neurons = action understanding = imitation engine" story is CONTESTED and should not drive design. Infant "neonatal imitation" is also contested by a large longitudinal null result. The robust parts are (i) shared sensorimotor representations of goals and object affordances and (ii) a later-developing goal-directed (rather than movement-copying) imitation. For robots this argues for inferring the goal or effect of a demo (object pose change, contact or release events) rather than copying joint trajectories.

### Cited Findings
- Mirror neurons in macaque F5 fire both when the monkey performs and when it observes a goal-directed hand action (ESTABLISHED as a single-unit phenomenon). — [Rizzolatti et al. 1996, Cognitive Brain Research 3:131-141, doi:10.1016/0926-6410(95)00038-0](https://doi.org/10.1016/0926-6410(95)00038-0) (biblio); [Rizzolatti & Craighero 2004, Annu. Rev. Neurosci., doi:10.1146/annurev.neuro.27.070203.144230](https://doi.org/10.1146/annurev.neuro.27.070203.144230) (biblio)
- CONTESTED: Hickok (2009, J. Cogn. Neurosci. 21:1229-1243) lists "eight problems". He argues that no monkey data directly test the action-understanding theory and that human evidence "makes a strong case against" it, and proposes a sensorimotor-association account instead. — [Hickok 2009 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2773693) (verified); [Hickok & Hauser follow-up (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2911438) (verified)
- Alternative "associative sequence learning" view: mirror properties arise from correlated sensorimotor experience rather than innate adaptation (Cook, Bird, Catmur, Press & Heyes 2014, Behav. Brain Sci. 37:177-192). — [doi:10.1017/S0140525X13000903](https://doi.org/10.1017/S0140525X13000903) (biblio)
- Neonatal imitation: Meltzoff & Moore (1977, Science 198:75-78) reported newborns imitating facial gestures. — [doi:10.1126/science.198.4312.75](https://doi.org/10.1126/science.198.4312.75) (biblio)
- CONTESTED: Oostenbroek et al. (2016, Current Biology 26:1334-1338) tested 106 infants at about 1, 3, 6 and 9 weeks with 11 models. Infants produced gestures as often after control models as after matching ones, so there was no evidence of neonatal imitation. Nagy et al. (2018) reanalyzed the data and claimed a tongue-protrusion imitation effect; the original authors responded. — [Oostenbroek et al. 2016 (summary and debate)](https://controversciences.github.io/references/comprehensive-longitudinal-study-challenges-the-existence-of-neonatal-imitation-in-humans.html) (verified); [BPS Research Digest](https://bps.org.uk/research-digest/classic-finding-about-newborn-babies-imitation-skills-probably-wrong) (verified)
- Goal-directed, "rational" imitation in 14-month-olds: infants copied an unusual head-touch action only when the model's hands were free, suggesting inference of the demonstrator's goals and constraints (Gergely, Bekkering & Király 2002, Nature 415:755). — [doi:10.1038/415755a](https://doi.org/10.1038/415755a) (biblio)
- Affordances: Gibson (1979) proposed that perception directly specifies action possibilities offered by objects. In robotics this motivates predicting graspable or placeable regions from object geometry (a conceptual framework, not a testable neural claim). — [Gibson 1979, The Ecological Approach to Visual Perception (Routledge reissue, doi:10.4324/9781315740218)](https://doi.org/10.4324/9781315740218) (biblio)

### Inferences
- Design implication: few-shot imitation should extract what changed and when (object pose trajectory, contact onset, gripper-open event), not low-level actions. This also makes demos transferable across embodiments, as in goal-directed imitation.
- Experiment E9 (effect-based demo reward): from 1-2 demos, extract keyframes (grasp onset, lift, at-goal, release, retreat) using gripper-width and contact signals. Build a dense, stage-wise reward or goal sequence (an "intent" representation) for TD3 or SAC plus HER. Compare against BC-from-demo and against no-demo. It is especially diagnostic if the release keyframe alone fixes the stall. This must go through AutoManip's narrow, editable reward and curriculum interface, not the frozen evaluator.
- Experiment E10 (affordance prior): learn a small classifier for "stable place pose" from simulation rollouts and use it as a goal proposer. This tests whether affordance-level abstraction transfers across cube sizes and shapes with one demo.

### Gaps
- Mirror-neuron literature gives no algorithmic detail beyond "shared representations". I found no evidence that mirror systems implement one-shot imitation learning.
- I did not verify the Nagy et al. 2018 reanalysis details directly; only secondary summaries were seen.

---

## Q5. Theory: sample complexity of imitation learning and RL (error compounding, horizon dependence)

### Takeaway
Classical analysis says naive behavior cloning suffers quadratic-in-horizon error compounding (about T² ε), and interactive correction (DAgger) reduces this to roughly linear. Minimax results (Rajaraman et al. 2020) show the H² dependence is unavoidable in the worst case for tabular offline BC with deterministic experts. Recent work (Foster, Block & Misra 2024) shows that with log-loss and controlled policy-class complexity, BC can be horizon-independent. So "few demos" is viable when the policy class is small, structured, and trained with likelihood loss on a stochastic (not deterministically saturated) policy. That bears directly on the unbounded-certainty failure.

### Cited Findings
- Ross & Bagnell (2010) / Ross, Gordon & Bagnell (2011, AISTATS, PMLR 15:627-635): BC cost gap can scale as T² ε. DAgger, an iterative no-regret reduction, gets bounds that scale roughly linearly in the effective horizon. — [Ross et al. 2011, arXiv:1011.0686](https://arxiv.org/pdf/1011.0686) (verified); [PMLR v15](https://proceedings.mlr.press/v15/ross11a) (verified)
- Rajaraman, Yang, Jiao & Ramchandran (2020, NeurIPS): minimax limits for offline imitation in tabular episodic MDPs. BC reaches about |S|H² log(N)/N suboptimality; a lower bound of order |S|H²/N holds even for deterministic experts. With known transitions an improved algorithm gets about |S|H^{3/2}/N. — [Rajaraman et al. 2020 NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2020/file/1e7875cf32d306989d80c14308f3a099-Paper.pdf) (verified)
- Foster, Block & Misra (2024, NeurIPS spotlight): with logarithmic loss and controlled policy-class complexity, offline BC can achieve horizon-independent sample complexity. Without further assumptions, online IL cannot improve on offline log-loss BC, even in benign MDPs. — [Foster et al. 2024, arXiv:2407.15007](https://arxiv.org/abs/2407.15007) (verified)
- Context for RL sample complexity: in general MDPs, tabular RL needs sample counts polynomial in |S|, |A| and H. Lower bounds show exploration over long horizons is costly without structure. For AutoManip this explains why a sparse release reward late in a long episode is hard. — [Azar, Osband & Munos 2017, minimax regret, arXiv:1703.05449](https://arxiv.org/abs/1703.05449) (biblio)

### Inferences
- The AutoManip "unbounded certainty to keep gripper closed" maps onto a theory-relevant property. Log-loss BC analyses assume the learner can represent the expert's conditional action distribution. A deterministically saturated gripper head (variance near 0, mean far past the clipping bound) makes the policy's log-likelihood of "open" vanishingly small. Gradients for correcting it then vanish, an optimization pathology on top of the statistical one.
- Experiment E11 (log-loss BC + bounded gripper): train BC from 1, 2, 5 and 10 demos with (a) MSE on actions, (b) Gaussian log-loss with a minimum std, (c) a Bernoulli gripper head with logit clipping. Report success vs. demo count and episode horizon (vary the time limit) to empirically check horizon dependence in ManiSkill. This is cheap and tests Foster et al.'s claim in a manipulation setting.
- Experiment E12 (DAgger with a scripted expert): ManiSkill PickCube permits a scripted motion-planning expert. Run DAgger vs. BC at equal expert-label budgets to quantify compounding error, specifically at the release step.

### Gaps
- No sharp theory found for few-shot imitation with continuous actions and high-dimensional states. The bounds above are tabular or depend on abstract policy-class complexity.
- The Foster et al. theorem statements were not checked in full text in this session.

---

## Q6. Theory of learning-to-learn: Baxter, PAC-Bayes meta-learning, Bayesian task inference

### Takeaway
Theory formalizes the Harlow intuition. If tasks are drawn from an environment (task distribution) and share structure (a common representation or prior), then sample complexity per new task can drop dramatically once that structure is learned from enough tasks. The per-task cost can approach the cost of learning only the task-specific parameters. PAC-Bayes meta-learning makes this concrete as "learn a prior over policies from many tasks; then a few demos update a posterior". Bounds are loose numerically but give the right scaling message: few-shot ability is paid for with task diversity.

### Cited Findings
- Baxter (2000, JAIR 12:149-198), "A Model of Inductive Bias Learning": extends PAC learning to learning a hypothesis space (bias) from multiple tasks sampled from an environment. It shows that learning a shared representation from many tasks reduces samples needed per new task, including for neural feature learning. — [Baxter 2000, arXiv:1106.0245](https://arxiv.org/abs/1106.0245) (verified)
- Pentina & Lampert (2014, ICML), PAC-Bayesian lifelong learning: bounds on transfer risk when a "hyper-prior" is learned over tasks. — [Pentina & Lampert 2014, PMLR 32](https://proceedings.mlr.press/v32/pentina14.html) (biblio); setup confirmed as basis for [Amit & Meir 2018](https://arxiv.org/abs/1711.01244v1) (verified)
- Amit & Meir (2018, ICML): "Meta-Learning by Adjusting Priors Based on Extended PAC-Bayes Theory" derives meta-learning PAC-Bayes bounds and a gradient-based algorithm for deep nets that learns the prior by minimizing the bound. — [Amit & Meir 2018, arXiv:1711.01244](https://arxiv.org/abs/1711.01244v1) (verified)
- Representation-learning theory for few-shot: with a shared low-dimensional representation learned from T source tasks, target-task error scales roughly with the representation dimension rather than the ambient dimension, given enough source tasks and task diversity (Tripuraneni, Jordan & Jin 2020; Du, Hu, Kakade, Lee & Lei 2021). — [Tripuraneni et al. 2020, arXiv:2002.11684](https://arxiv.org/abs/2002.11684); [Du et al. 2021, arXiv:2002.09434](https://arxiv.org/abs/2002.09434) (biblio)
- Meta-RL as Bayes-adaptive control: the optimal few-shot adapting agent is the Bayes-optimal policy in the BAMDP over the task posterior. VariBAD approximates this with a learned task-belief encoder (Zintgraf et al. 2020, ICLR). — [Zintgraf et al. 2020, arXiv:1910.08348](https://arxiv.org/abs/1910.08348) (biblio)

### Inferences
- A concrete, AutoManip-sized prediction from Baxter, Tripuraneni and Du: held-out few-shot performance should improve with the number and diversity of meta-training task variants, and saturate once the shared structure is captured. Experiment E13: train on K in {1, 4, 16, 64} PickPlace variants (fixed total env steps), then measure 1- and 2-demo adaptation on held-out variants. Plot adaptation success vs. K. This directly tests the "task diversity buys few-shot" claim, and the frozen evaluator can own the held-out variant split.
- Experiment E14 (learned prior as regularizer): fine-tune from a meta-learned prior with a KL or L2 penalty to the prior mean (PAC-Bayes-style), vs. unregularized fine-tuning, from 1-2 demos. Measure overfitting to the demo (success on perturbed initial states).

### Gaps
- No PAC-Bayes meta-RL bound found that is numerically non-vacuous for continuous-control manipulation.
- Tripuraneni et al. and Du et al. results are for supervised regression or linear settings. Extension to RL with distribution shift is incomplete in what I found.

---

## Q7. Invariance and equivariance as sample-complexity reducers (equivariant manipulation policies)

### Takeaway
Building known symmetries (planar rotation of the scene, translation, permutation of objects) into the architecture has both a provable generalization benefit (strict for linear models; Elesedy & Zaidi 2021) and large empirical sample-efficiency gains in manipulation RL and imitation (Wang, Walters & Platt 2022; Equivariant Diffusion Policy 2024). Among the strongest few-shot levers with direct robot evidence, and cheap to test on state observations by expressing positions relative to the gripper and applying SO(2) about the vertical axis.

### Cited Findings
- Elesedy & Zaidi (2021, ICML, PMLR 139:2959-2969): first provably strict (non-zero) generalization gain for invariant or equivariant models in linear least-squares when the target is equivariant under a compact group. The gain depends on sample size and properties of the group action. — [Elesedy & Zaidi 2021](https://proceedings.mlr.press/v139/elesedy21a.html) (verified)
- Wang, Walters & Platt (2022, ICLR spotlight), SO(2)-Equivariant RL: derive equivariance/invariance of the optimal Q-function and policy under planar rotations. Equivariant DQN and SAC (plus an equivariant LfD variant) "outperform recent data-augmentation baselines by a wide margin" on manipulation. — [Wang et al. 2022, arXiv:2203.04439](https://arxiv.org/pdf/2203.04439.pdf) (verified)
- Equivariant Diffusion Policy (Wang, Hart, Surovik, ..., Walters, Platt, CoRL 2024): SO(2)-equivariant denoiser. About 21.9% higher average success than Diffusion Policy across 12 MimicGen tasks, and effective real-robot policies from relatively few demos where the baseline fails. The baseline in the 21.9% figure is described inconsistently across summaries. — [Wang et al. 2024, PMLR v270](https://proceedings.mlr.press/v270/wang25a.html) (verified); [arXiv:2407.01812](https://arxiv.org/abs/2407.01812v3) (verified)
- Related one-shot/few-shot manipulation with SE(3) equivariance: Neural Descriptor Fields learn pick-and-place from about 5-10 demos with generalization to novel poses (Simeonov et al. 2022, ICRA). — [arXiv:2112.05124](https://arxiv.org/abs/2112.05124) (biblio)
- Transporter Networks: translation-equivariant pick-and-place from few demos (Zeng et al. 2020, CoRL). — [arXiv:2010.14406](https://arxiv.org/abs/2010.14406) (biblio)

### Inferences
- In state-based PickCube, much of the symmetry can be had for free with egocentric, goal-relative features (cube-minus-TCP, goal-minus-cube, rotated into a frame aligned with the cube yaw) or with an e2cnn/escnn SO(2) MLP. AutoManip v0 fixes observation and action interfaces, so this must be a policy-internal featurization, which is permitted as long as the raw obs/action vectors are unchanged.
- Experiment E15 (equivariant TD3): compare a TD3 actor/critic with (a) raw features, (b) goal-relative canonicalized features, (c) a discrete C8 or C16 equivariant MLP. Report steps to strict success and 1-5-demo BC success under rotated goal and cube yaw distributions. Prediction: (b) and (c) cut sample needs substantially, with the biggest gains on held-out rotations.

### Gaps
- No ManiSkill-specific equivariant-policy benchmark numbers found in this session.
- When symmetry is broken (gravity, table edge, robot base kinematics), gains shrink. Quantifying "approximate equivariance" benefits for arm-mounted tasks was not covered by the sources found.

---

## Q8. Object-centric, causal and compositional representations

### Takeaway
Causal and object-centric factorization is the main theoretical argument for out-of-distribution few-shot transfer. If a new task changes only one mechanism (one object's goal, one constraint), only that module needs relearning (the "sparse mechanism shift" hypothesis; Schölkopf et al. 2021). Compositionality, as in BPL, lets a few examples specify a new combination of known parts. Unsupervised discovery of such factors is provably impossible without inductive biases (Locatello et al. 2019). So supply structure: state-based ManiSkill already provides object poses, which is an advantage.

### Cited Findings
- Schölkopf, Locatello, Bauer, Ke, Kalchbrenner, Goyal & Bengio (2021, Proc. IEEE 109:612-634), "Toward Causal Representation Learning": identifies discovering high-level causal variables from low-level observations as a central challenge. Argues causal and modular mechanism structure underpins transfer and generalization, including the independent-mechanisms and sparse-mechanism-shift principles. — [Schölkopf et al. 2021, arXiv:2102.11107](https://arxiv.org/abs/2102.11107v1) (verified); [MPI record](https://ps.is.tuebingen.mpg.de/publications/scholkopfetal21) (verified)
- Locatello et al. (2019, ICML best paper): unsupervised learning of disentangled representations is fundamentally impossible without inductive biases on both models and data. — [arXiv:1811.12359](https://arxiv.org/abs/1811.12359) (biblio)
- BPL (Lake et al. 2015) attributes one-shot success to compositionality, causality and learning-to-learn. — [CBMM record](https://cbmm.mit.edu/publications/human-level-concept-learning-through-probabilistic-program-induction) (verified)
- Building machines that learn and think like people: an argument that intuitive physics, intuitive psychology, compositionality and learning-to-learn are required for human-like few-shot learning (Lake, Ullman, Tenenbaum & Gershman 2017, Behav. Brain Sci. 40:e253). — [arXiv:1604.00289](https://arxiv.org/abs/1604.00289) (biblio)

### Inferences
- Experiment E16 (sparse-mechanism-shift test): pretrain a modular policy with separate heads or modules for reach, grasp, transport and release, or an object-factored critic. Then change only the release mechanism, e.g., require a gentle place below a velocity threshold or a placement inside a bin. Measure whether fine-tuning only the release module from 1-2 demos matches full fine-tuning. Prediction from the causal view: modular fine-tuning is much more sample-efficient and causes less forgetting of earlier stages.
- Experiment E17 (compositional generalization): train pick and place sub-skills on separate task variants, then test zero- or one-shot on a new composition (pick A, place on B, stack). This measures whether the representation composes.

### Gaps
- Formal compositional-generalization theory for RL policies is thin. I found no single sample-complexity theorem specific to compositional skill reuse in manipulation.
- Causal representation learning identifiability results are mostly for observational or interventional data in toy settings. Direct manipulation evidence was not found in this session.

---

## Q9. Synthesis: mechanism-to-experiment map, evidence strength, and priority for AutoManip

### Takeaway
Ranked by (relevance to the release stall) x (cost) x (strength of evidence), the first experiments should be (1) a context-gated or bounded-entropy gripper head (MOSAIC/COIN plus BC log-loss theory), (2) demo-seeded prioritized replay with an episodic memory override (CLS/NEC), (3) goal-relative or equivariant features (equivariance theory plus robot evidence), and (4) keyframe-based effect rewards from 1-2 demos (goal-directed imitation). Meta-RL over a task family (Harlow/Wang/Baxter) is the route to genuine "show it once" behavior. It is larger, so it should follow once single-task learning is reliable.

### Cited Findings (summary table; sources as in Q1-Q8)

| # | Mechanism (source, year) | Evidence status | Algorithmic idea | Cheap ManiSkill test |
|---|---|---|---|---|
| E6 | Contextual inference / module selection (Heald, Lengyel & Wolpert 2021; Haruno, Wolpert & Kawato 2001) | COIN SUPPORTED (recent); MOSAIC as brain mechanism SPECULATIVE | Discrete hold/release context with own controller plus learned switch; entropy floor | PickCube/PlaceCube release success and steps-to-strict-success vs. PPO/TD3, 5 seeds |
| E11 | Log-loss BC horizon-independence (Foster, Block & Misra 2024); compounding error (Ross et al. 2011; Rajaraman et al. 2020) | Theory (proved under assumptions) | Stochastic, bounded-confidence gripper head; log-loss BC from 1-10 demos | Success vs. #demos and vs. time-limit horizon |
| E1/E3 | CLS + episodic control + prioritized replay (McClelland 1995; Kumaran et al. 2016; Blundell 2016; Pritzel 2017; Mattar & Daw 2018) | CLS ESTABLISHED framework; replay ESTABLISHED; episodic-control gains SUPPORTED (Atari) | kNN episodic memory overriding policy near release; demo-seeded PER | Steps to success with 1-2 demos vs. TD3 |
| E15 | Equivariance (Elesedy & Zaidi 2021; Wang, Walters & Platt 2022; EquiDiff 2024) | Theory (linear) + strong robot evidence | Goal-relative canonical features or C_n-equivariant actor/critic | Rotated-goal generalization, steps to success |
| E9 | Goal-directed imitation (Gergely et al. 2002); against mirror-neuron "copying" (Hickok 2009) | Goal inference SUPPORTED; mirror-neuron action understanding CONTESTED; neonatal imitation CONTESTED (Oostenbroek 2016) | Extract keyframes/effects from demo; stage reward via editable reward interface | Does release-keyframe reward alone fix stall? |
| E7 | Two-state adaptation (Smith et al. 2006) | ESTABLISHED | Fast + slow weights | Adaptation to mid-training goal shift; spontaneous recovery |
| E8/E5 | Primitives/DMPs (d'Avella 2003; Ijspeert 2013); program induction (Lake 2015) | Synergies CONTESTED as neural (Tresch & Jarc 2009); DMPs established engineering tool; BPL strong in its domain | Fit primitives from 1 demo; RL over parameters and switch times | Sample efficiency of residual RL over 1-demo DMP |
| E4/E13 | Learning sets (Harlow 1949); PFC meta-RL (Wang 2018); Baxter 2000; PAC-Bayes meta (Pentina & Lampert 2014; Amit & Meir 2018) | Harlow ESTABLISHED; PFC=meta-RL SUPPORTED/hypothesis; theory proved | Recurrent/latent-task meta-RL over a task family | Held-out-variant success after 1-3 episodes vs. #train variants K |
| E2 | Schemas enable one-trial learning (Tse 2007) | SUPPORTED | Schema pretraining + consistent/inconsistent task control | 1-demo gain on schema-consistent vs. inconsistent variants |
| E16/E17 | Causal/modular structure (Schölkopf 2021; Locatello 2019) | Conceptual + impossibility theorem | Modular per-stage policy; fine-tune only changed module | Sample efficiency and forgetting under release-mechanism change |

### Inferences
- The neuroscience and the theory converge on a few messages. Fast learning is rarely fast weight change: it is selection or inference over pre-structured hypotheses (contexts, schemas, task latents, primitives, symmetry-reduced policy classes) plus verbatim storage of rare high-value episodes. AutoManip's current setup (single task, monolithic Gaussian policy, on-policy PPO) lacks all of these.
- The release stall is a good first target because several independent lines (MOSAIC/COIN context switching, log-loss BC theory, episodic control) all predict that an explicit, bounded-confidence, context-gated gripper decision fixes it. That makes the result informative either way.
- Validation hygiene per AGENTS.md: each experiment above should be run with paired development episodes, independent confirmation seeds and repeated training seeds, with env steps and compute logged. Demo counts must be treated as part of the compute budget. Report ManiSkill stock "success" separately from strict place-and-release success.
- Protocol caution: experiments that change reward (E9), task distributions (E2, E4, E13) or held-out variants must keep the held-out distribution and evaluator under the trusted, frozen side. Only the training-side reward and curriculum interfaces should be agent-editable.

### Gaps
- No source found that directly benchmarks "show once, then self-practice" (one demo plus autonomous RL) on ManiSkill PickCube. Representative related methods (demo-augmented off-policy RL such as DDPGfD, Vecerik et al. 2017, arXiv:1707.08817 (biblio); RLPD, Ball et al. 2023, arXiv:2302.02948 (biblio)) were not reviewed in depth here and likely belong to an adjacent research thread.
- Several citations are marked (biblio) and were not re-fetched in this session; page numbers and specific figures for those should be spot-checked before publication.
