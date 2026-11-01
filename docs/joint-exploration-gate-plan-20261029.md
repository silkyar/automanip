# Joint versus coordinatewise exploration with matched conditional marginals

CPU gate before implementing or admitting a learner. The randomized physical
probe identified a combined arm/closure rescue; frozen coverage shows small
command mass in many pretreatment states. Neither establishes a learning remedy.

Fix epsilon=.1. Vector mixture is `.9*pi(a|s)+.1*Uniform([-1,1]^8)`; coordinate
mixture is `product_i(.9*pi_i(a_i|s)+.1*Uniform([-1,1]))`. With the diagonal
Gaussian actor, the two have **identical full conditional one-dimensional
marginals at the same fixed actor/state**, hence identical per-coordinate means
and variances. They differ in cross-coordinate dependence, not temporal memory.
Training feedback would change each arm's actor/states, so empirical marginals
over separate learning runs are not promised identical.

For two independent event regions with original probabilities p1,p2 and uniform
probabilities u1,u2, the joint versus coordinate event probabilities differ by
`eps*(1-eps)*(u1-p1)*(u2-p2)`. Check rare/rare p=.001,u=.25 (roughly9.26x),
single-coordinate equality, and a mixed rare/common negative control where the
vector mixture is worse. No universal improvement or novelty claim.

Reuse all78 pretreatment states and original fixed512/state Gaussian/uniform
arm-FK estimates from the coverage receipt. Sample only the additional
coordinatewise arm mixture: same regenerated Gaussian/uniform draws, independent
coordinate mask generator93314300. Apply exactly the previous primary valid
`q+.1*a` target /TCP rise>=5mm definition; no changed endpoint or hidden clipping.
Use exact gripper CDFs. Primary proxy is requested gripper target<=10mm. Derive
vector-mixture joint mass from the cached components, and coordinate-mixture mass
from the new arm sample and its matched mixed gripper marginal. Retain all states,
MC counts/resolution/intervals, input/source hashes and analytic work.

The cheap directional gate requires: mathematical positive/negative controls
pass; vector median primary proxy mass >=5 times coordinate median, positive
vector median; and vector mean >=coordinate mean. Report MC uncertainty without
claiming these finite estimates prove a distributional inequality. Missing/failed
gate means no automatic robot learner. Passing admits only a separately reviewed
proposal, not a GPU launch or sample-efficiency claim.

If later tested, this changes the **behavior collector only**. SAC actor, critic,
target-policy log probabilities and temperature objective remain original. Always
draw the original policy proposal once per post-warmup step in both arms, with
mask/uniform draws on a dedicated generator; keep initial/warmup/global learner
RNG counts unchanged. Stored executed actions must not be labelled as iid-policy
samples/log probabilities. No teacher, task-stage oracle, controller or diagnostic
data enters training. Compare against the untouched cached low-alpha reference
as well as the matched-marginal control; beating a damaged control alone is not
an improvement. Any training requires a new protocol and exact-source review.

Related primary work motivates exploration coordination, not novelty or direct
transfer. MAVEN uses shared latent exploration in decentralized multi-agent
control, a different setting. [Mahajan et al.,2019](https://papers.neurips.cc/paper/8978-maven-multi-agent-variational-exploration.pdf).
SAC's replay-based critic and policy expectations support separating a behavior
collector from its target policy, without guaranteeing finite-network performance.
[Haarnoja et al.,2019](https://arxiv.org/pdf/1812.05905).
