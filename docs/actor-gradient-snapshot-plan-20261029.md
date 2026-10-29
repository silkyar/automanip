# Does the saved actor receive a local signal to change its saturated actions?

Registered before executing the CPU snapshot diagnostic. The attempted physical
intervention failed its paired-state gate, so it cannot establish whether an
upward action would lift the cube. Do not retry that experiment or infer its
unobserved outcomes.

Use only the existing low-alpha 256k actor/two online critics and its 1,904
grasp-flagged successor observations. No new simulator query, critic fit,
optimizer update or protected evaluation. Missing target networks and replay
prevent reconstruction of a historical training update; this is a current
surrogate-gradient measurement only.

Competing descriptions: the saved critics may locally prefer the current
near-saturated actions; or they may provide a material action gradient that is
attenuated by tanh before reaching the Gaussian mean. Entropy may reinforce or
oppose that signal. Neither description establishes correct critic values or
a physical lifting direction.

Measure deterministic `min(Q1,Q2)` gradients with respect to bounded action,
their exact tanh-transformed mean gradients, and the entropy contribution.
Also measure the reparameterized current SAC surrogate on exactly 16 fixed
standard-normal samples per saved state (NumPy generator seed 93294000).
Report every joint, signs, saturation derivatives, and component magnitudes.
Count actor/critic rows and derivative work separately; this is additional CPU
compute, not extra experience. Pin input/source hashes and check safe policy
inference parity. Verify the entropy derivative `2*alpha*action` and the Q chain
rule numerically. A finite-difference check may cross ReLU boundaries and should
be reported as a check, not silently retuned until it passes.

Report observations without declaring a learning bottleneck solely from this
snapshot. Any proposed intervention must subsequently distinguish a causal
mechanism against matched controls and preserve the frozen benchmark.
