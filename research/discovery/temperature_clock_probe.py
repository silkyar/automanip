"""Fixed-entropy temperature clock check, no actor or robot learning."""
import hashlib,json,time
from pathlib import Path
import torch

started=time.monotonic();torch.set_num_threads(1);rows=[]
for initial in (1.,.005):
    log_alpha=torch.tensor(initial,dtype=torch.float64).log().requires_grad_()
    optimizer=torch.optim.Adam([log_alpha],lr=.001,betas=(.5,.999))
    for _ in range(2029):
        # Fixed mean logprob−1 and target entropy−8, not a learned-policy trace.
        loss=-log_alpha.exp()*(-1.-8.)
        optimizer.zero_grad();loss.backward();optimizer.step()
    rows.append(dict(initial_alpha=initial,final_alpha=float(log_alpha.detach().exp()),
        log_alpha_displacement=float(log_alpha.detach()-torch.tensor(initial,dtype=torch.float64).log()),
        scalar_adam_steps=2029))
ratio=rows[1]['final_alpha']/rows[0]['final_alpha']
assert abs(ratio-.005)<1e-6
r=dict(state='passed',seconds=time.monotonic()-started,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    fixed_mean_logprob=-1.,target_entropy=-8.,records=rows,final_alpha_ratio=ratio,
    scalar_optimizer_calls=4058,neural_optimizer_calls=0,actor_updates=0,robot_transitions=0,
    conclusion='With fixed entropy, Adam approximately preserves the initial alpha scale across the fast recipe update count. This exposes an interaction-relative warmup, not an actual actor learning prediction.',
    limits='Policy entropy/critic feedback held fixed; cannot predict grasping, final temperature, loss stability or sample efficiency.')
p=Path(__file__).with_name('temperature-clock-probe-20261028.json')
if p.exists():raise FileExistsError('Preserve receipt')
p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
