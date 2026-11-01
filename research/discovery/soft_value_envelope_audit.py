"""Frozen numerical consistency audit. No fitting, simulator, cloud or evaluator."""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from replay_learner_lowalpha import Learner

ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/'research/discovery/soft-value-envelope-plan-20261029.json'
started=time.monotonic();torch.set_num_threads(1)
plan=json.loads(PLAN.read_bytes())
if not plan['admitted']:raise PermissionError('Not admitted')
if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=plan['audit_source_sha256']:raise ValueError('Source differs')
for name,expected in plan['source_pins'].items():
 if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()!=expected:raise ValueError('Pinned source differs')

def stats(value):
 x=value.detach().cpu().numpy().astype(float).reshape(-1)
 return dict(min=float(x.min()),q01=float(np.quantile(x,.01)),median=float(np.median(x)),q99=float(np.quantile(x,.99)),max=float(x.max()),mean=float(x.mean()))

def digest(learner):
 h=hashlib.sha256()
 for prefix,module in [('actor',learner.actor),('q1',learner.q1),('q2',learner.q2),('target1',learner.target1),('target2',learner.target2)]:
  for key,value in module.state_dict().items():h.update((prefix+key).encode());h.update(value.detach().numpy().tobytes())
 h.update(learner.log_alpha.detach().numpy().tobytes());return h.hexdigest()

def run():
 pools={};arrays={};cells=[];snapshots=[];parity=[]
 work=dict(actor_forward_rows=0,critic_head_forward_rows=0,Gaussian_sample_rows=0,optimizer_calls=0)
 for case in plan['cases']:
  path=ROOT/case['trajectory'];raw=path.read_bytes()
  if hashlib.sha256(raw).hexdigest()!=case['trajectory_sha256']:raise ValueError('Trajectory differs')
  z=np.load(path,allow_pickle=False)
  data={k:torch.from_numpy(z[k][::4].reshape(-1,48 if k in ('obs','next_obs') else 8 if k=='action' else 1).copy()).float() for k in ('obs','next_obs','action','reward','stop')}
  if data['obs'].shape!=(832,48) or data['stop'].shape!=(832,1):raise ValueError('Shape mismatch')
  if not all(torch.isfinite(v).all() for v in data.values()):raise ValueError('Nonfinite pool')
  if not bool(((data['stop']==0)|(data['stop']==1)).all()):raise ValueError('Nonbinary true-terminal mask')
  if not bool(((data['reward']>=-1e-6)&(data['reward']<=1+1e-6)).all()):raise ValueError('Observed reward range differs')
  pools[case['name']]=data
 for si,case in enumerate(plan['cases']):
  if time.monotonic()-started>80:raise RuntimeError('CPU audit cap')
  raw=(ROOT/case['state']).read_bytes()
  if hashlib.sha256(raw).hexdigest()!=case['state_sha256']:raise ValueError('Snapshot differs')
  saved=np.load(ROOT/case['state'],allow_pickle=False);learner=Learner('target-sac',device='cpu')
  for prefix,module in [('actor',learner.actor),('q1',learner.q1),('q2',learner.q2),('target1',learner.target1),('target2',learner.target2)]:
   module.load_state_dict({k:torch.from_numpy(saved[prefix+'.'+k].copy()) for k in module.state_dict()},strict=True)
  with torch.no_grad():learner.log_alpha.copy_(torch.from_numpy(saved['log_alpha'].copy()))
  before=digest(learner);alpha=float(learner.log_alpha.exp());gamma=learner.gamma
  bj=learner.actor.mean.weight.detach().double().abs().sum(1)+learner.actor.mean.bias.detach().double().abs()
  c=math.sqrt(2/math.pi);constant=min(.5*math.log(2*math.pi*math.e)+ell-2*math.exp(ell)*c for ell in (-5.,2.))
  hmin=float((constant-2*bj.double()).sum());hmax=8*math.log(2)
  lower=gamma*alpha*hmin/(1-gamma);upper=(1+gamma*alpha*hmax)/(1-gamma)
  snapshots.append(dict(name=case['name'],alpha=alpha,mean_coordinate_absolute_bounds=bj.tolist(),entropy_lower=hmin,entropy_upper=hmax,conditional_expected_Q_interval=[lower,upper]))
  # Exact production formula gate on a fixed64-row block; CPU only.
  data=next(iter(pools.values()));small={k:v[:64] for k,v in data.items()}
  with torch.no_grad():
   torch.manual_seed(93502200);actual_a,actual_lp=learner.actor.sample(small['next_obs']);work['actor_forward_rows']+=64
   torch.manual_seed(93502200);mu,ell=learner.actor(small['next_obs']);u=mu+torch.randn_like(mu)*ell.exp();manual_a=u.tanh()
   lp=(torch.distributions.Normal(mu,ell.exp()).log_prob(u)-2*(math.log(2)-u-F.softplus(-2*u))).sum(-1,keepdim=True)
   work['actor_forward_rows']+=64;work['Gaussian_sample_rows']+=128
   if not torch.allclose(actual_a,manual_a,atol=2e-5,rtol=2e-5) or not torch.allclose(actual_lp,lp,atol=2e-5,rtol=2e-5):raise ValueError('Actor reconstruction differs')
   q1,q2,reference=learner.critic_target(small,actual_a,actual_lp,torch.tensor(alpha));work['critic_head_forward_rows']+=4*64
   future=torch.minimum(learner.target1(small['next_obs'],actual_a),learner.target2(small['next_obs'],actual_a));work['critic_head_forward_rows']+=2*64
   reconstructed=small['reward']+gamma*(1-small['stop'])*future-gamma*(1-small['stop'])*alpha*actual_lp
   if not torch.allclose(reference,reconstructed,atol=2e-5,rtol=2e-5):raise ValueError('Production target differs')
   parity.append(dict(snapshot=case['name'],actor_action_max_error=float((actual_a-manual_a).abs().max()),actor_logprob_max_error=float((actual_lp-lp).abs().max()),target_max_error=float((reference-reconstructed).abs().max())))
  for pi,(poolname,data) in enumerate(pools.items()):
   if time.monotonic()-started>80:raise RuntimeError('CPU audit cap')
   n=len(data['obs']);draws=plan['noise_draws'];gen=torch.Generator().manual_seed(plan['noise_seed'])
   with torch.no_grad():
    mean,ell=learner.actor(data['next_obs']);work['actor_forward_rows']+=n
    noise=torch.randn(draws,n,8,generator=gen);u=mean[None]+ell.exp()[None]*noise;a=u.tanh()
    normal=torch.distributions.Normal(mean[None],ell.exp()[None]);perdim=normal.log_prob(u)-2*(math.log(2)-u-F.softplus(-2*u));lp=perdim.sum(-1,keepdim=True)
    obs=data['next_obs'][None].expand(draws,n,48).reshape(-1,48);act=a.reshape(-1,8)
    t1=learner.target1(obs,act).reshape(draws,n,1);t2=learner.target2(obs,act).reshape(draws,n,1)
    work['critic_head_forward_rows']+=2*draws*n;work['Gaussian_sample_rows']+=draws*n
    online1=learner.q1(data['obs'],data['action']);online2=learner.q2(data['obs'],data['action'])
    fixedt1=learner.target1(data['obs'],data['action']);fixedt2=learner.target2(data['obs'],data['action']);work['critic_head_forward_rows']+=4*n
    factor=gamma*(1-data['stop'])[None];reward=data['reward'][None].expand(draws,n,1)
    critic=factor*torch.minimum(t1,t2);entropy=-factor*alpha*lp;target=reward+critic+entropy
    residual1=online1[None]-target;residual2=online2[None]-target
    key=f's{si}_p{pi}'
    for name,value in dict(reward=reward,critic=critic,entropy=entropy,target=target,lp=lp,t1=t1,t2=t2,residual1=residual1,residual2=residual2,online1=online1,online2=online2,fixedt1=fixedt1,fixedt2=fixedt2).items():arrays[key+'_'+name]=value.numpy()
    tails=[]
    indexes=np.argsort(-target.abs().numpy().reshape(-1),kind='stable')[:8]
    for index in indexes:
     draw=int(index//n);row=int(index%n)
     tails.append(dict(draw=draw,row=row,reward=float(reward[draw,row,0]),stop=float(data['stop'][row,0]),critic_component=float(critic[draw,row,0]),entropy_component=float(entropy[draw,row,0]),target=float(target[draw,row,0]),target_q1=float(t1[draw,row,0]),target_q2=float(t2[draw,row,0]),online_behavior_q1=float(online1[row,0]),online_behavior_q2=float(online2[row,0]),per_coordinate_logprob=perdim[draw,row].tolist(),next_mean=mean[row].tolist(),next_std=ell[row].exp().tolist(),sampled_next_action=a[draw,row].tolist(),next_cubeA_position=data['next_obs'][row,25:28].tolist(),next_tcp_position=data['next_obs'][row,18:21].tolist(),next_robot_qvel_norm=float(data['next_obs'][row,9:18].norm())))
    def outside(q):
     tolerance=1e-3+1e-5*max(abs(lower),abs(upper))
     return dict(below_fraction=float((q<lower-tolerance).float().mean()),above_fraction=float((q>upper+tolerance).float().mean()),numerical_margin=tolerance)
    finger_sum=data['obs'][:,7:9].sum(-1)
    next_fingers=data['next_obs'][:,7:9]
    finger_check=dict(next_per_joint_min=next_fingers.min(0).values.tolist(),next_per_joint_max=next_fingers.max(0).values.tolist(),next_strict_legal_outside_fraction=float(((next_fingers<0)|(next_fingers>.04)).float().mean()),next_outside_with_10micrometer_tolerance_fraction=float(((next_fingers<-1e-5)|(next_fingers>.04001)).float().mean()),conditional_bound_only=True)
    cells.append(dict(snapshot=case['name'],pool=poolname,rows=n,draws=draws,summary={name:stats(value) for name,value in dict(reward=reward,critic_component=critic,entropy_component=entropy,Y=target,target_q1=t1,target_q2=t2,online_behavior_q1=online1,online_behavior_q2=online2,target_behavior_q1=fixedt1,target_behavior_q2=fixedt2,current_q1_minus_Y=residual1,current_q2_minus_Y=residual2).items()},expected_Q_screen={name:outside(value) for name,value in dict(online_behavior_q1=online1,online_behavior_q2=online2,target_behavior_q1=fixedt1,target_behavior_q2=fixedt2,target_next_q1=t1,target_next_q2=t2).items()},observed_pre_step_finger_sum_range=[float(finger_sum.min()),float(finger_sum.max())],reward_state_fingers=finger_check,top8_abs_targets=tails))
  if before!=digest(learner):raise ValueError('Snapshot parameter mutation')
 return dict(state='complete',snapshots=snapshots,parity=parity,cells=cells,work=work,environment_transitions=0,GPU_seconds=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),elapsed_seconds=time.monotonic()-started,caveat=plan['scope']),arrays

initial_rng=torch.get_rng_state().clone()
with torch.random.fork_rng(devices=[]):result,arrays=run()
if not torch.equal(initial_rng,torch.get_rng_state()):raise ValueError('Global CPU RNG changed')
result['global_CPU_RNG_restored']=True
output=ROOT/'research/discovery/soft-value-envelope-audit-20261029.json'
with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
np.savez_compressed(output.with_suffix('.npz'),**arrays)
print(json.dumps(dict(state=result['state'],elapsed_seconds=result['elapsed_seconds'],work=result['work'],snapshots=result['snapshots'],cell_extremes=[dict(snapshot=c['snapshot'],pool=c['pool'],target_min=c['summary']['Y']['min'],target_max=c['summary']['Y']['max']) for c in result['cells']])))
