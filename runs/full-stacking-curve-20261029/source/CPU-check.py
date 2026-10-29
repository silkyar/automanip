import ast,hashlib,json,sys,tempfile,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path('research/discovery').resolve()))
from replay_learner_lowalpha import Learner
from sac_baseline import export_actor
from joint_exploration import Collector
start=time.monotonic(); torch.set_num_threads(1)
models=[]; results={}
for route,eps in [('baseline',0.),('coordinate',.1)]:
 torch.manual_seed(2); learner=Learner('target-sac',device='cpu')
 before=torch.get_rng_state().clone(); collector=Collector('coordinate',2,'cpu',epsilon=eps)
 assert torch.equal(before,torch.get_rng_state())
 state={p+'.'+k:v.detach().clone() for p,m in [('actor',learner.actor),('q1',learner.q1),('q2',learner.q2),('target1',learner.target1),('target2',learner.target2)] for k,v in m.state_dict().items()}
 state['log_alpha']=learner.log_alpha.detach().clone(); models.append(state)
 for k,v in learner.q1.state_dict().items():assert torch.equal(v,learner.target1.state_dict()[k])
 for k,v in learner.q2.state_dict().items():assert torch.equal(v,learner.target2.state_dict()[k])
 with tempfile.TemporaryDirectory() as temp:
  output=Path(temp); artifact=export_actor(learner.actor,output/'initial')
  tree=ast.parse(Path('research/discovery/replay_robot_full_curve.py').read_text())
  train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
  save=next(n for n in train.body if isinstance(n,ast.FunctionDef) and n.name=='save')
  at=next(i for i,n in enumerate(save.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='values' for t in n.targets))
  end=next(i for i,n in enumerate(save.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='savez_compressed')
  module=ast.fix_missing_locations(ast.Module(body=save.body[at:end+1],type_ignores=[]))
  scope=dict(learner=learner,np=np,torch=torch,output=output,label='probe')
  with patch('torch.cuda.get_rng_state',return_value=before.clone()):exec(compile(module,'production-save-extract','exec'),scope)
  raw=np.load(output/'probe-state.npz',allow_pickle=False)
  assert set(raw.files)==set(state)|{'torch_cpu_rng','torch_cuda_rng'}
  for k,v in state.items():assert np.array_equal(raw[k],v.numpy())
  assert torch.equal(before,torch.get_rng_state())
  for p,m in [('actor',learner.actor),('q1',learner.q1),('q2',learner.q2),('target1',learner.target1),('target2',learner.target2)]:
   for k,v in m.state_dict().items():assert torch.equal(v,state[p+'.'+k])
  results[route]=dict(epsilon=collector.epsilon,initial_actor_sha256=artifact['weights_sha256'],snapshot_tensors=len(raw.files),target_serialization_nonmutation=True,CPU_rng_unchanged=True,CUDA_rng_call_mocked_not_GPU_test=True)
 assert torch.equal(before,torch.get_rng_state())
 obs=torch.linspace(-1,1,48).repeat(64,1)
 torch.manual_seed(100); original=learner.actor.sample(obs); ref_rng=torch.get_rng_state()
 torch.manual_seed(100); sample=collector.sample(learner.actor,obs)
 assert torch.equal(sample[2],original[0]);assert torch.equal(sample[1],original[1]);assert torch.equal(ref_rng,torch.get_rng_state())
 if eps==0:assert torch.equal(sample[0],original[0]); assert not sample[3].any()
for k in models[0]:assert torch.equal(models[0][k],models[1][k])
assert results['baseline']['initial_actor_sha256']==results['coordinate']['initial_actor_sha256']
result=dict(state='passed',scope='CPU-only seed2 initialization, epsilon0 identity/proposal global RNG, exact production snapshot AST statements; no simulator/GPU training',seed=2,arms=results,all_initial_tensors_equal=len(models[0]),elapsed_seconds=time.monotonic()-start,source_sha256={name:hashlib.sha256(Path('research/discovery',name).read_bytes()).hexdigest() for name in ['full_stacking_curve.py','replay_robot_full_curve.py','replay_learner_lowalpha.py','joint_exploration.py','sac_baseline.py']},environment_transitions=0,learner_updates=0)
Path('.runtime/full-stacking-curve-CPU-gate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
