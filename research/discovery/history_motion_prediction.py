"""Fixed data-only history probe. No simulator, policy, cloud or evaluator imports."""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/'research/discovery/history-motion-plan-20261029.json'
started=time.monotonic()
plan=json.loads(PLAN.read_bytes())
if not plan['admitted']:raise PermissionError('Probe not admitted')
if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=plan['source_sha256']:raise ValueError('Probe source differs')

def checktime():
    if time.monotonic()-started>175:raise RuntimeError('CPU probe cap; no result admission')

xs=[];hs=[];ys=[];metas=[];continuity=[];relative=[]
for file in plan['files']:
    path=ROOT/file['path'];raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=file['sha256']:raise ValueError('Data hash differs')
    z=np.load(path,allow_pickle=False);o=z['obs'];n=z['next_obs'];a=z['action'];tr=z['truncated'];stop=z['stop']
    if o.shape!=(50,64,48) or n.shape!=o.shape or a.shape!=(50,64,8):raise ValueError('Shape differs')
    if not all(np.isfinite(v).all() for v in (o,n,a)):raise ValueError('Nonfinite input')
    rel=max(float(np.max(np.abs(o[:,:,39:42]-(o[:,:,25:28]-o[:,:,18:21])))),
            float(np.max(np.abs(o[:,:,42:45]-(o[:,:,32:35]-o[:,:,18:21])))),
            float(np.max(np.abs(o[:,:,45:48]-(o[:,:,32:35]-o[:,:,25:28])))))
    relative.append(rel)
    if rel>2e-6:raise ValueError('Flattened observation layout differs')
    code=np.eye(3)[['baseline','joint','coordinate'].index(file['collector'])]
    for t in range(3,49,3):
        for lane in range(64):
            if bool(tr[t-1,lane]) or bool(stop[t-1,lane]):continue
            err=float(np.max(np.abs(o[t,lane]-n[t-1,lane])));continuity.append(err)
            if err>2e-6:raise ValueError('Within-episode continuity differs')
            current=np.r_[o[t,lane],a[t,lane],t/49,np.log2(file['steps']/51200),code,np.eye(2)[file['seed']-1]]
            history=np.r_[o[t,lane,25:28]-o[t-1,lane,25:28],a[t-1,lane]]
            target=n[t,lane,25:28]-o[t,lane,25:28]
            xs.append(current);hs.append(history);ys.append(target)
            fold=int.from_bytes(hashlib.sha256(f"{file['seed']}:{lane}".encode()).digest()[:4],'big')%4
            phase=bool(o[t,lane,27]>.04 and np.linalg.norm(o[t,lane,39:42])<.06)
            metas.append((file['seed'],lane,file['steps'],['baseline','joint','coordinate'].index(file['collector']),t,t//12,fold,phase))
X=np.asarray(xs,dtype=np.float64);H=np.asarray(hs,dtype=np.float64);Y=np.asarray(ys,dtype=np.float64);M=np.asarray(metas,dtype=int)
if X.shape[1]!=63 or H.shape[1]!=11:raise ValueError('Feature dimensions differ')
from panda_kinematics import Kinematics,parse_chain,quaternion_matrix,angle_between
helper=Path(__file__).with_name('panda_kinematics.py');urdf=ROOT/plan['kinematic_urdf']
if hashlib.sha256(helper.read_bytes()).hexdigest()!=plan['kinematic_helper_sha256'] or hashlib.sha256(urdf.read_bytes()).hexdigest()!=plan['kinematic_urdf_sha256']:raise ValueError('Public geometry differs')
kin=Kinematics(parse_chain(urdf));pose,jac=kin.fk(X[:,:7],jacobian=True)
fk_error=float(np.max(np.linalg.norm(pose[:,:3,3]-X[:,18:21],axis=1)))
fk_angle=float(np.max(angle_between(pose[:,:3,:3],quaternion_matrix(X[:,21:25]))))
if fk_error>2e-5 or fk_angle>5e-5:raise ValueError('FK frame mismatch')
velocity=np.einsum('nij,nj->ni',jac,X[:,9:16])
proxy=velocity[:,:3]+np.cross(velocity[:,3:],X[:,25:28]-X[:,18:21])
K=np.c_[velocity,proxy,np.zeros((len(X),2))]
chosen=np.linspace(0,len(X)-1,64,dtype=int);direction=X[chosen,9:16].copy()
length=np.linalg.norm(direction,axis=1);direction[length<1e-10]=np.eye(7)[0];direction/=np.linalg.norm(direction,axis=1)[:,None]
eps=1e-6;plus=kin.fk(X[chosen,:7]+eps*direction);minus=kin.fk(X[chosen,:7]-eps*direction)
fd_linear=(plus[:,:3,3]-minus[:,:3,3])/(2*eps)
rotation=plus[:,:3,:3]@minus[:,:3,:3].transpose(0,2,1)
fd_angular=np.stack((rotation[:,2,1]-rotation[:,1,2],rotation[:,0,2]-rotation[:,2,0],rotation[:,1,0]-rotation[:,0,1]),axis=1)/(4*eps)
analytic=np.einsum('nij,nj->ni',jac[chosen],direction)
jac_error=float(np.max(np.abs(np.c_[fd_linear,fd_angular]-analytic)))
if jac_error>1e-6:raise ValueError('Jacobian directional/frame gate failed')
work=dict(ridge_fits=0,fit_design_rows=0,prediction_design_rows=0,history_neighbor_queries=0)
predictions={k:np.full_like(Y,np.nan) for k in plan['models']}
rng=np.random.default_rng(93422300);W=rng.normal(size=(74,256))/np.sqrt(74);bias=rng.uniform(0,2*np.pi,256)
match_distances=[]

def matched_history(indices,mean,scale):
    result=H[indices].copy();meta=M[indices]
    groups={}
    for j,m in enumerate(meta):groups.setdefault(tuple(m[[0,2,3,5]]),[]).append(j)
    for positions in groups.values():
        p=np.asarray(positions);norm=(X[indices[p]]-mean)/scale
        tree=cKDTree(norm);dist,near=tree.query(norm,k=min(64,len(p)))
        if near.ndim==1:near=near[:,None];dist=dist[:,None]
        for j,row in enumerate(p):
            ok=meta[p[near[j]],1]!=meta[row,1]
            if not ok.any():raise ValueError('No different-lane neighbor in registered matching cell')
            at=np.flatnonzero(ok)[0];source=p[near[j,at]]
            result[row]=H[indices[source]];match_distances.append(float(dist[j,at]/np.sqrt(63)))
            work['history_neighbor_queries']+=1
    return result

for fold in range(4):
    checktime();train=np.flatnonzero(M[:,6]!=fold);test=np.flatnonzero(M[:,6]==fold)
    # Seed+lane is held together across every policy/checkpoint/window.
    if set(map(tuple,M[train,:2]))&set(map(tuple,M[test,:2])):raise ValueError('Cluster leakage')
    cm=X[train].mean(0);cs=X[train].std(0);cs=np.where(cs<1e-8,1,cs)
    control_train=matched_history(train,cm,cs);control_test=matched_history(test,cm,cs)
    for model in plan['models']:
        checktime();width=256 if model=='current256' else 128
        htrain=H[train] if model=='history128' else K[train] if model=='kinematic128' else control_train if model=='shuffled128' else np.zeros((len(train),11))
        htest=H[test] if model=='history128' else K[test] if model=='kinematic128' else control_test if model=='shuffled128' else np.zeros((len(test),11))
        inp=np.c_[X[train],htrain];held=np.c_[X[test],htest]
        mean=inp.mean(0);scale=inp.std(0);scale=np.where(scale<1e-8,1,scale)
        inp=(inp-mean)/scale;held=(held-mean)/scale
        design=np.c_[inp,np.sqrt(2/width)*np.cos(inp@W[:,:width]+bias[:width])]
        query=np.c_[held,np.sqrt(2/width)*np.cos(held@W[:,:width]+bias[:width])]
        dm=design.mean(0);ym=Y[train].mean(0);design-=dm;query-=dm
        coef=np.linalg.solve(design.T@design+np.eye(design.shape[1]),design.T@(Y[train]-ym))
        predictions[model][test]=query@coef+ym
        work['ridge_fits']+=1;work['fit_design_rows']+=len(train);work['prediction_design_rows']+=len(test)

if not all(np.isfinite(p).all() for p in predictions.values()):raise ValueError('Missing prediction')
errors={k:1000*np.linalg.norm(p-Y,axis=1) for k,p in predictions.items()}
errors['zero']=1000*np.linalg.norm(Y,axis=1);errors['persistence']=1000*np.linalg.norm(H[:,:3]-Y,axis=1)
summary={};gates=[]
for seed in (1,2):
    mask=(M[:,0]==seed)&M[:,7].astype(bool);lanes=sorted(set(M[mask,1]));part={}
    for model,e in errors.items():
        cluster=np.array([e[mask&(M[:,1]==lane)].mean() for lane in lanes])
        part[model]=dict(cluster_equal_mean_distance_error_mm=float(cluster.mean()) if len(cluster) else None,
                         frame_distance_RMSE_mm=float(np.sqrt(np.mean(e[mask]**2))) if mask.any() else None)
    sufficient=len(lanes)>=16
    criteria={}
    for model in ('current128','current256','shuffled128','kinematic128'):
        b=part[model]['cluster_equal_mean_distance_error_mm'];h=part['history128']['cluster_equal_mean_distance_error_mm']
        criteria[model]=bool(sufficient and b-h>=.25 and h<=.8*b)
    if sufficient:
        delta=np.array([(errors['current256'][mask&(M[:,1]==lane)]-errors['history128'][mask&(M[:,1]==lane)]).mean() for lane in lanes])
        boot=np.random.default_rng(93422500+seed).choice(delta,size=(500,len(delta)),replace=True).mean(1)
        interval=np.quantile(boot,[.025,.975]).tolist()
    else:interval=None
    gate=bool(sufficient and all(criteria.values()) and interval[0]>0)
    gates.append(gate);summary[str(seed)]=dict(primary_lane_clusters=len(lanes),primary_frames=int(mask.sum()),sufficient=sufficient,models=part,criteria=criteria,current256_minus_history_paired95_mm=interval,gate=gate)
nonprimary={k:float(e[~M[:,7].astype(bool)].mean()) for k,e in errors.items()}
all_rows={k:dict(mean_distance_error_mm=float(e.mean()),distance_RMSE_mm=float(np.sqrt(np.mean(e**2)))) for k,e in errors.items()}
by_checkpoint=[]
for seed,step,mode in sorted(set(map(tuple,M[:,[0,2,3]]))):
    mask=(M[:,0]==seed)&(M[:,2]==step)&(M[:,3]==mode)
    by_checkpoint.append(dict(seed=int(seed),steps=int(step),collector=['baseline','joint','coordinate'][mode],frames=int(mask.sum()),mean_distance_error_mm={k:float(v[mask].mean()) for k,v in errors.items()},primary_frames=int((mask & M[:,7].astype(bool)).sum()),primary_mean_distance_error_mm={k:float(v[mask & M[:,7].astype(bool)].mean()) if (mask & M[:,7].astype(bool)).any() else None for k,v in errors.items()}))
phase_strata=[]
for seed,step,mode,phasebin,primary in sorted(set(map(tuple,M[:,[0,2,3,5,7]]))):
    mask=(M[:,0]==seed)&(M[:,2]==step)&(M[:,3]==mode)&(M[:,5]==phasebin)&(M[:,7]==primary)
    phase_strata.append(dict(seed=int(seed),steps=int(step),collector=['baseline','joint','coordinate'][mode],phase_bin=int(phasebin),primary=bool(primary),frames=int(mask.sum()),lane_clusters=len(set(M[mask,1])),mean_distance_error_mm={k:float(v[mask].mean()) for k,v in errors.items()}))
checktime()
result=dict(state='complete',verdict='predictive_gate_passed_not_RL_admission' if all(gates) else 'inconclusive_sparse_primary' if not all(s['sufficient'] for s in summary.values()) else 'registered_predictive_gate_failed',primary_by_seed=summary,nonprimary_mean_distance_error_mm=nonprimary,all_rows=all_rows,by_checkpoint=by_checkpoint,phase_strata=phase_strata,
    data_rows=len(X),lane_clusters=len(set(map(tuple,M[:,:2]))),continuity_max=max(continuity),relative_layout_max=max(relative),
    matched_neighbor_distance_median=float(np.median(match_distances)),matched_neighbor_distance_p90=float(np.quantile(match_distances,.9)),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
    kinematic_checks=dict(FK_max_position_error_m=fk_error,FK_max_rotation_error_rad=fk_angle,Jacobian_directional_max_error=jac_error,FK_pose_evaluations=kin.fk_pose_evaluations,Jacobian_pose_evaluations=kin.geometric_jacobian_pose_evaluations,assistance='Public URDF and known base pose for diagnostic predictor only; no oracle cube velocity or learner intervention'),work=work,elapsed_seconds=time.monotonic()-started,environment_transitions=0,simulator_queries=0,GPU_seconds=0,
    caveat='Fixed supervised predictors and recentwindows only. History can simplify nonlinear kinematics; neighborcontrol is not conditionally exchangeable. No irreducible aliasing,critictruth,PPO/SACcause or RLgain claim.')
output=ROOT/'research/discovery/history-motion-prediction-20261029.json'
with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
np.savez_compressed(output.with_suffix('.npz'),metadata=M,target=Y,**predictions)
print(json.dumps({k:result[k] for k in ('state','verdict','primary_by_seed','data_rows','kinematic_checks','work','elapsed_seconds')}))
