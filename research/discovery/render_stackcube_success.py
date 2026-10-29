"""Render exactly eight declared PUBLIC episodes of the fixed25M full-horizon actor.
Descriptive video, not development/confirmation/holdout scoring. No training or cloud API.
Root must separately admit/submit a bounded renderer. This file has not run on a GPU.
"""
from __future__ import annotations
import argparse,hashlib,json,os,time
from pathlib import Path
PUBLIC_RENDER_SEEDS=tuple(range(93_174_000,93_174_008))
WEIGHTS_SHA='39ba198d0c98af14a3d2965ef9a3be859f31aa3c4889db3da311da910c2da153'
EXPECTED_NATIVE_SHA='307a2bc5996d5def9f696ab35b27a9cf96a5cf7401c678ecc981b78170d1084b'

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--policy',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    started=time.monotonic();deadline=min(started+100,float(os.environ.get('AUTOMANIP_RENDER_DEADLINE',started+100)))
    def bounded():
        if time.monotonic()>deadline:raise TimeoutError('Original render deadline exceeded; no retry')
    args.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import imageio.v2 as imageio
    from automanip.evaluator import validate_actions
    from automanip.numpy_inference import NumpyPolicy
    from automanip.policy_io import load_policy
    from automanip.stackcube_adapter import StackCubeAdapter
    import importlib.metadata
    env=StackCubeAdapter(8,'physx_cuda')
    receipt=dict(kind='public illustrative render, not a benchmark',training_seed=1,training_transitions=24_985_600,
        training_seconds=1088.766732081771,training_gpu='L4',expected_native_checkpoint_sha256=EXPECTED_NATIVE_SHA,
        weights_sha256=WEIGHTS_SHA,public_render_seeds=list(PUBLIC_RENDER_SEEDS),episode_count=8,horizon=50,
        selection='First successful episode in the declared seed order at step50. If none succeeds, report no video; no resampling or retry.',
        new_training_transitions=0,development_queries=0,confirmation_queries=0,holdout_queries=0,
        maximum_new_public_simulator_transitions=400,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        state='started',versions={k:importlib.metadata.version(k) for k in ('mani_skill','sapien','torch','numpy','gymnasium')})
    frames=[[] for _ in PUBLIC_RENDER_SEEDS]
    try:
        actor=load_policy(args.policy,env.schema)
        if actor.weights_sha256!=WEIGHTS_SHA:raise ValueError('Only the confirmed25M full-horizon actor is eligible')
        policy=NumpyPolicy(actor,env.schema);obs=env.reset(PUBLIC_RENDER_SEEDS);receipt['actual_public_simulator_transitions']=0
        for step in range(51):
            bounded();images=env.env.render_rgb_array('render_camera')
            if hasattr(images,'detach'):images=images.detach().cpu().numpy()
            if images.shape!=(8,512,512,3):raise ValueError('Unexpected render shape')
            for index,frame in enumerate(images):
                frame=np.asarray(frame,dtype=np.uint8).copy()
                if frame.std()<1:raise ValueError('Blank rendered frame')
                frames[index].append(frame)
            if step==50:break
            actions=policy.act(obs);validate_actions(actions,batch_size=8,action_dim=8)
            obs,success,terms,trunc=env.step(actions);receipt['actual_public_simulator_transitions']+=8
            if any(trunc) and step+1<50:raise RuntimeError('Unexpected truncation')
        successes=[bool(t.last.success) for t in env.diagnostics]
        receipt['public_terminal_successes']=successes
        chosen=next((i for i,success in enumerate(successes) if success),None)
        receipt['selected_public_episode_index']=chosen
        if chosen is None:
            receipt['state']='complete_no_success';receipt['video']=None
        else:
            bounded();path=args.output/'stackcube-full-horizon-25m-public-success.mp4'
            with imageio.get_writer(str(path),fps=env.env.control_freq,codec='libx264',quality=8,macro_block_size=16) as writer:
                for frame in frames[chosen]:bounded();writer.append_data(frame)
            for step in (0,10,20,30,40,50):imageio.imwrite(args.output/f'public-success-step-{step:02d}.png',frames[chosen][step])
            receipt.update(state='complete',video=path.name,video_bytes=path.stat().st_size,video_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           frames=51,fps=env.env.control_freq,selected_public_seed=PUBLIC_RENDER_SEEDS[chosen],
                           final_consecutive_success_steps=env.diagnostics[chosen].stable_steps)
    except Exception as exc:
        receipt.update(state='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        env.close();receipt['render_process_seconds']=time.monotonic()-started
        (args.output/'render-receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
