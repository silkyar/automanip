"""Plot all registered public checkpoint scores; no smoothing/extrapolation."""
import argparse
import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=argparse.ArgumentParser();p.add_argument('--scores',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    fig,axes=plt.subplots(1,3,figsize=(12,4.1),sharex=True,sharey=True)
    for seed,ax in enumerate(axes,1):
        for method,label,color,marker,style in [('ppo-gae','Tuned PPO (GAE)','#1871aa','o','-'),('ppo-finite-mc','Finite-MC PPO','#d85736','x','--')]:
            index=(seed-1)*2+int(method=='ppo-finite-mc')
            r=json.loads((a.scores/f'trial-{index}/report.json').read_text())
            if r['state']!='complete' or any(p['failures'] for p in r['points']):raise ValueError('Incomplete/failed scoring')
            points=sorted(r['points'],key=lambda p:p['steps'])
            ax.plot([p['steps']/1e6 for p in points],[100*p['successes']/128 for p in points],
                    color=color,label=label,marker=marker,linestyle=style,lw=2,markersize=5)
        ax.set_title(f'Training seed {seed}',fontsize=11)
        ax.set_xlabel('Environment transitions (millions)')
        ax.set_xlim(-.1,10.7);ax.set_ylim(-3,103);ax.set_xticks([0,2,4,6,8,10])
        ax.set_yticks([0,20,40,60,80,100]);ax.grid(alpha=.18)
        ax.spines[['right','top']].set_visible(False)
    axes[0].set_ylabel('Final-episode stacking success (%)')
    axes[0].legend(frameon=False,loc='upper left',fontsize=9)
    fig.suptitle('StackCube: the assisted target benefit did not survive learning from scratch',fontsize=13,y=.98)
    fig.text(.5,.025,'Each point: 128 paired public-development episodes. Per-run training ceilings: 600 seconds and 10.24M transitions.',ha='center',fontsize=9,color='#444444')
    fig.tight_layout(rect=(0,.07,1,.92));a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output.with_suffix('.png'),dpi=170)
    fig.savefig(a.output.with_suffix('.svg'))


if __name__=='__main__':main()
