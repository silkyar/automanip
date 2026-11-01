"""Plot physical success over time in two completed manual diagnostics."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def curves(root):
    result=np.zeros((2,4,50));counts=np.zeros((2,4),int)
    for source in range(2):
        for batch in range(32):
            with np.load(root/f'remote/collection-{source}/run/batch-{batch}.npz',allow_pickle=False) as z:
                for group in range(4):
                    selected=z['assignment']==group
                    result[source,group]+=z['stage_flags'][:,selected,6].sum(1)
                    counts[source,group]+=selected.sum()
    if not (counts==512).all():raise ValueError('All assigned episodes required')
    return result/counts[:,:,None]


def main():
    base=Path('artifacts/cumulative');fig,axes=plt.subplots(1,2,figsize=(12.8,4.8),sharex=True,sharey=True)
    panels=[('release-randomized-20261030','Five opening commands',['Unchanged','Fixed-time open','Targeted open','Targeted open + zero arm']),('release-maintenance-20261030','After a common opening sequence',['Resume actor','Keep open','Zero arm','Keep open + zero arm'])]
    colors=['#64748b','#0e7490','#b45309','#6d28d9']
    for ax,(name,title,labels) in zip(axes,panels):
        values=curves(base/name)
        for group,(label,color) in enumerate(zip(labels,colors)):
            mean=values[:,group].mean(0)*100
            ax.plot(np.arange(1,51),mean,label=f'{label} ({mean[-1]:.1f}% final)',color=color,lw=2)
            ax.fill_between(np.arange(1,51),values[:,group].min(0)*100,values[:,group].max(0)*100,color=color,alpha=.1)
        ax.set_title(title,fontsize=12);ax.set_xlabel('Control step (20 Hz)');ax.set_ylim(0,100);ax.set_xlim(1,50);ax.grid(alpha=.18);ax.legend(fontsize=8,loc='upper left')
    axes[0].set_ylabel('Physical stacking success at this step (%)')
    fig.suptitle('Manual interventions diagnose transient success and maintenance failure',fontsize=14,y=.99)
    fig.text(.5,.015,'No learning or policy promotion. 1,024 assigned episodes per curve. Bands show the range of two fixed policies, not a confidence interval.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.94))
    destination=base/'release-maintenance-20261030'
    for suffix in ['png','svg']:fig.savefig(destination/f'success-trajectories.{suffix}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
