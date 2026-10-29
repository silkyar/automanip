"""Standalone scientific figure from recorded fixed-checkpoint scores only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]/'runs/full-stacking-curve-20261029'
s=json.loads((root/'reconciled-summary.json').read_bytes())
fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
for route,arm in s['arms'].items():
    curve=arm['learning_curve'];label='Ordinary SAC' if route=='baseline' else 'Coordinate exploration SAC'
    x=[p['steps']/1000 for p in curve]
    axes[0].plot(x,[100*p['successes']/128 for p in curve],marker='o',label=label)
    axes[1].plot(x,[100*p['diagnostics_rates']['lift'] for p in curve],marker='o',label=label)
    axes[2].plot([p['trainer_seconds']/60 for p in curve],[100*p['successes']/128 for p in curve],marker='o',label=label)
for ax,title in zip(axes,['Complete stacking vs interactions','Lift diagnostic vs interactions','Complete stacking vs trainer time']):
    ax.set_title(title,fontsize=10);ax.set_ylim(-2,102);ax.grid(alpha=.2);ax.set_ylabel('Development episodes (%)')
axes[0].set_xlabel('Environment transitions (thousands)');axes[1].set_xlabel('Environment transitions (thousands)');axes[2].set_xlabel('Elapsed trainer time (minutes)')
axes[0].axhline(80,color='gray',ls='--',lw=.8);axes[0].axhline(90,color='gray',ls=':',lw=.8)
axes[0].legend(fontsize=8,loc='upper left')
fig.suptitle('StackCube: fresh training seed 2, 128 frozen development episodes per checkpoint',fontsize=11)
fig.savefig(root/'learning-curves.png',dpi=180)
fig.savefig(root/'learning-curves.pdf')
print(root/'learning-curves.png')
