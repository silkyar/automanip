"""Rebuild publication figures from frozen aggregate JSON; no simulator or network.
Run: MPLCONFIGDIR=/tmp/automanip-mpl python scripts/plot_progress.py
Requires matplotlib and numpy. SVG/PDF retain vector geometry and selectable text.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
ROOT=Path(__file__).resolve().parents[1];ASSETS=ROOT/'assets'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'axes.labelcolor':'#28332e','text.color':'#1b2923','xtick.color':'#526159','ytick.color':'#526159'})
GREEN='#27815a';GRAY='#aab3ae'

def save(fig,name):
 for ext in ('png','svg','pdf'):fig.savefig(ASSETS/(name+'.'+ext),dpi=220,facecolor='white',bbox_inches='tight',metadata={'Creator':'AutoManip aggregate publication plot'})
 plt.close(fig)

def main():
 d=json.loads((ASSETS/'progress-data.json').read_bytes());families=d['families']
 fig=plt.figure(figsize=(14,8));ax=fig.add_axes([.075,.33,.895,.46]);fig.text(.075,.94,'StackCube: learning a stronger PPO recipe',fontsize=25,fontweight='bold');fig.text(.075,.885,'Same 24,985,600 training transitions per seed · 3 random initializations · L4 GPU',fontsize=13,color='#526159')
 selected=[r for r in families if r['selected']]
 for family in families:
  x=family['index'];rates=[100*r['development_successes']/r['development_episodes'] for r in family['runs']];color=GREEN if family['selected'] else GRAY
  ax.vlines(x,min(rates),max(rates),color=color,alpha=.28,lw=1.4,zorder=1)
  ax.scatter(x+np.array([-.07,0,.07]),rates,s=19,color=color,alpha=.35,zorder=2,edgecolors='none')
  ax.scatter([x],[family['development_mean_percent']],s=86 if family['selected'] else 46,color=color,edgecolors='white',linewidths=1.2,zorder=5)
 ax.step([r['index'] for r in selected],[r['development_mean_percent'] for r in selected],where='post',color=GREEN,lw=2.4,alpha=.85,zorder=3)
 annotation=[(selected[0],(.12,12),'Official PPO'),(selected[1],(.18,13),'Discount 0.8 → 0.9'),(selected[2],(-2.0,15),'Entropy 0 → 0.003'),(selected[3],(.18,9),'Full episode horizon')]
 for r,(dx,dy),label in annotation:
  ax.annotate(label+'\n'+f"{r['development_mean_percent']:.2f}%",(r['index'],r['development_mean_percent']),xytext=(r['index']+dx,r['development_mean_percent']+dy),ha='left',va='bottom',fontsize=11,color=GREEN,arrowprops=dict(arrowstyle='-',color=GREEN,alpha=.55,lw=.8))
 ax.set(xlim=(.6,10.5),ylim=(0,111),ylabel='Development success at episode end (%)',xlabel='Recipe comparisons (research sequence; some runs overlapped)')
 ax.set_yticks([0,25,50,75,100]);ax.grid(axis='y',color='#e1e7e3',lw=.8);ax.set_axisbelow(True)
 ticks=['Official','γ = .9','γ = .95','γ = .875','γ = .925','Entropy .001','Entropy .003','Full horizon','Entropy decay','γ .925 + entropy']
 ax.set_xticks(range(1,11),ticks,rotation=27,ha='right',fontsize=10)
 ax.legend(handles=[Line2D([0],[0],color=GREEN,marker='o',label='Selected recipe path'),Line2D([0],[0],marker='o',color='none',markerfacecolor=GRAY,markeredgecolor='white',label='Not retained (comparable recipe means)'),Line2D([0],[0],color=GRAY,marker='.',lw=1,label='Three-seed range / individual seeds')],frameon=False,loc='lower left',bbox_to_anchor=(-.005,1.02),ncol=3,fontsize=10)
 fig.text(.075,.195,'128 fixed development episodes per trained policy. Points are means across 3 training seeds; ranges are not confidence intervals.',fontsize=10,color='#526159')
 fig.text(.075,.168,'Selected path ≠ running best: the gray γ = .925 comparison exceeded the earlier selected recipe. No smoothing or interpolation.',fontsize=10,color='#526159')
 fig.text(.075,.105,'Independent confirmation',fontsize=12,fontweight='bold',color=GREEN)
 fig.text(.285,.105,'Official 32.23%   →   discount + entropy 82.10%   →   full horizon 94.08%',fontsize=12)
 fig.text(.075,.065,'3 training seeds × 512 confirmation episodes per recipe; discount-only recipe was not confirmed. Full horizon disables partial resets.',fontsize=10,color='#526159')
 save(fig,'stackcube-progress-25m')
 # Separate honest time-budget view: latest observed scored snapshot, never an interpolated22min endpoint.
 td=json.loads((ASSETS/'timing-checkpoints.json').read_bytes());groups={f['experiment']:f for f in selected};fig,axes=plt.subplots(1,2,figsize=(13.8,6.2),gridspec_kw={'width_ratios':[1.05,1]});fig.subplots_adjust(left=.07,right=.97,bottom=.30,top=.76,wspace=.3)
 fig.suptitle('What the existing checkpoints say about a 22-minute limit',x=.07,y=.96,ha='left',fontsize=20,fontweight='bold');fig.text(.07,.875,'Latest eligible scored checkpoint across equivalent recipe records by 1,320 training seconds; steps can differ.',fontsize=11,color='#526159')
 for index,(name,f) in enumerate(groups.items(),1):
  rr=[r for r in td['per_run'] if r['experiment']==name];pp=[r['latest_independently_scored_checkpoint_available_by1320s'] for r in rr]
  axes[0].scatter([index-.08,index,index+.08],[100*p['successes']/p['episodes'] for p in pp],s=45,color=GREEN)
  axes[0].plot([index-.08,index+.08],[np.mean([100*p['successes']/p['episodes'] for p in pp])]*2,color=GREEN,lw=2)
  axes[1].scatter([index-.08,index,index+.08],[p['environment_transitions']/1e6 for p in pp],s=45,color=GREEN)
 axes[0].set(ylabel='Development success (%)',ylim=(0,102));axes[1].set(ylabel='Checkpoint training transitions (millions)',ylim=(0,27));axes[1].axhline(24.9856,color=GRAY,lw=1,ls='--')
 labels=['Official','Discount .9','+ entropy .003','Full horizon']
 for ax in axes:
  ax.set_xticks(range(1,5),labels,rotation=18,ha='right');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 fig.text(.07,.15,'Each dot is one training seed. Left-panel strokes show three-seed means at the available snapshots, not equal-step comparisons.',fontsize=10,color='#526159')
 fig.text(.07,.115,'Intermediate times use logged update timestamps; final times use measured trainer duration. No claim of an exact 22-minute score.',fontsize=10,color='#526159')
 fig.text(.07,.08,'Entropy uses timed-control reruns with identical actor digests (1,280–1,299s); these are the same policies, not extra independent seeds.',fontsize=10,color='#526159')
 fig.text(.07,.045,'Entropy and full horizon reached 24.99M steps within the limit for all 3 seeds; official and discount-only still mix checkpoint steps.',fontsize=10,color='#526159')
 save(fig,'stackcube-observed-by-22min')
 print('Wrote PNG, SVG and PDF for both figures.')
if __name__=='__main__':main()
