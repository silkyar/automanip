"""Show training-time oracle coverage separately from deployable policy scores."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    first = json.loads(Path('cumulative/results/readout-score-20261031.json').read_text())
    final = json.loads(Path('cumulative/results/onpolicy-support-score-20261031.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=True)
    groups = [
        ('Training search oracle', ['Absolute', 'Anchor-relative', 'Current-relative'], [1,19,89],128),
        ('First deployed students', ['Original', 'Unbranched data', 'Search data'],
         [round(first['pooled_success'][k]*1024) for k in ['base','unbranched','search']],1024),
        ('One data aggregation step', ['Unchanged student', 'Original states', 'Student states'],
         [round(final['pooled_success'][k]*512) for k in ['baseline','original_states','student_states']],512)]
    for ax, (title, labels, counts, denominator) in zip(axes, groups):
        values = [100*n/denominator for n in counts]
        bars = ax.bar(range(3), values, color=['#8393a4','#4d91b0','#c57847'],width=.64)
        ax.set_title(title,fontsize=12)
        ax.set_xticks(range(3),labels,rotation=24,ha='right',fontsize=9)
        ax.set_ylim(0,100 if denominator==128 else 5)
        ax.set_ylabel('Covered states (%)' if denominator==128 else 'Final successful episodes (%)')
        ax.spines[['top','right']].set_visible(False)
        ax.bar_label(bars,labels=[f'{n}/{denominator}' for n in counts],padding=5,fontsize=9)
    fig.suptitle('Successful search continuations did not become a reliable policy',fontsize=14)
    out=Path('artifacts/cumulative/gpd-pilot-closeout-20261031');out.mkdir(exist_ok=True)
    fig.savefig(out/'mechanism-summary.png',dpi=180)
    fig.savefig(out/'mechanism-summary.svg')
    plt.close(fig)


if __name__=='__main__':main()
