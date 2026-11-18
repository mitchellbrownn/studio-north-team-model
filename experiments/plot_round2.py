"""Figures for retention round 2 (reads experiments/retention_round2/results.json)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent/'retention_round2'
d = json.loads((OUT/'results.json').read_text())
R = d['results']; cut = ['2024-12','2025-03','2025-06']
SURFACE, INK, INK2, GRID = '#fcfcfb', '#262624', '#5f5e5a', '#e8e7e2'
# Reference categorical palette, fixed order (validated: all checks pass; contrast WARN
# -> legend + line styles + companion table carry identity).
PAL = ['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#6250d6','#e34948']
MODELS = ['sBG (cell, cohort trend)','BdW','Logistic hazard','A: chain-ladder, sBG tail',
          'A: chain-ladder, flat tail','B: latent-class geometric K=2','B: latent-class geometric K=3',
          'C: recent churn flat, 3m']
STYLES = ['-','--','-.',':','-','--','-.',':']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.edgecolor':GRID,
                     'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'text.color':INK})

# Figure 1: mean contribution WAPE vs retention-only WAPE (dot plot, two series).
allm = list(R)
mean = lambda n, key: 100*sum(R[n]['cutoffs'][k][key] for k in cut)/3
order = sorted(allm, key=lambda n: mean(n,'contribution_wape'))
fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE); ax.set_facecolor(SURFACE)
for i, n in enumerate(order):
    a, b = mean(n,'contribution_wape'), mean(n,'retention_only_contribution_wape')
    ax.plot([b, a], [i, i], color=GRID, lw=2, zorder=1)
    ax.scatter(a, i, s=64, color=PAL[0], zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.scatter(b, i, s=64, color=PAL[1], zorder=3, edgecolor=SURFACE, linewidth=2, marker='D')
ax.set_yticks(range(len(order)), order); ax.invert_yaxis()
ax.set_xlabel('Mean WAPE across 3 cutoffs (%)'); ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ('top','right','left'): ax.spines[s].set_visible(False)
ax.tick_params(axis='y', length=0)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],marker='o',ls='',color=PAL[0],label='Contribution WAPE (with selected cash)'),
                   Line2D([],[],marker='D',ls='',color=PAL[1],label='Retention-only contribution WAPE')],
          loc='lower left', bbox_to_anchor=(0,1.0), ncol=2, frameon=False, fontsize=8)
ax.set_title('Retention models: total vs retention-only backtest error', loc='left', fontsize=11, color=INK, pad=26)
fig.tight_layout(); fig.savefig(OUT/'fig1_wape_comparison.png', dpi=170, facecolor=SURFACE); plt.close(fig)

# Figure 2: small multiples, Jan 2024 cohort observed share (ages 0-23) and each
# model's projected path ages 23-35 from its observed age-23 share (full-data fits).
cells = ['partner/build_skill','partner/save_time','search/build_skill','search/save_time']
ages = d['ages']
fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6), sharex=True, facecolor=SURFACE)
for ax, cell in zip(axes.flat, cells):
    ax.set_facecolor(SURFACE)
    o = d['observed_jan2024'][cell]; s23 = o['share'][o['age'].index(23)]
    ax.plot(o['age'], [100*x for x in o['share']], color=INK, lw=1.4, marker='o', ms=3, label='Observed (Jan 2024 cohort)')
    for j, n in enumerate(MODELS):
        path = [100*s23*x for x in d['curves_from_age23'][n][cell]]
        ax.plot(ages, path, color=PAL[j], lw=2, ls=STYLES[j], label=n)
    ax.axvline(23.5, color=GRID, lw=1); ax.text(24, ax.get_ylim()[1]-0.5, 'beyond observed ages', fontsize=7.5, color=INK2, va='top')
    ax.set_title(f"{cell.replace('/',' / ').replace('_',' ')}  (n={o['n_acquired']})", loc='left', fontsize=9.5, color=INK)
    lo=min(min(100*s23*x for x in d['curves_from_age23'][n][cell]) for n in MODELS); hi=100*o['share'][o['age'].index(18)]
    ax.set_ylim(max(0,lo-3), hi+3); ax.set_xlim(18, 35.3); ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ('top','right'): ax.spines[s].set_visible(False)
for ax in axes[1]: ax.set_xlabel('Customer age (months; 0 = first paid month)')
for ax in axes[:,0]: ax.set_ylabel('Retained, % of original cohort')
h, l = axes[0,0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=8)
fig.suptitle('How each retention model extrapolates the oldest cohort (Jan 2024) to age 35, zoomed to ages 18-35', x=0.01, ha='left', fontsize=11.5, color=INK)
fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.2, hspace=0.3, wspace=0.18); fig.savefig(OUT/'fig2_retention_curves.png', dpi=170, facecolor=SURFACE); plt.close(fig)
print('ok')
