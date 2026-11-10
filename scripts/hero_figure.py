"""The three-panel summary figure at the top of the README (figures/hero.png).

  python scripts/hero_figure.py

Reads only results/ (no raw data): the imbalance deciles of the training days, the
break-even fee of the first test and the execution savings of the second test.
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BLUE, ORANGE, GRAY = '#2a78d6', '#eb6834', '#85847f'
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
TAKER_FEE_BP = 5.5


def main():
    S = json.loads((ROOT / 'results/summary.json').read_text())
    with open(ROOT / 'results/imbalance_bins.csv') as f:
        bins = [r for r in csv.DictReader(f) if r['feature'] == 'imb_l1']
    f_star = S['trade_alone']['break_even_fee']['estimate_bp']
    E = S['execution_value']
    signal = E['Q4_vs_twap']['estimate']
    q4 = E['Q4_vs_twap']
    intervals = [q4['t_interval']] + list(q4['bootstrap'].values())   # report the widest of the four
    lo, hi = min(i[0] for i in intervals), max(i[1] for i in intervals)
    d = E['descriptive_is_minus_twap_base']          # IS_X - IS_twap; saving = -(that)
    savings = [('signal timing', signal, BLUE),
               ('random timing', -d['IS_random-twap'], GRAY),
               ('placebo forecast', -d['IS_shift-twap'], GRAY),
               ('signal reversed', -d['IS_negated-twap'], ORANGE)]

    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': MUTED,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3), facecolor=SURFACE,
                           gridspec_kw={'width_ratios': [1.05, 0.95, 1.1], 'wspace': 0.42})
    for a in ax:
        a.set_facecolor(SURFACE)

    def title(a, n, head, sub):
        a.text(0, 1.17, f'{n}  {head}', transform=a.transAxes, fontsize=12.5, fontweight='bold', va='bottom')
        a.text(0, 1.04, sub, transform=a.transAxes, fontsize=9.5, color=MUTED, va='bottom')

    # 1. the signal: mean 5 s return by imbalance decile
    a = ax[0]
    x = [float(r['mean_x']) for r in bins]
    y = [float(r['mean_y_bp']) for r in bins]
    a.axhline(0, color=MUTED, lw=0.8)
    a.grid(axis='y', color=GRID, lw=0.8)
    a.set_axisbelow(True)
    a.plot(x, y, '-', color=BLUE, lw=2)
    a.plot(x, y, 'o', color=BLUE, ms=7, mec=SURFACE, mew=2)
    a.set_xlabel('order-book imbalance  (bid − ask) / (bid + ask)')
    a.set_ylabel('mean return over next 5 s  [bp]')
    a.annotate('more size on the bid\n→ price tends to rise', xy=(x[-1], y[-1]), xytext=(0.05, 0.36),
               fontsize=9, color=MUTED, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    title(a, '1', 'The signal is real', 'out-of-sample R² ≈ 2.3%, gain on 63 of 65 test days')

    # 2. too small to trade alone
    a = ax[1]
    rows = [('break-even fee\nof the signal', f_star, BLUE), ('exchange taker fee', TAKER_FEE_BP, ORANGE)]
    for i, (name, v, c) in enumerate(rows):
        a.barh(i, v, height=0.5, color=c)
        a.text(v + 0.12, i, f'{v:.2f} bp', va='center', fontsize=10)
    a.set_yticks(range(len(rows)), [r[0] for r in rows])
    a.tick_params(axis='y', length=0)
    a.spines['left'].set_visible(False)
    a.set_xlim(0, 6.6)
    a.set_ylim(-0.6, 1.6)
    a.invert_yaxis()
    a.set_xlabel('bp per side')
    a.grid(axis='x', color=GRID, lw=0.8)
    a.set_axisbelow(True)
    a.text(0.6, 0.5, f'{TAKER_FEE_BP / f_star:.0f}× too small', transform=a.transAxes, ha='center',
           fontsize=11, fontweight='bold', color=INK)
    title(a, '2', 'Too small to trade on its own', '≈ 45% of the predicted move is lost to an assumed 100 ms delay')

    # 3. but it makes necessary trading cheaper
    a = ax[2]
    a.axvline(0, color=MUTED, lw=0.8)
    a.grid(axis='x', color=GRID, lw=0.8)
    a.set_axisbelow(True)
    for i, (name, v, c) in enumerate(savings):
        a.barh(i, v, height=0.55, color=c)
        label = f'{v:+.2f} bp' if abs(v) >= 0.005 else '≈ 0'
        x_text = hi + 0.012 if i == 0 else v + (0.012 if v >= 0 else -0.012)
        a.text(x_text, i, label, va='center',
               ha='left' if v >= 0 else 'right', fontsize=10)
    a.errorbar(signal, 0, xerr=[[signal - lo], [hi - signal]], color=INK, lw=1.2, capsize=3)
    a.set_yticks(range(len(savings)), [s[0] for s in savings])
    a.tick_params(axis='y', length=0)
    a.spines['left'].set_visible(False)
    a.set_xlim(-0.32, 0.32)
    a.invert_yaxis()
    a.set_xlabel('saving vs TWAP  [bp of order value]')
    a.text(0.015, 0.45, f'interval {lo:.2f} – {hi:.2f} bp', ha='left', va='center',
           fontsize=9.5, color=MUTED)
    title(a, '3', 'But it makes execution cheaper', 'timing the slices of a 0.3 BTC order, 45 later days')

    fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.15)
    out = ROOT / 'figures/hero.png'
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print('wrote', out.relative_to(ROOT))


if __name__ == '__main__':
    main()
