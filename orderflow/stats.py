"""Inference on daily outcomes: moving-block day bootstrap, Bonferroni level, daily
t-intervals and the classification rules fixed before each test period was opened.

Days are the sampling unit (seconds within a day are strongly dependent). A quantity is
"robustly positive" only if all three bootstrap intervals (blocks of 1, 2 and 5 days) and
the t-interval lie above zero. Each family of three quantities uses alpha = 0.05 / 3.
"""
import math

import numpy as np
from scipy import stats

ALPHA = 0.05 / 3                 # Bonferroni over three primary quantities
BLOCKS = (1, 2, 5)
R = 2000
SEED = 20251110
FEE_SCENARIO_BP = 5.5


def block_indices(T, b, r=R, seed=SEED):
    """(r, T) non-circular moving-block resample indices. Starts are uniform on {0..T-b},
    ceil(T/b) blocks are concatenated and truncated to T. The generator is default_rng([seed, b])."""
    if not 1 <= b <= T:
        raise ValueError('block length outside 1..T')
    rng = np.random.default_rng([seed, b])
    m = -(-T // b)
    starts = rng.integers(0, T - b + 1, size=(r, m))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(r, m * b)
    return idx[:, :T]


def percentile_interval(x, alpha=ALPHA):
    x = np.asarray(x, np.float64)
    return [float(np.quantile(x, alpha / 2, method='linear')),
            float(np.quantile(x, 1 - alpha / 2, method='linear'))]


def t_interval(d, alpha=ALPHA):
    """mean +- t_{T-1}(1-alpha/2) s/sqrt(T); None if T < 2 or s == 0."""
    d = np.asarray(d, np.float64)
    T = len(d)
    if T < 2:
        return None
    s = float(np.std(d, ddof=1))
    if not s > 0:
        return None
    h = float(stats.t.ppf(1 - alpha / 2, T - 1)) * s / math.sqrt(T)
    m = float(d.mean())
    return [m - h, m + h]


def ratio_bp(pi, n):
    """10^4 * sum(pi) / sum(n) from exact integers; None if sum(n) == 0."""
    sp, sn = sum(int(x) for x in pi), sum(int(x) for x in n)
    return None if sn == 0 else 1e4 * sp / sn


def bootstrap(D1, D2, Pi, N, blocks=BLOCKS, r=R, seed=SEED, alpha=ALPHA):
    """Joint resampling of the daily tuples. D1, D2 may be None (unidentified); Pi, N
    may be None (unidentified). Returns {b: {'Q1': [lo, hi] | None, ...}}."""
    T = len(next(x for x in (D1, D2, Pi, N) if x is not None))
    D1 = None if D1 is None else np.asarray(D1, np.float64)
    D2 = None if D2 is None else np.asarray(D2, np.float64)
    Pi = None if Pi is None else np.asarray(Pi, dtype=object)
    N = None if N is None else np.asarray(N, dtype=object)
    out = {}
    for b in blocks:
        idx = block_indices(T, b, r, seed)
        row = {}
        for name, d in (('Q1', D1), ('Q2', D2)):
            row[name] = None if d is None else percentile_interval(d[idx].mean(axis=1), alpha)
        if Pi is None:
            row['Q3'], row['Q3_zero_denominator_resamples'] = None, None
        else:
            sp = np.array([sum(Pi[i]) for i in idx], dtype=object)
            sn = np.array([sum(N[i]) for i in idx], dtype=object)
            zero = int(sum(1 for x in sn if x == 0))
            row['Q3_zero_denominator_resamples'] = zero
            row['Q3'] = None if zero else percentile_interval(
                [1e4 * int(p) / int(n) for p, n in zip(sp, sn)], alpha)
        out[b] = row
    return out


def classify(intervals, c):
    """intervals: list of [lo, hi] or None. Any None means unidentified."""
    if not intervals or any(iv is None for iv in intervals):
        return 'unidentified'
    if all(iv[0] > c for iv in intervals):
        return 'robustly_above'
    if all(iv[1] <= c for iv in intervals):
        return 'robustly_at_or_below'
    return 'inconclusive'


def sign_label(cls):
    return {'robustly_above': 'robustly_positive', 'robustly_at_or_below': 'robustly_negative'}.get(cls, cls)


def fee_reading(c0, c55):
    """Combined reading of the break-even fee f* against 0 and against 5.5 bp."""
    if 'unidentified' in (c0, c55):
        return 'unidentified'
    if c0 == 'robustly_at_or_below':
        return 'no_pre_fee_edge'
    if c0 == 'robustly_above' and c55 == 'robustly_at_or_below':
        return 'positive_fee_budget_below_5.5bp'
    if c55 == 'robustly_above':
        return 'survives_5.5bp_scenario_conditional_on_replay'
    return 'inconclusive'


def primary(D1, D2, Pi, N, q3_identified=True, blocks=BLOCKS, r=R, seed=SEED, alpha=ALPHA):
    """First sealed test (Nov 10-29): Q1 = mean D1 = MSE(zero) - MSE(ridge),
    Q2 = mean D2 = MSE(OFI) - MSE(ridge), Q3 = break-even fee f* = 1e4 sum(Pi) / sum(N)."""
    if not q3_identified:
        Pi = N = None
    T = len(next(x for x in (D1, D2, Pi, N) if x is not None))
    boot = bootstrap(D1, D2, Pi, N, blocks, r, seed, alpha)
    res = {'T': T, 'alpha': alpha, 'level': 1 - alpha, 'blocks': list(blocks), 'resamples': r,
           'seed': seed, 'bootstrap': {str(b): v for b, v in boot.items()}, 'quantities': {}}
    for name, d in (('Q1', D1), ('Q2', D2)):
        if d is None:
            res['quantities'][name] = {'estimate': None, 't_interval': None, 'classification': 'unidentified'}
            continue
        ti = t_interval(d, alpha)
        ivs = [boot[b][name] for b in blocks] + ([ti] if ti is not None else [])
        res['quantities'][name] = {'estimate': float(np.mean(np.asarray(d, np.float64))), 't_interval': ti,
                                   't_defined': ti is not None,
                                   'classification': sign_label(classify(ivs, 0.0))}
    est = None if Pi is None else ratio_bp(Pi, N)
    ivs = [boot[b]['Q3'] for b in blocks]
    if est is None:
        ivs = [None]
    c0, c55 = classify(ivs, 0.0), classify(ivs, FEE_SCENARIO_BP)
    res['quantities']['Q3'] = {'estimate_bp': est, 'identified': Pi is not None,
                               'vs_zero': sign_label(c0), 'vs_5.5bp': c55,
                               'reading': fee_reading(c0, c55)}
    return res


# ------------------------------------------------------------------ execution study (Dec 1 - Jan 14)
EXEC_SEED = 20251201
EXEC = {'Q4': 'D4', 'Q5': 'D5', 'Q7': 'D7', 'Q6': 'Q6'}

ROWS = (
    (1, 'unidentified', lambda c, g: 'unid' in c.values()),
    (2, 'inconclusive (unresolved quantity)', lambda c, g: g),
    (3, 'Execution value supported, consistent with forecast information',
     lambda c, g: c['Q4'] == 'pos' and c['Q5'] == 'pos' and c['Q7'] == 'pos'),
    (4, 'Execution value supported, information not isolated',
     lambda c, g: c['Q4'] == 'pos' and c['Q5'] == 'pos' and c['Q7'] != 'pos'),
    (5, 'Timing effect without evidence of information', lambda c, g: c['Q4'] == 'pos' and c['Q5'] != 'pos'),
    (6, 'Harmful', lambda c, g: c['Q4'] == 'neg'),
    (7, 'Inconclusive', lambda c, g: True),
)


def strict_class(intervals):
    """'pos' if every lower bound > 0, 'neg' if every upper bound < 0, else 'inc'."""
    if any(iv is None for iv in intervals):
        return 'inc'
    if all(iv[0] > 0 for iv in intervals):
        return 'pos'
    if all(iv[1] < 0 for iv in intervals):
        return 'neg'
    return 'inc'


def execution_study(days, blocks=BLOCKS, r=R, seed=EXEC_SEED, alpha=ALPHA):
    """days: list of dicts with eligible, D4, D5, D7, Q6, unresolved_parents (calendar order).
    One set of resample indices is shared by all four quantities."""
    kept = [d for d in days if d['eligible'] > 0]
    T = len(kept)
    idx = {b: block_indices(T, b, r, seed) for b in blocks}
    q = {}
    for name, key in EXEC.items():
        x = np.asarray([d[key] for d in kept], np.float64)
        boot = {str(b): percentile_interval(x[i].mean(axis=1), alpha) for b, i in idx.items()}
        ti = t_interval(x, alpha)
        q[name] = {'estimate': float(x.mean()), 'bootstrap': boot, 't_interval': ti,
                   'class': strict_class(list(boot.values()) + [ti])}
    n_el = sum(d['eligible'] for d in kept)
    n_unres = sum(d['unresolved_parents'] for d in kept)
    gate = 200 * n_unres > n_el            # unresolved parents above 0.5%
    classes = {k: q[k]['class'] for k in ('Q4', 'Q5', 'Q7')}
    row, text = next((n, t) for n, t, cond in ROWS if cond(classes, gate))
    return {'T': T, 'quantities': q, 'eligible': n_el, 'unresolved': n_unres,
            'reading': {'row': row, 'text': text}}
