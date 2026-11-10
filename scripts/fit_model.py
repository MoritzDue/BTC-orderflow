"""Fit and validate the forecast on the 40 training days (2025-10-01 .. 11-09).

  python scripts/download_sample.py 2025-10-01      # ... for each day you want (about 300 MB a day)
  python scripts/fit_model.py --jobs 6              # all 40 days: the full fit, about 45 minutes
  python scripts/fit_model.py --days 2025-10-01 2025-10-02 2025-10-03    # quick fit on a few days

Each day is replayed once from raw data (orderflow/book.py) and its d = 100 ms features and
targets are cached in data/cache/. With all 40 days the script repeats the original
procedure: ridge and OFI fits on two forward folds, lambda chosen by mean daily
validation MSE, refit on all 40 days, taker threshold = 90% quantile of |f| on the
validation days. It then compares the result with results/frozen_model.json and writes
results/training_check.json. With --days it fits on the given days only and shows the
coefficients next to the frozen ones (they will differ: fewer days).

It also writes two small data-exploration files used by 01_signal.ipynb (training days
only, so nothing about the test periods is looked at):
  results/imbalance_bins.csv   mean 5 s return by decile of book imbalance and of OFI
  results/sample_hour.npz      one hour (2025-10-15 14:00-15:00 UTC) of mid, spread, features
"""
import argparse
import csv
import json
from multiprocessing import Pool
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # import orderflow without installing
from orderflow import book, model
from orderflow import features as F

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / 'data/cache'
DELAY = 100
J = book.DELAYS.index(DELAY)
TRAIN = F.split_days('refit', 'train')
SAMPLE_DAY, SAMPLE_HOUR = '2025-10-15', 14


def cache_day(day):
    """Replay one day and cache features, target and masks at d = 100 ms."""
    path = CACHE / f'{day}.npz'
    if not path.exists():
        from run_day import build_day
        with tempfile.TemporaryDirectory() as tmp:
            grid, *_ = build_day(day, Path(tmp) / 'store.zlib')
        fe = F.day_features(grid)
        s = grid[J, :, book.COLUMNS.index('bid')] + grid[J, :, book.COLUMNS.index('ask')]
        tmp_path = path.with_suffix('.tmp.npz')
        np.savez(tmp_path, X=fe['X'][J], y=fe['y'], eligible=fe['eligible'][J], label=fe['label'], s=s)
        os.replace(tmp_path, path)
    return day


def load(day):
    """Usable rows (X, y) and threshold rows X for one day, as in the original fit."""
    with np.load(CACHE / f'{day}.npz') as z:
        X, y, e, lab = z['X'], z['y'], z['eligible'], z['label']
    purge = F.purge_mask(day)
    use = e & lab & ~purge
    return {'X': X[use], 'y': y[use], 'Xt': X[e & ~purge]}


def stack(data, days, key):
    return np.concatenate([data[d][key] for d in days])


def full_fit(data):
    """Folds, lambda selection, final refit and thresholds."""
    fold_models, daily = {}, {}
    for fold in ('fold1', 'fold2'):
        tr = F.split_days(fold, 'train')
        X, y = stack(data, tr, 'X'), stack(data, tr, 'y')
        fold_models[fold] = {lam: model.fit_ridge(X, y, lam) for lam in model.LAMBDAS}
        for d in F.split_days(fold, 'evaluate'):
            daily[d] = (fold, {lam: model.day_metrics(data[d]['y'], model.predict_ridge(m, data[d]['X']))['mse']
                               for lam, m in fold_models[fold].items()})
    scores = {lam: float(np.mean([daily[d][1][lam] for d in sorted(daily)])) for lam in model.LAMBDAS}
    lam = model.select_lambda(scores)
    f_val = np.concatenate([model.predict_ridge(fold_models[daily[d][0]][lam], data[d]['Xt']) for d in sorted(daily)])
    X, y = stack(data, TRAIN, 'X'), stack(data, TRAIN, 'y')
    return {'scores': scores, 'lambda': lam, 'ridge': model.fit_ridge(X, y, lam), 'ofi': model.fit_ofi(X, y),
            'thresholds': model.thresholds(f_val), 'rows': int(len(y))}


def compare(fit, frozen):
    """Every fitted number next to the frozen one, with the largest absolute difference."""
    fr = frozen['ridge']
    pairs = {
        'lambda scores': ([fit['scores'][lam] for lam in model.LAMBDAS],
                          [json.load(open(ROOT / 'results/validation.json'))['selection']['scores_mean_daily_mse_bp2']
                           ['%g' % lam] for lam in model.LAMBDAS]),
        'selected lambda': ([fit['lambda']], [frozen['lambda']]),
        'training rows': ([fit['rows']], [fr['n']]),
        'feature means': (fit['ridge']['mu'], fr['mu']),
        'feature std devs': (fit['ridge']['sd'], fr['sd']),
        'ridge intercept': ([fit['ridge']['a']], [fr['a']]),
        'ridge coefficients': (fit['ridge']['beta'], fr['beta']),
        'OFI model (a, b)': ([fit['ofi']['a'], fit['ofi']['b']], [frozen['ofi']['a'], frozen['ofi']['b']]),
        'taker threshold (bp)': ([fit['thresholds']['0.90']], [frozen['taker_threshold_bp']]),
    }
    out = {}
    for name, (a, b) in pairs.items():
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        out[name] = {'identical': bool(np.array_equal(a, b)), 'max_abs_diff': float(np.max(np.abs(a - b)))}
    return out


def exploration(days):
    """Decile tables (training days, usable rows) and one sample hour."""
    data = [load(d) for d in days]
    X, y = np.concatenate([d['X'] for d in data]), np.concatenate([d['y'] for d in data])
    rows = []
    for name in ('imb_l1', 'ofi_norm'):
        x = X[:, F.FEATURES.index(name)]
        edges = np.quantile(x, np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(edges, x, side='right') - 1, 0, 9)
        for i in range(10):
            yy = y[b == i]
            rows.append({'feature': name, 'decile': i + 1, 'mean_x': float(x[b == i].mean()),
                         'mean_y_bp': float(yy.mean()), 'se_y_bp': float(yy.std(ddof=1) / np.sqrt(len(yy))),
                         'n': int(len(yy))})
    with open(ROOT / 'results/imbalance_bins.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    with np.load(CACHE / f'{SAMPLE_DAY}.npz') as z:
        k = slice(SAMPLE_HOUR * 3600, (SAMPLE_HOUR + 1) * 3600)
        np.savez_compressed(ROOT / 'results/sample_hour.npz', day=SAMPLE_DAY, start_s=SAMPLE_HOUR * 3600,
                            mid_usdt=z['s'][k] / 200.0, features=z['X'][k], feature_names=np.array(F.FEATURES),
                            y=z['y'][k], eligible=z['eligible'][k])


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--days', nargs='+', help='fit on these training days only (quick demo)')
    p.add_argument('--jobs', type=int, default=1, help='days replayed in parallel')
    a = p.parse_args()
    days = a.days or list(TRAIN)
    if any(d not in TRAIN for d in days):
        raise SystemExit('training days are %s .. %s' % (TRAIN[0], TRAIN[-1]))
    CACHE.mkdir(parents=True, exist_ok=True)
    with Pool(a.jobs) as pool:
        for d in pool.imap_unordered(cache_day, days):
            print('cached', d, flush=True)
    data = {d: load(d) for d in days}
    _, _, frozen = model.load_model(ROOT / 'results/frozen_model.json')
    if a.days:
        X, y = stack(data, days, 'X'), stack(data, days, 'y')
        m = model.fit_ridge(X, y, frozen['lambda'])
        print('\n%d rows from %d days. ridge coefficients (z-scored features):' % (len(y), len(days)))
        print('  feature       this fit   frozen (40 days)')
        for f, b, b0 in zip(F.FEATURES, m['beta'], frozen['ridge']['beta']):
            print('  %-11s  %+8.4f   %+8.4f' % (f, b, b0))
        return
    fit = full_fit(data)
    check = compare(fit, frozen)
    exploration(days)
    res = {'description': 'fit_model.py on the 40 training days, rebuilt from raw data with this repository, '
                          'compared with results/frozen_model.json (the model used in both test periods)',
           'all_identical': all(v['identical'] for v in check.values()), 'checks': check,
           'lambda_scores': {'%g' % k: v for k, v in fit['scores'].items()}, 'rows': fit['rows']}
    (ROOT / 'results/training_check.json').write_text(json.dumps(res, indent=1) + '\n')
    print('\nrefit compared with the frozen model:')
    for name, v in check.items():
        print('  %-22s %s' % (name, 'identical' if v['identical'] else 'max |diff| %.3g' % v['max_abs_diff']))


if __name__ == '__main__':
    main()
