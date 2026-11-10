"""Run the whole pipeline on one day of raw data and compare it with the original study.

  python scripts/download_sample.py 2025-12-01
  python scripts/run_day.py 2025-12-01                  # about 5-10 minutes, pure Python

Steps: replay the book and build the one-second grid (orderflow/book.py), compute features
(orderflow/features.py), forecast with the frozen ridge model (results/frozen_model.json), then
replay in one pass over the day's book
  - the taker policy at four latencies plus a 2-lot variant (orderflow/taker.py), and
  - the execution study: TWAP, signal, random and placebo schedules (orderflow/execution.py).
For the 65 test days, results/reference_days.json holds what the original pipeline
recorded; every compared value must match exactly. Writes results/sample/<day>.json.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # import orderflow without installing
from orderflow import book, execution, model, taker
from orderflow import features as F
from download_sample import manifest, local_path, verified

ROOT = Path(__file__).resolve().parent.parent
DELAY = 100                                    # features at d = 100 ms
SPREAD = F.FEATURES.index('spread_bp')
BID, ASK = book.COLUMNS.index('bid'), book.COLUMNS.index('ask')


def losses(fe, day, ridge, ofi):
    """Daily MSE of zero / OFI / ridge forecasts on the usable rows at d = 100 ms."""
    j = book.DELAYS.index(DELAY)
    use = fe['eligible'][j] & fe['label'] & ~F.purge_mask(day)
    X, y = fe['X'][j][use], fe['y'][use]
    m = {'zero': model.day_metrics(y, np.zeros_like(y)), 'ofi': model.day_metrics(y, model.predict_ofi(ofi, X)),
         'ridge': model.day_metrics(y, model.predict_ridge(ridge, X))}
    return {'usable_rows': int(len(y)), 'D1': m['zero']['mse'] - m['ridge']['mse'],
            'D2': m['ofi']['mse'] - m['ridge']['mse'], 'r2_ridge': m['ridge']['r2_oos'], 'ic_ridge': m['ridge']['ic']}


def taker_agents(day, fe, grid, ridge, theta, midnight):
    """The diagnostic taker policy at every latency, plus fixed 2-lot orders at 100/100 ms."""
    purge = F.purge_mask(day)
    out = []
    for d_md, d_o in taker.LATENCIES:
        j = book.DELAYS.index(d_md)
        e, X = fe['eligible'][j], fe['X'][j]
        f = model.forecast(ridge, X, e)
        s = grid[j, :, BID] + grid[j, :, ASK]
        lat = '%d-%d' % (d_md, d_o)
        k, side, lots = taker.signals(f, e, purge, X[:, SPREAD], s, theta)
        out.append(taker.Agent('diag/' + lat, k, side, lots, midnight, d_o, taker.BASE_FEE))
        if (d_md, d_o) == taker.BASE_LATENCY:
            k, side, lots = taker.signals(f, e, purge, X[:, SPREAD], s, theta, lots=2)
            out.append(taker.Agent('diag2lot/' + lat, k, side, lots, midnight, d_o, taker.BASE_FEE))
    return out


def build_day(day, store_path):
    """Hash-check the day's raw files, replay them and return (grid, messages, counts, store info)."""
    rows = manifest(day)
    for r in rows:
        if not verified(r):
            raise SystemExit('missing or changed raw file %s; run: python scripts/download_sample.py %s'
                             % (local_path(r).name, day))
    book_zip, trades_csv = (local_path(r) for r in sorted(rows, key=lambda r: r['kind']))
    writer = book.StoreWriter(store_path)
    with zipfile.ZipFile(book_zip) as z, z.open(z.namelist()[0]) as f:
        grid, n_msg, counts = book.build_grid(f, book.trade_rows(trades_csv),
                                              F.midnight_s(day) * book.SECOND, writer)
    return grid, n_msg, counts, writer.finish()


def run(day, out_dir):
    ridge, ofi, frozen = model.load_model(ROOT / 'results/frozen_model.json')
    midnight = F.midnight_s(day) * book.SECOND
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        store_path = Path(tmp) / 'store.zlib'
        print('1/3 replaying the order book and building the one-second grid ...', flush=True)
        grid, n_msg, counts, store = build_day(day, store_path)
        t1 = time.time()
        print('    %d messages, %.0f s' % (n_msg, t1 - t0))

        print('2/3 features, forecasts and daily forecast losses ...', flush=True)
        fe = F.day_features(grid)
        loss = losses(fe, day, ridge, ofi)
        agents = taker_agents(day, fe, grid, ridge, frozen['taker_threshold_bp'], midnight)
        prm = frozen['execution']
        j = book.DELAYS.index(DELAY)
        f = model.forecast(ridge, fe['X'][j], fe['eligible'][j])
        plist = execution.parents(grid, fe['eligible'], DELAY)
        runs = execution.build_runs(day, plist, f, prm)
        del fe

        print('3/3 replaying %d taker agents and %d execution schedules against the book ...'
              % (len(agents), len(runs)), flush=True)
        results = taker.run(store_path, agents + [r for _, _, r in runs], midnight)
    t2 = time.time()

    tk = {a.name: r for a, r in zip(agents, results)}
    by = {}
    for (jp, name, _), r in zip(runs, results[len(agents):]):
        by.setdefault(jp, {})[name] = r
    costs = {p['j']: execution.parent_costs(p, by[p['j']]) for p in plist if p['eligible']}
    outcome = execution.day_outcome(plist, costs)
    res = {'day': day, 'messages': n_msg, 'book_counts': counts,
           'grid_sha256': hashlib.sha256(grid.tobytes()).hexdigest(),
           'store_sha256': store['sha256'], 'losses': loss,
           'taker': {n: {'pre_fee_units': r['pre_fee_units'], 'traded_units': r['traded_units'],
                         'round_trips': r['counts']['round_trips'], 'log_sha256': r['log_sha256'],
                         'f_star_bp': None if not r['traded_units'] else 1e4 * r['pre_fee_units'] / r['traded_units']}
                     for n, r in tk.items()},
           'execution': {**outcome, 'runs': len(runs), 'fill_log_sha256': execution.fill_log_sha256(runs)},
           'seconds': {'grid': round(t1 - t0), 'rest': round(t2 - t1)}}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{day}.json').write_text(json.dumps(res, indent=1) + '\n')
    return res


def compare(res):
    """Exact comparison with the original pipeline. Returns the number of mismatches."""
    ref = json.loads((ROOT / 'results/reference_days.json').read_text()).get(res['day'])
    if ref is None:
        print('no reference for %s (not a test day); nothing to compare' % res['day'])
        return 0
    checks = [('one-second grid (SHA-256)', res['grid_sha256'], ref['grid_sha256']),
              ('top-50 book store (SHA-256)', res['store_sha256'], ref['store_sha256']),
              ('usable forecast rows', res['losses']['usable_rows'], ref['usable_rows']),
              ('D1 = MSE(zero) - MSE(ridge)', res['losses']['D1'], ref['D1']),
              ('D2 = MSE(OFI) - MSE(ridge)', res['losses']['D2'], ref['D2'])]
    for n, r in ref.get('taker', {}).items():
        got = res['taker'][n]
        checks += [(n + ' pre-fee P&L / traded (units)', (got['pre_fee_units'], got['traded_units']),
                    (r['pre_fee_units'], r['traded_units'])),
                   (n + ' fill log (SHA-256)', got['log_sha256'], r['log_sha256'])]
    if 'execution' in ref:
        e, r = res['execution'], ref['execution']
        checks += [('execution: eligible parents', e['eligible'], r['eligible'])]
        checks += [('execution: %s (exact fraction)' % k, e[k + '_exact'], r[k + '_exact']) for k in ('D4', 'D5', 'D7')]
        checks += [('execution: fill log of %d runs (SHA-256)' % r['runs'], e['fill_log_sha256'], r['fill_log_sha256'])]
    bad = 0
    print('\ncomparison with the original study (%s period):' % ref['period'].replace('_', ' '))
    for name, got, want in checks:
        ok = got == want
        bad += not ok
        print('  %-48s %s' % (name, 'identical' if ok else 'DIFFERENT: %s vs %s' % (got, want)))
    return bad


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('day')
    p.add_argument('--out', type=Path, default=ROOT / 'results/sample')
    a = p.parse_args()
    res = run(a.day, a.out)
    L, E = res['losses'], res['execution']
    print('\n%s: R2 ridge %.4f, D1 %.4f bp^2 | taker f* at 100/100 ms %.3f bp | execution D4 %.3f, D5 %.3f, '
          'D7 %.3f bp (%d parents)' % (a.day, L['r2_ridge'], L['D1'], res['taker']['diag/100-100']['f_star_bp'] or 0,
                                       E['D4'], E['D5'], E['D7'], E['eligible']))
    sys.exit(1 if compare(res) else 0)
