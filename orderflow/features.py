"""Features, the 5-second target, eligibility and the train/test calendar.

Built from the one-second grid (book.py). For delay d and second k the decision time is
c = t_k - d and every feature uses only what was visible at c. s = bid + ask (0.01 USDT),
so the mid is s / 2; depths and volumes are in 0.001 BTC lots.

  1 ofi_norm   OFI over the last second / mean best-level depth at k-1 and k
  2 imb_l1     (B1 - A1) / (B1 + A1)            best-level depth imbalance
  3 imb_l5     (B5 - A5) / (B5 + A5)            five-level depth imbalance
  4 tvi        (buy lots - sell lots) / total   trade-volume imbalance, 0 without trades
  5 spread_bp  10000 (ask - bid) / mid
  6 ret_1s_bp  10000 ln(m_k / m_(k-1))
  7 log_trades ln(1 + number of trades in the last second)
  8 rv30_bp    root mean square of the last 30 one-second returns

A feature is NaN (mask False) when its inputs were not valid for the whole interval it
uses. Target: y_k = 10000 ln(m_(k+5) / m_k) on the zero-delay grid. The target mask uses
the future and is only for fitting and evaluation; eligibility never depends on it.
"""
from datetime import date, datetime, timedelta, timezone

import numpy as np

from .book import COLUMNS, DELAYS, VALID

FEATURES = ('ofi_norm', 'imb_l1', 'imb_l5', 'tvi', 'spread_bp', 'ret_1s_bp', 'log_trades', 'rv30_bp')
RV_WINDOW = 30          # one-second returns in the RMS
HORIZON = 5             # seconds ahead
PURGE = 120             # seconds removed on each side of a split boundary
N = 86400
C = {c: i for i, c in enumerate(COLUMNS)}

D = lambda s: date.fromisoformat(s)
DAYS = tuple((D('2025-10-01') + timedelta(days=i)).isoformat() for i in range(60))
SPLITS = {   # name -> (training days, evaluation days), inclusive
    'fold1': (('2025-10-01', '2025-10-30'), ('2025-10-31', '2025-11-04')),
    'fold2': (('2025-10-01', '2025-11-04'), ('2025-11-05', '2025-11-09')),
    'refit': (('2025-10-01', '2025-11-09'), None),
    'test': (None, ('2025-11-10', '2025-11-29')),
}
BOUNDARIES = ('2025-10-31', '2025-11-05', '2025-11-10')   # midnights UTC
SEALED = tuple(d for d in DAYS if d >= '2025-11-10')                       # first test, 20 days
SEALED2 = tuple((D('2025-12-01') + timedelta(days=i)).isoformat() for i in range(45))   # second test


def first_row(d_ms, lookback_s):
    """Smallest k with c_(d,k) - lookback >= midnight, i.e. 1000 k - d - 1000 L >= 0."""
    return -(-(d_ms + 1000 * lookback_s) // 1000)


def quote_ok(row):
    """Positive valid quote: V & 0 < bid < ask."""
    return (row[:, C['status']] == VALID) & (row[:, C['bid']] > 0) & (row[:, C['ask']] > row[:, C['bid']])


def log_mid_change_bp(s_new, s_old):
    """10000 ln(s_new / s_old) for integer s = bid + ask, computed as log1p of an exact
    integer difference to avoid cancellation; NaN where s_old <= 0."""
    s_new, s_old = np.asarray(s_new, np.float64), np.asarray(s_old, np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        return 1e4 * np.log1p((s_new - s_old) / s_old)


def delay_features(row, d_ms):
    """Features and masks for one delay. row: (86400, len(COLUMNS)) int64 grid slice.

    Returns (X, M, integrity): X float (86400, 8) with NaN where invalid, M bool (86400, 8),
    integrity: dict of integrity-failure counts (V rows only)."""
    k = np.arange(N)
    col = lambda name: row[:, C[name]]
    V, O = col('status') == VALID, col('ofi_ok') == 1
    q = quote_ok(row)
    b1, a1, b5, a5 = col('bid_l1'), col('ask_l1'), col('bid_l5'), col('ask_l5')
    point = V & (k >= first_row(d_ms, 0))
    interval = V & (k >= first_row(d_ms, 1))          # k >= 1 always, so row k-1 is same-day
    prev = lambda x, fill: np.concatenate(([fill], x[:-1]))

    X = np.full((N, len(FEATURES)), np.nan)
    M = np.zeros((N, len(FEATURES)), bool)

    # 1 normalized OFI
    Vp = prev(V, False)
    den = (prev(b1, 0) + prev(a1, 0) + b1 + a1) / 4.0
    m = interval & Vp & O & (den > 0)
    X[m, 0] = col('ofi')[m] / den[m]
    M[:, 0] = m

    # 2, 3 depth imbalances
    for f, (b, a) in ((1, (b1, a1)), (2, (b5, a5))):
        tot = b + a
        m = point & (tot > 0)
        X[m, f] = (b[m] - a[m]) / tot[m]
        M[:, f] = m

    # 4, 7 trade-interval features
    bn, sn, bl, sl = col('buy_n'), col('sell_n'), col('buy_lots'), col('sell_lots')
    negative = (bn < 0) | (sn < 0) | (bl < 0) | (sl < 0)
    inconsistent = ((bn > 0) != (bl > 0)) | ((sn > 0) != (sl > 0))   # trade lots are strictly positive
    m = interval & O & ~negative & ~inconsistent
    tot = bl + sl
    X[m, 3] = np.where(tot[m] > 0, (bl[m] - sl[m]) / np.where(tot[m] > 0, tot[m], 1), 0.0)
    X[m, 6] = np.log1p((bn + sn)[m].astype(np.float64))
    M[:, 3] = M[:, 6] = m

    # 5 spread
    s = col('bid') + col('ask')
    m = point & q
    X[m, 4] = 1e4 * 2.0 * (col('ask')[m] - col('bid')[m]) / s[m]
    M[:, 4] = m

    # 6 previous one-second log mid-return
    R = interval & q & prev(q, False) & O
    r = np.full(N, np.nan)
    r[R] = log_mid_change_bp(s[R], prev(s, 1)[R])
    X[:, 5], M[:, 5] = r, R

    # 8 30-second rolling RMS of r
    win = np.lib.stride_tricks.sliding_window_view
    ok = np.zeros(N, bool)
    ok[RV_WINDOW - 1:] = win(R, RV_WINDOW).all(axis=1)
    m = ok & V & (k >= first_row(d_ms, RV_WINDOW))
    r2 = np.where(R, r, 0.0) ** 2
    rv = np.full(N, np.nan)
    rv[RV_WINDOW - 1:] = np.sqrt(win(r2, RV_WINDOW).mean(axis=1))
    X[:, 7] = np.where(m, rv, np.nan)
    M[:, 7] = m

    depth = np.stack([b1, a1, b5, a5], axis=1)
    integrity = {'nonpositive_quote': int((V & ~q).sum()),
                 'nonpositive_depth': int((V & (depth <= 0).any(axis=1)).sum()),
                 'negative_trade_count_or_lots': int(negative.sum()),
                 'inconsistent_trade_count_lots': int(inconsistent.sum())}
    return X, M, integrity


def target(row0):
    """Five-second target from the zero-delay slice: (y, mask)."""
    q = quote_ok(row0)
    O = row0[:, C['ofi_ok']] == 1
    s = row0[:, C['bid']] + row0[:, C['ask']]
    m = np.zeros(N, bool)
    chain = np.lib.stride_tricks.sliding_window_view(O[1:], HORIZON).all(axis=1)   # O_(k+1..k+5)
    m[:N - HORIZON] = q[:N - HORIZON] & q[HORIZON:] & chain
    y = np.full(N, np.nan)
    y[m] = log_mid_change_bp(s[HORIZON:][m[:N - HORIZON]], s[:N - HORIZON][m[:N - HORIZON]])
    return y, m


def day_features(grid):
    """All delays for one day. grid: (len(DELAYS), 86400, len(COLUMNS)).

    Returns dict with X (4, N, 8), M (4, N, 8), eligible (4, N), y (N), label (N),
    integrity (per delay)."""
    if grid.shape != (len(DELAYS), N, len(COLUMNS)):
        raise ValueError('grid shape')
    out = [delay_features(grid[j], d) for j, d in enumerate(DELAYS)]
    X = np.stack([o[0] for o in out])
    M = np.stack([o[1] for o in out])
    y, label = target(grid[0])
    return {'X': X, 'M': M, 'eligible': M.all(axis=2), 'y': y, 'label': label,
            'integrity': {str(d): o[2] for d, o in zip(DELAYS, out)}}


# ---------------------------------------------------------------- splits and purge

def midnight_s(day):
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp())


def purge_mask(day):
    """True where t_k lies in the union of [b - 120 s, b + 120 s) over BOUNDARIES."""
    t = midnight_s(day) + np.arange(N)
    out = np.zeros(N, bool)
    for b in BOUNDARIES:
        b = midnight_s(b)
        out |= (t >= b - PURGE) & (t < b + PURGE)
    return out


def in_range(day, rng):
    return rng is not None and rng[0] <= day <= rng[1]


def role(day, split):
    """'train', 'evaluate' or None for a day in a named split."""
    train, evaluate = SPLITS[split]
    if in_range(day, train):
        return 'train'
    if in_range(day, evaluate):
        return 'evaluate'
    return None


def split_days(split, which):
    rng = SPLITS[split][0 if which == 'train' else 1]
    return tuple(d for d in DAYS if in_range(d, rng))
