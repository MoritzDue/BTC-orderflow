"""Order-book replay and the one-second decision grid.

Prices are integers in 0.01 USDT and sizes in 0.001 BTC lots, so the replay is exact.
Bybit publishes the top-200 book as one snapshot followed by deltas (about every 100 ms)
and the trades as a CSV. One pass over a day produces

  grid   int64 array (4 delays, 86400 seconds, 18 columns): the book and trade flow that
         was visible at t_k - d for every decision second t_k and market-data delay d;
  store  the top-50 book after every message, used later to fill simulated orders.

Timing rule (strict past): a query at time q sees the book messages with cts < q (in ms)
and the trades with T < q. Nothing stamped at or after q can leak into a decision.
"""
from bisect import bisect_left, insort
from collections import Counter
import csv
from decimal import Decimal
import gzip
import hashlib
import json
import zlib

import numpy as np

TOPIC = 'orderbook.200.BTCUSDT'
SECOND = 1_000_000                       # microseconds
DELAYS = (0, 100, 500, 1000)             # market-data delays d in ms
COLUMNS = ('status', 'last_cts', 'bid', 'ask', 'bid_l1', 'ask_l1', 'bid_l5', 'ask_l5',
           'ofi', 'ofi_groups', 'ofi_ok', 'buy_n', 'sell_n', 'buy_lots', 'sell_lots',
           'buy_notional', 'sell_notional', 'rpi_n')
VALID, WARMUP, STALE, INVALID = 0, 1, 2, 3
STATUS_NAMES = {VALID: 'valid', WARMUP: 'warmup', STALE: 'stale', INVALID: 'invalid'}
STORE_LEVELS = 50
TRADE_HEADER = ['timestamp', 'symbol', 'side', 'size', 'price', 'tickDirection',
                'trdMatchID', 'grossValue', 'homeNotional', 'foreignNotional', 'RPI']


def to_units(text, dp):
    """Exact decimal string -> integer units ('104250.5', 2 -> 10425050)."""
    if type(text) is not str:
        raise ValueError('not a string')
    whole, dot, frac = text.partition('.')
    if (not whole.isascii() or not whole.isdigit() or
            (dot and (not frac.isascii() or not frac.isdigit())) or len(frac) > dp):
        raise ValueError('representation')
    return int(whole + frac.ljust(dp, '0'))


class Book:
    """Integer limit order book. Any sequence problem (gap, regression, crossed book)
    marks the book unsynchronized until the next snapshot."""

    def __init__(self):
        self.bq, self.aq = {}, {}      # price -> lots
        self.bp, self.ap = [], []      # sorted prices (ascending)
        self.u = self.seq = self.cts = self.ts = None
        self.synchronized = False
        self.counts = Counter()

    def _set(self, q_map, prices, p, q):
        if q == 0:
            if q_map.pop(p, None) is not None:
                del prices[bisect_left(prices, p)]
            self.counts['delete'] += 1
        else:
            if p not in q_map:
                insort(prices, p)
            q_map[p] = q

    def apply(self, message):
        """Apply one archive message; returns the list of problems found (empty if fine)."""
        reasons = []
        try:
            data = message['data']
            if (message.get('topic') != TOPIC or data['s'] != 'BTCUSDT' or
                    message['type'] not in ('snapshot', 'delta')):
                raise ValueError('market/type')
            u, seq, cts, ts = data['u'], data['seq'], message['cts'], message['ts']
            if any(type(x) is not int or x < 0 for x in (u, seq, cts, ts)):
                raise ValueError('integer contract')
            levels = []
            for key in ('b', 'a'):
                side, seen = [], set()
                for p, q in data[key]:
                    p, q = to_units(p, 2), to_units(q, 3)
                    if p <= 0 or p in seen:
                        raise ValueError('invalid/repeated level')
                    seen.add(p)
                    side.append((p, q))
                levels.append(side)
        except (KeyError, TypeError, ValueError):
            self.synchronized = False
            self.counts['malformed'] += 1
            return ['malformed']
        snapshot = message['type'] == 'snapshot'
        if self.cts is not None and cts < self.cts:
            reasons.append('cts_regression')
        if self.ts is not None and ts < self.ts:
            reasons.append('ts_regression')
        if not snapshot and self.u is not None:
            if u <= self.u:
                reasons.append('update_nonincrease')
            elif u > self.u + 1:
                reasons.append('possible_gap')
            if seq <= self.seq:
                reasons.append('sequence_nonincrease')
        if u == 1 and not snapshot:
            reasons.append('restart_without_snapshot')
        self.u, self.seq, self.cts, self.ts = u, seq, cts, ts
        if reasons:
            self.synchronized = False
        if snapshot:
            self.counts['snapshot'] += 1
            self.bq, self.aq, self.bp, self.ap = {}, {}, [], []
            self.synchronized = not reasons
        else:
            self.counts['delta'] += 1
        if self.synchronized:
            for p, q in levels[0]:
                self._set(self.bq, self.bp, p, q)
            for p, q in levels[1]:
                self._set(self.aq, self.ap, p, q)
            if not self.bp or not self.ap:
                reasons.append('empty_side')
                self.synchronized = False
            elif self.bp[-1] >= self.ap[0]:
                reasons.append('crossed_or_locked')
                self.synchronized = False
        if not self.synchronized and not reasons:
            reasons.append('awaiting_snapshot')
        self.counts.update(reasons)
        return reasons

    def top(self, k):
        """Best-first (price, lots) ladders, k levels per side."""
        bids = [(p, self.bq[p]) for p in reversed(self.bp[-k:])]
        asks = [(p, self.aq[p]) for p in self.ap[:k]]
        return bids, asks


def status(book, q_us, warm_since_us):
    """State of the visible book at query time q: invalid, stale (> 1 s since the last
    message), warm-up (< 1 s since resynchronizing) or valid. First match wins."""
    if not book.synchronized:
        return INVALID
    if q_us // 1000 - book.cts > 1000:
        return STALE
    if warm_since_us is None or q_us - warm_since_us < SECOND:
        return WARMUP
    return VALID


def ofi_term(prev, cur):
    """Order-flow imbalance e_n (Cont, Kukanov and Stoikov 2014) between two best-level
    states (bid, bid lots, ask, ask lots)."""
    b0, qb0, a0, qa0 = prev
    b1, qb1, a1, qa1 = cur
    return ((qb1 if b1 >= b0 else 0) - (qb0 if b1 <= b0 else 0)
            - (qa1 if a1 <= a0 else 0) + (qa0 if a1 >= a0 else 0))


def trade_rows(path):
    """Trade CSV -> (T in µs, is_buy, price units, lots, rpi), exact, in file order."""
    with gzip.open(path, 'rt', newline='') as f:
        rows = csv.DictReader(f)
        if rows.fieldnames != TRADE_HEADER:
            raise ValueError('unexpected trade header')
        previous = None
        for r in rows:
            t = Decimal(r['timestamp']) * SECOND
            p = Decimal(r['price']) * 100
            q = Decimal(r['size']) * 1000
            for x in (t, p, q):
                if not x.is_finite() or x != x.to_integral_value():
                    raise ValueError('trade field not exact in integer units')
            t, p, q = int(t), int(p), int(q)
            if previous is not None and t < previous:
                raise ValueError('trade timestamps go backwards')
            previous = t
            rpi = r.get('RPI', '0')
            if r['symbol'] != 'BTCUSDT' or r['side'] not in ('Buy', 'Sell') or p <= 0 or q <= 0 \
                    or rpi not in ('0', '1'):
                raise ValueError('bad trade row')
            yield t, r['side'] == 'Buy', p, q, rpi == '1'


def queries(midnight_us):
    """All (q, delay index, k) in time order; k = -1 primes the first interval."""
    for k in range(-1, 86400):
        t = midnight_us + k * SECOND
        for j in (3, 2, 1, 0):                 # 1000, 500, 100, 0 ms
            yield t - DELAYS[j] * 1000, j, k


# ------------------------------------------------------------------ execution store
# One record per archive message holding the book after it. cts is the message clock,
# book_cts the book's last accepted clock, flags bit 0 = synchronized, bit 1 = snapshot.
# The file is a single zlib (level 6) stream, so its SHA-256 is deterministic.
STORE_DTYPE = np.dtype([('cts', '<i8'), ('book_cts', '<i8'), ('u', '<i8'), ('flags', '<i8'),
                        ('nb', '<i4'), ('na', '<i4'),
                        ('bid', '<i4', (STORE_LEVELS, 2)), ('ask', '<i4', (STORE_LEVELS, 2))])
CHUNK = 4096


class StoreWriter:
    def __init__(self, path):
        self.z = zlib.compressobj(6)
        self.h = hashlib.sha256()
        self.f = open(path, 'wb')
        self.rec = np.zeros(CHUNK, STORE_DTYPE)
        self.i = self.messages = 0

    def add(self, cts, book, snapshot):
        r = self.rec[self.i]
        r['cts'] = cts
        r['book_cts'] = -1 if book.cts is None else book.cts
        r['u'] = -1 if book.u is None else book.u
        r['flags'] = int(book.synchronized) | (2 * int(snapshot))
        if book.synchronized:
            bids, asks = book.top(STORE_LEVELS)
            r['nb'], r['na'] = len(bids), len(asks)
            r['bid'][:len(bids)] = bids
            r['ask'][:len(asks)] = asks
        self.i += 1
        self.messages += 1
        if self.i == CHUNK:
            self._flush()

    def _flush(self):
        out = self.z.compress(self.rec[:self.i].tobytes())
        self.h.update(out)
        self.f.write(out)
        self.rec[:] = 0
        self.i = 0

    def finish(self):
        self._flush()
        out = self.z.flush()
        self.h.update(out)
        self.f.write(out)
        self.f.close()
        return {'messages': self.messages, 'sha256': self.h.hexdigest()}


def read_store(path):
    """Yield store records in archive order (chunks of a structured array)."""
    d, rest = zlib.decompressobj(), b''
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            rest += d.decompress(block)
            n = len(rest) // STORE_DTYPE.itemsize
            if n:
                yield np.frombuffer(rest[:n * STORE_DTYPE.itemsize], STORE_DTYPE)
                rest = rest[n * STORE_DTYPE.itemsize:]
        rest += d.flush()
    if not d.eof or d.unused_data or len(rest) % STORE_DTYPE.itemsize:
        raise ValueError('truncated or corrupt store')
    if rest:
        yield np.frombuffer(rest, STORE_DTYPE)


# ------------------------------------------------------------------ the one-second grid

def build_grid(lines, trades, midnight_us, store=None):
    """One pass over a day's book messages (raw JSON lines) and trades.

    Grid columns per delay j and second k (book visible at q = t_k - d):
      status, last_cts      state of the visible book and the cts of its last message
      bid, ask              best prices (0 when invalid)
      bid_l1, ask_l1        best-level sizes; bid_l5, ask_l5 sizes over five levels
      ofi, ofi_groups       summed OFI over message groups (same cts) in [q_(k-1), q_k)
      ofi_ok                1 iff the book was valid throughout [q_(k-1), q_k]
      buy_n ... rpi_n       trade counts, lots and notionals with q_(k-1) <= T < q_k
    store, if given, is a StoreWriter that receives the book after every message.
    Returns (grid, number of messages, book counters)."""
    grid = np.zeros((len(DELAYS), 86400, len(COLUMNS)), dtype=np.int64)
    book = Book()
    warm = None                                  # µs at which warm-up started
    qs = queries(midnight_us)
    pending = next(qs, None)
    trades = iter(trades)
    trade = next(trades, None)
    tcum = [0] * 7                               # cumulative trade counters
    ofi_cum = groups_cum = bad_cum = 0
    prev_state = None                            # last visible best-level state
    group_cts, group_bad = None, False
    stale_peaks = []                             # cts c with a gap c - c_prev > 1 s
    last = {}                                    # delay -> counters at the previous query

    def stale_inside(lo_ms, hi_ms):
        i = bisect_left(stale_peaks, lo_ms)
        return i < len(stale_peaks) and stale_peaks[i] <= hi_ms

    def finalize():
        # end of a message group with one cts: update OFI against the previous group
        nonlocal ofi_cum, groups_cum, bad_cum, prev_state
        groups_cum += 1
        if book.synchronized:
            cur = (book.bp[-1], book.bq[book.bp[-1]], book.ap[0], book.aq[book.ap[0]])
            if prev_state is not None and not group_bad:
                ofi_cum += ofi_term(prev_state, cur)
            prev_state = cur
        else:
            prev_state = None
        if group_bad or not book.synchronized:
            bad_cum += 1

    def drain(bound_ms):
        # Called before the first message of a group with cts = bound: the book holds the
        # groups < bound, which is what every query or trade with floor_ms <= bound sees.
        nonlocal pending, trade
        while True:
            q_due = pending is not None and pending[0] // 1000 <= bound_ms
            t_due = trade is not None and trade[0] // 1000 <= bound_ms
            if t_due and (not q_due or trade[0] < pending[0]):
                t, buy, p, lots, rpi = trade
                s = 0 if buy else 1
                tcum[s] += 1
                tcum[2 + s] += lots
                tcum[4 + s] += p * lots
                tcum[6] += rpi
                trade = next(trades, None)
                continue
            if not q_due:
                return
            q, j, k = pending
            st = status(book, q, warm)
            if k < 0:
                last[j] = (ofi_cum, groups_cum, bad_cum, list(tcum), st, q // 1000)
                pending = next(qs, None)
                continue
            row = grid[j, k]
            row[0] = st
            row[1] = -1 if book.cts is None else book.cts
            if st != INVALID:
                bids, asks = book.top(5)
                row[2], row[3] = bids[0][0], asks[0][0]
                row[4], row[5] = bids[0][1], asks[0][1]
                row[6], row[7] = sum(x[1] for x in bids), sum(x[1] for x in asks)
            prior = last[j]
            ok = (prior[4] == VALID and st == VALID and bad_cum == prior[2]
                  and not stale_inside(prior[5], q // 1000))
            row[8], row[9], row[10] = ofi_cum - prior[0], groups_cum - prior[1], int(ok)
            row[11:18] = [a - b for a, b in zip(tcum, prior[3])]
            last[j] = (ofi_cum, groups_cum, bad_cum, list(tcum), st, q // 1000)
            pending = next(qs, None)

    n = 0
    for line in lines:
        m = json.loads(line)
        cts = m.get('cts')
        if type(cts) is not int:
            raise ValueError('message without an integer cts')
        if group_cts is not None and cts != group_cts:
            finalize()
            group_bad = False
            if cts - group_cts > 1000:
                stale_peaks.append(cts)
        group_cts = cts
        drain(cts)
        was = book.synchronized
        reasons = book.apply(m)
        n += 1
        snap = m.get('type') == 'snapshot'
        group_bad = group_bad or bool(reasons) or snap     # a snapshot restarts warm-up
        if not book.synchronized:
            warm = None
        elif not was or snap:
            warm = cts * 1000
        if store is not None:
            store.add(cts, book, snap)
    if group_cts is not None:
        finalize()
    drain(float('inf'))
    return grid, n, dict(book.counts)
