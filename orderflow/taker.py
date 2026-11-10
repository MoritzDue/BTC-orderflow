"""Taker replay: market orders filled against the historical top-50 book.

One pass over a day's store drives any number of independent agents. An agent
holds precomputed entry signals (decision second, side, planned lots) and a small state
machine: flat -> entry arrival -> exit attempts every second until flat -> cooldown.
Agents do not interact (no endogenous impact); each sees the same visible book.

Units: price 0.01 USDT, size 0.001 BTC lots, notional price*lots = 1e-5 USDT.
"""
import bisect
from fractions import Fraction
import hashlib
import heapq

import numpy as np

from .book import INVALID, STALE, VALID, WARMUP, STATUS_NAMES as NAMES, read_store

SECOND = 1_000_000
HOUR = 3600 * SECOND
HOLD_US = 5 * SECOND
RETRY_US = SECOND
COOLDOWN_US = SECOND
UNITS_PER_USDT = 100_000
CASH0 = 10_000 * UNITS_PER_USDT
TARGET_UNITS = 100 * UNITS_PER_USDT
LEVELS = 50
CAP_LEVELS = 5
CAP_PCT = 1                      # entry refused if 100 * lots > CAP_PCT * depth5
FUNDING_BLOCK_S = 60
LATENCIES = ((0, 0), (100, 100), (500, 500), (1000, 1000))
BASE_LATENCY = (100, 100)
FEES_BP = (Fraction(0), Fraction(1), Fraction(2), Fraction(3), Fraction(11, 2))
BASE_FEE = Fraction(11, 2)


def planned_lots(s):
    """max(1, round_half_up(100 USDT / (mid * 0.001 BTC))) with s = bid + ask (price units)."""
    # mid = s/2, so TARGET/mid + 1/2 = (4 TARGET + s) / (2 s); floor gives round-half-up
    s = np.asarray(s, np.int64)
    return np.maximum(1, (4 * TARGET_UNITS + s) // (2 * s))


def funding_blocked(k):
    """Decision seconds within FUNDING_BLOCK_S of a whole UTC hour (inclusive)."""
    s = np.asarray(k) % 3600
    return (s <= FUNDING_BLOCK_S) | (s >= 3600 - FUNDING_BLOCK_S)


def signals(f, eligible, purge, spread_bp, s, theta, fee_bp=None, lots=None):
    """Entry signals for one agent: (k array, side array +1/-1, lots array).

    f: forecast (NaN where not eligible); fee_bp None means no cost gate (diagnostic);
    lots None applies the planned_lots rule to s = bid + ask at the decision state."""
    k = np.arange(len(f))
    a = np.abs(np.where(eligible, f, 0.0))
    ok = eligible & ~purge & ~funding_blocked(k) & (a > theta)
    if fee_bp is not None:
        ok &= a > spread_bp + 2 * float(fee_bp)
    idx = np.flatnonzero(ok)
    side = np.where(f[idx] > 0, 1, -1)
    q = planned_lots(s[idx]) if lots is None else np.full(len(idx), lots, np.int64)
    return idx, side, q


def walk(levels, lots):
    """Best-first depth walk: [(price, lots taken)], residual cancelled."""
    out, left = [], lots
    for p, q in levels:
        if left == 0:
            break
        take = min(left, q)
        out.append((p, take))
        left -= take
    return out


class Agent:
    """The diagnostic taker policy in one latency/fee scenario for one day: enter with a
    market order when |forecast| > theta, hold 5 s, exit with market orders (retried every
    second until flat), wait 1 s, repeat."""

    def __init__(self, name, k, side, lots, midnight_us, d_o_ms, fee_bp):
        self.name, self.k, self.side, self.lots = name, list(map(int, k)), list(map(int, side)), list(map(int, lots))
        self.midnight, self.end = midnight_us, midnight_us + 86400 * SECOND
        self.d_o, self.fee = d_o_ms * 1000, Fraction(fee_bp)
        self.cash = Fraction(CASH0)
        self.pos = 0                 # signed lots
        self.free = midnight_us      # next entry decision must be >= free
        self.entry_fill = None       # arrival time of the entry fill of the open position
        self.pending = None          # ('entry', side, lots) or ('exit',)
        self.buy = self.sell = self.traded = 0
        self.log = []
        self.c = {'signals': len(self.k), 'entries': 0, 'entry_filled': 0, 'entry_partial': 0,
                  'rejected_state': 0, 'rejected_depth_cap': 0, 'rejected_cash': 0,
                  'exit_attempts': 0, 'exit_no_fill': 0, 'exit_partial': 0, 'round_trips': 0,
                  'max_exit_attempts': 0, 'funding_crossings': 0}
        self.attempts = 0
        self.terminal = 0
        self.expired = None

    # -- scheduling
    def next_entry(self):
        """Arrival time of the next entry order, or None."""
        lo = -(-(self.free - self.midnight) // SECOND)       # first k with t_k >= free
        i = bisect.bisect_left(self.k, lo)
        if i == len(self.k):
            self.pending = None
            return None
        self.pending = ('entry', self.side[i], self.lots[i])
        return self.midnight + self.k[i] * SECOND + self.d_o

    def first_event(self):
        return self.next_entry()

    # -- accounting
    def _fill(self, is_buy, consumed):
        n = sum(p * q for p, q in consumed)
        lots = sum(q for _, q in consumed)
        fee = self.fee * n / 10_000
        if is_buy:
            self.cash -= n
            self.buy += n
            self.pos += lots
        else:
            self.cash += n
            self.sell += n
            self.pos -= lots
        self.cash -= fee
        self.traded += n
        return lots, n

    def _line(self, a, leg, side, req, st, consumed):
        filled = sum(q for _, q in consumed)
        n = sum(p * q for p, q in consumed)
        self.log.append('%d|%s|%s|%d|%s|%d|%d|%s|%d' % (
            a, leg, side, req, st, filled, n, ';'.join('%d:%d' % x for x in consumed), req - filled))

    def _crossed_hour(self, x):
        """Position open at a whole-hour instant H with entry_fill < H <= x."""
        h = (self.entry_fill // HOUR + 1) * HOUR
        return h <= x

    # -- events
    def arrive(self, a, st, ladder):
        """Handle the pending order arriving at a; ladder(side) -> best-first levels of
        the side to be taken (asks for 'Buy'). Returns the next arrival time or None."""
        if self.pending[0] == 'entry':
            _, sgn, lots = self.pending
            side = 'Buy' if sgn > 0 else 'Sell'
            self.c['entries'] += 1
            if st != VALID:
                self.c['rejected_state'] += 1
                self._line(a, 'entry', side, lots, NAMES[st], [])
                self.free = a + 1
                return self.next_entry()
            levels = ladder(side)
            depth5 = sum(q for _, q in levels[:CAP_LEVELS])
            if 100 * lots > CAP_PCT * depth5:
                self.c['rejected_depth_cap'] += 1
                self._line(a, 'entry', side, lots, 'rejected_depth_cap', [])
                self.free = a + 1
                return self.next_entry()
            consumed = walk(levels, lots)
            n = sum(p * q for p, q in consumed)
            if n + self.fee * n / 10_000 > self.cash:
                self.c['rejected_cash'] += 1
                self._line(a, 'entry', side, lots, 'rejected_cash', [])
                self.free = a + 1
                return self.next_entry()
            filled, _ = self._fill(sgn > 0, consumed)
            self._line(a, 'entry', side, lots, 'filled' if filled == lots else 'partial', consumed)
            self.c['entry_filled' if filled == lots else 'entry_partial'] += 1
            if filled == 0:                       # nothing displayed: flat again
                self.free = a + 1
                return self.next_entry()
            self.entry_fill, self.attempts = a, 0
            self.pending = ('exit',)
            return a + HOLD_US + self.d_o
        # exit attempt
        self.c['exit_attempts'] += 1
        self.attempts += 1
        lots = abs(self.pos)
        side = 'Sell' if self.pos > 0 else 'Buy'
        if st != VALID:
            self.c['exit_no_fill'] += 1
            self._line(a, 'exit', side, lots, NAMES[st], [])
            return a + RETRY_US
        consumed = walk(ladder(side), lots)
        filled, _ = self._fill(side == 'Buy', consumed)
        self._line(a, 'exit', side, lots, 'filled' if filled == lots else 'partial', consumed)
        if self.pos != 0:
            self.c['exit_partial' if filled else 'exit_no_fill'] += 1
            return a + RETRY_US
        self.c['round_trips'] += 1
        self.c['max_exit_attempts'] = max(self.c['max_exit_attempts'], self.attempts)
        if self._crossed_hour(a):
            self.c['funding_crossings'] += 1
        self.entry_fill = None
        self.free = a + COOLDOWN_US
        return self.next_entry()

    def expire(self, a):
        """An arrival at or after the next midnight: never evaluated with this day's data."""
        self.expired = a
        if self.pos != 0:
            self.terminal = self.pos
            if self._crossed_hour(self.end):
                self.c['funding_crossings'] += 1
        self.pending = None

    # -- results
    def result(self):
        ident = self.terminal == 0 and self.c['funding_crossings'] == 0 and self.pos == 0
        pi = self.sell - self.buy if ident else None
        net = None if pi is None else Fraction(pi) - self.fee * self.traded / 10_000
        return {'name': self.name, 'fee_bp': str(self.fee), 'identified': ident,
                'terminal_lots': self.terminal if self.terminal else self.pos,
                'pre_fee_units': pi, 'traded_units': self.traded,
                'buy_units': self.buy, 'sell_units': self.sell,
                'net_units': None if net is None else [net.numerator, net.denominator],
                'net_usdt': None if net is None else float(net / UNITS_PER_USDT),
                'counts': dict(self.c),
                'log_sha256': hashlib.sha256(''.join(x + '\n' for x in self.log).encode()).hexdigest(),
                'log_lines': len(self.log)}


def net_at_fee(pre_fee_units, traded_units, fee_bp):
    """Fixed-log net P&L in units: Pi - fee/1e4 * N, exact."""
    return Fraction(pre_fee_units) - Fraction(fee_bp) * traded_units / 10_000


def break_even_bp(pre_fee_units, traded_units):
    """f* = 1e4 * Pi / N in bp per side; None if nothing traded."""
    return None if traded_units == 0 else Fraction(10_000 * pre_fee_units, traded_units)


def run(store_path, agents, midnight_us):
    """Drive agents through one day's store with the frozen visibility/status rules."""
    end = midnight_us + 86400 * SECOND
    heap = []
    for i, ag in enumerate(agents):
        e = ag.first_event()
        if e is not None:
            heapq.heappush(heap, (e, i))
    view = {'synced': False, 'book_cts': None, 'warm': None, 'rec': None}

    def ladder(side):
        r = view['rec']
        x = r['ask'][:r['na']] if side == 'Buy' else r['bid'][:r['nb']]
        return [tuple(v) for v in x[:LEVELS].tolist()]

    def drain(bound_ms):
        while heap and heap[0][0] // 1000 <= bound_ms:
            a, i = heapq.heappop(heap)
            if a >= end:
                agents[i].expire(a)
                continue
            if not view['synced']:
                st = INVALID
            elif a // 1000 - view['book_cts'] > 1000:
                st = STALE
            elif view['warm'] is None or a - view['warm'] < SECOND:
                st = WARMUP
            else:
                st = VALID
            nxt = agents[i].arrive(a, st, ladder)
            if nxt is not None:
                if nxt < a:
                    raise RuntimeError('agent scheduled an event in the past')
                heapq.heappush(heap, (nxt, i))

    for chunk in read_store(store_path):
        cts, flags, bcts = chunk['cts'].tolist(), chunk['flags'].tolist(), chunk['book_cts'].tolist()
        for j in range(len(chunk)):
            drain(cts[j])
            now = bool(flags[j] & 1)
            if not now:
                view['warm'] = None
            elif not view['synced'] or flags[j] & 2:
                view['warm'] = cts[j] * 1000
            view['synced'], view['book_cts'], view['rec'] = now, bcts[j], chunk[j]
    drain((end - 1) // 1000)
    while heap:
        a, i = heapq.heappop(heap)
        agents[i].expire(a)
    return [ag.result() for ag in agents]
