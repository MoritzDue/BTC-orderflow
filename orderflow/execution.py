"""Execution study: does the forecast help a desk that must trade anyway?

A parent order (buy or sell 60 x q lots) starts at S_j and is cut into 60 market-order
children, one per 10 s slot. Schedules decide *when inside its slot* each child is sent:

  twap      at the start of every slot
  signal    wait while the forecast says the price is moving in our favour
            (sigma f_t < -theta), at most 9 s; never wait on a missing forecast
  random    delay-matched random delays (same delay distribution as the signal)
  shift     the signal rule applied to a circularly shifted forecast (placebo)
  uniform   uniform random delays; negated: the signal rule with -f (descriptive)

Every schedule is replayed against the historical top-50 book (taker.run). A child takes
at most 10% of the displayed top-5 depth (kappa cap); what does not fill is resubmitted
the next second. Liquidity we took stays missing for tau = 5 s. Anything unfilled at the
deadline S_j + 660 s is valued at the deadline mid, plus a 50 bp penalty for the signal
schedule only (so leaving quantity behind can never look like a saving).

Cost: implementation shortfall IS = sigma 1e4 (VWAP - m) / m in bp against the mid m at
S_j. Per parent, Delta_X = IS_X - IS_signal for X in twap, random, shift (positive means
the signal schedule was cheaper). All arithmetic on costs is exact (Fractions).
"""
from collections import Counter
from fractions import Fraction
import hashlib
import math

import numpy as np

from . import taker
from .book import COLUMNS, DELAYS, VALID
from .features import midnight_s

SECOND = taker.SECOND
SLOTS, SLOT_S, WAIT_MAX_S, DEADLINE_S = 60, 10, 9, 660
STARTS_S = tuple(h * 3600 + m * 60 for h in range(24) for m in (5, 35))   # 48 starts a day
KAPPA = Fraction(1, 10)
CAP_LEVELS = 5
TAU_S = 5
P_BP = 50
B_RANDOM = 20                     # random schedules per parent
K_SHIFT = 20                      # shifted-forecast placebos per day
SHIFT_LO_S, SHIFT_HI_S = 3600, 82800
STATUS, BID, ASK = COLUMNS.index('status'), COLUMNS.index('bid'), COLUMNS.index('ask')
BENCH = ('twap', 'random', 'shift')


# ------------------------------------------------------------------ schedules

def twap_seconds(s0):
    return [s0 + SLOT_S * i for i in range(SLOTS)]


def delayed_seconds(s0, delays):
    return [s0 + SLOT_S * i + int(d) for i, d in enumerate(delays)]


def signal_seconds(s0, sigma, f, theta):
    """First second t in each slot with sigma f_t >= -theta; forced at +9 s."""
    out = []
    for i in range(SLOTS):
        t0 = s0 + SLOT_S * i
        t = t0
        while t < t0 + WAIT_MAX_S:
            x = f[t]
            if not math.isfinite(x) or sigma * x >= -theta:
                break
            t += 1
        out.append(t)
    return out


def random_delays(seed, day, j, b, cdf):
    """One uniform U_i per child from default_rng([seed, day, j, i, b]); delay = #{cdf <= U_i}."""
    cdf = np.asarray(cdf, np.float64)
    out = []
    for i in range(SLOTS):
        u = np.random.default_rng([seed, day, j, i, b]).random()
        out.append(min(WAIT_MAX_S, int(np.searchsorted(cdf, u, side='right'))))
    return out


def uniform_cdf():
    return [(d + 1) / 10 for d in range(WAIT_MAX_S + 1)]


def shift_offsets(seed, day, k=K_SHIFT):
    return [int(x) for x in np.random.default_rng([seed, day, 7]).integers(
        SHIFT_LO_S, SHIFT_HI_S, size=k, endpoint=True)]


def day_int(day):
    return int(day.replace('-', ''))


def schedules(day, p, f, shifts, prm):
    """[(name, 60 decision seconds)] for one parent, in a fixed order."""
    s0, sg, th, seed, cdf = p['s0'], p['sigma'], prm['theta_bp'], prm['seed'], prm['delay_cdf']
    d = day_int(day)
    out = [('twap', twap_seconds(s0)), ('signal', signal_seconds(s0, sg, f, th))]
    for b in range(B_RANDOM):
        out.append(('random/%d' % b, delayed_seconds(s0, random_delays(seed, d, p['j'], b, cdf))))
    for k, g in enumerate(shifts):
        out.append(('shift/%d' % k, signal_seconds(s0, sg, g, th)))
    for b in range(B_RANDOM):
        out.append(('uniform/%d' % b, delayed_seconds(s0, random_delays(seed, d, p['j'], b, uniform_cdf()))))
    out.append(('negated', signal_seconds(s0, sg, -f, th)))
    return out


# ------------------------------------------------------------------ one schedule of one parent

class ParentRun:
    """Replays one schedule of one parent (same interface as taker.Agent)."""

    def __init__(self, name, sigma, s0, decisions, q, midnight_us, d_o_ms, tau_s=TAU_S, kappa=KAPPA):
        self.name, self.sigma, self.s0, self.q = name, sigma, int(s0), int(q)
        self.side = 'Buy' if sigma > 0 else 'Sell'
        self.midnight, self.d_o, self.tau, self.kappa = midnight_us, d_o_ms * 1000, tau_s * SECOND, kappa
        self.deadline = midnight_us + (self.s0 + DEADLINE_S) * SECOND
        self.child = Counter(int(t) for t in decisions)
        self.times = sorted(self.child)
        self.ptr = 0                 # next child second
        self.res = 0                 # unfilled lots waiting for resubmission
        self.dep = []                # liquidity we consumed: [(until_us, price, lots)]
        self.fills = []
        self.log = []
        self.unresolved = None
        self.cur = None
        self.c = Counter()

    def _schedule(self, last_t):
        nxt_child = self.times[self.ptr] if self.ptr < len(self.times) else None
        nxt_retry = last_t + 1 if self.res > 0 else None
        cands = [t for t in (nxt_child, nxt_retry) if t is not None]
        if not cands:
            self.unresolved, self.cur = 0, None
            return None
        t = min(cands)
        a = self.midnight + t * SECOND + self.d_o
        if a > self.deadline:
            self.unresolved = self.res + self.q * sum(self.child[x] for x in self.times[self.ptr:])
            self.cur = None
            return None
        lots = 0
        if t == nxt_child:
            lots += self.q * self.child[t]
            self.ptr += 1
        if self.res > 0:
            self.c['retries'] += 1
        lots += self.res
        self.cur = (t, lots)
        self.res = 0
        return a

    def first_event(self):
        return self._schedule(None)

    def available(self, levels, a):
        """Historical levels minus our own consumption that is still active at a."""
        self.dep = [x for x in self.dep if x[0] > a]
        used = Counter()
        for _, p, n in self.dep:
            used[p] += n
        return [(p, max(0, n - used.get(p, 0))) for p, n in levels]

    def arrive(self, a, st, ladder):
        t, req = self.cur
        self.c['submissions'] += 1
        cap, consumed = None, []
        if st != VALID:
            state = taker.NAMES[st]
            self.c['not_valid'] += 1
        else:
            avail = self.available(ladder(self.side), a)
            cap = math.floor(self.kappa * sum(n for _, n in avail[:CAP_LEVELS]))
            take = min(req, cap)
            consumed = taker.walk([x for x in avail if x[1] > 0], take)
            filled = sum(n for _, n in consumed)
            state = 'filled' if filled == req else ('capped' if filled == take else 'partial')
            self.c[state] += 1
            if self.tau > 0:
                self.dep += [(a + self.tau, p, n) for p, n in consumed]
            self.fills += consumed
        filled = sum(n for _, n in consumed)
        self.log.append('%d|%d|%s|%d|%s|%s|%d|%d|%s|%d' % (
            t, a, self.side, req, '-' if cap is None else str(cap), state, filled,
            sum(p * n for p, n in consumed), ';'.join('%d:%d' % x for x in consumed), req - filled))
        self.res = req - filled
        return self._schedule(t)

    def expire(self, a):
        if self.cur is not None:
            self.res += self.cur[1]
        self.unresolved = self.res + self.q * sum(self.child[x] for x in self.times[self.ptr:])
        self.cur = None

    def result(self):
        lots = sum(n for _, n in self.fills)
        if self.unresolved is None or lots + self.unresolved != SLOTS * self.q:
            raise RuntimeError('quantity not conserved')
        return {'name': self.name, 'filled_lots': lots, 'notional': sum(p * n for p, n in self.fills),
                'unresolved_lots': self.unresolved, 'counts': dict(self.c)}


# ------------------------------------------------------------------ parents and costs

def last_valid(status, s, k):
    """(s, second) at the last second <= k with a valid book, or (None, None)."""
    for x in range(int(k), -1, -1):
        if status[x] == VALID:
            return int(s[x]), x
    return None, None


def parents(grid, eligible, d_md=100):
    """Two parents (buy and sell) per start. Eligible if the book visible at S_j - d_md is
    valid and all features are available; benchmark mid at S_j and deadline mid at
    S_j + 660 s from the zero-delay grid (last valid second at or before)."""
    jm = DELAYS.index(d_md)
    st0, s0 = grid[0, :, STATUS], grid[0, :, BID] + grid[0, :, ASK]
    out = []
    for n, S0 in enumerate(STARTS_S):
        ok = bool(grid[jm, S0, STATUS] == VALID) and bool(eligible[jm][S0])
        sb = last_valid(st0, s0, S0)[0] if ok else None
        sd = last_valid(st0, s0, S0 + DEADLINE_S)[0] if ok else None
        for sigma in (1, -1):
            out.append({'j': 2 * n + (sigma < 0), 'sigma': sigma, 's0': S0, 'eligible': ok,
                        's_bench': sb, 's_deadline': sd})
    return out


def build_runs(day, plist, f, prm, d_o_ms=100, tau_s=TAU_S):
    """Every schedule run of every eligible parent: [(j, name, ParentRun)]."""
    midnight = midnight_s(day) * SECOND
    shifts = [np.roll(np.asarray(f, np.float64), -int(s)) for s in shift_offsets(prm['seed'], day_int(day))]
    runs = []
    for p in plist:
        if p['eligible']:
            for name, dec in schedules(day, p, f, shifts, prm):
                runs.append((p['j'], name, ParentRun(name, p['sigma'], p['s0'], dec, prm['q_lots'],
                                                     midnight, d_o_ms, tau_s)))
    return runs


def shortfall_bp(sigma, notional, filled_lots, unresolved_lots, s_bench, s_deadline, penalty_bp):
    """Exact IS in bp: fills plus the unfilled lots valued at the deadline mid
    moved penalty_bp against us. s = bid + ask, so the mid is s / 2."""
    value = Fraction(notional)
    if unresolved_lots:
        value += unresolved_lots * Fraction(s_deadline, 2) * (1 + Fraction(sigma * penalty_bp, 10_000))
    m = Fraction(s_bench, 2)
    return sigma * 10_000 * (value / (filled_lots + unresolved_lots) - m) / m


def parent_costs(p, results):
    """IS per schedule family (mean over its runs) and Delta = IS_X - IS_signal.
    Returns None if the parent has no valid benchmark (never silently dropped)."""
    fam = {}
    for name, r in results.items():
        if p['s_bench'] is None or (r['unresolved_lots'] and p['s_deadline'] is None):
            return None
        pen = P_BP if name == 'signal' else 0
        IS = shortfall_bp(p['sigma'], r['notional'], r['filled_lots'], r['unresolved_lots'],
                          p['s_bench'], p['s_deadline'], pen)
        fam.setdefault(name.split('/')[0], []).append(IS)
    IS = {k: sum(v, Fraction(0)) / len(v) for k, v in fam.items()}
    unresolved = any(r['unresolved_lots'] for n, r in results.items()
                     if n.split('/')[0] in ('twap', 'signal', 'random', 'shift'))
    return {'IS': IS, 'Delta': {k: IS[k] - IS['signal'] for k in BENCH}, 'unresolved': unresolved}


def day_outcome(plist, costs):
    """Daily D4, D5, D7: mean Delta over the day's eligible parents against TWAP, random
    and shifted timing. None if any eligible parent is unidentified."""
    el = [p['j'] for p in plist if p['eligible']]
    out = {'eligible': len(el), 'unresolved_parents': sum(bool(costs.get(j) and costs[j]['unresolved']) for j in el)}
    ok = el and all(costs.get(j) is not None for j in el)
    for key, k in (('D4', 'twap'), ('D5', 'random'), ('D7', 'shift')):
        if not ok:
            out[key] = None
            continue
        tot = sum((costs[j]['Delta'][k] for j in el), Fraction(0)) / len(el)
        out[key], out[key + '_exact'] = float(tot), [tot.numerator, tot.denominator]
    return out


def fill_log_sha256(runs):
    """SHA-256 of every run's fill log (lines prefixed j|name|), in run order."""
    h = hashlib.sha256()
    for j, name, r in runs:
        for line in r.log:
            h.update(('%d|%s|%s\n' % (j, name, line)).encode())
    return h.hexdigest()
