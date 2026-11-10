from fractions import Fraction
import os
import unittest

import numpy as np

from orderflow import execution as X
from orderflow import taker
from tests.synthetic import MIDNIGHT_MS, quiet_day, write_store

MID_US = MIDNIGHT_MS * 1000


class TestTaker(unittest.TestCase):
    def setUp(self):
        self.store = write_store(quiet_day(until_s=60))

    def tearDown(self):
        os.unlink(self.store)

    def test_round_trip_pays_the_spread(self):
        a = taker.Agent('a', [10], [1], [100], MID_US, 0, Fraction(0))
        (r,) = taker.run(self.store, [a], MID_US)
        # buy 100 lots at the ask 100.10, sell 5 s later at the bid 100.00
        self.assertEqual(r['counts']['round_trips'], 1)
        self.assertEqual(r['pre_fee_units'], (10000 - 10010) * 100)
        self.assertEqual(r['traded_units'], (10000 + 10010) * 100)

    def test_fee_is_linear_in_traded_value(self):
        pi, n = -1000, 2_001_000
        self.assertEqual(taker.net_at_fee(pi, n, Fraction(11, 2)), Fraction(pi) - Fraction(11, 2) * n / 10_000)
        self.assertEqual(taker.break_even_bp(pi, n), Fraction(10_000 * pi, n))

    def test_depth_cap_rejects_large_orders(self):
        # top-5 ask depth is 18,000 lots; orders above 1% of it are refused
        a = taker.Agent('a', [10], [1], [181], MID_US, 0, Fraction(0))
        (r,) = taker.run(self.store, [a], MID_US)
        self.assertEqual(r['counts']['rejected_depth_cap'], 1)
        self.assertEqual(r['traded_units'], 0)

    def test_walk(self):
        self.assertEqual(taker.walk([(10, 3), (11, 5)], 4), [(10, 3), (11, 1)])
        self.assertEqual(taker.walk([(10, 3)], 5), [(10, 3)])          # residual is not filled


class TestSchedules(unittest.TestCase):
    def test_twap_and_signal(self):
        self.assertEqual(X.twap_seconds(300)[:3], [300, 310, 320])
        f = np.full(1000, -1.0)                                        # price expected to fall
        self.assertEqual(X.signal_seconds(0, 1, f, 0.0)[:2], [9, 19])  # a buyer waits the full 9 s
        self.assertEqual(X.signal_seconds(0, -1, f, 0.0)[:2], [0, 10]) # a seller goes now
        f[:5] = np.nan
        self.assertEqual(X.signal_seconds(0, 1, f, 0.0)[0], 0)         # never wait on a missing forecast

    def test_random_delays(self):
        cdf = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 1.0]
        a = X.random_delays(20251105, 20251201, 3, 0, cdf)
        self.assertEqual(a, X.random_delays(20251105, 20251201, 3, 0, cdf))
        self.assertTrue(all(0 <= d <= 9 for d in a))
        self.assertEqual(set(X.random_delays(1, 2, 3, 4, [1.0] * 10)), {0})


class TestParentRun(unittest.TestCase):
    def setUp(self):
        self.store = write_store(quiet_day(until_s=900))

    def tearDown(self):
        os.unlink(self.store)

    def run_parent(self, q):
        r = X.ParentRun('twap', 1, 10, X.twap_seconds(10), q, MID_US, 0)
        (res,) = taker.run(self.store, [r], MID_US)
        return r, res

    def test_twap_fills_at_the_ask(self):
        r, res = self.run_parent(100)
        self.assertEqual((res['filled_lots'], res['unresolved_lots']), (6000, 0))
        self.assertEqual(res['notional'], 6000 * 10010)
        IS = X.shortfall_bp(1, res['notional'], 6000, 0, 20010, 20010, 0)
        self.assertEqual(IS, Fraction(10_000 * 5, 10005))              # half spread over the mid

    def test_cap_limits_each_submission_and_conserves_quantity(self):
        r, res = self.run_parent(2000)        # cap = floor(0.1 x 18,000) = 1,800 lots
        self.assertGreater(res['counts']['capped'], 0)
        self.assertEqual(res['filled_lots'] + res['unresolved_lots'], 60 * 2000)
        self.assertTrue(all(int(x.split('|')[6]) <= 1800 for x in r.log))

    def test_unfilled_quantity_costs_the_penalty(self):
        base = X.shortfall_bp(1, 0, 0, 10, 20010, 20010, 0)
        pen = X.shortfall_bp(1, 0, 0, 10, 20010, 20010, 50)
        self.assertEqual(base, 0)
        self.assertEqual(pen, 50)


if __name__ == '__main__':
    unittest.main()
