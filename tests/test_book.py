import unittest

import numpy as np

from orderflow import book
from tests.synthetic import MIDNIGHT_MS, lines, msg, quiet_day

C = {c: i for i, c in enumerate(book.COLUMNS)}
MID_US = MIDNIGHT_MS * 1000


class TestBook(unittest.TestCase):
    def test_units_are_exact(self):
        self.assertEqual(book.to_units('104250.5', 2), 10425050)
        self.assertEqual(book.to_units('0.001', 3), 1)
        for bad in ('1.234', '-1', '1e3', ' 1'):
            with self.assertRaises(ValueError):
                book.to_units(bad, 2)

    def test_snapshot_delta_and_delete(self):
        b = book.Book()
        b.apply(msg(1, 1, 'snapshot', [('100.00', '2')], [('100.10', '3')]))
        b.apply(msg(2, 2, 'delta', [('100.05', '1')], [('100.10', '0'), ('100.20', '4')]))
        self.assertTrue(b.synchronized)
        self.assertEqual(b.top(2), ([(10005, 1000), (10000, 2000)], [(10020, 4000)]))

    def test_gap_and_crossed_book_unsynchronize(self):
        b = book.Book()
        b.apply(msg(1, 1, 'snapshot', [('100.00', '2')], [('100.10', '3')]))
        self.assertIn('possible_gap', b.apply(msg(2, 5, 'delta', [], [])))
        self.assertFalse(b.synchronized)
        b.apply(msg(3, 6, 'snapshot', [('100.00', '2')], [('100.10', '3')]))
        self.assertTrue(b.synchronized)
        self.assertIn('crossed_or_locked', b.apply(msg(4, 7, 'delta', [('100.10', '1')], [])))
        self.assertFalse(b.synchronized)

    def test_ofi_term(self):
        # bid size grows at the same price: buying pressure, positive OFI
        self.assertEqual(book.ofi_term((100, 2, 101, 3), (100, 5, 101, 3)), 3)
        # ask price moves up and the old ask level disappears: positive
        self.assertEqual(book.ofi_term((100, 2, 101, 3), (100, 2, 102, 4)), 3)


class TestGrid(unittest.TestCase):
    def setUp(self):
        # at exactly 3.000 s the best bid moves up to 100.05
        self.m = quiet_day({3000: ([('100.05', '1')], [])})

    def grid(self, trades=()):
        g, n, _ = book.build_grid(lines(self.m), iter(trades), MID_US)
        return g

    def test_strict_past_visibility(self):
        g = self.grid()[0]                      # zero delay
        self.assertEqual(g[3, C['bid']], 10000)  # query at 3.000 s: the 3.000 s message is not visible
        self.assertEqual(g[4, C['bid']], 10005)

    def test_delay_shifts_visibility(self):
        g = self.grid()
        j = book.DELAYS.index(1000)             # 1 s market-data delay
        self.assertEqual(g[j, 4, C['bid']], 10000)
        self.assertEqual(g[j, 5, C['bid']], 10005)

    def test_warmup_valid_and_stale(self):
        g = self.grid()[0]
        self.assertEqual(g[0, C['status']], book.INVALID)   # nothing visible yet
        self.assertEqual(g[1, C['status']], book.WARMUP)    # synced at 0.1 s, < 1 s ago
        self.assertEqual(g[2, C['status']], book.VALID)
        self.assertEqual(g[200, C['status']], book.STALE)   # messages stop at 120 s

    def test_trade_at_query_time_counts_next_second(self):
        t_us = MID_US + 10 * book.SECOND
        g = self.grid([(t_us, True, 10010, 7, False)])[0]
        self.assertEqual(g[10, C['buy_lots']], 0)
        self.assertEqual(g[11, C['buy_lots']], 7)
        self.assertEqual(g[11, C['buy_notional']], 10010 * 7)

    def test_ofi_counted_once(self):
        g = self.grid()[0]
        self.assertEqual(int(g[:, C['ofi']].sum()), 1000)   # the bid improvement, 1 lot = 1000 units
        self.assertEqual(g[2, C['ofi_ok']], 0)                 # the previous second was warm-up
        self.assertTrue(np.all(g[3:100, C['ofi_ok']] == 1))


if __name__ == '__main__':
    unittest.main()
