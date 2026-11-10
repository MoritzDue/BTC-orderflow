import unittest

import numpy as np

from orderflow import book
from orderflow import features as F

C = F.C


def random_grid(seed=1):
    """A plausible, fully valid one-second grid."""
    rng = np.random.default_rng(seed)
    g = np.zeros((len(book.DELAYS), F.N, len(book.COLUMNS)), np.int64)
    mid = 10_000_000 + np.cumsum(rng.integers(-3, 4, F.N))
    for j in range(len(book.DELAYS)):
        g[j, :, C['status']] = book.VALID
        g[j, :, C['ofi_ok']] = 1
        g[j, :, C['bid']] = mid - 5
        g[j, :, C['ask']] = mid + 5
        g[j, :, C['bid_l1']] = rng.integers(1, 2000, F.N)
        g[j, :, C['ask_l1']] = rng.integers(1, 2000, F.N)
        g[j, :, C['bid_l5']] = g[j, :, C['bid_l1']] + rng.integers(1, 9000, F.N)
        g[j, :, C['ask_l5']] = g[j, :, C['ask_l1']] + rng.integers(1, 9000, F.N)
        g[j, :, C['ofi']] = rng.integers(-500, 500, F.N)
        g[j, :, C['buy_n']] = rng.integers(0, 3, F.N)
        g[j, :, C['buy_lots']] = g[j, :, C['buy_n']] * 10
    return g


def same(a, b):
    return np.array_equal(a, b, equal_nan=True)


class TestFeatures(unittest.TestCase):
    def test_no_look_ahead(self):
        g = random_grid()
        k0 = 50_000
        before = F.day_features(g)
        g2 = g.copy()
        g2[:, k0 + 1:, C['bid']] -= 40                 # change only the future
        g2[:, k0 + 1:, C['ofi']] = 0
        after = F.day_features(g2)
        self.assertTrue(same(before['X'][:, :k0 + 1], after['X'][:, :k0 + 1]))
        self.assertTrue(np.array_equal(before['eligible'][:, :k0 + 1], after['eligible'][:, :k0 + 1]))
        self.assertFalse(same(before['X'][:, k0 + 1:], after['X'][:, k0 + 1:]))

    def test_target_is_five_second_log_return(self):
        g = random_grid()
        y, ok = F.target(g[0])
        s = (g[0, :, C['bid']] + g[0, :, C['ask']]).astype(float)
        k = 12_345
        self.assertTrue(ok[k])
        self.assertAlmostEqual(y[k], 1e4 * np.log(s[k + 5] / s[k]), places=9)
        self.assertFalse(ok[-1])                    # no label beyond the day

    def test_invalid_book_removes_features(self):
        g = random_grid()
        g[1, 1000, C['status']] = book.STALE
        fe = F.day_features(g)
        self.assertFalse(fe['eligible'][1, 1000])
        self.assertFalse(fe['M'][1, 1001, F.FEATURES.index('ofi_norm')])     # needs k-1 valid too
        self.assertFalse(fe['M'][1, 1000 + 29, F.FEATURES.index('rv30_bp')])  # in the 30 s window
        self.assertTrue(fe['eligible'][1, 1000 + 31])

    def test_purge_bands_and_split_order(self):
        p = F.purge_mask('2025-11-10')
        self.assertTrue(p[:120].all() and not p[120:].any())
        self.assertTrue(F.purge_mask('2025-11-09')[-120:].all())
        self.assertEqual(len(F.split_days('refit', 'train')), 40)
        self.assertEqual(len(F.SEALED), 20)
        self.assertEqual(len(F.SEALED2), 45)


if __name__ == '__main__':
    unittest.main()
