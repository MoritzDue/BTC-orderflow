import csv
import json
from pathlib import Path
import unittest

import numpy as np
from scipy import stats as sps

from orderflow import model, stats

ROOT = Path(__file__).resolve().parent.parent


def read_csv(name):
    with open(ROOT / 'results' / name) as f:
        return list(csv.DictReader(f))


class TestModel(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(5000, 8)) * np.arange(1, 9)
        self.y = self.X @ rng.normal(size=8) * 0.01 + rng.normal(size=5000)

    def test_ridge_matches_augmented_least_squares(self):
        lam = 0.1
        m = model.fit_ridge(self.X, self.y, lam)
        Xs = (self.X - m['mu']) / m['sd']
        n = len(self.y)
        A = np.vstack([Xs, np.sqrt(n * lam) * np.eye(8)])
        b = np.concatenate([self.y - self.y.mean(), np.zeros(8)])
        np.testing.assert_allclose(m['beta'], np.linalg.lstsq(A, b, rcond=None)[0], rtol=1e-10)

    def test_small_lambda_is_ols(self):
        m = model.fit_ridge(self.X, self.y, 1e-12)
        A = np.column_stack([np.ones(len(self.y)), self.X])
        coef = np.linalg.lstsq(A, self.y, rcond=None)[0]
        np.testing.assert_allclose(model.predict_ridge(m, self.X), A @ coef, rtol=1e-8, atol=1e-10)

    def test_lambda_ties_go_to_more_regularization(self):
        self.assertEqual(model.select_lambda({0.001: 1.0, 0.01: 1.0 + 1e-14, 0.1: 1.1}), 0.01)

    def test_r2_against_zero_forecast(self):
        y = np.array([1.0, -1.0, 2.0])
        self.assertEqual(model.day_metrics(y, np.zeros(3))['r2_oos'], 0.0)
        self.assertEqual(model.day_metrics(y, y)['r2_oos'], 1.0)


class TestStats(unittest.TestCase):
    def test_block_indices(self):
        idx = stats.block_indices(20, 5, r=100, seed=1)
        self.assertEqual(idx.shape, (100, 20))
        self.assertTrue(((idx >= 0) & (idx < 20)).all())
        self.assertTrue((np.diff(idx[:, :5], axis=1) == 1).all())      # contiguous blocks

    def test_t_interval(self):
        d = np.array([0.1, 0.3, 0.2, 0.4, 0.25])
        lo, hi = stats.t_interval(d, 0.05)
        h = sps.t.ppf(0.975, 4) * d.std(ddof=1) / np.sqrt(5)
        self.assertAlmostEqual(lo, d.mean() - h)
        self.assertAlmostEqual(hi, d.mean() + h)

    def test_classification(self):
        self.assertEqual(stats.classify([[0.1, 0.2], [0.05, 0.3]], 0.0), 'robustly_above')
        self.assertEqual(stats.classify([[0.1, 0.2], [-0.1, 0.3]], 0.0), 'inconclusive')
        self.assertEqual(stats.strict_class([[-0.3, -0.1], [-0.2, -0.05]]), 'neg')


class TestPublishedNumbers(unittest.TestCase):
    """The intervals in results/summary.json, recomputed from the committed per-day tables."""

    def setUp(self):
        self.S = json.loads((ROOT / 'results/summary.json').read_text())

    def test_first_test_period(self):
        rows = read_csv('sealed_test_days.csv')
        res = stats.primary([float(r['D1']) for r in rows], [float(r['D2']) for r in rows],
                            [int(r['lat_100_100_pi']) for r in rows], [int(r['lat_100_100_n']) for r in rows])
        q1, fs = self.S['signal']['first_test_gain_vs_zero'], self.S['trade_alone']['break_even_fee']
        self.assertEqual(res['quantities']['Q1']['estimate'], q1['estimate'])
        self.assertEqual(res['quantities']['Q1']['t_interval'], q1['t_interval'])
        self.assertEqual({b: res['bootstrap'][b]['Q1'] for b in ('1', '2', '5')}, q1['bootstrap'])
        self.assertEqual(res['quantities']['Q3']['estimate_bp'], fs['estimate_bp'])
        self.assertEqual({b: res['bootstrap'][b]['Q3'] for b in ('1', '2', '5')}, fs['bootstrap'])
        self.assertEqual(res['quantities']['Q3']['reading'], fs['reading'])

    def test_execution_period(self):
        rows = read_csv('execution_days.csv')
        days = [{'eligible': int(r['eligible']), 'unresolved_parents': int(r['unresolved_parents']),
                 **{k: float(r[k]) for k in ('D4', 'D5', 'D7', 'Q6')}} for r in rows]
        res = stats.execution_study(days)
        E = self.S['execution_value']
        for q, key in (('Q4', 'Q4_vs_twap'), ('Q5', 'Q5_vs_random'), ('Q7', 'Q7_vs_shift')):
            got = res['quantities'][q]
            self.assertEqual(got['estimate'], E[key]['estimate'])
            self.assertEqual(got['bootstrap'], E[key]['bootstrap'])
            self.assertEqual(got['t_interval'], E[key]['t_interval'])
            self.assertEqual(got['class'], E[key]['class'])
        self.assertEqual(res['quantities']['Q6']['estimate'], self.S['signal']['second_test_gain_vs_zero']['estimate'])
        self.assertEqual(res['reading']['text'], E['reading']['text'])


if __name__ == '__main__':
    unittest.main()
