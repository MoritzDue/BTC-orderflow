"""Forecast models: ridge regression on the eight features and a one-feature OFI benchmark.

X (n, 8) raw features in the features.FEATURES order, y (n,) 5-second target in bp.
Features are z-scored with training means and standard deviations only.
"""
import numpy as np

LAMBDAS = (0.0001, 0.001, 0.01, 0.1)
TIE_RTOL = 1e-12
QUANTILES = (0.80, 0.90, 0.95, 0.99)
OFI = 0                      # column of ofi_norm


def zscore(X):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    if not np.all(sd > 0):
        raise ValueError('a training feature has zero variance')
    return mu, sd


def ridge_beta(Xs, yc, lam):
    """argmin (1/n)||yc - Xs b||^2 + lam ||b||^2  <=>  (Xs'Xs/n + lam I) b = Xs'yc/n."""
    n, p = Xs.shape
    G = Xs.T @ Xs / n
    g = Xs.T @ yc / n
    return np.linalg.solve(G + lam * np.eye(p), g), np.linalg.cond(G)


def fit_ridge(X, y, lam):
    mu, sd = zscore(X)
    ybar = y.mean()
    beta, cond = ridge_beta((X - mu) / sd, y - ybar, lam)
    return {'lambda': lam, 'n': int(len(y)), 'mu': mu, 'sd': sd, 'a': float(ybar), 'beta': beta,
            'cond_G': float(cond)}


def predict_ridge(m, X):
    return m['a'] + ((X - m['mu']) / m['sd']) @ m['beta']


def fit_ofi(X, y):
    """OLS of y on (1, ofi_norm)."""
    x = X[:, OFI]
    xc = x - x.mean()
    b = (xc @ (y - y.mean())) / (xc @ xc)
    return {'n': int(len(y)), 'a': float(y.mean() - b * x.mean()), 'b': float(b)}


def predict_ofi(m, X):
    return m['a'] + m['b'] * X[:, OFI]


def forecast(model, X, eligible):
    """Ridge forecast where eligible, NaN elsewhere."""
    f = np.full(len(eligible), np.nan)
    f[eligible] = predict_ridge(model, X[eligible])
    return f


def day_metrics(y, f):
    """n, MSE, information coefficient and out-of-sample R^2 against the zero forecast."""
    e = y - f
    sse, sst0 = float(e @ e), float(y @ y)
    ic = float(np.corrcoef(f, y)[0, 1]) if np.ptp(f) > 0 and np.ptp(y) > 0 else None
    return {'n': int(len(y)), 'mse': sse / len(y), 'ic': ic, 'r2_oos': 1.0 - sse / sst0,
            'sse': sse, 'sst0': sst0}


def select_lambda(scores):
    """scores: {lam: mean validation MSE}. Ties go to the larger (more regularized) lambda."""
    best = min(scores.values())
    return max(lam for lam, s in scores.items() if abs(s - best) <= TIE_RTOL * abs(best))


def thresholds(f):
    """Quantiles of |forecast| in bp (NumPy linear method); 0.90 is the taker threshold."""
    a = np.abs(np.asarray(f, np.float64))
    return {f'{q:.2f}': float(np.quantile(a, q, method='linear')) for q in QUANTILES}


def load_model(path):
    """The frozen ridge and OFI models (results/frozen_model.json)."""
    import json
    m = json.loads(open(path).read())
    ridge = {k: np.asarray(v) if isinstance(v, list) else v for k, v in m['ridge'].items()}
    return ridge, m['ofi'], m
