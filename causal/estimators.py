"""
Causal effect estimators for a binary treatment t, outcome y and confounders X.

  naive_diff        E[y | t=1] - E[y | t=0]                      (confounded)
  ols_adjusted      coefficient of t in y ~ t + X                (linear adjustment)
  ipw               inverse-probability weighting (ATE / ATT)     (re-weights to balance X)
  ps_matching_att   1:1 nearest-neighbour matching on the logit propensity score, with a caliper
  aipw              augmented IPW, "doubly robust": consistent if EITHER the outcome model
                    OR the propensity model is right; SE from the influence function
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Estimate:
    method: str
    effect: float
    ci_low: float
    ci_high: float


def naive_diff(y, t) -> Estimate:
    y, t = np.asarray(y, float), np.asarray(t)
    a, b = y[t == 0], y[t == 1]
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    d = b.mean() - a.mean()
    return Estimate("Naive difference", d, d - 1.96 * se, d + 1.96 * se)


def ols_adjusted(y, t, X) -> Estimate:
    design = sm.add_constant(pd.concat([pd.Series(np.asarray(t), name="t", index=X.index), X], axis=1))
    fit = sm.OLS(np.asarray(y, float), design).fit(cov_type="HC1")     # heteroskedasticity-robust SE
    lo, hi = fit.conf_int().loc["t"]
    return Estimate("Regression adjustment (OLS)", float(fit.params["t"]), float(lo), float(hi))


def propensity_scores(X, t, clip=0.01):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    model.fit(X, t)
    return np.clip(model.predict_proba(X)[:, 1], clip, 1 - clip)


def ipw(y, t, ps, estimand="ATE", n_boot=200, seed=42) -> Estimate:
    """Normalised (Hajek) IPW. Bootstrap CI with the propensity scores held fixed."""
    y, t = np.asarray(y, float), np.asarray(t)

    def est(idx):
        yy, tt, pp = y[idx], t[idx], ps[idx]
        if estimand == "ATE":
            w1, w0 = tt / pp, (1 - tt) / (1 - pp)
        else:                                   # ATT: treated keep weight 1, controls get odds weights
            w1, w0 = tt.astype(float), (1 - tt) * pp / (1 - pp)
        return (w1 * yy).sum() / w1.sum() - (w0 * yy).sum() / w0.sum()

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    boots = [est(rng.choice(idx, idx.size)) for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return Estimate(f"Inverse-probability weighting ({estimand})", float(est(idx)), float(lo), float(hi))


def ps_matching_att(y, t, ps, caliper_sd=0.2, n_boot=200, seed=42):
    """Each treated unit is matched (with replacement) to the control with the closest logit
    propensity score, within caliper_sd x SD(logit ps). Returns the estimate and the share matched."""
    y, t = np.asarray(y, float), np.asarray(t)
    logit = np.log(ps / (1 - ps)).reshape(-1, 1)
    treated, control = np.where(t == 1)[0], np.where(t == 0)[0]
    nn = NearestNeighbors(n_neighbors=1).fit(logit[control])
    dist, pos = nn.kneighbors(logit[treated])
    ok = dist[:, 0] <= caliper_sd * logit.std()
    pairs_t, pairs_c = treated[ok], control[pos[ok, 0]]
    diffs = y[pairs_t] - y[pairs_c]
    rng = np.random.default_rng(seed)
    boots = [rng.choice(diffs, diffs.size).mean() for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return Estimate("Propensity-score matching (ATT)", float(diffs.mean()), float(lo), float(hi)), float(ok.mean())


def aipw(y, t, X, ps, binary_outcome=False):
    """Augmented IPW (doubly robust) ATE with an influence-function standard error.
    Returns the estimate plus the potential-outcome means (mu1, mu0) for risk ratios."""
    y, t = np.asarray(y, float), np.asarray(t)
    if binary_outcome:
        m = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))  # noqa: E731
        mu1 = m().fit(X[t == 1], y[t == 1]).predict_proba(X)[:, 1]
        mu0 = m().fit(X[t == 0], y[t == 0]).predict_proba(X)[:, 1]
    else:
        mu1 = LinearRegression().fit(X[t == 1], y[t == 1]).predict(X)
        mu0 = LinearRegression().fit(X[t == 0], y[t == 0]).predict(X)
    phi1 = mu1 + t * (y - mu1) / ps
    phi0 = mu0 + (1 - t) * (y - mu0) / (1 - ps)
    psi = phi1 - phi0
    ate, se = psi.mean(), psi.std(ddof=1) / np.sqrt(len(y))
    return Estimate("Doubly robust (AIPW)", float(ate), float(ate - 1.96 * se), float(ate + 1.96 * se)), \
        float(phi1.mean()), float(phi0.mean())


def standardized_mean_differences(X, t, weights=None) -> pd.Series:
    """|SMD| per covariate; < 0.1 is the usual threshold for 'balanced'."""
    t = np.asarray(t)
    w = np.ones(len(t)) if weights is None else np.asarray(weights)
    out = {}
    for c in X.columns:
        x = X[c].to_numpy(float)
        m1 = np.average(x[t == 1], weights=w[t == 1]); m0 = np.average(x[t == 0], weights=w[t == 0])
        v1 = np.average((x[t == 1] - m1) ** 2, weights=w[t == 1]); v0 = np.average((x[t == 0] - m0) ** 2, weights=w[t == 0])
        pooled = np.sqrt((v1 + v0) / 2)
        out[c] = abs(m1 - m0) / pooled if pooled > 0 else 0.0
    return pd.Series(out).sort_values(ascending=False)


def e_value(rr: float) -> float:
    """VanderWeele & Ding E-value: how strongly an unmeasured confounder would have to be associated
    with both treatment and outcome (risk-ratio scale) to explain away the observed risk ratio."""
    rr = rr if rr >= 1 else 1 / rr
    return float(rr + np.sqrt(rr * (rr - 1)))
