"""The estimators must recover a KNOWN effect on simulated confounded data where the naive
comparison is badly biased.

    python -m pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from causal.estimators import (aipw, e_value, ipw, naive_diff, ols_adjusted, propensity_scores,
                               ps_matching_att, standardized_mean_differences)

TRUE_EFFECT = -1.0


@pytest.fixture(scope="module")
def confounded():
    rng = np.random.default_rng(7)
    n = 20_000
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    t = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * x1 - 0.5 * x2)))).astype(int)   # x1 drives treatment...
    y = 2.0 * x1 + 0.5 * x2 + TRUE_EFFECT * t + rng.normal(size=n)               # ...and the outcome
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return y, t, X, propensity_scores(X, t)


def test_naive_estimate_is_biased(confounded):
    y, t, X, ps = confounded
    assert naive_diff(y, t).effect > 0.5          # confounding even flips the sign


@pytest.mark.parametrize("estimator", ["ols", "ipw_ate", "ipw_att", "matching", "aipw"])
def test_adjusted_estimators_recover_the_true_effect(confounded, estimator):
    y, t, X, ps = confounded
    est = {"ols": lambda: ols_adjusted(y, t, X),
           "ipw_ate": lambda: ipw(y, t, ps, "ATE", n_boot=50),
           "ipw_att": lambda: ipw(y, t, ps, "ATT", n_boot=50),
           "matching": lambda: ps_matching_att(y, t, ps, n_boot=50)[0],
           "aipw": lambda: aipw(y, t, X, ps)[0]}[estimator]()
    assert est.effect == pytest.approx(TRUE_EFFECT, abs=0.15)
    assert est.ci_low < est.effect < est.ci_high


def test_weighting_balances_covariates(confounded):
    y, t, X, ps = confounded
    before = standardized_mean_differences(X, t)
    after = standardized_mean_differences(X, t, np.where(t == 1, 1 / ps, 1 / (1 - ps)))
    assert before["x1"] > 0.5 and after.max() < 0.1


def test_e_value_known_values():
    assert e_value(1.0) == pytest.approx(1.0)
    assert e_value(2.0) == pytest.approx(2 + np.sqrt(2))       # VanderWeele & Ding worked example
    assert e_value(0.5) == pytest.approx(e_value(2.0))
