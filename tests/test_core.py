import numpy as np
import pytest

from quant_gbm import (GBMParams, estimate_gbm, filtered_probabilities, fit_hmm, gbm_return_quantile,
                       kupiec_pof, max_drawdowns, select_n_states, simulate_gbm, simulate_regime_gbm,
                       var_cvar)
from quant_gbm.gbm import draw_shocks


def _regime_returns(n=3000, seed=0):
    """Two-state Markov chain: calm (0.6% daily vol) and turbulent (2% daily vol)."""
    rng = np.random.default_rng(seed)
    A = np.array([[0.98, 0.02], [0.05, 0.95]])
    s, states = 0, []
    for _ in range(n):
        s = rng.choice(2, p=A[s])
        states.append(s)
    states = np.array(states)
    sd = np.where(states == 0, 0.006, 0.02)
    return rng.normal(0.0002, sd), states


def test_estimate_recovers_parameters():
    p = GBMParams(mu=0.08, sigma=0.20)
    paths = simulate_gbm(100.0, p, n_steps=252 * 40, n_paths=1, seed=1)
    est = estimate_gbm(np.diff(np.log(paths[0])))
    assert est.sigma == pytest.approx(0.20, rel=0.03)
    assert est.mu == pytest.approx(0.08, abs=0.07)  # drift is notoriously noisy


def test_simulation_shape_and_mean():
    p = GBMParams(mu=0.05, sigma=0.15)
    paths = simulate_gbm(200.0, p, n_steps=252, n_paths=50_000, seed=2)
    assert paths.shape == (50_000, 253)
    assert np.all(paths[:, 0] == 200.0)
    assert paths[:, -1].mean() == pytest.approx(200.0 * np.exp(0.05), rel=0.01)


def test_student_t_shocks_have_unit_variance():
    z = draw_shocks(np.random.default_rng(3), 400_000, "t", df=5)
    assert z.var() == pytest.approx(1.0, rel=0.03)


def test_closed_form_quantile_matches_simulation():
    p = GBMParams(mu=0.06, sigma=0.18)
    paths = simulate_gbm(1.0, p, n_steps=21, n_paths=200_000, seed=4)
    assert np.quantile(paths[:, -1] - 1, 0.05) == pytest.approx(gbm_return_quantile(p, 21, 0.05), abs=0.002)


def test_var_cvar_ordering():
    r = np.random.default_rng(5).normal(0, 0.1, 100_000)
    v, c = var_cvar(r, 0.05)
    assert v == pytest.approx(0.1645, abs=0.003)
    assert c > v


def test_max_drawdown():
    paths = np.array([[100, 120, 60, 90, 130.0]])
    assert max_drawdowns(paths)[0] == pytest.approx(0.5)


def test_hmm_recovers_regimes_sorted_by_vol():
    r, states = _regime_returns()
    m = fit_hmm(r, 2, n_init=5)
    assert m.stds[0] < m.stds[1]
    assert m.stds[0] == pytest.approx(0.006, rel=0.15)
    assert m.stds[1] == pytest.approx(0.02, rel=0.15)
    probs = filtered_probabilities(m, r)
    assert np.allclose(probs.sum(axis=1), 1)
    assert (probs.argmax(axis=1) == states).mean() > 0.85


def test_bic_prefers_two_states():
    r, _ = _regime_returns()
    best, table = select_n_states(r, candidates=(1, 2, 3), n_init=3)
    assert best.n_states == 2
    assert list(table.index) == [1, 2, 3]


def test_regime_simulation():
    r, _ = _regime_returns()
    m = fit_hmm(r, 2, n_init=3)
    paths, states = simulate_regime_gbm(100.0, m, np.array([1.0, 0.0]), 252, 2000, seed=6)
    assert paths.shape == (2000, 253) and states.shape == (2000, 252)
    assert states.mean() == pytest.approx(m.stationary_distribution()[1], abs=0.08)


def test_kupiec():
    ok = kupiec_pof(np.r_[np.ones(5), np.zeros(95)].astype(bool), 0.05)
    bad = kupiec_pof(np.r_[np.ones(20), np.zeros(80)].astype(bool), 0.05)
    assert ok["p_value"] > 0.5 and bad["p_value"] < 0.001
