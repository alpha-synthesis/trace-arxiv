"""Tests for the v2 corrections that separate R1 and R2 from R0.

Each test pins one corrected behaviour; the legacy behaviour stays the default
and is covered by the existing suite.
"""
import numpy as np
import pytest
import torch

from sde_causal_generator.data_structures import ImpactMatrix
from sde_causal_generator.generate import ImpactDrivenGenerator
from sde_causal_generator.factor_impact_network import train_impact_network


def _im(K=3, b=None, units="daily", estimator="ablation"):
    M = 5
    base = np.zeros((K, M))
    if b is not None:
        base[:, 3] = b
    return ImpactMatrix(
        factor_names=[f"f{k}" for k in range(K)],
        base_impact=base,
        impact_std=np.full((K, M), 0.02),
        occurrence_prob=np.full(K, 0.3),
        temporal_profile=np.ones((K, 10)) / 10,
        interaction_matrix=np.zeros((K, K)),
        feature_names=["open", "high", "low", "close", "volume"],
        estimator=estimator, units=units,
    )


def _real(T=400, seed=0):
    rng = np.random.RandomState(seed)
    lr = rng.standard_t(5, size=(T, 5)) * 0.012
    prices = 100 * np.exp(np.cumsum(lr, axis=0))
    prices[:, 4] = rng.lognormal(15, 0.3, T)        # volume
    return prices, np.diff(np.log(prices), axis=0)


def _close_paths(gen, presence, rp, rlr, seed, **kw):
    np.random.seed(seed)
    out = gen.generate_scenario(
        n_steps=len(rlr), initial_prices=rp[0], n_samples=4,
        real_log_returns=rlr, historical_presence=presence, real_prices=rp, **kw)
    return out[:, :, 3]


def test_impact_matrix_roundtrip_and_legacy_default():
    im = _im()
    d = im.to_dict()
    assert ImpactMatrix.from_dict(d).units == "daily"
    d.pop("units"); d.pop("estimator")
    old = ImpactMatrix.from_dict(d)
    assert old.units == "legacy" and old.estimator == "legacy"
    assert ImpactDrivenGenerator(old).mode == "legacy"
    assert ImpactDrivenGenerator(im).mode == "corrected"


def test_corrected_matched_counterfactual_is_exact_removed_drift():
    rp, rlr = _real()
    T = len(rlr)
    presence = np.zeros((T, 3)); presence[:, 0] = 1.0; presence[::3, 1] = 1.0
    b0 = 1e-3
    gen = ImpactDrivenGenerator(_im(b=[b0, -5e-4, 0.0]))
    fac = _close_paths(gen, presence, rp, rlr, 11, record_ref=True)
    ref = gen._last_ref
    cf = _close_paths(gen, presence, rp, rlr, 11, matched_ref=ref, remove_factors=["f0"])
    d = np.diff(np.log(cf), axis=1) - np.diff(np.log(fac), axis=1)
    # every daily difference is exactly 0 (factor inactive that day) or -b0
    near0 = np.isclose(d, 0.0, atol=1e-12)
    nearb = np.isclose(d, -b0, atol=1e-12)
    assert np.all(near0 | nearb)
    assert nearb.mean() > 0.8          # always present except jitter-padded edges


def test_corrected_has_no_factor_noise_and_no_phantom_floor():
    rp, rlr = _real()
    T = len(rlr)
    gen = ImpactDrivenGenerator(_im(b=[0.0, 0.0, 0.0]))
    p1 = np.zeros((T, 3)); p1[:, 0] = 1.0
    p2 = np.zeros((T, 3)); p2[:, 1] = 1.0
    a = _close_paths(gen, p1, rp, rlr, 3)
    b = _close_paths(gen, p2, rp, rlr, 3)
    # with zero effects and no factor noise the presence pattern is irrelevant
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)
    # schedule floor 0: an absent factor never fires
    np.random.seed(0)
    sched = gen._build_schedule(T, None, 4, historical_presence=np.zeros((T, 3)))
    assert sched.sum() == 0


def _train(prices, presence, seed=0, epochs=8, **kw):
    torch.manual_seed(seed); np.random.seed(seed)
    return train_impact_network(
        factor_matrix=presence, price_data=prices, factor_names=["f0", "f1", "f2"],
        n_lags=10, n_epochs=epochs, batch_size=64, learning_rate=1e-3,
        device="cpu", patience=50, architecture="tft", **kw)[1]


def test_ablation_estimator_units_and_drift_invariance():
    rng = np.random.RandomState(1)
    T = 300
    presence = (rng.rand(T, 3) < 0.3).astype(np.float32)
    lr = rng.normal(0, 0.01, (T, 5))
    p_a = 100 * np.exp(np.cumsum(lr, axis=0))
    p_b = 100 * np.exp(np.cumsum(lr + 0.002, axis=0))   # every log-return shifted
    p_a[:, 4] = p_b[:, 4] = 1e6
    im_a = _train(p_a, presence, impact_estimator="ablation")
    im_b = _train(p_b, presence, impact_estimator="ablation")
    assert im_a.estimator == "ablation" and im_a.units == "daily"
    assert im_a.response_curves is None
    # adding a constant to all returns must not change the estimated effects
    np.testing.assert_allclose(im_a.base_impact[:, 3], im_b.base_impact[:, 3], atol=1e-6)


def test_planted_effect_sign_recovered_without_llm_prior():
    rng = np.random.RandomState(2)
    T = 800
    presence = (rng.rand(T, 3) < 0.25).astype(np.float32)
    sigma = 0.01
    lr = rng.normal(0, sigma, (T, 5))
    # pipeline convention (prepare_training_data): presence on day t is paired
    # with the log-return from t to t+1, i.e. lr[t+1] here.
    lr[1:, 3] += 0.8 * sigma * presence[:-1, 0] - 0.8 * sigma * presence[:-1, 1]
    prices = 100 * np.exp(np.cumsum(lr, axis=0))
    prices[:, 4] = rng.lognormal(15, 0.3, T)   # non-degenerate volume target
    torch.manual_seed(0); np.random.seed(0)
    im = train_impact_network(
        factor_matrix=presence, price_data=prices, factor_names=["f0", "f1", "f2"],
        n_lags=10, n_epochs=300, batch_size=64, learning_rate=1e-3, device="cpu",
        patience=30, architecture="tft", impact_estimator="ablation", llm_directions=None)[1]
    b = im.base_impact[:, 3] / sigma
    # planted +0.8 / -0.8 / 0 sigma; recovered 0.69 / -0.81 / 0.02 on 2026-09-30
    assert b[0] > 0.4 and b[1] < -0.4 and abs(b[2]) < 0.3


# ── v2.1 corrections ───────────────────────────────────────────────────

def test_generator_refuses_mode_units_mismatch():
    with pytest.raises(ValueError):
        ImpactDrivenGenerator(_im(units="legacy", estimator="legacy"), mode="corrected")
    with pytest.raises(ValueError):
        ImpactDrivenGenerator(_im(units="daily"), mode="legacy")


def _small_tft(K=3, seed=0):
    from sde_causal_generator.factor_impact_network import TemporalFusionTransformerFIN
    import inspect
    torch.manual_seed(seed)
    sig = inspect.signature(TemporalFusionTransformerFIN.__init__)
    kw = {}
    for name, val in (("n_factors", K), ("n_features", 5), ("n_lags", 10)):
        if name in sig.parameters:
            kw[name] = val
    m = TemporalFusionTransformerFIN(**kw)
    m.eval()
    return m


def test_pooled_ablation_matches_slow_reference_and_training_windows():
    model = _small_tft()
    rng = np.random.RandomState(5)
    T, K, L = 60, 3, 10
    P = torch.tensor((rng.rand(T, K) < 0.3).astype(np.float32))
    fast = model._ablation_all_windows(P, "cpu", batch=7)
    # slow reference: training windows s = 0 .. T-L-2 (prepare_training_data),
    # pooled mean over active (window, day) pairs
    n_w = T - L - 1
    num = np.zeros((K, 5)); den = np.zeros(K)
    with torch.no_grad():
        for s in range(n_w):
            w = P[s:s + L].unsqueeze(0)
            fp = model(w)["predicted_returns"].numpy()[0]
            for k in range(K):
                act = P[s:s + L, k].numpy() > 0
                if act.sum() == 0:
                    continue
                ab = w.clone(); ab[:, :, k] = 0.0
                ap = model(ab)["predicted_returns"].numpy()[0]
                num[k] += (fp[act] - ap[act]).sum(0); den[k] += act.sum()
    ref = num / np.maximum(den, 1)[:, None]
    np.testing.assert_allclose(fast, ref, atol=1e-5)
    # a factor active only on the last day (no training target) gets no effect
    P2 = torch.zeros(T, K); P2[-1, 0] = 1.0
    assert np.allclose(model._ablation_all_windows(P2, "cpu")[0], 0.0)


def test_training_is_reproducible_with_fixed_seed_on_cpu():
    rng = np.random.RandomState(3)
    T = 200
    presence = (rng.rand(T, 3) < 0.3).astype(np.float32)
    lr = rng.normal(0, 0.01, (T, 5))
    prices = 100 * np.exp(np.cumsum(lr, axis=0)); prices[:, 4] = rng.lognormal(15, 0.3, T)
    a = _train(prices, presence, impact_estimator="ablation", llm_directions=None)
    b = _train(prices, presence, impact_estimator="ablation", llm_directions=None)
    np.testing.assert_array_equal(a.base_impact, b.base_impact)


def test_remove_unknown_factor_raises():
    rp, rlr = _real()
    gen = ImpactDrivenGenerator(_im(b=[0.001, 0.0, 0.0]))
    presence = np.ones((len(rp), 3), dtype=np.float32)
    with pytest.raises(KeyError, match="unknown factor"):
        _close_paths(gen, presence, rp, rlr, 0, remove_factors=["f0__typo"])
