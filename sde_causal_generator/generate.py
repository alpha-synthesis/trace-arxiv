# -*- coding: utf-8 -*-
"""
Impact-Driven Synthetic Data Generator  (v5 — non-linear response curves).

Improvements over v4
---------------------
NL1. **Non-linear factor response** — uses response curves probed at
     multiple intensities [0.5, 1.0, 2.0] to interpolate the true
     non-linear relationship between activation and price impact.
     Factors with negligible non-linearity fall back to linear.
NL2. **Vectorised temporal profiles** — temporal lag convolutions use
     ``np.convolve`` instead of triple-nested Python loops (O(K×S×L)
     → O(K×S) with C-level inner loops).
NL3. **Non-linear interactions** — pairwise interactions use ``tanh``
     saturation, geometric-mean scaling, and non-linearity amplification
     to prevent blow-up and capture higher-order synergies.

Retained from v4
-----------------
R1. **GJR-GARCH(1,1) conditional volatility** — time-varying σ_t with
    asymmetric leverage term (γ·ε²·𝟙(ε<0)) reproduces both volatility
    clustering and the leverage effect (negative returns → higher vol).
    Uses skewed Student-t innovations for realistic return asymmetry.
R2. **Continuous drift correction** — every 63 days the cumulative
    drift is nudged toward the real trajectory (50 % correction).
R3. **Bear-market forcing** — bearish regimes use 100 % drift
    calibration and guarantee negative mean.
R4. **Volume rescale** — post-generation rescale so mean(synth) ≈
    mean(real).
R5. **Regime-dependent jump intensity** — crisis regimes get 3-5×
    baseline λ for fatter tails where they belong.
R6. **Drift-only regime calibration** — regime calibration corrects
    only drift (mean), preserving GARCH temporal memory intact.
    Volatility is calibrated via ω in the GARCH recursion, not by
    post-hoc segment rescaling that would destroy clustering.

Retained from v3
-----------------
1. Uses **real trading dates** instead of ``pd.bdate_range``.
2. **Informed temporal schedule** — factor activation mirrors the
   historical presence extracted by the LLM, with stochastic jitter.
3. **Volume AR(1) + day-of-week** — autoregressive volume with
   intra-week seasonality, conditioned on |returns|.
4. **t-Student noise** for fat tails, with degrees-of-freedom
   estimated from real kurtosis.
5. **Regime-switching calibration** — detects drift/vol regimes
   via rolling statistics and calibrates per-segment instead of
   using one global (μ, σ).
6. Works with daily-granularity presence matrices (onset + decay).
7. **Merton jump-diffusion** — Poisson jumps for crash / rally events.
8. **OHLC microstructure** — generates Close via SDE, then derives
   Open/High/Low from a calibrated intraday microstructure model.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .data_structures import ImpactMatrix, ScenarioSet


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _estimate_t_df(real_log_returns: np.ndarray) -> float:
    """Estimate t-Student degrees-of-freedom from empirical kurtosis.

    Excess kurtosis of a t-distribution is  6 / (ν − 4)  for ν > 4.
    Solving: ν = 4 + 6 / kurtosis  (clamp to [3, 30]).
    """
    kurt = float(sp_stats.kurtosis(real_log_returns.ravel(), fisher=True))
    if kurt <= 0:
        return 30.0          # practically Gaussian
    nu = 4.0 + 6.0 / kurt
    # STAT-7: ν < 4 has infinite kurtosis; clamp to [5, 30] for
    # well-defined fourth moment (finite kurtosis requires ν > 4).
    return float(np.clip(nu, 5.0, 30.0))


def _estimate_jump_params(
    real_log_returns: np.ndarray,
    threshold_sigma: float = 3.0,
    target_excess_kurt: float = 6.0,
) -> Tuple[float, float, float]:
    """Estimate Merton jump-diffusion parameters from tail events.

    Returns (λ, μ_J, σ_J):
        λ   — daily jump intensity (probability per day)
        μ_J — mean jump size  (log-return)
        σ_J — jump size std   (log-return)

    Jump sizes are scaled so the 4th-moment contribution does
    not exceed ``target_excess_kurt``.
    """
    # Use close column (index 3) if available, else first column
    col = min(3, real_log_returns.shape[1] - 1)
    r = real_log_returns[:, col]
    sigma = r.std()
    threshold = threshold_sigma * sigma

    jumps = r[np.abs(r) > threshold]
    if len(jumps) < 3:
        return 0.0, 0.0, 0.0          # not enough tail events

    lam = len(jumps) / len(r)          # ≈ P(jump on any day)
    mu_j = float(jumps.mean())
    sigma_j = float(jumps.std())

    # Scale σ_J so excess kurtosis from jumps ≈ target_excess_kurt.
    # For Poisson jumps:  excess_kurt ≈ λ·(3σ_J⁴ + 6μ_J²σ_J² + μ_J⁴) / σ_total⁴
    # Approximate with symmetric jumps: ≈ 3λ·σ_J⁴ / σ⁴
    # → σ_J = σ · (target / (3λ))^{1/4}
    if lam > 0 and target_excess_kurt > 0:
        desired_sig_j = sigma * (target_excess_kurt / (3.0 * lam)) ** 0.25
        if sigma_j > desired_sig_j:
            sigma_j = desired_sig_j

    return lam, mu_j, sigma_j


def _fit_gjr_garch_params(
    real_log_returns: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Fit GJR-GARCH(1,1) to real close returns via ``arch`` package.

    The GJR (Glosten-Jagannathan-Runkle) model adds an asymmetric
    leverage term γ that increases volatility after negative shocks:

        σ²_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·𝟙(ε_{t-1}<0) + β·σ²_{t-1}

    This captures the *leverage effect* — a core stylized fact where
    negative returns increase subsequent volatility more than positive
    returns of the same magnitude.

    Returns
    -------
    (omega, alpha, gamma, beta)
        omega — baseline variance constant (decimal scale)
        alpha — ARCH coefficient (symmetric shock response)
        gamma — leverage coefficient (extra response to negative shocks)
        beta  — GARCH persistence (volatility memory)
    """
    try:
        from arch import arch_model

        r = np.asarray(real_log_returns).ravel()
        r_pct = r * 100.0  # arch expects percentage returns
        am = arch_model(
            r_pct, vol="Garch", p=1, o=1, q=1,  # o=1 → GJR asymmetric
            dist="normal", mean="Zero", rescale=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(disp="off", show_warning=False)

        omega = float(res.params.get("omega", 0.01)) / 1e4
        raw_alpha = float(res.params.get("alpha[1]", 0.05))
        raw_gamma = float(res.params.get("gamma[1]", 0.05))
        raw_beta = float(res.params.get("beta[1]", 0.88))

        alpha = float(np.clip(raw_alpha, 0.01, 0.20))
        gamma = float(np.clip(raw_gamma, 0.0, 0.25))
        beta = float(np.clip(raw_beta, 0.50, 0.98))

        # R1-FIX: Detect degenerate fit (reference data has no real
        # ARCH effect — e.g. i.i.d. or very short series).
        # Indicators of a spurious fit:
        #   1. Very low ARCH reactivity (α + γ/2 < 0.03)
        #   2. Negative raw leverage coefficient (unphysical)
        #   3. Very low persistence (insufficient GARCH memory)
        # Fall back to canonical equity GJR-GARCH parameters
        # (Bollerslev 1986; Engle & Ng 1993; typical equity estimates).
        arch_reactivity = alpha + gamma / 2.0
        persistence = alpha + gamma / 2.0 + beta
        degenerate = (
            arch_reactivity < 0.03
            or raw_gamma < -0.05
            or persistence < 0.90
        )
        if degenerate:
            alpha, gamma, beta = 0.06, 0.10, 0.88
            omega = float(np.var(r)) * (1.0 - alpha - gamma / 2.0 - beta)
            omega = max(omega, 1e-8)

        # Stationarity: α + γ/2 + β < 1
        if alpha + gamma / 2.0 + beta >= 0.999:
            beta = 0.999 - alpha - gamma / 2.0
            beta = max(beta, 0.5)
        return omega, alpha, gamma, beta
    except Exception:
        # Fallback: typical equity GJR-GARCH(1,1) parameters
        return 1e-6, 0.06, 0.10, 0.88


def _estimate_skewed_t_params(
    real_log_returns: np.ndarray,
) -> Tuple[float, float]:
    """Estimate skewed Student-t parameters (df, skew) from data.

    Uses Hansen's skewed-t parameterisation:
        df   — degrees of freedom (tail weight, ≥ 5)
        skew — asymmetry parameter in [-1, +1], negative = left tail heavier

    Returns
    -------
    (df, skew_param)
    """
    r = np.asarray(real_log_returns).ravel()
    # Estimate df from kurtosis
    kurt = float(sp_stats.kurtosis(r, fisher=True))
    if kurt <= 0:
        df = 30.0
    else:
        df = 4.0 + 6.0 / kurt
    df = float(np.clip(df, 5.0, 30.0))

    # Estimate skewness direction
    empirical_skew = float(sp_stats.skew(r))
    # Map empirical skewness to a [-0.5, +0.5] asymmetry parameter
    # (clamped, since very extreme skewness is unreliable)
    skew_param = float(np.clip(empirical_skew * 0.3, -0.5, 0.5))

    return df, skew_param


def _sample_skewed_t(
    df: float, skew_param: float, size: int,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Draw samples from a skewed Student-t distribution.

    Uses Fernández-Steel (1998) skewing mechanism:
        X = (1/ξ)·|Z|  if U < 1/(1+ξ²)
        X = -ξ·|Z|      otherwise
    where Z ~ t(df), ξ = exp(skew_param), and the result is centred
    to zero mean.  Variance is NOT rescaled because the GARCH loop
    multiplies by √σ² anyway; rescaling here would destroy kurtosis.
    """
    if rng is None:
        z = np.random.standard_t(df, size)
        u = np.random.uniform(0, 1, size)
    else:
        z = rng.standard_t(df, size)
        u = rng.uniform(0, 1, size)

    xi = np.exp(skew_param)  # asymmetry ratio
    threshold = 1.0 / (1.0 + xi ** 2)
    result = np.where(
        u < threshold,
        np.abs(z) / xi,
        -np.abs(z) * xi,
    )
    # Centre to zero mean only — do NOT rescale variance,
    # as that would destroy the excess kurtosis from the t-distribution.
    result = result - result.mean()
    # Normalise to unit variance so the GARCH σ² controls scale
    std = result.std()
    if std > 1e-10:
        result = result / std
    return result


def _detect_regimes(
    real_log_returns: np.ndarray,
    min_regime_len: int = 63,
    n_regimes: int = 0,
) -> List[Tuple[int, int, float, float]]:
    """Detect drift/vol regimes via rolling statistics.

    Uses a simple approach: compute rolling annual drift and partition
    into segments where drift magnitude shifts significantly.

    Parameters
    ----------
    real_log_returns : (T,) or (T, P) — uses first column if 2D
    min_regime_len : minimum segment length in days
    n_regimes : 0 = auto-detect

    Returns
    -------
    regimes : list[(start, end, mu_daily, sigma_daily)]
    """
    if real_log_returns.ndim == 2:
        r = real_log_returns[:, 0]
    else:
        r = real_log_returns
    T = len(r)

    if T < min_regime_len * 2:
        return [(0, T, float(r.mean()), float(r.std()))]

    # Rolling annual drift (252-day window)
    window = min(252, T // 3)
    roll_mu = pd.Series(r).rolling(window, min_periods=window // 2).mean().values
    roll_mu = np.nan_to_num(roll_mu, nan=float(r.mean()))

    # Auto-detect number of regimes from period length
    if n_regimes <= 0:
        n_regimes = max(2, min(8, T // (252 * 4) + 1))

    # Cumulative sum of deviations for changepoint detection
    global_mu = float(r.mean())
    cusum = np.cumsum(roll_mu - global_mu)

    # Find changepoints via even partitioning refined by cusum
    boundaries: List[int] = [0]
    segment_len = T // n_regimes
    for i in range(1, n_regimes):
        cp = i * segment_len
        search_lo = max(boundaries[-1] + min_regime_len, cp - segment_len // 4)
        search_hi = min(T - min_regime_len, cp + segment_len // 4)
        if search_lo >= search_hi:
            boundaries.append(cp)
            continue
        diffs = np.abs(np.diff(cusum[search_lo:search_hi]))
        if len(diffs) > 0:
            best = search_lo + int(np.argmax(diffs))
            boundaries.append(best)
        else:
            boundaries.append(cp)
    boundaries.append(T)

    # Clean out too-short segments
    clean: List[int] = [boundaries[0]]
    for b in boundaries[1:]:
        if b - clean[-1] >= min_regime_len:
            clean.append(b)
        elif b == T:
            clean[-1] = T
    if clean[-1] != T:
        clean.append(T)

    regimes: List[Tuple[int, int, float, float]] = []
    for i in range(len(clean) - 1):
        s, e = clean[i], clean[i + 1]
        seg = r[s:e]
        regimes.append((s, e, float(seg.mean()), float(seg.std())))

    return regimes


def _estimate_microstructure_params(
    real_prices: np.ndarray,
) -> Dict[str, float]:
    """Estimate intraday microstructure params from real OHLC.

    Calibrates:
        gap_std  :  std of overnight log-gap   log(O_t / C_{t-1})
        high_std :  std of log-excursion        log(H / max(O,C))
        low_std  :  std of log-excursion        log(min(O,C) / L)
    """
    if real_prices.shape[1] < 4:
        return {"gap_std": 0.005, "high_std": 0.008, "low_std": 0.008}

    O = real_prices[:, 0]
    H = real_prices[:, 1]
    L = real_prices[:, 2]
    C = real_prices[:, 3]

    # Overnight gap
    with np.errstate(divide="ignore", invalid="ignore"):
        gaps = np.log(O[1:] / np.clip(C[:-1], 1e-8, None))
    gaps = np.nan_to_num(gaps, nan=0.0)
    gap_std = float(np.clip(np.std(gaps), 0.001, 0.05))

    # High excursion (always ≥ 0)
    max_oc = np.maximum(O, C)
    with np.errstate(divide="ignore", invalid="ignore"):
        high_exc = np.log(np.clip(H / np.clip(max_oc, 1e-8, None), 1.0, None))
    high_exc = np.nan_to_num(high_exc, nan=0.0)
    high_std = float(np.clip(np.std(high_exc), 0.001, 0.05))

    # Low excursion (always ≥ 0)
    min_oc = np.minimum(O, C)
    with np.errstate(divide="ignore", invalid="ignore"):
        low_exc = np.log(np.clip(min_oc / np.clip(L, 1e-8, None), 1.0, None))
    low_exc = np.nan_to_num(low_exc, nan=0.0)
    low_std = float(np.clip(np.std(low_exc), 0.001, 0.05))

    return {"gap_std": gap_std, "high_std": high_std, "low_std": low_std}


# ══════════════════════════════════════════════════════════════════════
# Generator
# ══════════════════════════════════════════════════════════════════════


class ImpactDrivenGenerator:
    """
    Generates synthetic price data using the learned impact matrix.

    The generation process:
        1. Build factor-activation schedule (informed by real history).
        2. Apply learned impacts with non-linear response curves.
        3. Apply temporal profiles via vectorised convolution.
        4. Apply pairwise interactions with non-linear synergy.
        5. Add t-Student noise (fat tails).
        6. Add Merton jump-diffusion component.
        7. Regime-switching calibration (per-segment drift/vol).
        8. Convert Close log-returns → Close prices.
        9. Derive Open/High/Low via microstructure model.
       10. Generate volume (AR(1) + day-of-week).
       11. Enforce OHLCV constraints.
    """

    def __init__(
        self,
        impact_matrix: ImpactMatrix,
        window_trading_days: int = 63,
        mode: Optional[str] = None,
    ):
        # mode "legacy" reproduces R0's generator. "corrected"
        # (criteria v2): impacts in daily units, linear drift
        # act*b_k only, no factor noise, no temporal-lag or interaction
        # terms, schedule floor 0 and Bernoulli on the presence indicator.
        # Default: corrected when the impact matrix declares daily units.
        if mode is None:
            mode = "corrected" if getattr(impact_matrix, "units", "legacy") == "daily" else "legacy"
        if mode not in ("legacy", "corrected"):
            raise ValueError(f"unknown generator mode: {mode}")
        units = getattr(impact_matrix, "units", "legacy")
        if (mode == "corrected") != (units == "daily"):
            raise ValueError(
                f"generator mode {mode!r} is incompatible with impact-matrix units "
                f"{units!r}: corrected mode needs daily-unit ablation impacts and "
                f"legacy mode needs legacy impacts")
        self.mode = mode
        self.im = impact_matrix
        self.K = len(impact_matrix.factor_names)
        self.M = len(impact_matrix.feature_names)
        # Number of price features (OHLC, excluding volume)
        self._price_features = min(4, self.M)
        self.window_days = max(1, window_trading_days)

        # Pre-compute non-linear response interpolators per factor.
        # If response_curves are available (K, 3, M) at intensities
        # [0.5, 1.0, 2.0], build a piecewise-linear lookup for each
        # factor that maps activation intensity → impact.
        self._has_response_curves = (
            impact_matrix.response_curves is not None
            and len(impact_matrix.response_curves) == self.K
        )
        self._response_intensities = np.array([0.5, 1.0, 2.0])

    # ── non-linear factor response ──────────────────────────────────

    def _apply_nonlinear_impact(
        self,
        schedule: np.ndarray,   # (S, T, K)
        close_col: int,
        daily_scale: float,
    ) -> np.ndarray:
        """Apply factor impacts using non-linear response curves.

        Instead of  ``impact = schedule × base_impact[k]``  (linear),
        this interpolates the response curves probed at multiple
        intensities [0.5, 1.0, 2.0] to capture the non-linear
        relationship between activation intensity and price impact.

        For factors without significant non-linearity (score < 0.005),
        falls back to the faster linear application.

        Returns
        -------
        base : (S, T) — per-step Close log-return contribution.
        """
        n_samples, n_steps, K = schedule.shape

        # Response curves: (K, 3, M) at intensities [0.5, 1.0, 2.0]
        rc = self.im.response_curves  # may be None
        nl_scores = self.im.nonlinearity_scores  # (K,) or None
        base_impact = self.im.base_impact[:, close_col] * daily_scale  # (K,)

        base = np.zeros((n_samples, n_steps))

        for k in range(K):
            act = schedule[:, :, k]  # (S, T)

            # Check if this factor is meaningfully non-linear
            use_nonlinear = (
                self._has_response_curves
                and nl_scores is not None
                and float(nl_scores[k]) > 0.005
            )

            if use_nonlinear:
                # Piecewise-linear interpolation of the response curve
                # at the Close column.
                curve_k = rc[k, :, close_col] * daily_scale  # (3,)
                # np.interp clamps outside [0.5, 2.0]; extend to [0, 3]
                # with linear extrapolation at the boundaries.
                #   Below 0.5: scale from origin (0→0, 0.5→curve[0])
                #   Above 2.0: linear extrapolation from [1.0, 2.0]
                intensities = self._response_intensities
                impact = np.interp(
                    act.ravel(),
                    np.concatenate([[0.0], intensities, [3.0]]),
                    np.concatenate([
                        [0.0],
                        curve_k,
                        [curve_k[2] + (curve_k[2] - curve_k[1])],
                    ]),
                ).reshape(n_samples, n_steps)
                base += impact
            else:
                # Linear fallback (faster)
                base += act * base_impact[k]

        return base

    # ── vectorised temporal profiles ────────────────────────────────

    def _apply_temporal_profiles(
        self,
        schedule: np.ndarray,   # (S, T, K)
        base: np.ndarray,       # (S, T)
        close_col: int,
        daily_scale: float,
    ) -> np.ndarray:
        """Apply temporal lag profiles via vectorised convolution.

        Instead of triple-nested Python loops (O(K×S×L)), uses
        ``np.convolve`` per factor to apply the learned decay profiles
        — O(K × S × T) with C-level inner loops.
        """
        n_samples, n_steps, K = schedule.shape
        daily_impact = self.im.base_impact[:, close_col] * daily_scale

        for k in range(K):
            profile = self.im.temporal_profile[k]
            # Skip factors with no lag structure or negligible lag weights
            if np.abs(profile[1:]).max() < 1e-6:
                continue

            # Build causal convolution kernel: [0, profile[1], profile[2], ...]
            # (profile[0] is the contemporaneous effect, already in base)
            L_prof = len(profile)
            kernel = np.zeros(L_prof)
            kernel[1:] = profile[1:] * daily_impact[k]

            # Skip if the kernel has no energy
            if np.abs(kernel).max() < 1e-10:
                continue

            for s in range(n_samples):
                # Convolve activation with kernel (causal: output[t] depends
                # only on input[0..t]).  mode='full' then truncate.
                conv = np.convolve(schedule[s, :, k], kernel, mode="full")
                base[s] += conv[:n_steps]

        return base

    # ── non-linear interactions ─────────────────────────────────────

    def _apply_interactions(
        self,
        schedule: np.ndarray,   # (S, T, K)
        close_col: int,
        daily_scale: float,
    ) -> np.ndarray:
        """Apply pairwise factor interactions with non-linear blending.

        Improvements over v3:
        - Uses ``tanh`` saturation so strong co-activations don't
          produce unbounded interaction contributions.
        - Scales interaction by the geometric mean of per-factor impacts
          (not arithmetic mean) for better dynamic range.
        - Factors with strong non-linearity scores get an amplified
          interaction contribution.
        """
        n_samples, n_steps, K = schedule.shape
        daily_impact = self.im.base_impact[:, close_col] * daily_scale
        nl_scores = self.im.nonlinearity_scores  # (K,) or None

        inter = np.zeros((n_samples, n_steps))
        for i in range(K):
            for j in range(i + 1, K):
                strength = self.im.interaction_matrix[i, j]
                if abs(strength) < 1e-6:
                    continue

                joint = schedule[:, :, i] * schedule[:, :, j]  # (S, T)

                # Geometric mean of absolute impacts for better scaling
                geo_imp = np.sqrt(
                    np.abs(daily_impact[i] * daily_impact[j])
                ).clip(1e-10)

                # Non-linearity amplification: if either factor is
                # highly non-linear, amplify the interaction contribution
                nl_amp = 1.0
                if nl_scores is not None:
                    pair_nl = max(float(nl_scores[i]), float(nl_scores[j]))
                    nl_amp = 1.0 + 0.5 * min(pair_nl / (pair_nl + 0.01), 1.0)

                # Saturating interaction via tanh to prevent blow-up
                # when both factors have strong simultaneous activation.
                raw_inter = joint * strength * geo_imp * nl_amp
                inter += np.tanh(raw_inter / (geo_imp * 5 + 1e-10)) * geo_imp * 5

        return inter

    # ── main entry ──────────────────────────────────────────────────

    def generate_scenario(
        self,
        n_steps: int = 252,
        initial_prices: Optional[np.ndarray] = None,
        active_factors: Optional[Dict[str, float]] = None,
        n_samples: int = 1,
        noise_scale: float = 1.0,
        seed: Optional[int] = None,
        real_log_returns: Optional[np.ndarray] = None,
        real_volume: Optional[np.ndarray] = None,
        historical_presence: Optional[np.ndarray] = None,
        real_prices: Optional[np.ndarray] = None,
        real_dates: Optional[np.ndarray] = None,
        counterfactual: bool = False,
        remove_factors: Optional[List[str]] = None,
        matched_ref: Optional[Dict[str, np.ndarray]] = None,
        record_ref: bool = False,
    ) -> np.ndarray:
        """
        Generate synthetic price trajectories for a given scenario.

        Parameters
        ----------
        n_steps : int
            Number of trading days.
        initial_prices : ndarray, shape (M,)
            Starting OHLCV values.  Default all 100.
        active_factors : dict | None
            ``{factor_name: intensity}``.
        n_samples : int
            Parallel trajectories.
        noise_scale : float
            Scale of residual noise (1.0 = calibrated).
        seed : int | None
            Reproducibility seed.
        real_log_returns : ndarray, shape (T, M) or None
            Real log-returns for calibration and jump estimation.
        real_volume : ndarray, shape (T,) or None
            Real volume series for volume model calibration.
        historical_presence : ndarray, shape (T_hist, K) or None
            Daily factor presence from extraction phase.
        real_prices : ndarray, shape (T, M) or None
            Real price matrix for microstructure calibration.
        real_dates : ndarray or None
            Real trading dates (for day-of-week volume effects).

        Returns
        -------
        ndarray, shape (n_samples, n_steps, M)
        """
        if seed is not None:
            np.random.seed(seed)

        if initial_prices is None:
            initial_prices = np.full(self.M, 100.0)

        P = self._price_features      # typically 4 (OHLC)
        daily_scale = 1.0 if self.mode == "corrected" else 1.0 / self.window_days
        close_col = min(3, P - 1)     # Close column index

        # ── (#4) estimate t-Student df from real data ───────────────
        t_df = 30.0

        # ── (#7) estimate jump parameters ───────────────────────────
        jump_lam, jump_mu, jump_sig = 0.0, 0.0, 0.0
        if real_log_returns is not None and len(real_log_returns) > 30:
            jump_lam, jump_mu, jump_sig = _estimate_jump_params(
                real_log_returns[:, :P]
            )

        if jump_lam <= 0 or jump_sig <= 0:
            if real_log_returns is not None and len(real_log_returns) > 30:
                t_df = _estimate_t_df(real_log_returns[:, :P])
                # STAT-7: clamp ν ≥ 5 (ν < 4 has infinite kurtosis)
                t_df = max(t_df, 5.0)

        # ── 1. factor schedule (#2 — informed) ─────────────────────
        schedule = self._build_schedule(
            n_steps, active_factors, n_samples,
            historical_presence=historical_presence,
        )
        # Matched counterfactual: the schedule is drawn from the factual
        # presence (same random draws), then the removed factors are fully
        # switched off. Zeroing presence alone leaves them firing at the
        # 0.5 * base_prob floor of _build_schedule.
        if remove_factors:
            names = list(self.im.factor_names)
            unknown = [n for n in remove_factors if n not in names]
            if unknown:
                # a misspelt name would otherwise be a silent no-op removal
                raise KeyError(f"remove_factors: unknown factor(s) {unknown}")
            idx = [names.index(n) for n in remove_factors]
            if idx:
                schedule[:, :, idx] = 0.0

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # F2: From here, evolve only CLOSE.  O/H/L via microstructure.
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        if self.mode == "corrected":
            # ── 2'. linear drift only: act * b_k (daily units) ──────
            base = np.einsum(
                "stk,k->st", schedule, self.im.base_impact[:, close_col],
            ) * daily_scale
        else:
            # ── 2. non-linear factor impacts (Close only) ──────────────
            base = self._apply_nonlinear_impact(schedule, close_col, daily_scale)

            # ── 3. temporal profiles via vectorised convolution ────────
            base = self._apply_temporal_profiles(
                schedule, base, close_col, daily_scale,
            )

            # ── 4. non-linear interactions (Close only) ────────────────
            inter = self._apply_interactions(schedule, close_col, daily_scale)
            base += inter

        # ── 5. background volatility + factor noise ────────────────
        if real_log_returns is not None and len(real_log_returns) > 30:
            real_close_std = float(real_log_returns[:, close_col].std())
        else:
            real_close_std = 0.02

        noise = np.zeros_like(base)
        # corrected mode: impact_std is the network's predictive return
        # std (~1 sigma), not effect uncertainty; GARCH supplies variance.
        for k in (range(self.K) if self.mode == "legacy" else ()):
            active = schedule[:, :, k]
            std = self.im.impact_std[k, close_col] * noise_scale * daily_scale
            if std < 1e-10:
                continue
            fn = np.random.normal(0, std, (n_samples, n_steps))
            noise += active * fn

        signal = base + noise
        signal_var = np.var(signal, axis=1, keepdims=True).clip(1e-12)
        target_var = real_close_std ** 2
        # STAT-6: empirical variance decomposition instead of magic 0.85.
        # bg_var = target_var - signal_var, floored at 15 % of target_var
        # so background noise never collapses even when factors dominate.
        bg_var = np.clip(target_var - signal_var, target_var * 0.15, None)
        if matched_ref is not None:
            # keep the factual run's noise level: the difference between
            # the two runs is then the removed factors' drift only
            bg_var = matched_ref["bg_var"].copy()
        bg_std = np.sqrt(bg_var)

        if t_df < 25:
            scale_t = bg_std * np.sqrt((t_df - 2) / t_df) if t_df > 2 else bg_std
            bg_noise = np.random.standard_t(t_df, base.shape) * scale_t
        else:
            bg_noise = np.random.normal(0, 1, base.shape) * bg_std

        # R1: GJR-GARCH(1,1) conditional volatility ─────────────────
        #   σ²_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·𝟙(ε<0) + β·σ²_{t-1}
        #   Replaces i.i.d. bg_noise with time-varying volatility to
        #   reproduce volatility-clustering AND leverage effect.
        #   Uses skewed-t innovations for realistic asymmetry.
        if real_log_returns is not None and len(real_log_returns) > 60:
            g_omega, g_alpha, g_gamma, g_beta = _fit_gjr_garch_params(
                real_log_returns[:, close_col],
            )
            persistence = g_alpha + g_gamma / 2.0 + g_beta
            if persistence >= 0.999:
                g_beta = max(0.5, 0.999 - g_alpha - g_gamma / 2.0)
                persistence = g_alpha + g_gamma / 2.0 + g_beta

            # Estimate skewed-t parameters for innovations
            skew_df, skew_param = _estimate_skewed_t_params(
                real_log_returns[:, close_col],
            )

            # Set per-sample ω so unconditional variance = bg_var_s
            target_v = bg_var.ravel()  # (S,)
            omega_s = target_v * (1.0 - persistence)  # (S,)
            sigma2 = target_v.copy()               # start at unc. var

            garch_noise = np.zeros((n_samples, n_steps))
            # Store sigma2 trajectory for regime calibration to use
            sigma2_path = np.zeros((n_samples, n_steps))
            prev_eps = np.zeros(n_samples)  # for leverage indicator

            for t in range(n_steps):
                sigma2_path[:, t] = sigma2
                z = _sample_skewed_t(skew_df, skew_param, n_samples)
                eps = z * np.sqrt(sigma2)
                garch_noise[:, t] = eps
                # GJR asymmetric update: γ·ε² only when ε < 0
                leverage_ind = (prev_eps < 0).astype(np.float64)
                sigma2 = (
                    omega_s
                    + g_alpha * eps ** 2
                    + g_gamma * eps ** 2 * leverage_ind
                    + g_beta * sigma2
                )
                sigma2 = np.clip(sigma2, 1e-10, target_v * 10)
                prev_eps = eps

            bg_noise = garch_noise
            # Store sigma2 path for regime calibration to preserve
            self._last_sigma2_path = sigma2_path

        close_ret = signal + bg_noise  # (S, T)

        # ── 6. jump-diffusion (R5: regime-dependent intensity) ──
        if jump_lam > 0 and jump_sig > 0:
            # R5: scale jump intensity per regime (×3–5 in crises)
            lam_schedule = np.full(n_steps, jump_lam)
            if real_log_returns is not None and len(real_log_returns) > 63:
                regimes = _detect_regimes(real_log_returns[:, close_col])
                T_real = len(real_log_returns)
                global_std = float(real_log_returns[:, close_col].std())
                for r_s, r_e, r_mu, r_std in regimes:
                    s_s = max(0, min(int(round(r_s / T_real * n_steps)), n_steps - 1))
                    s_e = max(s_s + 1, min(int(round(r_e / T_real * n_steps)), n_steps))
                    is_crisis = (r_std > global_std * 1.3) or (r_mu < -0.0002)
                    if is_crisis:
                        mult = 3.0 + 2.0 * max(0, r_std / global_std - 1.0)
                        mult = float(np.clip(mult, 3.0, 5.0))
                        lam_schedule[s_s:s_e] = jump_lam * mult

            lam_grid = np.broadcast_to(
                lam_schedule[None, :], (n_samples, n_steps),
            ).copy()
            jump_mask = np.random.binomial(1, lam_grid).astype(np.float64)
            jump_sizes = np.random.normal(
                jump_mu, jump_sig, (n_samples, n_steps)
            )
            close_ret += jump_mask * jump_sizes

        # ── 7. regime-switching calibration (F1) ──────────────────
        # In counterfactual mode, skip drift calibration so that
        # factor removal actually affects the generated trajectory.
        # In matched mode, replay the factual run's corrections instead of
        # recomputing them, so the factual path keeps reproducing history
        # and the counterfactual differs only by the removed factors.
        pre_calib = close_ret.copy() if record_ref else None
        if matched_ref is not None:
            close_ret = close_ret + matched_ref["calib_delta"]
            counterfactual = True   # skip both calibration blocks below
        if (not counterfactual
                and real_log_returns is not None
                and len(real_log_returns) > 1):
            close_ret = self._regime_calibrate(
                close_ret, real_log_returns[:, close_col], blend=0.6,
            )

        # ── R2: continuous drift correction (quarterly) ────────────
        #   Every ~63 days, compare cumulative synthetic drift to
        #   real and apply a correction with exponential decay
        #   (STAT-8: avoids uniform shift that injects autocorrelation).
        if (not counterfactual
                and real_log_returns is not None
                and len(real_log_returns) > 63):
            real_cum = np.cumsum(real_log_returns[:, close_col])
            correction_period = 63
            half_life = 21  # days — most correction applied early
            _alpha = 1.0 - np.exp(-np.log(2.0) / half_life)
            for t_chk in range(correction_period, n_steps, correction_period):
                t_real = min(
                    int(round(t_chk / n_steps * len(real_cum))),
                    len(real_cum) - 1,
                )
                target_cum = real_cum[t_real]
                synth_cum = close_ret[:, :t_chk].sum(axis=1)  # (S,)
                gap = target_cum - synth_cum
                remaining = min(correction_period, n_steps - t_chk)
                # STAT-8: exp-decay weights instead of uniform
                weights = _alpha * (1.0 - _alpha) ** np.arange(remaining)
                weights = weights / weights.sum()  # normalise
                close_ret[:, t_chk:t_chk + remaining] += (
                    gap[:, None] * 0.5 * weights[None, :]
                )

        if record_ref:
            self._last_ref = {"bg_var": np.asarray(bg_var).copy(),
                              "calib_delta": close_ret - pre_calib,
                              # factual activity schedule (S, T, K): defines the
                              # active days of the attribution check
                              "schedule": np.asarray(schedule).copy()}

        # ── 8. clamp daily log-returns ─────────────────────────────
        close_ret = np.clip(close_ret, -0.20, 0.20)

        # ── 9. Close prices ────────────────────────────────────────
        close_prices = np.zeros((n_samples, n_steps))
        close_prices[:, 0] = initial_prices[close_col] * np.exp(close_ret[:, 0])
        for t in range(1, n_steps):
            close_prices[:, t] = close_prices[:, t - 1] * np.exp(close_ret[:, t])

        # ── 10. OHLC via microstructure (F2) ───────────────────────
        micro = {"gap_std": 0.005, "high_std": 0.008, "low_std": 0.008}
        if real_prices is not None and real_prices.shape[1] >= 4:
            micro = _estimate_microstructure_params(real_prices)

        prices = np.zeros((n_samples, n_steps, self.M))
        prices[:, :, close_col] = close_prices

        # Open = C_{t-1} × exp(gap_noise)
        gap_noise = np.random.normal(0, micro["gap_std"],
                                     (n_samples, n_steps))
        prices[:, 0, 0] = initial_prices[0] * np.exp(gap_noise[:, 0])
        for t in range(1, n_steps):
            prices[:, t, 0] = close_prices[:, t - 1] * np.exp(gap_noise[:, t])

        # High = max(O, C) × exp(|ε_h|)
        if P >= 2:
            max_oc = np.maximum(prices[:, :, 0], close_prices)
            high_exc = np.abs(np.random.normal(
                0, micro["high_std"], (n_samples, n_steps)))
            prices[:, :, 1] = max_oc * np.exp(high_exc)

        # Low = min(O, C) × exp(-|ε_l|)
        if P >= 3:
            min_oc = np.minimum(prices[:, :, 0], close_prices)
            low_exc = np.abs(np.random.normal(
                0, micro["low_std"], (n_samples, n_steps)))
            prices[:, :, 2] = min_oc * np.exp(-low_exc)

        # ── 11. Volume AR(1) + DOW (F3) ───────────────────────────
        if self.M >= 5:
            dow = None
            if real_dates is not None:
                try:
                    dt = pd.to_datetime(real_dates[:n_steps])
                    dow = dt.dayofweek.values
                except Exception:
                    pass

            prices[:, :, 4] = self._generate_volume(
                log_returns=close_ret,
                n_samples=n_samples,
                n_steps=n_steps,
                real_volume=real_volume,
                initial_volume=(
                    initial_prices[4] if len(initial_prices) > 4 else 1e8
                ),
                day_of_week=dow,
            )

        prices = self._enforce_ohlcv(prices)
        return prices

    # ── regime-switching calibration (F1 — drift-only) ────────────────

    @staticmethod
    def _regime_calibrate(
        close_ret: np.ndarray,
        real_close_lr: np.ndarray,
        blend: float = 0.6,
    ) -> np.ndarray:
        """Per-regime **drift-only** calibration.

        CRITICAL DESIGN DECISION: This function calibrates ONLY the
        drift (mean) per regime segment.  It does NOT rescale the
        volatility per segment, because that would destroy the GARCH
        temporal memory (σ²_t depends on σ²_{t-1} and ε²_{t-1}).

        The GJR-GARCH loop already produces the correct volatility
        dynamics (clustering + leverage).  Segment-wise std rescaling
        breaks the conditional heteroskedasticity by treating each
        regime as i.i.d., which is the exact opposite of what GARCH
        models.  Therefore volatility calibration is handled entirely
        by the GARCH parameterisation (ω scaled to match unconditional
        variance), not by post-hoc rescaling.

        Parameters
        ----------
        close_ret : (S, T) — synthetic Close log-returns
        real_close_lr : (T_real,) — real Close log-returns
        blend : float in [0, 1] — calibration strength (for drift)
        """
        S, T = close_ret.shape
        T_real = len(real_close_lr)

        regimes = _detect_regimes(real_close_lr)

        result = close_ret.copy()

        for r_start, r_end, r_mu, r_std in regimes:
            # Map regime boundaries from real → synthetic timeline
            s_start = int(round(r_start / T_real * T))
            s_end = int(round(r_end / T_real * T))
            s_start = max(0, min(s_start, T - 1))
            s_end = max(s_start + 1, min(s_end, T))

            seg = result[:, s_start:s_end]
            seg_len = seg.shape[1]
            if seg_len < 2:
                continue

            synth_mu = seg.mean(axis=1, keepdims=True)

            # Drift: 90% correction (100% for bearish regimes — R3)
            corr_strength = 1.0 if r_mu < 0 else 0.9
            drift_shift = blend * corr_strength * (r_mu - synth_mu)
            seg = seg + drift_shift

            # R3: in bearish regimes, guarantee negative mean
            if r_mu < 0:
                seg_mu_post = seg.mean(axis=1, keepdims=True)
                still_positive = seg_mu_post > 0
                if still_positive.any():
                    seg = seg - seg_mu_post * still_positive + r_mu * still_positive

            # NOTE: No per-segment volatility rescaling here.
            # GARCH temporal structure is preserved intact.

            result[:, s_start:s_end] = seg

        return result

    # ── volume model (F3 — AR(1) + day-of-week) ────────────────────

    @staticmethod
    def _generate_volume(
        log_returns: np.ndarray,
        n_samples: int,
        n_steps: int,
        real_volume: Optional[np.ndarray] = None,
        initial_volume: float = 1e8,
        day_of_week: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate volume via AR(1) + |returns| + day-of-week.

        Model::

            log(V_t) = γ·log(V_{t-1}) + (1-γ)·α + β·|r_t| + DOW_d + ε_t

        where γ is the AR(1) persistence, DOW_d captures intra-week
        seasonality, and ε_t is calibrated residual noise.
        """
        # Handle both 2D and 3D log_returns
        if log_returns.ndim == 3:
            col = min(3, log_returns.shape[2] - 1)
            abs_ret = np.abs(log_returns[:, :, col])
        else:
            abs_ret = np.abs(log_returns)  # (S, T)

        if real_volume is not None and len(real_volume) > 10:
            log_vol = np.log(np.clip(real_volume, 1.0, None))
            vol_mu = float(log_vol.mean())
            vol_std = float(log_vol.std())

            # AR(1) coefficient γ
            if len(log_vol) > 2:
                gamma = float(np.clip(
                    np.corrcoef(log_vol[:-1], log_vol[1:])[0, 1],
                    0.3, 0.98,
                ))
            else:
                gamma = 0.85

            # Day-of-week effects
            dow_effects = np.zeros(5)
            if day_of_week is not None and len(real_volume) > 20:
                n_common = min(len(real_volume), len(day_of_week))
                for d in range(5):
                    mask = day_of_week[:n_common] == d
                    if mask.sum() > 5:
                        dow_effects[d] = float(
                            log_vol[:n_common][mask].mean() - vol_mu
                        )

            # Residual noise std (after AR(1))
            if len(log_vol) > 2:
                ar_residual = (
                    log_vol[1:]
                    - gamma * log_vol[:-1]
                    - (1 - gamma) * vol_mu
                )
                noise_std = float(np.clip(ar_residual.std(), 0.01, vol_std))
            else:
                noise_std = vol_std * 0.3

            # DISC-1: calibrate β from data instead of hardcoded 5.0.
            # Regress AR(1) residuals on |returns| to get data-driven β.
            if log_returns.ndim >= 2:
                col = min(3, log_returns.shape[-1] - 1) if log_returns.ndim == 3 else 0
                _ret_for_beta = np.abs(log_returns[:, col] if log_returns.ndim == 2 else log_returns[:, :, col].mean(axis=0))
            else:
                _ret_for_beta = np.abs(log_returns)
            if len(log_vol) > 2:
                _n_common = min(len(ar_residual), len(_ret_for_beta))
                if _n_common > 10:
                    _ret_slice = _ret_for_beta[:_n_common]
                    _resid_slice = ar_residual[:_n_common]
                    _r_denom = np.dot(_ret_slice, _ret_slice)
                    if _r_denom > 1e-15:
                        beta = float(np.clip(
                            np.dot(_resid_slice, _ret_slice) / _r_denom,
                            0.5, 20.0
                        ))
                    else:
                        beta = 5.0
                else:
                    beta = 5.0
            else:
                beta = 5.0

            # Generate AR(1) process
            volume = np.zeros((n_samples, n_steps))
            state = np.full(n_samples, vol_mu)

            for t in range(n_steps):
                dow_t = 0.0
                if day_of_week is not None and t < len(day_of_week):
                    dow_t = dow_effects[int(day_of_week[t]) % 5]

                innovation = beta * abs_ret[:, t] + dow_t
                eps = np.random.normal(0, noise_std, n_samples)
                state = (
                    gamma * state
                    + (1 - gamma) * vol_mu
                    + innovation
                    + eps
                )
                volume[:, t] = np.exp(state)

            v_min = max(1.0, real_volume.min() * 0.01)
            v_max = real_volume.max() * 10.0
            volume = np.clip(volume, v_min, v_max)

            # R4: rescale volume so synth mean ≈ real mean
            real_mean = float(real_volume.mean())
            synth_mean = volume.mean(axis=1, keepdims=True).clip(1e-8)
            volume = volume * (real_mean / synth_mean)
        else:
            log_v0 = np.log(max(initial_volume, 1.0))
            noise = np.random.normal(0, 0.5, (n_samples, n_steps))
            volume = np.exp(log_v0 + noise)

        return volume

    # ── multi-scenario ──────────────────────────────────────────────

    def generate_from_scenario_set(
        self,
        scenario_set: ScenarioSet,
        n_total_samples: int = 100,
        n_steps: int = 252,
        initial_prices: Optional[np.ndarray] = None,
        noise_scale: float = 1.0,
        real_log_returns: Optional[np.ndarray] = None,
        real_volume: Optional[np.ndarray] = None,
        historical_presence: Optional[np.ndarray] = None,
        real_prices: Optional[np.ndarray] = None,
        real_dates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate proportional samples from a ScenarioSet."""
        all_prices: List[np.ndarray] = []

        for sc in scenario_set.scenarios:
            n = max(1, round(sc.scenario_prob * n_total_samples))
            prices = self.generate_scenario(
                n_steps=n_steps,
                initial_prices=initial_prices,
                active_factors=sc.factor_probs,
                n_samples=n,
                noise_scale=noise_scale,
                real_log_returns=real_log_returns,
                real_volume=real_volume,
                historical_presence=historical_presence,
                real_prices=real_prices,
                real_dates=real_dates,
            )
            all_prices.append(prices)

        return np.concatenate(all_prices, axis=0)[:n_total_samples]

    # ── FinRL format ────────────────────────────────────────────────

    def generate_finrl_df(
        self,
        ticker: str,
        n_steps: int = 252,
        initial_prices: Optional[np.ndarray] = None,
        active_factors: Optional[Dict[str, float]] = None,
        n_samples: int = 5,
        real_dates: Optional[np.ndarray] = None,
        start_date: str = "2020-01-01",
        real_log_returns: Optional[np.ndarray] = None,
        real_volume: Optional[np.ndarray] = None,
        historical_presence: Optional[np.ndarray] = None,
        real_prices: Optional[np.ndarray] = None,
        counterfactual: bool = False,
        remove_factors: Optional[List[str]] = None,
        matched_ref: Optional[Dict[str, np.ndarray]] = None,
        record_ref: bool = False,
    ) -> pd.DataFrame:
        """Generate in FinRL DataFrame format.

        (#1) If ``real_dates`` is provided, uses those exact dates
        instead of generating via ``pd.bdate_range``.
        """
        prices = self.generate_scenario(
            n_steps=n_steps,
            initial_prices=initial_prices,
            active_factors=active_factors,
            n_samples=n_samples,
            real_log_returns=real_log_returns,
            real_volume=real_volume,
            historical_presence=historical_presence,
            real_prices=real_prices,
            real_dates=real_dates,
            counterfactual=counterfactual,
            remove_factors=remove_factors,
            matched_ref=matched_ref,
            record_ref=record_ref,
        )

        # (#1) Use real dates when available
        if real_dates is not None:
            dates = pd.to_datetime(real_dates[:n_steps])
        else:
            dates = pd.bdate_range(start=start_date, periods=n_steps)

        dfs: List[pd.DataFrame] = []
        for s in range(prices.shape[0]):
            sdf = pd.DataFrame(
                prices[s],
                columns=self.im.feature_names[: prices.shape[2]],
            )
            sdf["date"] = dates.strftime("%Y-%m-%d")
            sdf["tic"] = ticker
            sdf["sample"] = s
            dfs.append(sdf)

        return pd.concat(dfs, ignore_index=True)

    # ── informed schedule (#2) ──────────────────────────────────────

    def _build_schedule(
        self,
        n_steps: int,
        active_factors: Optional[Dict[str, float]],
        n_samples: int,
        historical_presence: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build ``(n_samples, n_steps, K)`` activation schedule.

        (#2) If ``historical_presence`` is given, the schedule is derived
        from actual factor timing with stochastic perturbation:
            - The real daily activation is smoothed and used as a
              time-varying probability.
            - Each sample draws from that probability with jitter,
              so events happen "around" the real time (±days).
        Fallback: original Markov schedule when no history is provided.
        """
        schedule = np.zeros((n_samples, n_steps, self.K))

        for k, name in enumerate(self.im.factor_names):
            # Explicit scenario overrides
            if active_factors and name in active_factors:
                schedule[:, :, k] = active_factors[name]
                continue

            # (#2) Informed schedule from history
            if historical_presence is not None and k < historical_presence.shape[1]:
                hist = historical_presence[:, k].astype(np.float64)
                if self.mode == "corrected":
                    hist = (hist > 0).astype(np.float64)   # presence indicator

                # Resample history to match n_steps if sizes differ
                if len(hist) != n_steps:
                    x_old = np.linspace(0, 1, len(hist))
                    x_new = np.linspace(0, 1, n_steps)
                    hist = np.interp(x_new, x_old, hist)

                # Smooth to get time-varying activation probability
                kernel = min(21, max(1, n_steps // 5))
                if kernel > 1:
                    smoothed = np.convolve(
                        hist, np.ones(kernel) / kernel, mode="same"
                    )
                else:
                    smoothed = hist.copy()

                # Clamp to [base_prob * 0.5, 1.0]
                base_prob = self.im.occurrence_prob[k]
                floor = 0.0 if self.mode == "corrected" else base_prob * 0.5
                smoothed = np.clip(smoothed, floor, 1.0)
                pad = 0.0 if self.mode == "corrected" else base_prob

                for s in range(n_samples):
                    # BUG-C2: shift activation ± random days without
                    # circular wrap — pad edges with base_prob instead
                    jitter = np.random.randint(-10, 11)
                    if jitter > 0:
                        shifted = np.concatenate([
                            np.full(jitter, pad),
                            smoothed[:n_steps - jitter],
                        ])
                    elif jitter < 0:
                        shifted = np.concatenate([
                            smoothed[-jitter:],
                            np.full(-jitter, pad),
                        ])
                    else:
                        shifted = smoothed.copy()
                    # Draw from time-varying Bernoulli
                    draws = np.random.random(n_steps) < shifted
                    schedule[s, :, k] = draws.astype(np.float64)
                continue

            # Fallback: original Markov process
            prob = self.im.occurrence_prob[k]
            for s in range(n_samples):
                state = np.random.random() < prob
                persistence = 0.95
                for t in range(n_steps):
                    if state:
                        if np.random.random() < persistence:
                            schedule[s, t, k] = 1.0
                        else:
                            state = False
                    else:
                        if np.random.random() < prob * (1 - persistence):
                            state = True
                            schedule[s, t, k] = 1.0

        return schedule

    # ── OHLCV constraints ───────────────────────────────────────────

    def _enforce_ohlcv(self, prices: np.ndarray) -> np.ndarray:
        """Enforce: High >= max(O,C), Low <= min(O,C), Vol >= 0."""
        if self.M >= 4:
            o = prices[:, :, 0]
            c = prices[:, :, 3]
            prices[:, :, 1] = np.maximum(prices[:, :, 1], np.maximum(o, c))
            prices[:, :, 2] = np.minimum(prices[:, :, 2], np.minimum(o, c))
        if self.M >= 5:
            prices[:, :, 4] = np.clip(prices[:, :, 4], 0, None)
        return prices
