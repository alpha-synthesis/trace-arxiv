# -*- coding: utf-8 -*-
"""
Publication-Quality Evaluation Report.

Generates two self-contained evaluation folders ready for academic
publication:

1. ``llm_quality/``  — compares factor extraction between different
   LLM models (e.g. Gemini vs Gemma) by loading cached factor JSONs.
2. ``generation_quality/`` — comprehensive assessment of synthetic data
   fidelity including an LSTM discriminator (real-vs-synthetic classifier).

Designed to work **without re-running** LLM inference — reads cached
``WindowAnalysis`` JSONs directly from factor cache directories.

Usage::

    from sde_causal_generator.quality_report import QualityReportGenerator
    gen = QualityReportGenerator(output_dir="results/evaluation")
    gen.run(
        real_df=real_df,
        synth_df=synth_df,
        factor_cache_dirs={"gemini": "cache/back_up/llm_factors_gemini_flash_lite",
                           "gemma":  "cache/causal_sde/AAPL/llm_factors"},
    )
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _load_factor_cache(cache_dir: str) -> List[Dict]:
    """Load all WindowAnalysis JSONs from a cache directory."""
    analyses = []
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"  ⚠ Cache directory not found: {cache_dir}")
        return analyses
    for fp in sorted(cache_path.glob("*.json")):
        try:
            with open(fp) as f:
                analyses.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return analyses


def _extract_factor_info(analyses: List[Dict]) -> pd.DataFrame:
    """Flatten factor data from analyses into a DataFrame."""
    rows = []
    for a in analyses:
        start = a.get("start_date", "")
        end = a.get("end_date", "")
        wtype = a.get("window_type", "unknown")
        for f in a.get("factors", []):
            rows.append({
                "window_start": start,
                "window_end": end,
                "window_type": wtype,
                "name": f.get("name", ""),
                "category": f.get("category", "unknown"),
                "direction": float(f.get("direction", 0)),
                "magnitude": float(f.get("magnitude", 0)),
                "persistence": f.get("persistence", "medium"),
                "description": f.get("description", ""),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════
# Part 1: LLM Quality Comparison
# ══════════════════════════════════════════════════════════════════════

class LLMQualityComparison:
    """Compare factor extraction quality across LLM models."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(
        self,
        factor_cache_dirs: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Compare multiple LLM factor caches.

        Parameters
        ----------
        factor_cache_dirs : dict
            ``{model_label: cache_dir_path}``

        Returns
        -------
        report : dict
            Summary metrics for each model and pairwise comparisons.
        """
        print("\n" + "─" * 60)
        print("  LLM QUALITY COMPARISON")
        print("─" * 60)

        # Load all caches
        model_data: Dict[str, Tuple[List[Dict], pd.DataFrame]] = {}
        for label, path in factor_cache_dirs.items():
            analyses = _load_factor_cache(path)
            if not analyses:
                print(f"  ⚠ No data for '{label}' — skipping")
                continue
            df_factors = _extract_factor_info(analyses)
            model_data[label] = (analyses, df_factors)
            print(f"  Loaded '{label}': {len(analyses)} windows, "
                  f"{len(df_factors)} factor entries")

        if len(model_data) < 1:
            print("  ⚠ Not enough data for comparison")
            return {}

        report: Dict[str, Any] = {}

        # Per-model metrics
        for label, (analyses, df_f) in model_data.items():
            metrics = self._per_model_metrics(label, analyses, df_f)
            report[label] = metrics

        # Pairwise comparisons
        labels = list(model_data.keys())
        if len(labels) >= 2:
            pairwise = {}
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    la, lb = labels[i], labels[j]
                    pw = self._pairwise_comparison(
                        la, model_data[la],
                        lb, model_data[lb],
                    )
                    pairwise[f"{la}_vs_{lb}"] = pw
            report["pairwise"] = pairwise

        # Generate plots
        self._plot_category_distribution(model_data)
        self._plot_magnitude_distribution(model_data)
        self._plot_temporal_coverage(model_data)
        self._plot_direction_distribution(model_data)
        self._plot_genericness(model_data)

        # Save CSV and JSON
        self._save_summary_csv(report, model_data)
        with open(os.path.join(self.output_dir, "llm_comparison_report.json"), "w") as f:
            json.dump(report, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))

        print(f"\n  ✓ LLM quality report saved to {self.output_dir}/")
        return report

    # ── per-model ───────────────────────────────────────────────────

    def _per_model_metrics(
        self, label: str, analyses: List[Dict], df_f: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Compute per-model quality metrics."""
        unique_factors = df_f["name"].nunique()
        total_entries = len(df_f)
        n_windows = len(analyses)

        # Category distribution
        cat_counts = df_f["category"].value_counts().to_dict()

        # Direction balance
        n_bull = int((df_f["direction"] > 0.15).sum())
        n_bear = int((df_f["direction"] < -0.15).sum())
        n_neutral = int(total_entries - n_bull - n_bear)

        # Magnitude stats
        mag_mean = float(df_f["magnitude"].mean()) if len(df_f) > 0 else 0
        mag_std = float(df_f["magnitude"].std()) if len(df_f) > 0 else 0
        mag_median = float(df_f["magnitude"].median()) if len(df_f) > 0 else 0

        # Genericness: count how many factor names appear >10 times
        name_counts = df_f["name"].value_counts()
        n_generic = int((name_counts > 10).sum())
        top_repeat = name_counts.head(10).to_dict()

        # Factors per window
        factors_per_window = df_f.groupby(
            ["window_start", "window_end"]
        ).size()
        fpw_mean = float(factors_per_window.mean())
        fpw_std = float(factors_per_window.std())

        # market_noise attribution
        noise_entries = df_f[
            df_f["name"].str.contains("noise|residual", case=False, na=False)
        ]
        noise_pct = len(noise_entries) / max(total_entries, 1) * 100

        metrics = {
            "n_windows": n_windows,
            "total_factor_entries": total_entries,
            "unique_factors": unique_factors,
            "factors_per_window_mean": round(fpw_mean, 2),
            "factors_per_window_std": round(fpw_std, 2),
            "category_distribution": cat_counts,
            "direction_balance": {
                "bullish": n_bull, "bearish": n_bear, "neutral": n_neutral
            },
            "magnitude_mean": round(mag_mean, 4),
            "magnitude_std": round(mag_std, 4),
            "magnitude_median": round(mag_median, 4),
            "n_generic_factors": n_generic,
            "top_10_repeated_factors": top_repeat,
            "noise_attribution_pct": round(noise_pct, 2),
        }

        print(f"\n  ── {label} ──")
        print(f"    Windows: {n_windows}, Unique factors: {unique_factors}")
        print(f"    Factors/window: {fpw_mean:.1f} ± {fpw_std:.1f}")
        print(f"    Direction: {n_bull} bull / {n_bear} bear / {n_neutral} neutral")
        print(f"    Magnitude: {mag_mean:.4f} ± {mag_std:.4f}")
        print(f"    Generic (>10 repeats): {n_generic}")
        print(f"    Noise attribution: {noise_pct:.1f}%")

        return metrics

    # ── pairwise ────────────────────────────────────────────────────

    def _pairwise_comparison(
        self,
        label_a: str, data_a: Tuple[List[Dict], pd.DataFrame],
        label_b: str, data_b: Tuple[List[Dict], pd.DataFrame],
    ) -> Dict[str, Any]:
        """Compare two models on shared windows."""
        analyses_a, df_a = data_a
        analyses_b, df_b = data_b

        # Build window → factor sets
        def _window_factors(analyses):
            wf = {}
            for a in analyses:
                key = (a["start_date"], a["end_date"])
                factors = set(f["name"] for f in a.get("factors", []))
                wf[key] = factors
            return wf

        wf_a = _window_factors(analyses_a)
        wf_b = _window_factors(analyses_b)

        # Shared windows
        shared_keys = set(wf_a.keys()) & set(wf_b.keys())
        n_shared = len(shared_keys)

        if n_shared == 0:
            print(f"    ⚠ No shared windows between {label_a} and {label_b}")
            return {"n_shared_windows": 0}

        # Jaccard per window
        jaccards = []
        dir_agreements = []
        for key in shared_keys:
            fa, fb = wf_a[key], wf_b[key]
            inter = len(fa & fb)
            union = len(fa | fb)
            jaccards.append(inter / union if union > 0 else 0)

        # Direction agreement for shared factors
        # Build dicts: window → factor_name → direction
        def _window_factor_dirs(analyses):
            wfd = {}
            for a in analyses:
                key = (a["start_date"], a["end_date"])
                wfd[key] = {
                    f["name"]: float(f.get("direction", 0))
                    for f in a.get("factors", [])
                }
            return wfd

        wfd_a = _window_factor_dirs(analyses_a)
        wfd_b = _window_factor_dirs(analyses_b)

        for key in shared_keys:
            da, db = wfd_a.get(key, {}), wfd_b.get(key, {})
            shared_factors = set(da.keys()) & set(db.keys())
            for fn in shared_factors:
                sign_a = 1 if da[fn] > 0 else (-1 if da[fn] < 0 else 0)
                sign_b = 1 if db[fn] > 0 else (-1 if db[fn] < 0 else 0)
                dir_agreements.append(1.0 if sign_a == sign_b else 0.0)

        avg_jaccard = float(np.mean(jaccards))
        avg_dir_agree = float(np.mean(dir_agreements)) if dir_agreements else 0.0

        # Unique factor overlap
        all_a = set(df_a["name"].unique())
        all_b = set(df_b["name"].unique())
        global_jaccard = len(all_a & all_b) / len(all_a | all_b) if (all_a | all_b) else 0

        result = {
            "n_shared_windows": n_shared,
            "avg_jaccard_per_window": round(avg_jaccard, 4),
            "global_factor_jaccard": round(global_jaccard, 4),
            "direction_agreement": round(avg_dir_agree, 4),
            "only_in_a": len(all_a - all_b),
            "only_in_b": len(all_b - all_a),
            "shared_factors": len(all_a & all_b),
        }

        print(f"\n  ── {label_a} vs {label_b} ──")
        print(f"    Shared windows: {n_shared}")
        print(f"    Jaccard (per window): {avg_jaccard:.4f}")
        print(f"    Jaccard (global factors): {global_jaccard:.4f}")
        print(f"    Direction agreement: {avg_dir_agree*100:.1f}%")
        print(f"    Factors only in {label_a}: {len(all_a - all_b)}")
        print(f"    Factors only in {label_b}: {len(all_b - all_a)}")
        print(f"    Shared factors: {len(all_a & all_b)}")

        return result

    # ── plots ───────────────────────────────────────────────────────

    def _plot_category_distribution(self, model_data):
        """Bar chart comparing category distributions across models."""
        labels = list(model_data.keys())
        all_cats = set()
        for _, (_, df_f) in model_data.items():
            all_cats |= set(df_f["category"].unique())
        cats = sorted(all_cats)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(cats))
        width = 0.8 / max(len(labels), 1)

        for i, label in enumerate(labels):
            _, df_f = model_data[label]
            counts = df_f["category"].value_counts()
            total = len(df_f)
            vals = [counts.get(c, 0) / total * 100 for c in cats]
            ax.bar(x + i * width, vals, width, label=label, alpha=0.8)

        ax.set_xlabel("Category", fontsize=12)
        ax.set_ylabel("Percentage (%)", fontsize=12)
        ax.set_title("Factor Category Distribution by Model", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width * (len(labels) - 1) / 2)
        ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        path = os.path.join(self.output_dir, "category_distribution.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _plot_magnitude_distribution(self, model_data):
        """Violin/box plot comparing magnitude distributions."""
        labels = list(model_data.keys())
        data = [model_data[l][1]["magnitude"].values for l in labels]

        fig, ax = plt.subplots(figsize=(10, 6))
        parts = ax.violinplot(data, showmeans=True, showmedians=True)
        colors = ["#1565C0", "#E65100", "#2E7D32", "#7B1FA2"]
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_alpha(0.6)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel("Factor Magnitude", fontsize=12)
        ax.set_title("Factor Magnitude Distribution by Model", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        path = os.path.join(self.output_dir, "magnitude_distribution.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _plot_temporal_coverage(self, model_data):
        """Heatmap showing factor density by year per model."""
        labels = list(model_data.keys())

        fig, axes = plt.subplots(len(labels), 1, figsize=(16, 4 * len(labels)),
                                  sharex=True)
        if len(labels) == 1:
            axes = [axes]

        for ax, label in zip(axes, labels):
            _, df_f = model_data[label]
            df_f = df_f.copy()
            df_f["year"] = pd.to_datetime(df_f["window_start"]).dt.year
            year_cats = df_f.groupby(["year", "category"]).size().unstack(fill_value=0)

            ax.imshow(year_cats.T.values, aspect="auto", cmap="YlOrRd",
                      interpolation="nearest")
            ax.set_yticks(range(len(year_cats.columns)))
            ax.set_yticklabels([c.replace("_", " ") for c in year_cats.columns], fontsize=8)
            ax.set_xticks(range(len(year_cats.index)))
            ax.set_xticklabels(year_cats.index, fontsize=8, rotation=45)
            ax.set_title(f"{label} — Factor Density by Year & Category", fontsize=11)

        fig.suptitle("Temporal Factor Coverage", fontsize=14, fontweight="bold", y=1.02)
        path = os.path.join(self.output_dir, "temporal_coverage.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _plot_direction_distribution(self, model_data):
        """Histogram of direction values per model."""
        labels = list(model_data.keys())
        fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 5),
                                  sharey=True)
        if len(labels) == 1:
            axes = [axes]

        for ax, label in zip(axes, labels):
            _, df_f = model_data[label]
            dirs = df_f["direction"].values
            ax.hist(dirs, bins=40, color="#1565C0", alpha=0.7, edgecolor="white")
            ax.axvline(0, color="red", linestyle="--", alpha=0.5)
            ax.set_xlabel("Direction", fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            ax.set_title(f"{label}", fontsize=12)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Factor Direction Distribution", fontsize=14, fontweight="bold")
        path = os.path.join(self.output_dir, "direction_distribution.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _plot_genericness(self, model_data):
        """Bar chart of top-20 most repeated factor names per model."""
        labels = list(model_data.keys())
        fig, axes = plt.subplots(1, len(labels),
                                  figsize=(8 * len(labels), 8))
        if len(labels) == 1:
            axes = [axes]

        for ax, label in zip(axes, labels):
            _, df_f = model_data[label]
            top20 = df_f["name"].value_counts().head(20)
            y_pos = np.arange(len(top20))
            ax.barh(y_pos, top20.values, color="#E65100", alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(
                [n[:40] for n in top20.index], fontsize=8
            )
            ax.invert_yaxis()
            ax.set_xlabel("Occurrences", fontsize=10)
            ax.set_title(f"{label} — Top 20 Repeated Factors", fontsize=11)
            ax.grid(True, alpha=0.3, axis="x")

        fig.suptitle("Factor Name Repetition (Genericness Check)",
                     fontsize=14, fontweight="bold")
        path = os.path.join(self.output_dir, "factor_genericness.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ── CSV export ──────────────────────────────────────────────────

    def _save_summary_csv(self, report, model_data):
        """Save per-window comparison CSV."""
        rows = []
        for label, (analyses, df_f) in model_data.items():
            for a in analyses:
                for f in a.get("factors", []):
                    rows.append({
                        "model": label,
                        "window_start": a["start_date"],
                        "window_end": a["end_date"],
                        "window_type": a["window_type"],
                        "price_change_pct": a["price_change_pct"],
                        "factor_name": f["name"],
                        "category": f.get("category", ""),
                        "direction": f.get("direction", 0),
                        "magnitude": f.get("magnitude", 0),
                        "persistence": f.get("persistence", ""),
                    })
        if rows:
            df = pd.DataFrame(rows)
            path = os.path.join(self.output_dir, "per_window_comparison.csv")
            df.to_csv(path, index=False)
            print(f"  ✓ Per-window CSV: {path} ({len(df)} rows)")

        # Summary CSV
        summary_rows = []
        for label in model_data:
            if label in report:
                row = {"model": label}
                for k, v in report[label].items():
                    if isinstance(v, (int, float)):
                        row[k] = v
                summary_rows.append(row)
        if summary_rows:
            df_s = pd.DataFrame(summary_rows)
            path = os.path.join(self.output_dir, "model_comparison_summary.csv")
            df_s.to_csv(path, index=False)


# ══════════════════════════════════════════════════════════════════════
# Part 2: Generation Quality — LSTM Discriminator
# ══════════════════════════════════════════════════════════════════════

class GenerationQualityReport:
    """
    Comprehensive synthetic data quality assessment.

    Includes an LSTM discriminator (Train-on-Synthetic, Test-on-Real
    paradigm from TimeGAN), plus statistical fidelity plots for
    publication.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        ticker: str = "AAPL",
        window_size: int = 20,
        disc_epochs: int = 100,
        disc_hidden: int = 64,
        disc_n_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Run full generation quality assessment.

        Parameters
        ----------
        real_df : pd.DataFrame
            Real OHLCV data with ``date, open, high, low, close, volume``.
        synth_df : pd.DataFrame
            Synthetic data. If multi-sample, uses sample 0.
        ticker : str
            For titles and filenames.
        window_size : int
            Sliding window length for discriminator input.
        disc_epochs : int
            Training epochs for LSTM discriminator.
        disc_hidden : int
            LSTM hidden dimension.
        disc_n_folds : int
            Cross-validation folds.

        Returns
        -------
        report : dict
        """
        print("\n" + "─" * 60)
        print("  GENERATION QUALITY REPORT")
        print("─" * 60)

        report: Dict[str, Any] = {}

        # Prepare data
        real_df = real_df.copy()
        real_df["date"] = pd.to_datetime(real_df["date"])
        real_df = real_df.sort_values("date")

        synth_eval = synth_df.copy()
        if "sample" in synth_eval.columns:
            synth_eval = synth_eval[synth_eval["sample"] == 0].copy()
        synth_eval["date"] = pd.to_datetime(synth_eval["date"])
        synth_eval = synth_eval.sort_values("date")

        feature_cols = [c for c in ["open", "high", "low", "close", "volume"]
                        if c in real_df.columns and c in synth_eval.columns]

        # 1. Statistical tests
        print("\n  Computing statistical fidelity metrics...")
        stat_report = self._statistical_tests(real_df, synth_eval, feature_cols)
        report["statistical_tests"] = stat_report
        self._save_stat_csv(stat_report)

        # 2. Stylized facts
        print("  Generating stylized facts plots...")
        self._plot_stylized_facts(real_df, synth_eval, ticker, feature_cols)

        # 3. Return distributions
        print("  Generating distribution comparison plots...")
        self._plot_return_distributions(real_df, synth_eval, ticker, feature_cols)

        # 4. LSTM Discriminator
        print(f"\n  Training LSTM discriminator ({disc_n_folds}-fold CV)...")
        disc_report = self._lstm_discriminator(
            real_df, synth_eval, feature_cols,
            window_size=window_size,
            epochs=disc_epochs,
            hidden_dim=disc_hidden,
            n_folds=disc_n_folds,
        )
        report["discriminator"] = disc_report

        # 5. Save report
        with open(os.path.join(self.output_dir, "generation_quality_report.json"), "w") as f:
            json.dump(report, f, indent=2,
                      default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))

        # 6. Summary CSV
        self._save_discriminator_csv(disc_report)

        print(f"\n  ✓ Generation quality report saved to {self.output_dir}/")
        return report

    # ── Statistical Tests ───────────────────────────────────────────

    def _statistical_tests(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Dict[str, Any]:
        """Compute KS, Wasserstein, moments for each feature."""
        from scipy import stats as sp_stats

        results = {}
        for col in feature_cols:
            if col not in real_df.columns or col not in synth_df.columns:
                continue

            real_vals = real_df[col].values.astype(float)
            synth_vals = synth_df[col].values.astype(float)

            # Log returns
            real_lr = np.diff(np.log(np.clip(real_vals, 1e-8, None)))
            synth_lr = np.diff(np.log(np.clip(synth_vals, 1e-8, None)))
            real_lr = real_lr[np.isfinite(real_lr)]
            synth_lr = synth_lr[np.isfinite(synth_lr)]

            if len(real_lr) < 5 or len(synth_lr) < 5:
                continue

            ks_stat, ks_pval = sp_stats.ks_2samp(real_lr, synth_lr)
            wass = float(sp_stats.wasserstein_distance(real_lr, synth_lr))

            results[col] = {
                "ks_statistic": round(float(ks_stat), 6),
                "ks_pvalue": round(float(ks_pval), 6),
                "wasserstein": round(wass, 6),
                "real_mean": round(float(np.mean(real_lr)), 6),
                "synth_mean": round(float(np.mean(synth_lr)), 6),
                "real_std": round(float(np.std(real_lr)), 6),
                "synth_std": round(float(np.std(synth_lr)), 6),
                "real_skew": round(float(sp_stats.skew(real_lr)), 4),
                "synth_skew": round(float(sp_stats.skew(synth_lr)), 4),
                "real_kurtosis": round(float(sp_stats.kurtosis(real_lr)), 4),
                "synth_kurtosis": round(float(sp_stats.kurtosis(synth_lr)), 4),
            }

        return results

    def _save_stat_csv(self, stat_report: Dict):
        """Save statistical tests as CSV."""
        rows = []
        for col, metrics in stat_report.items():
            row = {"feature": col}
            row.update(metrics)
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            path = os.path.join(self.output_dir, "statistical_tests.csv")
            df.to_csv(path, index=False)

    # ── Stylized Facts ──────────────────────────────────────────────

    def _plot_stylized_facts(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        ticker: str,
        feature_cols: List[str],
    ):
        """Generate stylized-facts comparison plot."""
        col = "close" if "close" in feature_cols else feature_cols[0]
        real_prices = real_df[col].values.astype(float)
        synth_prices = synth_df[col].values.astype(float)

        real_lr = np.diff(np.log(np.clip(real_prices, 1e-8, None)))
        synth_lr = np.diff(np.log(np.clip(synth_prices, 1e-8, None)))
        real_lr = real_lr[np.isfinite(real_lr)]
        synth_lr = synth_lr[np.isfinite(synth_lr)]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. QQ plot
        ax = axes[0, 0]
        from scipy import stats as sp_stats
        real_sorted = np.sort(real_lr)
        synth_sorted = np.sort(synth_lr)
        n = min(len(real_sorted), len(synth_sorted))
        q_real = np.quantile(real_sorted, np.linspace(0, 1, n))
        q_synth = np.quantile(synth_sorted, np.linspace(0, 1, n))
        ax.scatter(q_real, q_synth, s=4, alpha=0.5, color="#1565C0")
        lims = [min(q_real.min(), q_synth.min()), max(q_real.max(), q_synth.max())]
        ax.plot(lims, lims, "r--", alpha=0.5, linewidth=1)
        ax.set_xlabel("Real Quantiles")
        ax.set_ylabel("Synthetic Quantiles")
        ax.set_title("QQ Plot (Log Returns)")
        ax.grid(True, alpha=0.3)

        # 2. ACF of returns
        # BUG-C8: use proper ACF (normalised by full-series variance)
        # instead of np.corrcoef on sub-slices.
        def _proper_acf(x, max_lag):
            n = len(x)
            x_c = x - x.mean()
            var = float(np.dot(x_c, x_c) / n)
            if var < 1e-15:
                return np.zeros(max_lag)
            result = np.correlate(x_c, x_c, mode="full")
            result = result[n - 1:] / (var * n)
            return result[:max_lag]

        ax = axes[0, 1]
        max_lag = min(50, len(real_lr) // 3)
        real_acf = _proper_acf(real_lr, max_lag)
        synth_acf = _proper_acf(synth_lr, max_lag)
        ax.bar(np.arange(max_lag) - 0.15, real_acf, 0.3, label="Real", color="#1565C0", alpha=0.7)
        ax.bar(np.arange(max_lag) + 0.15, synth_acf, 0.3, label="Synthetic", color="#E65100", alpha=0.7)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.set_title("ACF of Returns")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 3. ACF of squared returns (volatility clustering)
        ax = axes[0, 2]
        real_sq = real_lr ** 2
        synth_sq = synth_lr ** 2
        real_acf_sq = _proper_acf(real_sq, max_lag)
        synth_acf_sq = _proper_acf(synth_sq, max_lag)
        ax.bar(np.arange(max_lag) - 0.15, real_acf_sq, 0.3, label="Real", color="#1565C0", alpha=0.7)
        ax.bar(np.arange(max_lag) + 0.15, synth_acf_sq, 0.3, label="Synthetic", color="#E65100", alpha=0.7)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.set_title("ACF of Squared Returns\n(Volatility Clustering)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 4. Rolling volatility (21-day)
        ax = axes[1, 0]
        window = 21
        if len(real_lr) > window and len(synth_lr) > window:
            real_rvol = pd.Series(real_lr).rolling(window).std() * np.sqrt(252)
            synth_rvol = pd.Series(synth_lr).rolling(window).std() * np.sqrt(252)
            ax.plot(real_rvol.values, color="#1565C0", alpha=0.8, label="Real", linewidth=1)
            ax.plot(synth_rvol.values, color="#E65100", alpha=0.8, label="Synthetic", linewidth=1)
        ax.set_xlabel("Trading Day")
        ax.set_ylabel("Annualised Volatility")
        ax.set_title("Rolling 21-day Volatility")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 5. Tail comparison (empirical CDF)
        ax = axes[1, 1]
        real_sorted = np.sort(real_lr)
        synth_sorted_lr = np.sort(synth_lr)
        ax.plot(real_sorted, np.linspace(0, 1, len(real_sorted)),
                color="#1565C0", label="Real", linewidth=1.5)
        ax.plot(synth_sorted_lr, np.linspace(0, 1, len(synth_sorted_lr)),
                color="#E65100", label="Synthetic", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Log Return")
        ax.set_ylabel("CDF")
        ax.set_title("Empirical CDF Comparison")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 6. Log-log tail plot (power law)
        ax = axes[1, 2]
        # Right tail
        real_pos = np.sort(real_lr[real_lr > 0])[::-1]
        synth_pos = np.sort(synth_lr[synth_lr > 0])[::-1]
        if len(real_pos) > 5 and len(synth_pos) > 5:
            ax.loglog(np.arange(1, len(real_pos)+1), real_pos,
                      ".", color="#1565C0", alpha=0.5, markersize=3, label="Real (right tail)")
            ax.loglog(np.arange(1, len(synth_pos)+1), synth_pos,
                      ".", color="#E65100", alpha=0.5, markersize=3, label="Synthetic (right tail)")
        ax.set_xlabel("Rank")
        ax.set_ylabel("|Return|")
        ax.set_title("Power Law Tail (Log-Log)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"{ticker} — Stylized Facts Comparison",
                     fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(self.output_dir, f"stylized_facts_{ticker}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ── Return Distributions ────────────────────────────────────────

    def _plot_return_distributions(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        ticker: str,
        feature_cols: List[str],
    ):
        """Histogram + KDE comparison for each feature's log returns."""
        cols = [c for c in feature_cols if c != "volume"]
        n = len(cols)
        if n == 0:
            return

        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
        if n == 1:
            axes = [axes]

        for ax, col in zip(axes, cols):
            real_vals = real_df[col].values.astype(float)
            synth_vals = synth_df[col].values.astype(float)
            real_lr = np.diff(np.log(np.clip(real_vals, 1e-8, None)))
            synth_lr = np.diff(np.log(np.clip(synth_vals, 1e-8, None)))
            real_lr = real_lr[np.isfinite(real_lr)]
            synth_lr = synth_lr[np.isfinite(synth_lr)]

            bins = np.linspace(
                min(real_lr.min(), synth_lr.min()),
                max(real_lr.max(), synth_lr.max()),
                80,
            )
            ax.hist(real_lr, bins=bins, density=True, alpha=0.5,
                    color="#1565C0", label="Real")
            ax.hist(synth_lr, bins=bins, density=True, alpha=0.5,
                    color="#E65100", label="Synthetic")
            ax.set_xlabel("Log Return", fontsize=10)
            ax.set_title(col.title(), fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"{ticker} — Log Return Distributions",
                     fontsize=14, fontweight="bold")
        path = os.path.join(self.output_dir, f"return_distributions_{ticker}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ── LSTM Discriminator ──────────────────────────────────────────

    def _lstm_discriminator(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        feature_cols: List[str],
        window_size: int = 20,
        epochs: int = 100,
        hidden_dim: int = 64,
        n_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Train LSTM classifiers (real vs synthetic) with k-fold CV on
        **multiple feature subsets** for granular evaluation.

        Subsets evaluated:
        - ``close``    — Close price only
        - ``volume``   — Volume only
        - ``high_low`` — High + Low prices
        - ``ohlc``     — All price features (no volume)
        - ``ohlcv``    — All features combined

        Follows the evaluation protocol of Yoon et al., "Time-series
        Generative Adversarial Networks" (NeurIPS 2019), with
        **group-aware shuffled k-fold** to prevent data leakage while
        keeping folds representative of all market regimes.

        Design choices
        --------------
        - **Normalisation**: each series (real / synthetic) is z-scored
          independently so the discriminator tests *temporal dynamics*
          rather than marginal level / scale differences.
        - **Group-aware k-fold**: the time axis is divided into
          non-overlapping groups of ``window_size`` consecutive indices.
          All stride-1 windows within the same group share data and must
          stay together.  Groups are *shuffled* and dealt into k folds,
          so every fold sees a mix of market regimes (eliminating
          non-stationarity bias) while no overlapping windows leak
          across train/test.
        - **Early stopping**: training halts if validation loss does not
          improve for ``patience`` consecutive epochs.

        Returns
        -------
        dict with per-subset accuracy, AUC-ROC, F1, and fold-level details.
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            print("  ⚠ PyTorch not available — skipping LSTM discriminator")
            return {"error": "pytorch_not_available"}

        # Build feature subsets from what is actually available
        feature_subsets: Dict[str, List[str]] = {}

        if "close" in feature_cols:
            feature_subsets["close"] = ["close"]
        if "volume" in feature_cols:
            feature_subsets["volume"] = ["volume"]
        hl = [c for c in ["high", "low"] if c in feature_cols]
        if hl:
            feature_subsets["high_low"] = hl
        ohlc = [c for c in feature_cols if c != "volume"]
        if len(ohlc) >= 2:
            feature_subsets["ohlc"] = ohlc
        feature_subsets["ohlcv"] = list(feature_cols)  # all

        # Sliding-window helper.
        # Each series is z-scored independently so the discriminator
        # evaluates temporal *dynamics*, not marginal level/scale.
        # DISC-1: log-transform volume before z-scoring (volume is
        # heavily right-skewed; z-scoring raw values leaks distributional
        # shape as a discriminative feature).
        def _make_windows(df, cols, wsize):
            vals = df[cols].values.astype(np.float32).copy()
            # Log-transform volume column(s) before z-scoring
            for ci, c in enumerate(cols):
                if c == "volume":
                    vals[:, ci] = np.log1p(np.clip(vals[:, ci], 0, None))
            mu = vals.mean(axis=0, keepdims=True)
            sigma = vals.std(axis=0, keepdims=True) + 1e-8
            vals = (vals - mu) / sigma
            windows = []
            for i in range(len(vals) - wsize):
                windows.append(vals[i:i + wsize])
            return np.array(windows)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        es_patience = 15  # early-stopping patience (epochs)

        # LSTM architecture (defined once, instantiated per subset/fold)
        class LSTMDiscriminator(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                                    batch_first=True, dropout=0.2)
                self.fc = nn.Sequential(
                    nn.Linear(hidden_dim, 32),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(32, 1),
                )

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        # ── Per-subset evaluation ───────────────────────────────────
        all_subset_results: Dict[str, Dict] = {}
        roc_data: Dict[str, tuple] = {}  # subset → (labels, probs, avg_auc)

        for subset_name, subset_cols in feature_subsets.items():
            print(f"\n    ── Subset: {subset_name} ({', '.join(subset_cols)}) ──")

            real_w = _make_windows(real_df, subset_cols, window_size)
            synth_w = _make_windows(synth_df, subset_cols, window_size)

            # Balance by the shorter series
            n_windows = min(len(real_w), len(synth_w))
            if n_windows < 2 * n_folds * window_size:
                print(f"      ⚠ Not enough windows ({n_windows}) — skipped")
                all_subset_results[subset_name] = {"error": "too_few_windows"}
                continue

            real_w = real_w[:n_windows]
            synth_w = synth_w[:n_windows]

            input_dim = real_w.shape[2]

            # Adaptive hidden dimension: scale down for low-dim inputs
            # to avoid over-parameterisation that causes training collapse.
            eff_hidden = max(16, min(hidden_dim, hidden_dim * input_dim // 3))

            # ── Group-aware shuffled k-fold ─────────────────────────
            # Assign each window index to a non-overlapping temporal
            # group.  Windows i and j share data iff |i-j| < window_size,
            # so grouping by (i // window_size) ensures that all windows
            # sharing ANY data belong to the same group.
            # Groups are shuffled and dealt into k folds, so each fold
            # sees a mix of time periods (no regime bias) and no
            # overlapping windows leak across train/test.
            n_groups = max(1, n_windows // window_size)
            group_of = np.arange(n_windows) // window_size  # group id per window
            unique_groups = np.arange(n_groups)

            rng = np.random.RandomState(42)
            rng.shuffle(unique_groups)

            # Deal groups into folds (round-robin)
            fold_group_sets: List[set] = [set() for _ in range(n_folds)]
            for i, g in enumerate(unique_groups):
                fold_group_sets[i % n_folds].add(g)

            fold_metrics = []
            all_probs: List[float] = []
            all_labels: List[float] = []

            for fold in range(n_folds):
                test_groups = fold_group_sets[fold]
                test_mask = np.array([group_of[i] in test_groups
                                      for i in range(n_windows)])
                test_idx = np.where(test_mask)[0]
                train_idx = np.where(~test_mask)[0]

                if len(train_idx) < 20 or len(test_idx) < 10:
                    continue  # degenerate fold

                # Same time indices for both classes → balanced by design
                X_test = np.concatenate([real_w[test_idx], synth_w[test_idx]])
                y_test = np.concatenate([np.ones(len(test_idx)),
                                         np.zeros(len(test_idx))])

                X_train_all = np.concatenate([real_w[train_idx],
                                              synth_w[train_idx]])
                y_train_all = np.concatenate([np.ones(len(train_idx)),
                                              np.zeros(len(train_idx))])

                # Shuffle train (safe — no leakage)
                perm = rng.permutation(len(X_train_all))
                X_train_all = X_train_all[perm]
                y_train_all = y_train_all[perm]

                # Hold out 10% of train for early-stopping validation
                n_val = max(2, len(X_train_all) // 10)
                X_val, y_val = X_train_all[:n_val], y_train_all[:n_val]
                X_train = X_train_all[n_val:]
                y_train = y_train_all[n_val:]

                ds = TensorDataset(
                    torch.FloatTensor(X_train),
                    torch.FloatTensor(y_train).unsqueeze(1),
                )
                loader = DataLoader(ds, batch_size=64, shuffle=True,
                                    generator=torch.Generator().manual_seed(42))

                model = LSTMDiscriminator(input_dim, eff_hidden).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                                             weight_decay=1e-5)
                criterion = nn.BCEWithLogitsLoss()

                best_val_loss = float("inf")
                no_improve = 0
                best_state = None

                for epoch in range(epochs):
                    # ── Train ──
                    model.train()
                    for Xb, yb in loader:
                        Xb, yb = Xb.to(device), yb.to(device)
                        loss = criterion(model(Xb), yb)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # ── Early-stopping check ──
                    model.eval()
                    with torch.no_grad():
                        val_loss = float(criterion(
                            model(torch.FloatTensor(X_val).to(device)),
                            torch.FloatTensor(y_val).unsqueeze(1).to(device),
                        ))
                    if val_loss < best_val_loss - 1e-4:
                        best_val_loss = val_loss
                        no_improve = 0
                        best_state = {k: v.cpu().clone()
                                      for k, v in model.state_dict().items()}
                    else:
                        no_improve += 1
                    if no_improve >= es_patience:
                        break

                # Restore best checkpoint
                if best_state is not None:
                    model.load_state_dict(best_state)
                    model.to(device)

                # ── Evaluate on held-out test fold ──
                model.eval()
                with torch.no_grad():
                    logits = model(torch.FloatTensor(X_test).to(device))
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    preds = (probs > 0.5).astype(float)
                    acc = float(np.mean(preds == y_test))
                    try:
                        from sklearn.metrics import roc_auc_score, f1_score
                        auc = float(roc_auc_score(y_test, probs))
                        f1 = float(f1_score(y_test, preds))
                    except ImportError:
                        auc, f1 = acc, acc

                fold_metrics.append({
                    "subset": subset_name,
                    "fold": fold,
                    "accuracy": round(acc, 4),
                    "auc_roc": round(auc, 4),
                    "f1": round(f1, 4),
                    "stopped_epoch": epoch + 1,
                })
                all_probs.extend(probs.tolist())
                all_labels.extend(y_test.tolist())

                print(f"      Fold {fold + 1}/{n_folds}: "
                      f"acc={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}  "
                      f"(stopped @ epoch {epoch + 1})")

            # Aggregate for this subset
            if not fold_metrics:
                all_subset_results[subset_name] = {"error": "no_valid_folds"}
                continue

            accs = [m["accuracy"] for m in fold_metrics]
            aucs = [m["auc_roc"] for m in fold_metrics]
            f1s = [m["f1"] for m in fold_metrics]

            avg_acc = float(np.mean(accs))
            avg_auc = float(np.mean(aucs))
            avg_f1 = float(np.mean(f1s))

            if avg_acc <= 0.55:
                verdict = "EXCELLENT — practically indistinguishable"
            elif avg_acc <= 0.65:
                verdict = "GOOD — minor distinguishable patterns"
            elif avg_acc <= 0.75:
                verdict = "MODERATE — some systematic differences"
            else:
                verdict = "POOR — clearly distinguishable"

            print(f"      Avg acc={avg_acc:.4f}  AUC={avg_auc:.4f}  "
                  f"F1={avg_f1:.4f}  → {verdict}")

            all_subset_results[subset_name] = {
                "features": subset_cols,
                "avg_accuracy": round(avg_acc, 4),
                "std_accuracy": round(float(np.std(accs)), 4),
                "avg_auc_roc": round(avg_auc, 4),
                "std_auc_roc": round(float(np.std(aucs)), 4),
                "avg_f1": round(avg_f1, 4),
                "std_f1": round(float(np.std(f1s)), 4),
                "verdict": verdict,
                "folds": fold_metrics,
            }
            roc_data[subset_name] = (all_labels, all_probs, avg_auc)

        # ── Print combined summary ──────────────────────────────────
        print(f"\n    ── DISCRIMINATOR SUMMARY (per subset) ──")
        print(f"    {'Subset':<12} {'Accuracy':>10} {'AUC-ROC':>10} "
              f"{'F1':>10}  Verdict")
        print(f"    {'─' * 65}")
        for sn, sr in all_subset_results.items():
            if "error" in sr:
                print(f"    {sn:<12} {'—':>10} {'—':>10} {'—':>10}  {sr['error']}")
            else:
                print(f"    {sn:<12} {sr['avg_accuracy']:>10.4f} "
                      f"{sr['avg_auc_roc']:>10.4f} {sr['avg_f1']:>10.4f}  "
                      f"{sr['verdict']}")

        # ── Plot multi-curve ROC ────────────────────────────────────
        self._plot_roc_multi(roc_data)

        # ── Build top-level report (backward-compatible) ────────────
        # Use 'ohlcv' subset as the primary metric for backward compat
        primary = all_subset_results.get("ohlcv", {})
        return {
            "avg_accuracy": primary.get("avg_accuracy", 0.0),
            "std_accuracy": primary.get("std_accuracy", 0.0),
            "avg_auc_roc": primary.get("avg_auc_roc", 0.0),
            "std_auc_roc": primary.get("std_auc_roc", 0.0),
            "avg_f1": primary.get("avg_f1", 0.0),
            "std_f1": primary.get("std_f1", 0.0),
            "verdict": primary.get("verdict", ""),
            "folds": primary.get("folds", []),
            "subsets": all_subset_results,
            "window_size": window_size,
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "n_folds": n_folds,
        }

    def _plot_roc_multi(self, roc_data: Dict[str, tuple]):
        """
        Plot ROC curves from concatenated cross-validation predictions.

        Parameters
        ----------
        roc_data : dict
            ``{subset_name: (all_labels, all_probs, avg_auc)}``
            where *all_labels* and *all_probs* are flat lists aggregated
            across all CV folds.
        """
        try:
            from sklearn.metrics import roc_curve, roc_auc_score
        except ImportError:
            return

        if not roc_data:
            return

        # Colour palette per subset
        palette = {
            "close":    {"color": "#1565C0", "ls": "-"},    # blue solid
            "volume":   {"color": "#E65100", "ls": "-"},    # orange solid
            "high_low": {"color": "#2E7D32", "ls": "--"},   # green dashed
            "ohlc":     {"color": "#C62828", "ls": "-."},   # red dash-dot
            "ohlcv":    {"color": "#6A1B9A", "ls": "-"},    # purple solid
        }
        default_colors = ["#00838F", "#F9A825", "#4E342E", "#546E7A"]
        color_idx = 0

        fig, ax = plt.subplots(figsize=(8, 8))

        display_names = {
            "close": "Close",
            "volume": "Volume",
            "high_low": "High + Low",
            "ohlc": "OHLC (no Volume)",
            "ohlcv": "All OHLCV",
        }

        # Sort subsets: plot highest AUC first (background) so lower-AUC
        # curves render on top and remain visible when curves overlap.
        sorted_subsets = sorted(
            roc_data.items(),
            key=lambda kv: kv[1][2], reverse=True,   # kv[1][2] = avg_auc
        )

        for subset_name, (labels, probs, avg_auc) in sorted_subsets:
            if not labels:
                continue

            labels_arr = np.array(labels)
            probs_arr = np.array(probs)
            fpr, tpr, _ = roc_curve(labels_arr, probs_arr)
            # AUC consistent with the plotted curve
            plot_auc = float(roc_auc_score(labels_arr, probs_arr))

            style = palette.get(subset_name, None)
            if style is None:
                style = {"color": default_colors[color_idx % len(default_colors)],
                         "ls": "-"}
                color_idx += 1

            display = display_names.get(subset_name, subset_name)

            ax.plot(
                fpr, tpr,
                color=style["color"], linestyle=style["ls"], linewidth=2.2,
                label=f"{display} (AUC = {plot_auc:.3f})",
            )

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4,
                label="Random (AUC = 0.500)")

        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves — LSTM Discriminator per Feature Subset",
                      fontsize=13, fontweight="bold")
        ax.legend(fontsize=11, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

        path = os.path.join(self.output_dir, "discriminative_roc.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _save_discriminator_csv(self, disc_report: Dict):
        """Save discriminator fold details and per-subset summary as CSV."""
        # Per-fold details (all subsets combined)
        all_folds = []
        subsets = disc_report.get("subsets", {})
        for sn, sr in subsets.items():
            if "folds" in sr:
                all_folds.extend(sr["folds"])
        if not all_folds:
            # Backward compat: use top-level folds
            if "folds" in disc_report:
                all_folds = disc_report["folds"]
        if all_folds:
            df = pd.DataFrame(all_folds)
            path = os.path.join(self.output_dir, "discriminator_folds.csv")
            df.to_csv(path, index=False)

        # Per-subset summary table
        summary_rows = []
        for sn, sr in subsets.items():
            if "error" in sr:
                continue
            row = {"subset": sn, "features": ", ".join(sr.get("features", []))}
            for k in ["avg_accuracy", "std_accuracy", "avg_auc_roc",
                       "std_auc_roc", "avg_f1", "std_f1", "verdict"]:
                row[k] = sr.get(k, "")
            summary_rows.append(row)
        if summary_rows:
            df_s = pd.DataFrame(summary_rows)
            path = os.path.join(self.output_dir, "discriminator_summary.csv")
            df_s.to_csv(path, index=False)
        else:
            # Backward compat: single summary row
            summary = {k: v for k, v in disc_report.items()
                       if isinstance(v, (int, float, str))}
            if summary:
                df_s = pd.DataFrame([summary])
                path = os.path.join(self.output_dir, "discriminator_summary.csv")
                df_s.to_csv(path, index=False)


# ══════════════════════════════════════════════════════════════════════
# Top-level orchestrator
# ══════════════════════════════════════════════════════════════════════

class QualityReportGenerator:
    """
    Top-level orchestrator for publication-quality evaluation.

    Creates two sub-folders:
    - ``llm_quality/``        — LLM factor comparison
    - ``generation_quality/`` — synthetic data fidelity + discriminator
    """

    def __init__(self, output_dir: str = "results/evaluation"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        ticker: str = "AAPL",
        factor_cache_dirs: Optional[Dict[str, str]] = None,
        disc_epochs: int = 100,
        disc_hidden: int = 64,
        disc_window_size: int = 20,
        disc_n_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Run both LLM comparison and generation quality reports.

        Parameters
        ----------
        real_df : pd.DataFrame
            Real OHLCV data.
        synth_df : pd.DataFrame
            Synthetic data.
        ticker : str
            Ticker symbol.
        factor_cache_dirs : dict | None
            ``{model_label: cache_path}``. If None or empty, skips LLM
            comparison.
        disc_epochs, disc_hidden, disc_window_size, disc_n_folds
            LSTM discriminator hyperparameters.

        Returns
        -------
        report : dict
        """
        print("\n" + "═" * 60)
        print("  PUBLICATION QUALITY REPORT")
        print("═" * 60)

        report: Dict[str, Any] = {}

        # Part 1: LLM comparison
        if factor_cache_dirs:
            llm_dir = os.path.join(self.output_dir, "llm_quality")
            llm_comp = LLMQualityComparison(output_dir=llm_dir)
            report["llm_quality"] = llm_comp.run(factor_cache_dirs)
        else:
            print("\n  ℹ  No factor_cache_dirs provided — skipping LLM comparison")

        # Part 2: Generation quality
        gen_dir = os.path.join(self.output_dir, "generation_quality")
        gen_qual = GenerationQualityReport(output_dir=gen_dir)
        report["generation_quality"] = gen_qual.run(
            real_df=real_df,
            synth_df=synth_df,
            ticker=ticker,
            disc_epochs=disc_epochs,
            disc_hidden=disc_hidden,
            window_size=disc_window_size,
            disc_n_folds=disc_n_folds,
        )

        # Final summary
        print(f"\n{'═' * 60}")
        print(f"  QUALITY REPORT COMPLETE")
        print(f"  Output: {self.output_dir}/")
        print(f"{'═' * 60}\n")

        return report
