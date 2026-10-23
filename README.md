# TRACE: code and data

This repository accompanies the paper *TRACE: A Three-block Reproducible Assessment Protocol for LLM-Conditioned Event Generators* (arXiv, 2026). It contains the Event-Conditioned Stochastic Differential Equation (ECSDE) generator, the scripts that run the TRACE blocks (B1 to B3), the sign-recovery check, the fidelity guard and the positive control, and the saved results behind the paper's tables and figures. Names follow the paper.

## Layout

| Path | Contents |
|---|---|
| `sde_causal_generator/` | The ECSDE generator: P1 price-blind extraction, P2 effect estimation (TFT), P3 generation |
| `configs/` | One configuration per run: `r0`, `a1`, `r1_datasign`, `r1_llmsign`, `r2_datasign`, `r2_llmsign` |
| `scripts/` | Generator entry point (`run_pipeline.py`), corpus and context builders, run scripts for R1 and R2 |
| `experiments/scripts/` | TRACE blocks, checks, diagnostics, positive control and figures |
| `experiments/results/` | Saved results read by the paper |
| `experiments/audit/` | Human audit of the R2 news contexts (workbook and sample) |
| `tools/` | Price download and restoration of window returns |
| `tests/` | Unit tests of the generator |

## Runs and label sets

| Paper | Run folder and config | Description |
|---|---|---|
| A0 | labels in `r0` | Labels extracted with realized prices in the prompt from the uncurated corpus. Used by R1 |
| A1 | `a1` | Price-blind labels from the uncurated corpus, with market-move sentences removed. Used only in B1 |
| R0 | `r0` | Reference configuration of the fidelity guard (Appendix A.2). `r0` holds the A0 labels and R0's impact matrices, and `implementation_checks.py` regenerates R0's paths from them with the same code and seed as R1 and R2 |
| R1 | `r1_datasign` | A0 labels with the corrected estimator and generator. Reference of B2's factual-fidelity criterion |
| R2 | `r2_datasign` | Primary arm: price-blind labels from curated, pinned and redacted news contexts |
| LLM-sign variants | `r1_llmsign`, `r2_llmsign` | Secondary variants whose effect signs follow the labels (Appendix A.10) |

In the B1 scripts and results, the label sets are keyed `A0`, `A1` and `R2`.

## Setup

```bash
conda env create -f environment.yml && conda activate trace
pip install -e ".[full,dev]"
```

Tested on Linux x86_64 with Python 3.12. The positive control used 32 CPU workers and one NVIDIA A100 GPU. Every other script runs on a laptop, a CUDA GPU being optional.

## Data

Two inputs are not in the repository itself.

1. **Data bundle** (`trace-data.tar.gz`, archived on Zenodo at [10.5281/zenodo.23600332](https://doi.org/10.5281/zenodo.23600332) and attached to the [v1.0 release](https://github.com/alpha-synthesis/trace-arxiv/releases/tag/v1.0) of this repository, SHA-256 `026dce092059ff1af848d8d06f87f3e0b34e719cf44ae49946b04d257812d4ad`). It holds the label sets, impact matrices and run reports of each run, the R2 news contexts, the Wikipedia revision manifest and the five-seed impact matrices of R2. The free-text fields of the labels (descriptions and raw responses) are empty, since some quoted prices. Extract it at the repository root:

   ```bash
   tar -xzf trace-data.tar.gz     # creates results/, cache/ and experiments/results/five_seed__r2_datasign__data/
   ```

2. **Market prices** (Yahoo Finance, through `yfinance`), which are not redistributed. Download them and restore each window's return in the label files:

   ```bash
   python tools/download_prices.py          # writes results/<run>/training_data.csv
   python tools/attach_window_returns.py    # restores price_change_pct in every analyses.json
   ```

   Yahoo revises dividend-adjusted history, so a fresh download can differ slightly from the one used for the paper. Restored window returns match the original ones to within 1e-4 percentage points, but a revision can flip the sign of a window whose return is almost zero. With prices downloaded on 22 October 2026, one such window (KO, 5 to 19 August 2018, +0.00003% at run time) changed sign. For the ten windows with an absolute return below 0.01%, the tool therefore uses the value recorded at run time (`tools/near_zero_window_returns.json`).

   From a clean clone with a fresh download, `b1_event_grounding.py`, `b1_sensitivity.py`, `llm_recall_probe.py --score`, `score_human_audit.py` and `make_figures.py` reproduced the saved results exactly (769 numbers).

## Reproducing the paper

Run from `experiments/scripts/`. Outputs go to `experiments/results/`, where the saved versions already are. `TRACE_RUN` selects the run folder.

| Paper | Command | Saved output |
|---|---|---|
| 4.1 Implementation checks, fidelity guard (R0) | `TRACE_RUN=r1_datasign python implementation_checks.py` (also `r2_datasign`). The guard compares each arm with R0 regenerated from `r0`. A side comparison with R0's archived synthetic paths (Appendix A.2) is skipped, see below. | `implementation_checks__<run>/` |
| 4.1 Fidelity guard insensitivity to the effects | `python exploratory_generator_diagnostics.py` | `exploratory_generator_diagnostics/` |
| 4.2 B1 agreement, placebo, within-stock null | `python b1_event_grounding.py` | `b1_event_grounding/` |
| 4.2, A.5 B1 calendar-window tests and sensitivity | `python b1_sensitivity.py` | `b1_sensitivity/` |
| 4.2, A.5 Label information and other diagnostics | `python exploratory_diagnostics.py` | `exploratory_diagnostics/` |
| 4.2 Recall probe | `python llm_recall_probe.py --model openai/gpt-4o-mini --score` | `llm_recall_probe/` |
| 4.2, A.4 Human audit of the R2 contexts | `python score_human_audit.py` | `human_audit/` |
| 4.3, A.7 B2 | `TRACE_RUN=<run> python b2_counterfactual.py` for `r1_datasign`, `r1_llmsign`, `r2_datasign` and `r2_llmsign` | `b2_counterfactual__<run>/` |
| 4.3, A.7 Tracking against the engine without effects | `python tracking_vs_no_effects.py` | `tracking_vs_no_effects/` |
| 4.4, A.6 Sign-recovery check | `TRACE_RUN=<run> python sign_recovery.py --seed S` for S = 0..4, then `--aggregate` and `--aggregate-majority` | `sign_recovery__<run>/`. The pooled least-squares rows of the sign-recovery table come from `figure_data/figure_data.json` (`power.ols_pooled_over_planted`), which pools over planted factors, while `sign_recovery__<run>/summary.json` averages per-stock rates. |
| A.6 Estimator comparison | `TRACE_RUN=r1_datasign python sign_recovery_estimators.py`, then `--aggregate` | `sign_recovery_estimators__r1_datasign/` |
| 4.5.1, A.8 B3 rolling origin | `TRACE_RUN=r1_datasign python b3_rolling_origin.py --origin YYYY-01-01` for 2021 to 2023, then `--aggregate` (add `--estimator ols` for the OLS secondary) | `b3_rolling_origin__<run>__data*/` |
| 4.5.2 B3 documented effects | `TRACE_RUN=r2_datasign python five_seed_estimates.py --variant data`, then `TRACE_RUN=r2_datasign python b3_documented_effects.py --score --variant data --frozen ../results/b3_documented_effects__r2_datasign/classified_FROZEN.csv` (add `--estimator ols` for the secondary) | `b3_documented_effects__r2_datasign__data*/` |
| 4.6, A.9 Positive control | `bash launch_positive_control.sh selftest`, then `cpu`, `gpu`, `cal` and `calf` (runs go to `pc_runs/`), then `positive_control_aggregate.py` with `--ols` and `--tft-work` (main settings), `--cal [--cal-arm CALF] --pair` (real-calendar setting) or `--compare`, and `positive_control_leakage_diag.py` for the real-calendar leakage-gate investigation. Each script's docstring gives its usage. | `positive_control/` |
| Figures | `python make_figures.py` | `experiments/figures/`, `figure_data/figure_data.json` |

The runs come from `scripts/run_r1.sh` and `scripts/run_r2.sh` (`python scripts/run_pipeline.py configs/<run>.yaml`). R2's extraction calls GPT-4o-mini through OpenRouter (`OPENROUTER_API_KEY` in `.env`, see `.env.example`). The saved labels let every later step run without an API key.

In the positive control (Appendix A.9), `M0` to `M3` are the main worlds, `EV` and `EV0` the worlds with window-specific names (with and without effects), `IN` the worlds with independent noise, `HET` the worlds with effects that vary across events, `RR` the worlds built on circularly shifted real returns, `CAL` the real-calendar setting and `CALF` the real-calendar worlds regenerated with a per-world offset of the auxiliary channels. `TFT` marks the worlds scored with the five-seed TFT.

## What cannot be rerun from this repository

- The extractions behind A0 (`r0`) and A1 (`a1`) read a news store that included FRED market series and is not redistributed. Their labels are in the data bundle.
- The cross-model labels (`experiments/results/cross_model_labels/`) were extracted with prices in the prompt from the same store. The folder also holds Gemini 2.0 Flash labels for three stocks, which `b1_sensitivity.py` scores but the paper does not report.
- The side comparison of the fidelity guard with R0's archived synthetic paths needs those paths, which are tied to the price history and are not redistributed. The saved summaries keep its counts (8 of 28 stocks pass for R1 and 13 of 28 for R2, the 20 and 15 failures reported in Appendix A.2).
- `scripts/run_r2.sh` requires fresh output folders, so rerun R2 in a copy without the bundle's `results/r2_*` folders.
- The positive-control worlds are rebuilt from the downloaded prices, so their recorded hashes will not match a new download exactly.
- LLM answers are not deterministic, so a new extraction or recall probe gives different labels. The saved ones are in the data bundle and in `experiments/results/`.

## Notes

- The analysis choices that affected interpretation and the known implementation issues are listed in the paper (Appendices A.2 and A.3).
- Two notes of the human auditor were translated from Portuguese ("unsure").
- The scripts and result folders were renamed to the paper's nomenclature and edited for release after the runs (names, comments, option names, file locations and the optional stop time of the positive-control queues). The computations are unchanged, but script hashes recorded in saved outputs (for example `aggregator_sha256`) do not match the released files, and a few saved fields keep their original internal names (for example `exp_o_tft` in the positive-control summary).
- The TFT trains for 300 epochs in the runs (configs) and for 200 epochs in the rolling-origin refits (`experiments/scripts/shared.py`).
- `b1_event_grounding.py` writes a `reading` field that maps each label set's agreement to the interpretation levels fixed in advance. It is not a verdict. The paper's B1 verdicts are in Section 4.2.
- `b3_rolling_origin__r2_datasign__data__ols/verdict.json` says FAIL with no eligible stock. The paper reports that test as not run (Appendix A.2).
- Effect estimation uses the GPU when available. Results can differ in the last digits across hardware.

## Licenses

- Code: MIT (`LICENSE`).
- Text from the Wikipedia Current Events Portal (`cache/contexts_v2.1`, `experiments/audit/`): CC BY-SA 4.0, attributed in `ATTRIBUTION.md`.
- Labels, impact matrices and other results: CC BY 4.0.
- Market data is not included. Prices downloaded with `tools/download_prices.py` are subject to Yahoo's terms.

## Citation

See `CITATION.cff`.
