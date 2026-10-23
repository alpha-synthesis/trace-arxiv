#!/usr/bin/env python3
"""Preflight for the R2 runs. No LLM, no network.

The YAML loader silently ignores unknown keys, so a typo would fall back to a
default (e.g. prices back in the prompt). This script loads each R2 config the
way scripts/run_pipeline.py does and asserts every field fixed in advance, then checks the corpus-v2 contexts and that the output/cache dirs are
fresh.

Usage: python scripts/preflight_r2.py configs/r2_datasign.yaml [...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sde_causal_generator.pipeline import PipelineConfig  # noqa: E402

CORPUS_TAG = "rag_store_v2@5dc822a7a220d920+ctx2.1"
COMMON = dict(llm_prompt_mode="price_blind", use_rag=False, fin_impact_estimator="ablation",
              use_factor_scoring=False, max_factors=0, gen_mode="")


def check(path: str) -> list[str]:
    raw = yaml.safe_load(open(path))
    cfg = PipelineConfig._from_flat_or_nested(dict(raw))
    errs = []
    pre = cfg.precomputed_analyses_dir
    want = dict(COMMON, fin_direction_prior="llm" if pre else "none")
    if not pre:
        want.update(rag_redact_market_moves="strict2", rag_context_dir="cache/contexts_v2.1",
                    rag_corpus_tag=CORPUS_TAG, generate_scenarios=False)
    for k, v in want.items():
        if getattr(cfg, k) != v:
            errs.append(f"{k} = {getattr(cfg, k)!r}, expected {v!r}")
    for k in raw:
        if k not in cfg.__dataclass_fields__ and k not in (
                "tickers", "train_start", "train_end", "fin", "generation", "evaluation",
                "neural_sde", "cross_asset", "factor_scoring", "transfer_learning", "dcc_garch",
                "interactive", "user_factors", "quality_report", "timegan", "tstr",
                "resume", "validate"):
            errs.append(f"unknown top-level key (ignored by the loader): {k}")
    tickers = raw["tickers"]
    if len(tickers) != 28:
        errs.append(f"{len(tickers)} tickers, expected 28")
    out = ROOT / "results" / raw["output_dir"]
    if out.exists() and any(out.iterdir()):
        errs.append(f"output dir not empty: {out}")
    if not pre:
        cache = ROOT / raw["cache_dir"]
        if cache.exists() and any(cache.rglob("*.json")):
            errs.append(f"LLM cache dir not empty: {cache}")
        idx = json.load(open(ROOT / cfg.rag_context_dir / "INDEX.json"))
        if idx.get("corpus_tag") != CORPUS_TAG or idx.get("redaction_mode") != "strict2":
            errs.append(f"contexts INDEX: corpus_tag={idx.get('corpus_tag')} "
                        f"redaction={idx.get('redaction_mode')}")
        for g in [k for k in idx if k.startswith("gate_G1_")] + ["store_unchanged"]:
            if idx.get(g) is not True:
                errs.append(f"contexts INDEX: {g} = {idx.get(g)}")
        if idx.get("window_sizes") != raw.get("window_sizes"):
            errs.append(f"contexts window_sizes {idx.get('window_sizes')} != {raw.get('window_sizes')}")
        for t in tickers:
            f = ROOT / cfg.rag_context_dir / f"{t}.json"
            if not f.exists():
                errs.append(f"no contexts for {t}")
                continue
            d = json.load(open(f))
            if d.get("corpus_tag") != CORPUS_TAG or len(d.get("windows", [])) != 155:
                errs.append(f"{t}: corpus_tag={d.get('corpus_tag')} windows={len(d.get('windows', []))}")
    else:
        if not (ROOT / pre).exists():
            print(f"  note: {pre} does not exist yet (written by the data-sign run)")
    return errs


def main() -> int:
    bad = 0
    for p in sys.argv[1:]:
        errs = check(p)
        print(f"{p}: {'OK' if not errs else 'FAIL'}")
        for e in errs:
            print(f"  - {e}")
        bad += bool(errs)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
