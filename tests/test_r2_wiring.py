"""Pipeline wiring for the R2 run (corpus v2 contexts, strict redaction).

The extractor hooks are covered by tests/test_corpus_v2_extractor.py; these
tests pin how PipelineConfig and _phase1_extract pass them through.
"""
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from sde_causal_generator import factor_extractor as fe
from sde_causal_generator.pipeline import CausalSDEPipeline, PipelineConfig

ROOT = Path(__file__).resolve().parents[1]
TAG = "rag_store_v2@5dc822a7a220d920+ctx2.1"


def _df():
    d = pd.bdate_range("2020-01-01", periods=30)
    return pd.DataFrame({"date": d, "open": 1.0, "high": 1.0, "low": 1.0,
                         "close": 1.0, "volume": 1.0})


def _ctx_dir(tmp_path, tag=TAG):
    (tmp_path / "INDEX.json").write_text(json.dumps({"corpus_tag": tag}))
    (tmp_path / "AAA.json").write_text(json.dumps(
        {"ticker": "AAA", "corpus_tag": tag, "windows": []}))
    return str(tmp_path)


class _Spy:
    """Stands in for LLMFactorExtractor; records kwargs and close()."""
    last = None

    def __init__(self, **kw):
        self.kw, self.closed, self.n_llm_calls = kw, False, 0
        self._consistency_stats = []
        _Spy.last = self

    def extract_all_windows(self, *a, **k):
        raise RuntimeError("boom")      # close() must still run

    def close(self):
        self.closed = True


def test_r2_configs_load_the_fixed_fields():
    for name, prior in (("r2_datasign", "none"), ("r2_llmsign", "llm")):
        raw = yaml.safe_load(open(ROOT / "configs" / f"{name}.yaml"))
        cfg = PipelineConfig._from_flat_or_nested(dict(raw))
        assert cfg.llm_prompt_mode == "price_blind"
        assert cfg.use_rag is False and cfg.max_factors == 0
        assert cfg.fin_impact_estimator == "ablation" and cfg.fin_direction_prior == prior
        assert cfg.use_factor_scoring is False
    raw = yaml.safe_load(open(ROOT / "configs" / "r2_datasign.yaml"))
    cfg = PipelineConfig._from_flat_or_nested(dict(raw))
    assert cfg.rag_redact_market_moves == "strict2"
    assert cfg.rag_context_dir == "cache/contexts_v2.1" and cfg.rag_corpus_tag == TAG
    assert cfg.generate_scenarios is False and cfg.llm_call_log
    assert not cfg.precomputed_analyses_dir


def test_context_dir_requires_use_rag_false(tmp_path):
    cfg = PipelineConfig(output_dir=str(tmp_path), use_rag=True,
                         rag_context_dir=_ctx_dir(tmp_path))
    with pytest.raises(ValueError, match="use_rag"):
        CausalSDEPipeline(config=cfg)._phase1_extract(_df(), "AAA")


def test_phase1_passes_contexts_and_closes_client(tmp_path, monkeypatch):
    monkeypatch.setattr(fe, "LLMFactorExtractor", _Spy)
    cfg = PipelineConfig(output_dir=str(tmp_path), use_rag=False,
                         rag_context_dir=_ctx_dir(tmp_path), rag_corpus_tag=TAG,
                         llm_call_log=str(tmp_path / "calls.jsonl"),
                         llm_prompt_mode="price_blind", rag_redact_market_moves="strict2")
    with pytest.raises(RuntimeError, match="boom"):
        CausalSDEPipeline(config=cfg)._phase1_extract(_df(), "AAA")
    kw = _Spy.last.kw
    assert isinstance(kw["context_provider"], fe.PrecomputedContexts)
    assert kw["context_provider"].corpus_tag == TAG
    assert kw["news_rag"] is None
    assert kw["redact_market_moves"] == "strict2" and kw["prompt_mode"] == "price_blind"
    assert kw["call_log_path"] == str(tmp_path / "calls.jsonl")
    assert _Spy.last.closed


def test_corpus_tag_mismatch_is_refused(tmp_path):
    cfg = PipelineConfig(output_dir=str(tmp_path), use_rag=False,
                         rag_context_dir=_ctx_dir(tmp_path, tag="other@0"), rag_corpus_tag=TAG)
    with pytest.raises(ValueError, match="corpus_tag"):
        CausalSDEPipeline(config=cfg)._phase1_extract(_df(), "AAA")
