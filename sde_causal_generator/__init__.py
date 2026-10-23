# -*- coding: utf-8 -*-
"""
Causal SDE Generator
====================

Event-Conditioned Stochastic Differential Equation (ECSDE) generator.

A three-phase synthetic data generator that:

1. **Factor Extraction**:
   - Extracts causal factors from historical data via LLM + RAG
   - Builds a temporal news corpus (Wikipedia, FRED, SEC EDGAR)
   - Produces a binary presence matrix F ∈ {0,1}^(T×K)

2. **Impact Learning**:
   - Trains a Factor Impact Network (FIN) on F + real prices
   - Produces an ImpactMatrix with direction, magnitude,
     temporal profiles, and factor interactions
   - Generates plausible future scenarios via LLM

3. **Generation & Evaluation**:
   - Generates realistic synthetic price trajectories
   - Provides interactive UI for user validation of events
   - Evaluates quality via discriminative/predictive scores

References:
    - Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via ML"
    - Ross (1976), "Arbitrage Pricing Theory"
    - Ke, Kelly & Xiu (2019), "Predicting Returns with Text Data"
"""

__version__ = "0.1.0"

# ── Public API ──────────────────────────────────────────────────────
from .data_structures import (
    CausalFactor,
    ImpactMatrix,
    Scenario,
    ScenarioSet,
    WindowAnalysis,
    NewsArticle,
    FACTOR_CATEGORIES,
)
from .factor_extractor import LLMFactorExtractor
from .news_rag import NewsRAGManager, TemporalNewsStore
from .factor_impact_network import (
    FactorImpactNetwork,
    TemporalFusionTransformerFIN,
    train_impact_network,
    prepare_training_data,
)
from .scenario_generator import ScenarioGenerator
from .generate import ImpactDrivenGenerator
from .neural_sde import NeuralSDEModel, NeuralSDEGenerator, train_neural_sde, sample_neural_sde
from .cross_asset import (
    estimate_cross_asset_correlation,
    estimate_dcc_garch,
    extract_returns_from_dataframes,
    cholesky_factor,
    recorrelate_prices,
    recorrelate_returns,
    recorrelate_returns_dynamic,
)
from .factor_relevance import (
    score_factors,
    prune_irrelevant_factors,
    print_relevance_report,
)
from .oos_validation import oos_validate, plot_oos_comparison, print_oos_report
from .evaluate_data import CausalSDEEvaluator
from .event_validator import (
    UserEvent,
    create_dash_app,
    deduplicate_events,
    events_to_presence_matrix,
    events_to_scenario_set,
    merge_llm_and_user_events,
    plot_price_with_events,
)
from .generation_editor import (
    create_generation_editor_app,
)
from .pipeline import CausalSDEPipeline, PipelineConfig
from .download_data import (
    download_training_data,
    download_test_data,
    split_by_ticker,
    save_training_data,
    load_training_data,
    validate_data,
)

__all__ = [
    # Data structures
    "CausalFactor",
    "ImpactMatrix",
    "Scenario",
    "ScenarioSet",
    "WindowAnalysis",
    "NewsArticle",
    "FACTOR_CATEGORIES",
    # Phase 1: Extraction
    "LLMFactorExtractor",
    "NewsRAGManager",
    "TemporalNewsStore",
    # Phase 2: Impact Learning
    "FactorImpactNetwork",
    "TemporalFusionTransformerFIN",
    "train_impact_network",
    "prepare_training_data",
    # Phase 2: Scenarios
    "ScenarioGenerator",
    # Phase 3: Generation
    "ImpactDrivenGenerator",
    # Phase 3: Neural SDE (improvement #8)
    "NeuralSDEModel",
    "NeuralSDEGenerator",
    "train_neural_sde",
    "sample_neural_sde",
    # Phase 3: Cross-asset (improvement #9)
    "estimate_cross_asset_correlation",
    "estimate_dcc_garch",
    "extract_returns_from_dataframes",
    "cholesky_factor",
    "recorrelate_prices",
    "recorrelate_returns",
    "recorrelate_returns_dynamic",
    # Phase 1: Factor Relevance (F4)
    "score_factors",
    "prune_irrelevant_factors",
    "print_relevance_report",
    # Phase 3: OOS validation (improvement #10)
    "oos_validate",
    "plot_oos_comparison",
    "print_oos_report",
    # Phase 3: Evaluation
    "CausalSDEEvaluator",
    # Phase 3: Interactive Validation & Generation Editing
    "UserEvent",
    "create_dash_app",
    "events_to_presence_matrix",
    "events_to_scenario_set",
    "merge_llm_and_user_events",
    "plot_price_with_events",
    "create_generation_editor_app",
    # Pipeline
    "CausalSDEPipeline",
    "PipelineConfig",
    # Data Download
    "download_training_data",
    "download_test_data",
    "split_by_ticker",
    "save_training_data",
    "load_training_data",
    "validate_data",
]
