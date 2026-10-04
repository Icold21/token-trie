"""Unit Tests for Prediction Mechanics, Sampling Controls, and Beam Search.

Validates inference parameters including temperature scaling, top-k filtering,
top-p nucleus sampling, unigram smoothing fallbacks, and masked breadth-first search.
"""

import math
from typing import Dict, Optional

import pytest

from tokentrie import TokenTrieModel


def test_predict_empty_model(empty_model: TokenTrieModel) -> None:
    """Verifies that unpopulated models return safe empty structures."""
    probas = empty_model.predict_proba()
    token = empty_model.predict()

    assert probas == {}
    assert token is None


def test_temperature_scaling(trained_model: TokenTrieModel) -> None:
    """Validates that temperature modifies output distribution flatness.

    High temperatures should flatten the probability mass (increasing entropy),
    while near-zero temperatures sharpen the distribution toward greedy argmax.
    """
    trained_model.fill_context(["click", "buy"])

    probas_greedy = trained_model.predict_proba(temperature=0.0)
    probas_flat = trained_model.predict_proba(temperature=5.0)

    max_greedy = max(probas_greedy.values())
    max_flat = max(probas_flat.values())

    assert max_greedy > max_flat


def test_top_k_filtering(trained_model: TokenTrieModel) -> None:
    """Verifies that top_k strictly limits candidates to the top K ranks."""
    trained_model.fill_context(["click"])
    probas = trained_model.predict_proba(top_k=1)

    assert len(probas) == 1
    assert math.isclose(sum(probas.values()), 1.0, rel_tol=1e-5)


def test_top_p_nucleus_sampling(trained_model: TokenTrieModel) -> None:
    """Verifies that Nucleus Sampling restricts candidates to cumulative bounds."""
    trained_model.fit(["a", "b", "a", "c", "a", "d", "a", "e"], verbose=False)
    trained_model.fill_context(["a"])

    probas = trained_model.predict_proba(top_p=0.5)

    assert len(probas) > 0
    assert math.isclose(sum(probas.values()), 1.0, rel_tol=1e-5)


def test_fallback_strategies() -> None:
    """Compares uniform distribution fallback against interpolated unigram prior."""
    model_unigram = TokenTrieModel(fallback_mode="interpolated_unigram")
    model_uniform = TokenTrieModel(fallback_mode="uniform")

    train_data = ["rare", "frequent", "frequent", "frequent"]
    model_unigram.fit(train_data, verbose=False)
    model_uniform.fit(train_data, verbose=False)

    model_unigram.fill_context(["unseen_context"])
    model_uniform.fill_context(["unseen_context"])

    probas_unigram = model_unigram.predict_proba()
    probas_uniform = model_uniform.predict_proba()

    assert probas_unigram["frequent"] > probas_unigram["rare"]
    assert math.isclose(
        probas_uniform["frequent"],
        probas_uniform["rare"],
        rel_tol=1e-5,
    )


def test_return_log_scores(trained_model: TokenTrieModel) -> None:
    """Ensures return_log_scores bypasses softmax normalization."""
    trained_model.fill_context(["click", "buy"])
    log_scores = trained_model.predict_proba(return_log_scores=True)

    for score in log_scores.values():
        assert isinstance(score, float)

    top_token = max(log_scores, key=log_scores.get)
    assert top_token is not None


def test_masked_mode_beam_search() -> None:
    """Validates masked BFS beam search across noisy or corrupted context tokens.

    Tests that exact matching fails on corrupted suffix tokens, while
    linear and squared masked beam searches resolve the correct path via wildcards.
    """
    model = TokenTrieModel(max_depth=3, decay=1.0)
    # Train distinct disambiguating context paths:
    # [open, door] -> enter
    # [close, door] -> exit
    model.fit([["open", "door", "enter"], ["close", "door", "exit"]], verbose=False)

    # 1. Exact match with masked_mode='none'
    model.fill_context(["open", "door"])
    assert model.predict(masked_mode="none") == "enter"

    # 2. Context with a corrupted/typo token at the end: ['open', 'CORRUPTED_DOOR']
    model.fill_context(["open", "CORRUPTED_DOOR"])

    # Standard exact match breaks at depth 0 on 'CORRUPTED_DOOR', falling back to equal unigrams
    probas_none = model.predict_proba(masked_mode="none")
    assert math.isclose(probas_none["enter"], probas_none["exit"], rel_tol=1e-3)

    # Masked beam search wildcards 'CORRUPTED_DOOR' and matches 'open',
    # resolving the distribution overwhelmingly in favor of 'enter'
    probas_linear = model.predict_proba(masked_mode="linear")
    probas_squared = model.predict_proba(masked_mode="squared")

    assert probas_linear["enter"] > probas_linear["exit"] * 2.0
    assert probas_squared["enter"] > probas_squared["exit"] * 2.0
    assert model.predict(masked_mode="linear") == "enter"
    assert model.predict(masked_mode="squared") == "enter"