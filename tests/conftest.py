"""Shared Pytest Fixtures and Environment Configuration for TokenTrie Tests.

This module provides initialized and pre-trained instances of TokenTrieModel
configured for deterministic unit testing across the test suite.
"""

from typing import Generator

import pytest

from tokentrie import TokenTrieModel


@pytest.fixture
def empty_model() -> TokenTrieModel:
    """Provides a fresh, unpopulated TokenTrieModel instance.

    Returns:
        TokenTrieModel: Initialized model with max_depth=5, decay=0.9.
    """
    return TokenTrieModel(max_depth=5, decay=0.9)


@pytest.fixture
def trained_model() -> TokenTrieModel:
    """Provides a TokenTrieModel trained on a deterministic e-commerce sequence.

    Returns:
        TokenTrieModel: Pre-fitted model instance with established context paths.
    """
    model = TokenTrieModel(max_depth=5, decay=0.95)
    model.fit(
        ["click", "buy", "click", "buy", "click", "buy", "exit"],
        verbose=False,
    )
    return model


@pytest.fixture
def static_model() -> TokenTrieModel:
    """Provides a TokenTrieModel with decay math completely disabled.

    Returns:
        TokenTrieModel: Model instance operating in fast-path static mode (decay=1.0).
    """
    return TokenTrieModel(max_depth=3, decay=1.0)