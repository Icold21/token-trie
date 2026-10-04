"""Unit Tests for State Persistence and Cross-Session Deserialization.

Validates JSON schema exports, binary Pickle serialization, volatile cache
invalidation, and post-load prediction fidelity using public accessors.
"""

import os
from pathlib import Path

from tokentrie import TokenTrieModel


def test_json_serialization_deserialization(
    trained_model: TokenTrieModel,
    tmp_path: Path,
) -> None:
    """Verifies lossless JSON state export, disk dumping, and reconstitution.

    Ensures that steps, depths, vocabulary sets, buffer items,
    and prediction distributions match identically across JSON serialization.
    """
    filepath = str(tmp_path / "model.json")
    trained_model.depth_list = [2, 4]

    # Pre-save predictions
    trained_model.fill_context(["click", "buy"])
    expected_probas = trained_model.predict_proba()

    trained_model.save_json(filepath)
    assert os.path.exists(filepath)

    loaded_model = TokenTrieModel.load_json(filepath)

    assert loaded_model.step == trained_model.step
    assert loaded_model.max_depth == trained_model.max_depth
    assert loaded_model.depth_list == [2, 4]
    assert loaded_model.known_vocabulary == trained_model.known_vocabulary
    assert loaded_model.skip_decay == trained_model.skip_decay
    assert loaded_model.buffer.to_tuple() == trained_model.buffer.to_tuple()

    # Assert prediction fidelity
    loaded_model.fill_context(["click", "buy"])
    assert loaded_model.predict_proba() == expected_probas


def test_pickle_serialization_deserialization(
    trained_model: TokenTrieModel,
    tmp_path: Path,
) -> None:
    """Verifies binary Pickle round-trips correctly restore model state and fidelity.

    Asserts that serialized models restore all structural counters, vocabulary sets,
    and output distributions identically upon deserialization.
    """
    filepath = str(tmp_path / "model.pkl")

    # Pre-save state and predictions
    trained_model.fill_context(["click", "buy"])
    expected_probas = trained_model.predict_proba()

    trained_model.save(filepath)
    assert os.path.exists(filepath)

    loaded_model = TokenTrieModel.load(filepath)

    assert loaded_model is not None
    assert loaded_model.step == trained_model.step
    assert loaded_model.node_count == trained_model.node_count
    assert loaded_model.vocab_size == trained_model.vocab_size
    assert loaded_model.buffer.to_tuple() == trained_model.buffer.to_tuple()

    # Verify prediction fidelity after unpickling
    loaded_model.fill_context(["click", "buy"])
    assert loaded_model.predict_proba() == expected_probas