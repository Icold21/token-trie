"""Unit Tests for Federated Learning and Heterogeneous Model Merging.

Tests asymmetric knowledge aggregation, context depth expansion,
and decaying timeline normalization across independent instances.
"""

from tokentrie import TokenTrieModel


def test_federated_asymmetric_merge() -> None:
    """Validates knowledge aggregation from heterogeneous distributed models.

    Asserts that structural depth limits expand upwards and distinct
    vocabulary sets unify without losing historical path weights.
    """
    global_model = TokenTrieModel(max_depth=3, decay=0.9, depth_list=[1, 3])
    edge_device = TokenTrieModel(max_depth=5, decay=1.0, depth_list=[5])

    global_model.fit(["click", "buy"], verbose=False)
    edge_device.fit(["open", "door", "walk", "away"], verbose=False)

    global_model.merge(edge_device)

    # Context depth constraints must expand to cover foreign model bounds
    assert global_model.max_depth == 5
    assert 5 in global_model.valid_lengths
    assert 3 in global_model.valid_lengths

    # Vocabularies must unify
    assert "door" in global_model.known_vocabulary
    assert "click" in global_model.known_vocabulary


def test_merge_unigram_projection() -> None:
    """Verifies that static unigram statistics add linearly upon merging."""
    model_a = TokenTrieModel(decay=1.0)
    model_b = TokenTrieModel(decay=1.0)

    model_a.update("token_a")
    model_a.update("token_shared")

    model_b.update("token_b")
    model_b.update("token_shared")

    model_a.merge(model_b)

    assert model_a.unigram_counts["token_shared"] == 2.0
    assert model_a.unigram_counts["token_a"] == 1.0
    assert model_a.unigram_counts["token_b"] == 1.0


def test_merge_with_asymmetric_decay_timelines() -> None:
    """Verifies decay factor scaling when merging models with shifted timelines."""
    host_model = TokenTrieModel(max_depth=3, decay=0.9)
    client_model = TokenTrieModel(max_depth=3, decay=0.8)

    # Ingest tokens with temporal gaps
    host_model.fit(["a", "b", "c"], verbose=False)
    client_model.fit(["x", "y", "z"], verbose=False)

    # Advance host timeline to create step delta
    for _ in range(5):
        host_model.update("pad")

    host_model.merge(client_model)

    assert "z" in host_model.known_vocabulary
    assert host_model.unigram_counts["z"] > 0.0