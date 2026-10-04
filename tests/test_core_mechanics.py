"""Unit Tests for Core Mechanical Operations of TokenTrieModel.

Tests context management, sliding window operations, garbage collection,
dynamic tree restructuring, surgical branch modifications, and token validation.
"""

from typing import Dict, List

import pytest

from tokentrie import TokenTrieModel


def test_token_type_validation(empty_model: TokenTrieModel) -> None:
    """Verifies that the engine strictly enforces 'str' or 'int' token types.

    Asserts that boolean types, floats, and arbitrary objects raise TypeError.
    """
    empty_model.update("valid_string")
    empty_model.update(42)

    with pytest.raises(TypeError, match=r"TokenTrieModel strictly (accepts|requires)"):
        empty_model.update(True)

    with pytest.raises(TypeError, match=r"TokenTrieModel strictly (accepts|requires)"):
        empty_model.update(3.1415)

    with pytest.raises(TypeError, match=r"TokenTrieModel strictly (accepts|requires)"):
        empty_model.update(None)


def test_fit_batch_vs_stream() -> None:
    """Verifies stream learning vs batch sequence boundary isolation.

    Asserts that continuous streams learn transitions across sequence boundaries,
    while batch mode isolates sequences by clearing the working buffer between items.
    """
    model_stream = TokenTrieModel(max_depth=3)
    model_batch = TokenTrieModel(max_depth=3)

    tokens = ["start", "run", "stop", "start", "run", "stop"]
    batches = [["start", "run", "stop"], ["start", "run", "stop"]]

    model_stream.fit(tokens, verbose=False)
    model_batch.fit(batches, verbose=False)

    # Unigram counts and vocabulary cardinality match identically via public properties
    assert model_stream.vocab_size == model_batch.vocab_size
    assert model_stream.unigram_counts == model_batch.unigram_counts

    # Batching isolates sequences (buffer.clear), preventing boundary bleeding (stop -> start).
    # Stream creates additional cross-boundary transitions (10 nodes vs 4 nodes).
    assert model_batch.node_count == 4
    assert model_stream.node_count == 10


def test_context_management(empty_model: TokenTrieModel) -> None:
    """Tests manual working memory manipulation without affecting model weights.

    Verifies that update_context, fill_context, and reset_context manipulate
    the working buffer without updating the underlying Trie or unigram counts.
    """
    empty_model.update_context("ghost_token")
    assert empty_model.buffer.size == 1

    empty_model.fill_context(["new_1", "new_2"])
    assert empty_model.buffer.to_tuple() == ("new_1", "new_2")

    assert "ghost_token" not in empty_model.known_vocabulary
    assert "new_1" not in empty_model.known_vocabulary
    assert empty_model.unigram_counts.get("new_1", 0.0) == 0.0

    empty_model.reset_context()
    assert empty_model.buffer.size == 0


def test_min_depth_constraint() -> None:
    """Verifies that context depths below min_depth do not record or match tree counts."""
    model = TokenTrieModel(min_depth=3, max_depth=5, decay=1.0)
    model.fit(["a", "b", "c", "d"], verbose=False)

    # Sub-depth branches (length 1 and 2) must not store prediction counts
    node_c = model.root.children.get("c")
    assert node_c is not None
    assert node_c.counts == {}

    node_b = node_c.children.get("b")
    assert node_b is not None
    assert node_b.counts == {}

    # Context length 1 ([c]) is below min_depth=3 -> falls back to unigram distribution
    model.fill_context(["c"])
    probas_shallow = model.predict_proba()
    assert probas_shallow["d"] == pytest.approx(0.25, rel=1e-2)

    # Context length 3 ([a, b, c]) >= min_depth -> resolves contextually to 'd'
    # Under v1.1.0 shrinkage on N_c=1, sample support w_node=0.5 yields ~0.75 probability
    model.fill_context(["a", "b", "c"])
    probas_deep = model.predict_proba()
    assert probas_deep["d"] > 0.70
    assert model.predict() == "d"


def test_fixed_garbage_collection() -> None:
    """Verifies fixed interval pruning sweeps sub-threshold transition counts.

    Accelerates timeline decay to force weights below the threshold and ensures
    dead branches and exhausted vocabulary entries are completely purged.
    """
    model = TokenTrieModel(
        decay=0.5,
        pruning_mode="fixed",
        pruning_step=5,
        pruning_threshold=0.1,
    )

    model.update("a")
    model.update("b")
    assert "a" in model.known_vocabulary

    for _ in range(10):
        model.update("noise")

    assert "a" not in model.known_vocabulary
    assert "b" not in model.known_vocabulary
    
    
def test_dynamic_garbage_collection() -> None:
    """Verifies that dynamic GC threshold escalates proportionally to node density."""
    model = TokenTrieModel(
            decay=0.5,
            pruning_mode="dynamic",
            pruning_step=10,
            pruning_threshold=0.1,
    )
    assert model.next_prune_target == 10

    for i in range(25):
        model.update(f"token_{i}")

    # Dynamic threshold target escalates as tree node count exceeds step baseline
    assert model.next_prune_target > 10


def test_lazy_decay_cache_mechanics() -> None:
    """Verifies observable transition decay dynamics as global timeline advances."""
    model = TokenTrieModel(max_depth=2, decay=0.85)

    model.update("a")
    model.update("b")  # 'a' -> 'b' observed at step 2

    # Advance the timeline significantly
    for _ in range(8):
        model.update("pad")

    # Upon re-querying, historical transition 'b' is decayed relative to recent tokens
    model.fill_context(["a"])
    probas = model.predict_proba()
    assert "b" in probas
    assert probas["b"] < 0.50


def test_restructure_tree_depth_trimming() -> None:
    """Verifies physical branch trimming when max_depth is lowered via reconfigure."""
    model = TokenTrieModel(max_depth=5)
    model.fit(["1", "2", "3", "4", "5", "6"], verbose=False)

    old_node_count = model.node_count

    model.reconfigure({"max_depth": 2})

    assert model.max_depth == 2
    assert model.node_count < old_node_count
    assert max(model.valid_lengths) == 2


def test_set_branches_basic_and_overwrite(empty_model: TokenTrieModel) -> None:
    """Verifies explicit surgical injection and full branch replacement.

    Asserts that existing predictive logic can be forcefully overridden
    with custom distribution weights without corrupting ancestor nodes.
    """
    empty_model.set_branches([
        (["hello", "world"], {"!": 5.0, "?": 1.0})
    ])

    empty_model.fill_context(["hello", "world"])
    probas = empty_model.predict_proba()

    assert probas["!"] > probas["?"]
    assert "!" in empty_model.known_vocabulary

    # Overwrite the branch with a new target
    empty_model.set_branches([
        (["hello", "world"], {".": 10.0})
    ])

    probas_updated = empty_model.predict_proba()
    assert "." in probas_updated
    # Target '.' overwhelmingly dominates over legacy unigram smoothed prior of '!'
    assert probas_updated["."] > probas_updated["!"] * 10.0
    assert empty_model.predict() == "."


def test_set_branches_auto_expansion() -> None:
    """Ensures setting a deeper branch automatically upgrades model depth limits."""
    model = TokenTrieModel(max_depth=2)

    model.set_branches([
        ([1, 2, 3, 4, 5], {6: 1.0})
    ])

    assert model.max_depth == 5
    assert 5 in model.valid_lengths
    assert model.buffer.maxlen == 5

    model.fill_context([1, 2, 3, 4, 5])
    assert model.predict() == 6


def test_delete_branches_structural_pruning() -> None:
    """Verifies recursive deletion of target branches and descendant sub-paths."""
    model = TokenTrieModel(max_depth=3)

    model.fit([["a", "b", "c"], ["x", "a", "b", "d"]], verbose=False)

    model.fill_context(["x", "a", "b"])
    assert model.predict() == "d"

    model.delete_branches([["a", "b"]])

    node_b = model.root.children.get("b")
    assert node_b is not None
    assert "a" not in node_b.children