"""Unit Tests for Low-Level Core Components of TokenTrie.

Validates invariant properties, boundary checks, sliding window operations,
and serialization mechanisms of TokenBuffer and TokenTrieNode using public APIs.
"""

import pickle
from typing import Dict, Union

import pytest

from tokentrie import TokenBuffer, TokenTrieNode


def test_token_buffer_initialization() -> None:
    """Verifies default initialization states and invariants of TokenBuffer.

    Ensures that a new buffer starts with zero elements, an empty tuple
    representation, and respects the specified capacity boundary.
    """
    buffer = TokenBuffer(maxlen=3)
    assert buffer.size == 0
    assert buffer.maxlen == 3
    assert buffer.to_tuple() == ()


def test_token_buffer_append_and_overflow() -> None:
    """Tests FIFO eviction, size tracking, and cache invalidation on overflow.

    Appends tokens exceeding capacity and verifies that the oldest elements
    are evicted in O(1) while maintaining cache consistency.
    """
    buffer = TokenBuffer(maxlen=3)
    buffer.append("a")
    buffer.append("b")

    assert buffer.size == 2
    assert buffer.to_tuple() == ("a", "b")

    # Induce sliding overflow
    buffer.append("c")
    buffer.append("d")

    assert buffer.size == 3
    assert buffer.to_tuple() == ("b", "c", "d")


def test_token_buffer_extend_and_clear() -> None:
    """Tests batch extending, tail truncation, and hard clearing mechanics.

    Verifies that extending with an iterable larger than maxlen keeps only
    the most recent elements and that clear() resets all internal state.
    """
    buffer = TokenBuffer(maxlen=3)
    buffer.extend([1, 2, 3, 4])

    assert buffer.size == 3
    assert buffer.to_tuple() == (2, 3, 4)

    buffer.clear()
    assert buffer.size == 0
    assert buffer.to_tuple() == ()


def test_token_buffer_pickle_serialization() -> None:
    """Ensures TokenBuffer state can be pickled and unpickled losslessly.

    Verifies that state dictionaries accurately restore internal contents,
    size counters, and capacity bounds via public accessors.
    """
    buffer = TokenBuffer(maxlen=4)
    buffer.extend(["tok_1", "tok_2", "tok_3"])

    serialized = pickle.dumps(buffer)
    restored: TokenBuffer = pickle.loads(serialized)

    assert restored.size == 3
    assert restored.maxlen == 4
    assert restored.to_tuple() == ("tok_1", "tok_2", "tok_3")


def test_token_trie_node_dict_serialization() -> None:
    """Ensures nested TokenTrieNode hierarchies serialize to JSON-safe dictionaries.

    Tests that token frequency maps, children links, and step timestamps
    are preserved across from_dict() and to_dict() transitions.
    """
    node = TokenTrieNode()
    node.counts["target_token"] = 2.5
    node.last_visit_step = 42

    child = TokenTrieNode()
    child.counts["deep_token"] = 1.0
    node.children["context_token"] = child

    serialized = node.to_dict()
    restored = TokenTrieNode.from_dict(serialized)

    assert restored.counts["target_token"] == 2.5
    assert restored.last_visit_step == 42
    assert "context_token" in restored.children
    assert restored.children["context_token"].counts["deep_token"] == 1.0


def test_token_trie_node_pickle_serialization() -> None:
    """Ensures TokenTrieNode pickle serialization retains nested child links."""
    node = TokenTrieNode()
    node.counts["target"] = 3.0
    node.last_visit_step = 100

    child = TokenTrieNode()
    child.counts["sub_target"] = 1.5
    node.children["sub_context"] = child

    serialized = pickle.dumps(node)
    restored: TokenTrieNode = pickle.loads(serialized)

    assert restored.counts["target"] == 3.0
    assert restored.last_visit_step == 100
    assert "sub_context" in restored.children
    assert restored.children["sub_context"].counts["sub_target"] == 1.5