"""TokenTrie package initialization.

Exports the core Variable Order Markov Model classes and typing structures.
"""

from .core import TokenTrieModel, TokenTrieNode, TokenBuffer, Token

__version__ = "1.0.0"
__all__ = ["TokenTrieModel", "TokenTrieNode", "TokenBuffer", "Token"]