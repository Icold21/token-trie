# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

"""Core Engine for TokenTrieModel.

This module implements the Variable Order Markov Model (VOMM) using a 
Reverse Suffix Trie, accelerated via Cython for high-performance sequential 
inference and online learning.
"""

import json
import pickle
import random
import warnings
from collections import defaultdict, deque
from typing import List, Dict, Optional, Any, Iterable, Union, Tuple, Set

# Import highly optimized C-level mathematical functions
from libc.math cimport log as c_log, exp as c_exp, log1p as c_log1p

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

# Strict typing for the internal engine
Token = Union[str, int]


class TokenBuffer:
    """Optimized stateful buffer for sliding window operations.

    Wraps `collections.deque` and maintains a manual size counter to bypass 
    O(N) operations when continuously checking size or converting to tuples 
    in hot loops.

    Attributes:
        _maxlen (int): The maximum capacity of the buffer. Range: [1, inf).
        _deque (deque): Double-ended queue storing the token history.
        _cache_tuple (Optional[Tuple[Token, ...]]): Cached tuple representation.
        _size (int): Current number of elements inside the buffer.
    """
    __slots__ = ['_maxlen', '_deque', '_cache_tuple', '_size']

    def __init__(self, int maxlen):
        """Initializes the token buffer.

        Args:
            maxlen (int): The maximum capacity of the buffer. Range: [1, inf).
        """
        self._maxlen = maxlen
        self._deque = deque(maxlen=maxlen)
        self._cache_tuple = None
        self._size = 0  

    def append(self, item: Token):
        """Appends a token, invalidating the tuple cache. O(1).
        
        Args:
            item (Token): The token to add to the buffer.
        """
        self._deque.append(item)
        # Invalidate cache on modification
        self._cache_tuple = None
        if self._size < self._maxlen:
            self._size += 1

    def extend(self, items: Iterable[Token]):
        """Extends the buffer with multiple tokens and updates the size.
        
        Args:
            items (Iterable[Token]): Tokens to add.
        """
        self._deque.extend(items)
        # Invalidate cache on batch modification
        self._cache_tuple = None
        self._size = len(self._deque)

    def clear(self):
        """Clears the buffer and resets all internal trackers."""
        self._deque.clear()
        self._cache_tuple = None
        self._size = 0

    @property
    def size(self) -> int:
        """Returns the current size of the buffer. O(1)."""
        return self._size

    def to_tuple(self) -> Tuple[Token, ...]:
        """Returns an immutable tuple representation of the buffer."""
        if self._cache_tuple is None:
            self._cache_tuple = tuple(self._deque)
        return self._cache_tuple

    def __getstate__(self) -> Dict[str, Union[int, deque]]:
        return {'_maxlen': self._maxlen, '_deque': self._deque, '_size': self._size}

    def __setstate__(self, state: Dict[str, Union[int, deque]]):
        self._maxlen = state.get('_maxlen', 10)
        self._deque = state.get('_deque', deque(maxlen=self._maxlen))
        self._size = state.get('_size', len(self._deque))
        self._cache_tuple = None


class TokenTrieNode:
    """Lightweight Node for the Suffix Trie structure.

    Attributes:
        counts (Dict[Token, float]): Map from a predicted next token to its frequency.
        children (Dict[Token, TokenTrieNode]): Subtrees representing preceding contexts.
        last_visit_step (int): Global timeline step of the last update to this node.
    """
    __slots__ = ['counts', 'children', 'last_visit_step']
    
    def __init__(self):
        """Initializes an empty Trie node."""
        self.counts = defaultdict(float) 
        self.children = {}
        self.last_visit_step = 0

    def to_dict(self) -> Dict[str, Union[Dict, int]]:
        """Serializes the node into a JSON-safe dictionary."""
        return {
            'c': dict(self.counts),
            'ch': {str(k): v.to_dict() for k, v in self.children.items()},
            'lvs': self.last_visit_step
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Union[Dict, int]]) -> 'TokenTrieNode':
        """Deserializes a node from a state dictionary."""
        node = cls()
        node.counts = defaultdict(float, data.get('c', {}))
        node.children = {k: cls.from_dict(v) for k, v in data.get('ch', {}).items()}
        node.last_visit_step = data.get('lvs', 0)
        return node

    def __getstate__(self) -> Dict[str, Union[Dict, int]]:
        return {'counts': self.counts, 'children': self.children, 'last_visit_step': self.last_visit_step}

    def __setstate__(self, state: Dict[str, Union[Dict, int]]):
        self.counts = state.get('counts', defaultdict(float))
        self.children = state.get('children', {})
        self.last_visit_step = state.get('last_visit_step', 0)


class TokenTrieModel:
    """Variable-order Markov Model utilizing a Reverse Suffix Trie.

    Provides O(N) context traversal, lazy weight decay, structural reconfiguration,
    Katz-style backoff fallback, and asymmetric federated merging.

    Attributes:
        max_depth (int): Maximum depth of context (Markov order). Range: [1, inf).
        min_depth (int): Minimum context length required for a valid match. Range: [1, max_depth].
        depth_list (Optional[List[int]]): Specific depths tracked outside [min_depth, max_depth].
        decay (float): Exponential forgetting rate. Range: [0.0, 1.0].
        skip_decay (bool): Accelerator flag. True if decay is 1.0 or None.
        valid_lengths (Set[int]): Precomputed set of active context depths.
    """

    def __init__(self, 
                 max_depth: Optional[int] = 10, 
                 min_depth: Optional[int] = 1, 
                 depth_list: Optional[List[int]] = None,
                 decay: Optional[float] = 0.99, 
                 alphabet_autoscale: bool = True,
                 fallback_mode: str = 'katz_backoff',
                 pruning_mode: str = 'fixed',
                 pruning_step: int = 1000,
                 pruning_threshold: float = 1e-6,
                 max_beams: int = 1000,
                 cache_size: int = 4096):
        """Initializes the sequence model with hyperparameter constraints.

        Args:
            max_depth (Optional[int]): Maximum depth of context. Range: [1, inf).
                If None, defaults to max(depth_list) or min_depth. If all N params are None, defaults to 1.
            min_depth (Optional[int]): Minimum context length required. Range: [1, inf).
                If None, defaults to 1.
            depth_list (Optional[List[int]]): Specific context lengths to track.
                If provided, expands tracked depths. If None, ignored.
            decay (Optional[float]): Exponential forgetting rate. Range: [0.0, 1.0].
                If None or 1.0, lazy decay is strictly bypassed for maximum performance.
            alphabet_autoscale (bool): Enables dynamic entropy scaling. Defaults to True.
            fallback_mode (str): Smoothing mode in {'katz_backoff', 'uniform'}.
            pruning_mode (str): GC strategy in {'fixed', 'dynamic'}.
            pruning_step (int): Pruning step target. Range: [1, inf).
            pruning_threshold (float): Limit below which weights are deleted. Range: [0.0, inf).
            max_beams (int): Bounded path limit in masked search. Range: [1, inf).
            cache_size (int): Math cache capacity limit. Range: [1, inf).
        """
        # Automatically infer N-limits if completely unprovided
        if max_depth is None and min_depth is None and depth_list is None:
            max_depth, min_depth = 1, 1
            
        if max_depth is None:
            max_depth = max(depth_list) if depth_list else (min_depth if min_depth is not None else 1)
            
        if min_depth is None and not depth_list:
            min_depth = 1
            
        # Ensure max_depth covers the deepest target requested by depth_list
        if depth_list is not None and max_depth < max(depth_list):
            warnings.warn(f"Invalid max_depth. Adjusting to max(depth_list).")
            max_depth = max(depth_list)
            
        if not isinstance(max_depth, int) or max_depth < 1: max_depth = 10
        if min_depth is not None and (not isinstance(min_depth, int) or min_depth < 1): min_depth = 1
        if decay is not None and (not isinstance(decay, (int, float)) or not (0.0 <= decay <= 1.0)): decay = 0.99

        if fallback_mode not in {'katz_backoff', 'uniform'}: fallback_mode = 'katz_backoff'
        if pruning_mode not in {'fixed', 'dynamic'}: pruning_mode = 'fixed'

        self.max_depth = max_depth
        self.min_depth = min_depth
        self.depth_list = depth_list
        self.decay = decay
        
        # Fast path execution flag: disables heavy mathematical timeline tracking
        self.skip_decay = (self.decay is None or self.decay == 1.0)
        
        self.alphabet_autoscale = alphabet_autoscale
        self.fallback_mode = fallback_mode
        self.pruning_mode = pruning_mode
        self.pruning_step = max(1, pruning_step)
        self.pruning_threshold = max(0.0, pruning_threshold)
        self.max_beams = max(1, max_beams)
        self.cache_size = max(1, cache_size)
        
        self.reset()

    def _update_valid_lengths(self):
        """Precalculates valid depths (n-grams) to avoid continuous runtime checks."""
        self.valid_lengths = set()
        if self.min_depth is not None and self.max_depth is not None:
            self.valid_lengths.update(range(self.min_depth, self.max_depth + 1))
        if self.depth_list is not None:
            self.valid_lengths.update(self.depth_list)

    def reset(self) -> 'TokenTrieModel':
        """Resets the model back to an empty initial state."""
        self.root = TokenTrieNode()
        self.buffer = TokenBuffer(maxlen=self.max_depth) 
        self.step = 0
        self.known_vocabulary = set()
        
        self.unigram_counts = defaultdict(float)
        self.unigram_last_update = defaultdict(int)

        self._node_count = 1
        self._next_prune_target = self.pruning_step
        self._vocab_len = 0
        self._last_computed_vocab_len = 0
        self._cached_log_base = 0.6931471805599453

        self._power_cache = {}
        self._power_cache_len = 0
        # Precompute decay logarithm if active
        self.log_decay = c_log(self.decay) if not self.skip_decay and self.decay > 0 else -float('inf')

        self._int_log_cache = {}
        self._log_cache_len = 0
        
        self._update_valid_lengths()
        return self

    def reconfigure(self, params: Dict[str, Any]) -> 'TokenTrieModel':
        """Dynamically reconfigures model parameters and restructures the internal tree."""
        old_decay = self.decay
        old_skip = self.skip_decay
        new_decay = params.get('decay', self.decay)
        new_skip = (new_decay is None or new_decay == 1.0)
        
        # If decay rate modifies, forcefully evaluate and bake historic weights 
        if old_decay != new_decay or old_skip != new_skip:
            self.prune_tree() 
            
        # Completely flush dynamic math caches to avoid numerical bleeding 
        self._power_cache.clear()
        self._power_cache_len = 0
        self._int_log_cache.clear()
        self._log_cache_len = 0
        self._last_computed_vocab_len = 0
        self._cached_log_base = 0.6931471805599453
            
        if 'decay' in params:
            self.decay = params['decay']
            self.skip_decay = new_skip
            self.log_decay = c_log(self.decay) if not self.skip_decay and self.decay > 0 else -float('inf')
            
        # Adjust structural depth bounds and reconstruct valid lengths
        if any(k in params for k in ('max_depth', 'min_depth', 'depth_list')):
            new_max_depth = params.get('max_depth', self.max_depth)
            new_min_depth = params.get('min_depth', getattr(self, 'min_depth', None))
            new_depth_list = params.get('depth_list', getattr(self, 'depth_list', None))
            
            if new_max_depth is None and new_min_depth is None and new_depth_list is None:
                new_max_depth, new_min_depth = 1, 1
            if new_max_depth is None:
                new_max_depth = max(new_depth_list) if new_depth_list else (new_min_depth if new_min_depth is not None else 1)
            if new_min_depth is None and not new_depth_list:
                new_min_depth = 1
                
            if new_depth_list is not None and new_max_depth < max(new_depth_list):
                new_max_depth = max(new_depth_list)
                
            self.max_depth = new_max_depth
            self.min_depth = new_min_depth
            self.depth_list = new_depth_list
            self._update_valid_lengths()
            
            # Physically restructure and truncate invalid branches inside the Trie
            self._restructure_tree(self.root, 0)
            self._node_count = self._count_nodes(self.root)
            
            # Resize token buffer safely utilizing positive offsets
            if self.buffer._maxlen != self.max_depth:
                old_items = self.buffer.to_tuple()
                new_buffer = TokenBuffer(maxlen=self.max_depth)
                if self.buffer.size > self.max_depth:
                    new_buffer.extend(old_items[len(old_items) - self.max_depth:])
                else:
                    new_buffer.extend(old_items)
                self.buffer = new_buffer
                
        # Remainder parameters are mapped directly
        for k in ['alphabet_autoscale', 'fallback_mode', 'pruning_mode', 'pruning_step', 'pruning_threshold', 'max_beams', 'cache_size']:
            if k in params:
                setattr(self, k, params[k])
                
        return self

    def _restructure_tree(self, node, int current_depth):
        """Recursively removes node counts out of valid depths and prunes excess tree depth."""
        # Clean local predictions if this precise depth is unrequested
        if current_depth > 0 and current_depth not in self.valid_lengths:
            node.counts.clear()
            
        # Hard truncate branches exceeding absolute depth boundary
        if current_depth >= self.max_depth:
            node.children.clear()
            return
            
        for child in node.children.values():
            self._restructure_tree(child, current_depth + 1)
            
    def _count_nodes(self, node) -> int:
        """Utility to deeply recount nodes accurately post-reconfiguration."""
        return 1 + sum(self._count_nodes(c) for c in node.children.values())
    
    @property
    def log_scaling_base(self) -> float:
        """Retrieves dynamic scaling log base mapped to vocabulary entropy."""
        if not self.alphabet_autoscale:
            return 0.6931471805599453
        
        # Calculate vocabulary entropy modifier utilizing C-level logarithm
        if self._vocab_len != self._last_computed_vocab_len:
             self._last_computed_vocab_len = self._vocab_len
             self._cached_log_base = c_log(max(2, self._vocab_len))
             
        return self._cached_log_base

    def _get_decay_factor(self, int delta) -> float:
        """Retrieves or computes exponential decay multiplier mapped by time deltas."""
        cdef double val
        if self.decay <= 0: 
            return 0.0
        # Serve immediately from RAM bounds
        if delta in self._power_cache:
            return self._power_cache[delta]
            
        # Compute and conditionally cache heavy exponentiation
        val = self.decay ** delta
        if self._power_cache_len < self.cache_size:
            self._power_cache[delta] = val
            self._power_cache_len += 1
        return val

    def _get_log_count(self, double count) -> float:
        """Optimized logarithm fetcher preferring integers for fast caching."""
        cdef int ix
        cdef double val
        if count <= 1.0: 
            return 0.0 
        
        # Exploit integer frequencies commonly encountered in Markov tracks
        if count.is_integer():
            ix = int(count)
            if ix in self._int_log_cache:
                return self._int_log_cache[ix]
            
            val = c_log(count)
            if self._log_cache_len < self.cache_size:
                self._int_log_cache[ix] = val
                self._log_cache_len += 1
            return val
            
        return c_log(count)

    def _prune_recursive(self, node, int current_step) -> int:
        """Deep recursion to accurately apply late decay and strip dead edges."""
        cdef int delta, child_survivors, surviving_nodes
        cdef double decay_factor, real_count
        
        # Apply mathematical fade scaling and identify exhausted prediction targets
        if not self.skip_decay:
            delta = current_step - node.last_visit_step
            decay_factor = self._get_decay_factor(delta) if delta > 0 else 1.0
            
            keys_to_remove = []
            for token, count in node.counts.items():
                real_count = count * decay_factor
                if real_count < self.pruning_threshold:
                    keys_to_remove.append(token)
                else:
                    node.counts[token] = real_count
                    
            for token in keys_to_remove: 
                del node.counts[token]
        else:
            # Bypass math entirely and sweep static structural debris
            keys_to_remove = [t for t, c in node.counts.items() if c < self.pruning_threshold]
            for token in keys_to_remove:
                del node.counts[token]
                
        node.last_visit_step = current_step
        empty_children = []
        surviving_nodes = 1
        
        # Recursively penetrate children extracting survivor layouts
        for token, child in node.children.items():
            child_survivors = self._prune_recursive(child, current_step)
            # Remove branch structure unconditionally if entirely depleted
            if not child.counts and not child.children:
                empty_children.append(token)
            else:
                surviving_nodes += child_survivors
                
        for token in empty_children: 
            del node.children[token]
            
        return surviving_nodes

    def prune_tree(self):
        """Forces a Garbage Collection pass across Trie structure and Unigrams."""
        cdef int surviving_nodes = self._prune_recursive(self.root, self.step)
        cdef int delta
        cdef double val
        
        self._node_count = surviving_nodes
        keys_to_remove = []
        
        # Evaluate global fallback unigrams against threshold limits
        if not self.skip_decay:
            for t, c in self.unigram_counts.items():
                delta = self.step - self.unigram_last_update.get(t, 0)
                val = c * (self._get_decay_factor(delta) if delta > 0 else 1.0)
                if val < self.pruning_threshold:
                    keys_to_remove.append(t)
                else:
                    self.unigram_counts[t] = val
                    self.unigram_last_update[t] = self.step
        else:
            for t, c in self.unigram_counts.items():
                if c < self.pruning_threshold:
                    keys_to_remove.append(t)
                else:
                    self.unigram_last_update[t] = self.step

        for t in keys_to_remove:
            del self.unigram_counts[t]
            del self.unigram_last_update[t]
            self.known_vocabulary.discard(t)
            
        self._vocab_len = len(self.known_vocabulary)

        if self.pruning_mode == 'dynamic':
            self._next_prune_target = max(self.pruning_step, int(self._node_count * 1.5))

    def _get_context_nodes(self, str mode, tuple buffer_tuple, int max_depth) -> List[Tuple[TokenTrieNode, int]]:
        """O(N) memory-safe path search algorithm avoiding external allocations."""
        cdef int i, eff_len, depth, phase, match_len, beam_iters
        cdef int buf_len = len(buffer_tuple)
        
        if max_depth == 0 or buf_len == 0: 
            return []
        
        visited = {}
        curr_node = self.root
        
        # Extract direct exact sequential match utilizing reversed indexing mapping
        for i in range(max_depth):
            token = buffer_tuple[buf_len - 1 - i]
            if token not in curr_node.children: 
                break
            curr_node = curr_node.children[token]
            eff_len = i + 1
            if eff_len in self.valid_lengths:
                visited[id(curr_node)] = (curr_node, eff_len)
                
        # Handle combinatorial and linear token-skip masking searches
        if mode in ('linear', 'squared'):
            queue = deque([(self.root, 0, 0, 0)]) if mode == 'linear' else deque([(self.root, 0, 0)])
            beam_iters = 0
            
            while queue:
                beam_iters += 1
                if beam_iters > self.max_beams:
                    break
                    
                if mode == 'linear':
                    curr_node, depth, phase, eff_len = queue.popleft()
                else:
                    curr_node, depth, eff_len = queue.popleft()
                
                # Register valid sub-pattern alignments overwriting shallow hits
                if depth > 0 and eff_len in self.valid_lengths:
                    nid = id(curr_node)
                    if eff_len > visited.get(nid, (None, -1))[1]: 
                        visited[nid] = (curr_node, eff_len)
                        
                if depth == max_depth: 
                    continue
                    
                target_token = buffer_tuple[buf_len - 1 - depth]
                
                # Linearly limit wildcard evaluation or run squared branching expansion
                if mode == 'linear':
                    if phase == 0:
                        for t, child in curr_node.children.items():
                            queue.append((child, depth + 1, 0, eff_len))
                            if t == target_token:
                                queue.append((child, depth + 1, 1, eff_len + 1))
                    else: 
                        if target_token in curr_node.children:
                            queue.append((curr_node.children[target_token], depth + 1, 1, eff_len + 1))
                else:
                    for t, child in curr_node.children.items():
                        match_len = eff_len + 1 if t == target_token else eff_len
                        queue.append((child, depth + 1, match_len))
                        
        return list(visited.values())

    def _validate_inference_params(self, temperature, top_k, top_p, masked_mode):
        """Safely coerces generation arguments to their fallback algebraic equivalents."""
        temp = 0.0 if temperature in (None, "none", "None") else temperature
        k = 0 if top_k in (None, "none", "None") else top_k
        p = 1.0 if top_p in (None, "none", "None") else top_p
        
        if not isinstance(temp, (int, float)) or temp < 0.0: temp = 0.0
        if not isinstance(k, int) or k < 0: k = 0
        if not isinstance(p, (int, float)) or not (0.0 <= p <= 1.0): p = 1.0
        if masked_mode not in {'none', 'linear', 'squared'}: masked_mode = 'none'

        return temp, k, p, masked_mode

    def predict_proba(self, 
                      temperature: Union[float, str, None] = "none", 
                      top_k: Union[int, str, None] = "none", 
                      top_p: Union[float, str, None] = "none", 
                      masked_mode: str = 'none', 
                      *, 
                      return_log_scores: bool = False, 
                      _validated: bool = False) -> Dict[Token, float]:
        """Computes probability distribution for next token given actual buffer state.

        Args:
            temperature (Union[float, str, None]): Distribution flatness. Defaults to "none".
            top_k (Union[int, str, None]): Filters out below K rank. Defaults to "none".
            top_p (Union[float, str, None]): Nucleus sampling bound. Defaults to "none".
            masked_mode (str): Evaluation mode. Values in {'none', 'linear', 'squared'}.
            return_log_scores (bool): If True, yields unnormalized raw logits. Defaults to False.
            _validated (bool): internal bypass flag. Defaults to False.

        Returns:
            Dict[Token, float]: Probabilities per token.
        """
        cdef int hist_len, max_depth, current_step, delta, length, i
        cdef double log_scale_base, log_decay_val, node_factor, count, log_weight, curr, max_log, total_sum, val, factor
        
        if not _validated:
            temperature, top_k, top_p, masked_mode = self._validate_inference_params(temperature, top_k, top_p, masked_mode)

        hist_len = self.buffer.size
        if hist_len == 0: 
            return {}

        candidate_log_scores = defaultdict(lambda: -float('inf'))
        log_scale_base = self.log_scaling_base
        log_decay_val = self.log_decay
        current_step = self.step
        
        max_depth = min(self.max_depth, hist_len)
        buffer_tup = self.buffer.to_tuple()
        
        valid_nodes = self._get_context_nodes(masked_mode, buffer_tup, max_depth)
        found_pattern = False
        
        # Accumulate scaled log potentials bridging context weights and timeline degradation
        for node, length in valid_nodes:
            if not self.skip_decay:
                delta = current_step - node.last_visit_step
                node_factor = (delta * log_decay_val) + (length * log_scale_base)
            else:
                node_factor = length * log_scale_base
            
            for t, count in node.counts.items():
                if count <= 1e-9: 
                    continue
                found_pattern = True
                
                log_weight = self._get_log_count(count) + node_factor
                curr = candidate_log_scores[t]
                
                if curr == -float('inf'):
                    candidate_log_scores[t] = log_weight
                else:
                    # Execute mathematically stable LogSumExp arithmetic mapping via C runtime
                    if curr > log_weight: 
                        candidate_log_scores[t] = curr + c_log1p(c_exp(log_weight - curr))
                    else: 
                        candidate_log_scores[t] = log_weight + c_log1p(c_exp(curr - log_weight))

        # Gracefully handle blind sequences deploying backoff mechanisms
        if not found_pattern:
            if self._vocab_len == 0: 
                return {}
            
            if self.fallback_mode == 'katz_backoff' and self.unigram_counts:
                for t, c in self.unigram_counts.items():
                    if not self.skip_decay:
                        delta = current_step - self.unigram_last_update.get(t, 0)
                        factor = self._get_decay_factor(delta) if delta > 0 else 1.0
                        val = c * factor
                    else:
                        val = c
                        
                    if val > 1e-9:
                        candidate_log_scores[t] = c_log(val)
            else:
                prob = 1.0 / self._vocab_len
                log_prob = c_log(prob)
                for tk in self.known_vocabulary:
                    candidate_log_scores[tk] = log_prob

        # Deploy distribution temperature scaling if demanded
        if temperature != 1.0 and temperature > 1e-4:
            for t in candidate_log_scores: 
                candidate_log_scores[t] /= temperature

        # Halt directly mapping pre-normalized values
        if return_log_scores:
            return dict(sorted(candidate_log_scores.items(), key=lambda x: x[1], reverse=True))

        max_log = max(candidate_log_scores.values())
        linear_scores = {}
        total_sum = 0.0
        
        # Softmax normalization loop shifting values securing exponential stability
        for token, log_score in candidate_log_scores.items():
            val = c_exp(log_score - max_log)
            linear_scores[token] = val
            total_sum += val
            
        probas = {t: v / total_sum for t, v in linear_scores.items()}
        
        if top_k <= 0 and top_p >= 1.0:
            return dict(sorted(probas.items(), key=lambda x: x[1], reverse=True))

        sorted_items = sorted(probas.items(), key=lambda x: x[1], reverse=True)
        
        # Execute Top-K truncation 
        if 0 < top_k < len(sorted_items): 
            sorted_items = sorted_items[:top_k]

        # Execute Top-P Nucleus Sampling logic dynamically capturing cumulative mass bounds
        if top_p < 1.0:
            target_prob = top_p * sum(prob for _, prob in sorted_items) 
            cumulative_prob = 0.0
            for i, (_, prob) in enumerate(sorted_items):
                cumulative_prob += prob
                if cumulative_prob >= target_prob:
                    sorted_items = sorted_items[:i + 1]
                    break

        new_total = sum(prob for _, prob in sorted_items)
        if new_total > 0: 
            return {tk: prob / new_total for tk, prob in sorted_items}
        return dict(sorted_items)

    def predict(self, 
                temperature: Union[float, str, None] = "none", 
                top_k: Union[int, str, None] = "none", 
                top_p: Union[float, str, None] = "none", 
                masked_mode: str = 'none') -> Optional[Token]:
        """Samples a token from internal distributions utilizing generation controls.

        Args:
            temperature (Union[float, str, None]): Adjusts creative variance. Defaults to "none" (Greedy).
            top_k (Union[int, str, None]): Truncates tail values. Defaults to "none".
            top_p (Union[float, str, None]): Truncates accumulated mass. Defaults to "none".
            masked_mode (str): Wildcard evaluation parameter. Defaults to 'none'.

        Returns:
            Optional[Token]: Real token sampled or None if stream is fully empty.
        """
        temperature, top_k, top_p, masked_mode = self._validate_inference_params(temperature, top_k, top_p, masked_mode)
        # Exploit Argmax shortcuts avoiding structural randomness routines
        if temperature < 1e-4:
            probas = self.predict_proba(temperature=1.0, top_k=top_k, top_p=top_p, masked_mode=masked_mode, _validated=True)
            if not probas: return None
            return max(probas, key=probas.get)
        
        probas = self.predict_proba(temperature=temperature, top_k=top_k, top_p=top_p, masked_mode=masked_mode, _validated=True)
        if not probas: return None
        return random.choices(list(probas.keys()), weights=list(probas.values()), k=1)[0]

    def _validate_token(self, token: Any):
        """Halts instantly upon sensing unmapped data types inside the token pipe."""
        if not isinstance(token, (str, int)) or isinstance(token, bool):
            raise TypeError(f"TokenTrieModel strictly accepts 'str' or 'int' tokens. Got: {type(token).__name__}")

    def update(self, actual: Token):
        """Consumes a sequential target token updating relevant associative paths.

        Args:
            actual (Token): The recently resolved outcome value.
        """
        cdef int current_step, hist_len, i, delta_uni, delta
        cdef double factor, new_val
        
        self._validate_token(actual)
        self.step += 1
        current_step = self.step
        
        if actual not in self.known_vocabulary:
            self.known_vocabulary.add(actual)
            self._vocab_len += 1
            
        # Update Unigram fallback tracking structures directly scaling delayed weights
        if not self.skip_decay:
            delta_uni = current_step - self.unigram_last_update.get(actual, 0)
            if delta_uni > 0 and actual in self.unigram_counts:
                self.unigram_counts[actual] *= self._get_decay_factor(delta_uni)
        self.unigram_counts[actual] += 1.0
        self.unigram_last_update[actual] = current_step
        
        hist_len = self.buffer.size
        history_tuple = self.buffer.to_tuple()
        node = self.root
        
        # Traverse suffix layout constructing new branches and assigning actual hits
        for i in range(1, min(self.max_depth, hist_len) + 1):
            token = history_tuple[hist_len - i]
            
            if token not in node.children:
                node.children[token] = TokenTrieNode()
                self._node_count += 1
            node = node.children[token]
            
            # Sub-threshold evaluation evaluating late structural decay scaling
            if not self.skip_decay and node.last_visit_step != 0:
                delta = current_step - node.last_visit_step
                if delta > 0:
                    factor = self._get_decay_factor(delta)
                    keys_to_remove = []
                    for t, c in node.counts.items():
                        new_val = c * factor
                        if new_val < self.pruning_threshold: 
                            keys_to_remove.append(t)
                        else: 
                            node.counts[t] = new_val
                    for t in keys_to_remove: 
                        del node.counts[t]
            
            node.last_visit_step = current_step
            # Safely avoid memory inflation ignoring skipped distinct context distances
            if i in self.valid_lengths:
                node.counts[actual] += 1.0
            
        self.buffer.append(actual)
        
        if self.pruning_mode == 'fixed' and self.step % self.pruning_step == 0: 
            self.prune_tree()
        elif self.pruning_mode == 'dynamic' and self._node_count >= self._next_prune_target:
            self.prune_tree()

    def fit(self, X: Union[Iterable[Token], Iterable[Iterable[Token]]], verbose: bool = True) -> 'TokenTrieModel':
        """Sequentially passes batch sequence tuples or straight stream data.

        Args:
            X (Iterable): Train target structures.
            verbose (bool): Utilizes tqdm tracking when enabled. Defaults to True.

        Returns:
            TokenTrieModel: Self reference.
        """
        is_batch = False
        if hasattr(X, '__len__') and len(X) > 0:
            first_element = next(iter(X))
            if isinstance(first_element, (list, tuple)) or (hasattr(first_element, '__iter__') and not isinstance(first_element, (str, bytes))):
                is_batch = True

        iterator = X
        if verbose and _tqdm:
            total = len(X) if hasattr(X, '__len__') else None
            iterator = _tqdm(X, total=total, desc="TMP Fitting", unit="seq" if is_batch else "tok")

        if is_batch:
            for sequence in iterator:
                self.buffer.clear()
                for token in sequence: 
                    self.update(token)
        else:
            for token in iterator: 
                self.update(token)
        return self

    def set_branches(self, branches: Iterable[Tuple[Iterable[Token], Dict[Token, float]]]) -> 'TokenTrieModel':
        """Explicitly adds or overwrites target predictions for specific context paths.

        If the context branch already exists, its prediction weights are completely replaced. 
        If it does not exist, the branch is created. The model will automatically 
        adjust its depth boundaries (max_depth, depth_list) if the inserted context is deeper.

        Args:
            branches: An iterable of (context_sequence, prediction_weights).
                Example: [ (["open", "door"], {"walk": 10.0, "look": 2.5}) ]
                
        Returns:
            TokenTrieModel: Self reference for method chaining.
        """
        cdef int needs_reconfig = 0
        cdef int max_new_len = self.max_depth
        cdef int seq_len, i, delta
        cdef double fl_weight
        
        new_depth_list = set(self.depth_list) if self.depth_list else set()
        branch_list = list(branches)
        
        # 1. Pre-scan to conditionally identify boundaries expanding model structural limits
        for seq, counts in branch_list:
            seq_len = len(list(seq))
            if seq_len > 0 and seq_len not in self.valid_lengths:
                needs_reconfig = 1
                new_depth_list.add(seq_len)
                if seq_len > max_new_len:
                    max_new_len = seq_len
                    
        # Apply bounds expansion prior to inserting underlying Trie logic
        if needs_reconfig:
            self.reconfigure({'max_depth': max_new_len, 'depth_list': list(new_depth_list)})
            
        # 2. Iteratively inject specific context paths mapping target dependencies
        for seq, counts in branch_list:
            seq_list = list(seq)
            seq_len = len(seq_list)
            if seq_len == 0:
                continue
                
            node = self.root
            for i in range(seq_len):
                token = seq_list[seq_len - 1 - i]
                self._validate_token(token)
                self.known_vocabulary.add(token)
                
                if token not in node.children:
                    node.children[token] = TokenTrieNode()
                    self._node_count += 1
                node = node.children[token]
                
            # Intentionally overwrite legacy internal targets rendering explicit knowledge mapping
            node.counts.clear()
            for target_token, weight in counts.items():
                self._validate_token(target_token)
                fl_weight = float(weight)
                node.counts[target_token] = fl_weight
                self.known_vocabulary.add(target_token)
                
                # Sync localized unigram timeline validating Katz Backoff integrity 
                if target_token not in self.unigram_counts:
                    self.unigram_counts[target_token] = fl_weight
                    self.unigram_last_update[target_token] = self.step
                else:
                    if not self.skip_decay:
                        delta = self.step - self.unigram_last_update.get(target_token, 0)
                        if delta > 0:
                            self.unigram_counts[target_token] *= self._get_decay_factor(delta)
                    self.unigram_counts[target_token] += fl_weight
                    self.unigram_last_update[target_token] = self.step
                
            node.last_visit_step = self.step
            
        self._vocab_len = len(self.known_vocabulary)
        return self

    def delete_branches(self, sequences: Iterable[Iterable[Token]]):
        """Removes specific contextual branches and all their descending sub-paths.

        Args:
            sequences (Iterable[Iterable[Token]]): A list of sequences to delete.
                Example: [["click", "buy"]] deletes the exact ["click", "buy"] 
                context and any deeper contexts (like ["ad", "click", "buy"]).
        """
        cdef int nodes_removed = 0
        cdef int seq_len, i
        cdef bint path_exists
        
        for seq in sequences:
            seq_list = list(seq)
            seq_len = len(seq_list)
            if seq_len == 0:
                continue
                
            node = self.root
            path_exists = True
            
            # Navigate internal reversed path mapping up to the parent block exclusively
            for i in range(seq_len - 1):
                token = seq_list[seq_len - 1 - i]
                if token not in node.children:
                    path_exists = False
                    break
                node = node.children[token]
                
            # Physically isolate and destroy branch subtrees safely traversing the trie architecture
            if path_exists:
                target_token = seq_list[0]
                if target_token in node.children:
                    del node.children[target_token]
                    nodes_removed = 1
                    
        if nodes_removed:
            self._node_count = self._count_nodes(self.root)

    def _merge_recursive(self, node_self, node_other, int current_step_self, int current_step_other, other_model):
        """Applies mathematical step transformation to assimilate independent foreign models."""
        cdef int delta_self, delta_other
        cdef double factor_self, factor_other
        
        if not self.skip_decay:
            delta_self = current_step_self - node_self.last_visit_step
            factor_self = self._get_decay_factor(delta_self) if delta_self > 0 else 1.0
        else: factor_self = 1.0
        
        if not other_model.skip_decay:
            delta_other = current_step_other - node_other.last_visit_step
            factor_other = other_model._get_decay_factor(delta_other) if delta_other > 0 else 1.0
        else: factor_other = 1.0
        
        for t in list(node_self.counts.keys()): 
            node_self.counts[t] *= factor_self
            
        for t, c in node_other.counts.items():
            node_self.counts[t] += (c * factor_other)
            
        node_self.last_visit_step = current_step_self

        for t, child_other in node_other.children.items():
            if t not in node_self.children:
                node_self.children[t] = TokenTrieNode()
            self._merge_recursive(node_self.children[t], child_other, current_step_self, current_step_other, other_model)

    def merge(self, other: 'TokenTrieModel') -> 'TokenTrieModel':
        """Aggregates an asymmetrical parallel model's weights and capabilities.

        Args:
            other (TokenTrieModel): A separate instance to integrate.

        Returns:
            TokenTrieModel: Self reference augmented.
        """
        # Accommodate structural constraints dynamically extending boundaries upwards
        if other.max_depth > self.max_depth:
            self.reconfigure({'max_depth': other.max_depth})
            
        if getattr(other, 'min_depth', None) is not None and (self.min_depth is None or other.min_depth < self.min_depth):
            self.min_depth = other.min_depth
            
        if getattr(other, 'depth_list', None) is not None:
            if self.depth_list is None: self.depth_list = list(other.depth_list)
            else: self.depth_list = list(set(self.depth_list + other.depth_list))
            
        self._update_valid_lengths()
        self.known_vocabulary.update(other.known_vocabulary)
        
        cdef int delta_other, delta_self
        cdef double true_weight_other, true_weight_self
        
        # Merge individual unigram baseline statistics accurately assessing external decaying states
        for t, c in other.unigram_counts.items():
            true_weight_other = c
            if not other.skip_decay:
                delta_other = other.step - other.unigram_last_update.get(t, 0)
                true_weight_other *= (other._get_decay_factor(delta_other) if delta_other > 0 else 1.0)
            
            true_weight_self = self.unigram_counts.get(t, 0.0)
            if not self.skip_decay:
                delta_self = self.step - self.unigram_last_update.get(t, 0)
                true_weight_self *= (self._get_decay_factor(delta_self) if delta_self > 0 else 1.0)
            
            self.unigram_counts[t] = true_weight_self + true_weight_other
            self.unigram_last_update[t] = self.step

        self._merge_recursive(self.root, other.root, self.step, other.step, other)
        self.prune_tree()
        return self

    def update_context(self, token: Token): 
        """Pushes sequential element silently without training model weights."""
        self._validate_token(token)
        self.buffer.append(token)
        
    def fill_context(self, context: Iterable[Token]): 
        """Completely overwrites working memory with explicit buffer content."""
        for token in context: self._validate_token(token)
        self.buffer.clear()
        self.buffer.extend(context)
        
    def reset_context(self): 
        """Purges contextual buffer queue heavily."""
        self.buffer.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Provides raw JSON-compatible model state dump."""
        return {
            'max_depth': self.max_depth, 'min_depth': self.min_depth, 'depth_list': self.depth_list, 'decay': self.decay,
            'alphabet_autoscale': self.alphabet_autoscale, 'fallback_mode': self.fallback_mode,
            'pruning_mode': self.pruning_mode, 'pruning_step': self.pruning_step,
            'pruning_threshold': self.pruning_threshold, 'max_beams': self.max_beams,
            'step': self.step, 'known_vocabulary': list(self.known_vocabulary),
            'unigram_counts': {str(k): v for k, v in self.unigram_counts.items()},
            'unigram_last_update': {str(k): v for k, v in self.unigram_last_update.items()},
            'buffer': list(self.buffer._deque), 'root': self.root.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TokenTrieModel':
        """Generates functional instance from extracted dictionary content."""
        model = cls(
            max_depth=data.get('max_depth', 10), min_depth=data.get('min_depth', 1), depth_list=data.get('depth_list', None),
            decay=data.get('decay', 0.99), alphabet_autoscale=data.get('alphabet_autoscale', True),
            fallback_mode=data.get('fallback_mode', 'katz_backoff'), pruning_mode=data.get('pruning_mode', 'fixed'),
            pruning_step=data.get('pruning_step', 1000), pruning_threshold=data.get('pruning_threshold', 1e-6),
            max_beams=data.get('max_beams', 1000)
        )
        model.step = data.get('step', 0)
        model.known_vocabulary = set(data.get('known_vocabulary', []))
        model._vocab_len = len(model.known_vocabulary)
        model.unigram_counts = defaultdict(float, data.get('unigram_counts', {}))
        model.unigram_last_update = defaultdict(int, data.get('unigram_last_update', {}))
        model.buffer.extend(data.get('buffer', []))
        model.root = TokenTrieNode.from_dict(data.get('root', {}))
        return model

    def save_json(self, filepath: str):
        """Dumps dictionary format structure into valid JSON format."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load_json(cls, filepath: str) -> 'TokenTrieModel':
        """Resurrects model layout out of saved JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
        return cls.from_dict(data)

    def __getstate__(self) -> Dict[str, Any]:
        """Packs class dictionary securely avoiding volatile memory footprints."""
        state = self.__dict__.copy()
        for k in ['_power_cache', '_int_log_cache']: 
            if k in state: del state[k]
        return state

    def __setstate__(self, state: Dict[str, Any]):
        """Unpacks layout efficiently ensuring volatile cache reinjection."""
        self.__dict__.update(state)
        
        self.skip_decay = (self.decay is None or self.decay == 1.0)
        self._power_cache = {}
        self._power_cache_len = 0
        self.log_decay = c_log(self.decay) if not self.skip_decay and self.decay > 0 else -float('inf')
        self._int_log_cache = {} 
        self._log_cache_len = 0
        self._last_computed_vocab_len = 0
        self._update_valid_lengths()

    def save(self, filepath: str):
        """Drops binary instance using secure OS writing mechanisms."""
        try:
            with open(filepath, 'wb') as f: pickle.dump(self, f)
        except Exception as e: print(f"Error saving model: {e}")

    @classmethod
    def load(cls, filepath: str) -> Optional['TokenTrieModel']:
        """Ingests standard pickle payload format cleanly into layout."""
        try:
            with open(filepath, 'rb') as f: return pickle.load(f)
        except Exception as e: 
            print(f"Error loading model: {e}")
            return None