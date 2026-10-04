# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False

"""Core Engine for TokenTrieModel (TTM).

Adaptive Variable Order Markov Model (VOMM) utilizing a Reverse Suffix Trie,
Depth-Stratified Empirical Bayes shrinkage, and Information-Theoretic
Redundancy weighting. Accelerated via Cython primitives and libc math.
"""

import json
import pickle
import random
import warnings
from collections import deque
from typing import List, Dict, Optional, Any, Iterable, Union, Tuple, Set

from libc.math cimport (
    log as c_log,
    exp as c_exp,
    log1p as c_log1p,
    fabs as c_fabs,
    fmax as c_fmax,
    fmin as c_fmin,
    INFINITY
)
from libc.string cimport memset

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

Token = Union[str, int]

DEF MAX_TRACKED_DEPTH = 64
DEF NUM_HIST_BINS = 96


cdef inline int _count_to_bin(double count) noexcept nogil:
    """Maps continuous count to logarithmic histogram bin index in O(1)."""
    if count <= 1.0:
        return 1
    if count <= 32.0:
        return <int>count
    cdef int b = 32 + <int>(c_log(count / 32.0) * 5.484806552945484)  # 1 / ln(1.2)
    if b >= NUM_HIST_BINS - 1:
        return NUM_HIST_BINS - 1
    return b


cdef inline double _bin_to_count(int b) noexcept nogil:
    """Maps logarithmic bin index back to continuous count value in O(1)."""
    if b <= 32:
        return <double>b
    return 32.0 * c_exp((<double>(b - 32) + 0.5) * 0.1823215567939546)  # ln(1.2)


cdef inline double _c_logsumexp(double a, double b) noexcept nogil:
    """Computes pairwise LogSumExp in a numerically stable manner without GIL."""
    if a == -INFINITY or a <= -1e300:
        return b
    if b == -INFINITY or b <= -1e300:
        return a
    if a > b:
        return a + c_log1p(c_exp(b - a))
    else:
        return b + c_log1p(c_exp(a - b))


def _extract_item_value(item):
    """Module-level key extractor to avoid closure creation inside class methods."""
    return item[1]


cdef class TokenBuffer:
    """Stateful ring buffer for maintaining sliding token observation windows."""
    cdef public int maxlen
    cdef object _deque
    cdef tuple _cached_tuple

    def __init__(self, int maxlen):
        if maxlen < 1:
            maxlen = 1
        self.maxlen = maxlen
        self._deque = deque(maxlen=maxlen)
        self._cached_tuple = None

    cpdef void append(self, object item):
        self._deque.append(item)
        self._cached_tuple = None

    cpdef void extend(self, object items):
        self._deque.extend(items)
        self._cached_tuple = None

    cpdef void clear(self):
        self._deque.clear()
        self._cached_tuple = None

    @property
    def size(self) -> int:
        return len(self._deque)

    cpdef tuple to_tuple(self):
        if self._cached_tuple is None:
            self._cached_tuple = tuple(self._deque)
        return self._cached_tuple

    def __getstate__(self) -> Dict[str, Any]:
        return {'maxlen': self.maxlen, 'items': list(self._deque)}

    def __setstate__(self, dict state):
        self.maxlen = state.get('maxlen', 10)
        self._deque = deque(state.get('items', []), maxlen=self.maxlen)
        self._cached_tuple = None


cdef class TokenTrieNode:
    """Compact node representation for the Reverse Suffix Trie."""
    cdef public dict counts
    cdef public dict children
    cdef public int last_visit_step
    cdef public double total_mass

    def __init__(self):
        self.counts = {}
        self.children = {}
        self.last_visit_step = 0
        self.total_mass = 0.0

    def to_dict(self) -> Dict[str, Any]:
        cdef dict root_dict = {
            'c': dict(self.counts),
            'ch': {},
            'lvs': self.last_visit_step,
            'tm': self.total_mass
        }
        cdef object queue = deque([(self, root_dict)])
        cdef TokenTrieNode curr_node
        cdef dict curr_dict, child_dict
        cdef object token, child_node

        while queue:
            curr_node, curr_dict = queue.popleft()
            for token, child_node in curr_node.children.items():
                child_dict = {
                    'c': dict((<TokenTrieNode>child_node).counts),
                    'ch': {},
                    'lvs': (<TokenTrieNode>child_node).last_visit_step,
                    'tm': (<TokenTrieNode>child_node).total_mass
                }
                curr_dict['ch'][str(token)] = child_dict
                queue.append((child_node, child_dict))

        return root_dict

    @classmethod
    def from_dict(cls, dict data) -> TokenTrieNode:
        cdef TokenTrieNode root_node = cls()
        root_node.counts = dict(data.get('c', {}))
        root_node.last_visit_step = data.get('lvs', 0)
        root_node.total_mass = data.get('tm', sum(root_node.counts.values()) if root_node.counts else 0.0)

        cdef object queue = deque([(root_node, data.get('ch', {}))])
        cdef TokenTrieNode parent_node, child_node
        cdef dict children_dict, child_data
        cdef str token_key

        while queue:
            parent_node, children_dict = queue.popleft()
            for token_key, child_data in children_dict.items():
                child_node = cls()
                child_node.counts = dict(child_data.get('c', {}))
                child_node.last_visit_step = child_data.get('lvs', 0)
                child_node.total_mass = child_data.get('tm', sum(child_node.counts.values()) if child_node.counts else 0.0)
                parent_node.children[token_key] = child_node
                queue.append((child_node, child_data.get('ch', {})))

        return root_node


cdef class TokenTrieModel:
    """Variable Order Markov Model (VOMM) with Information-Theoretic Shrinkage."""
    cdef public int max_depth
    cdef public int min_depth
    cdef public object depth_list
    cdef public double decay
    cdef public bint skip_decay
    cdef public bint alphabet_autoscale
    cdef public str fallback_mode
    cdef public str pruning_mode
    cdef public int pruning_step
    cdef public double pruning_threshold
    cdef public int max_beams
    cdef public int cache_size
    cdef public double smoothing_prior
    cdef public bint entropy_weighting

    cdef TokenTrieNode root
    cdef TokenBuffer buffer
    cdef int step
    cdef set known_vocabulary
    cdef dict unigram_counts
    cdef dict unigram_last_update
    cdef double total_unigram_mass

    cdef set valid_lengths
    cdef int _node_count
    cdef int _next_prune_target
    cdef int _vocab_len
    cdef int _last_computed_vocab_len
    cdef double _cached_log_base
    cdef dict _power_cache
    cdef int _power_cache_len
    cdef double log_decay
    cdef dict _int_log_cache
    cdef int _log_cache_len

    # Depth-Stratified C-Arrays for O(1) Median and Entropy Tracking
    cdef int _depth_hist[MAX_TRACKED_DEPTH][NUM_HIST_BINS]
    cdef int _depth_total_nodes[MAX_TRACKED_DEPTH]
    cdef double _depth_median_cache[MAX_TRACKED_DEPTH]
    cdef bint _depth_median_dirty[MAX_TRACKED_DEPTH]
    cdef double _depth_total_trans[MAX_TRACKED_DEPTH]
    cdef double _depth_sigma_node[MAX_TRACKED_DEPTH]
    cdef double _depth_sigma_trans[MAX_TRACKED_DEPTH]

    def __init__(self,
                 max_depth: Optional[int] = 10,
                 min_depth: Optional[int] = 1,
                 depth_list: Optional[List[int]] = None,
                 decay: Optional[float] = 0.99,
                 alphabet_autoscale: bool = True,
                 fallback_mode: str = 'interpolated_unigram',
                 pruning_mode: str = 'fixed',
                 pruning_step: int = 1000,
                 pruning_threshold: float = 1e-6,
                 max_beams: int = 1000,
                 cache_size: int = 4096,
                 smoothing_prior: float = 1e-3,
                 entropy_weighting: bool = True):
        if max_depth is None and min_depth is None and depth_list is None:
            max_depth, min_depth = 1, 1
        if max_depth is None:
            max_depth = max(depth_list) if depth_list else (min_depth if min_depth is not None else 1)
        if min_depth is None and not depth_list:
            min_depth = 1

        if depth_list is not None and max_depth < max(depth_list):
            warnings.warn("Specified max_depth is less than maximum depth_list element. Adjusting max_depth.")
            max_depth = max(depth_list)

        self.max_depth = max(1, min(max_depth, MAX_TRACKED_DEPTH - 1))
        self.min_depth = max(1, min_depth if min_depth is not None else 1)
        self.depth_list = depth_list
        self.decay = decay if decay is not None else 1.0
        if not (0.0 <= self.decay <= 1.0):
            self.decay = 0.99

        self.skip_decay = (self.decay == 1.0)
        self.alphabet_autoscale = alphabet_autoscale

        if fallback_mode in {'interpolated_unigram', 'unigram', 'katz_backoff'}:
            self.fallback_mode = 'interpolated_unigram'
        elif fallback_mode == 'uniform':
            self.fallback_mode = 'uniform'
        else:
            self.fallback_mode = 'interpolated_unigram'

        self.pruning_mode = pruning_mode if pruning_mode in {'fixed', 'dynamic'} else 'fixed'
        self.pruning_step = max(1, pruning_step)
        self.pruning_threshold = max(0.0, pruning_threshold)
        self.max_beams = max(1, max_beams)
        self.cache_size = max(1, cache_size)
        self.smoothing_prior = max(1e-9, smoothing_prior)
        self.entropy_weighting = entropy_weighting

        self.reset()

    def _update_valid_lengths(self):
        self.valid_lengths = set()
        if self.min_depth is not None and self.max_depth is not None:
            self.valid_lengths.update(range(self.min_depth, self.max_depth + 1))
        if self.depth_list is not None:
            self.valid_lengths.update(self.depth_list)

    cpdef TokenTrieModel reset(self):
        self.root = TokenTrieNode()
        self.buffer = TokenBuffer(maxlen=self.max_depth)
        self.step = 0
        self.known_vocabulary = set()
        self.unigram_counts = {}
        self.unigram_last_update = {}
        self.total_unigram_mass = 0.0

        self._node_count = 1
        self._next_prune_target = self.pruning_step
        self._vocab_len = 0
        self._last_computed_vocab_len = 0
        self._cached_log_base = 0.6931471805599453

        self._power_cache = {}
        self._power_cache_len = 0
        self.log_decay = c_log(self.decay) if not self.skip_decay and self.decay > 0.0 else -INFINITY

        self._int_log_cache = {}
        self._log_cache_len = 0

        # Zero out depth-stratified statistics
        memset(self._depth_hist, 0, sizeof(self._depth_hist))
        memset(self._depth_total_nodes, 0, sizeof(self._depth_total_nodes))
        memset(self._depth_median_cache, 0, sizeof(self._depth_median_cache))
        memset(self._depth_median_dirty, 0, sizeof(self._depth_median_dirty))
        memset(self._depth_total_trans, 0, sizeof(self._depth_total_trans))
        memset(self._depth_sigma_node, 0, sizeof(self._depth_sigma_node))
        memset(self._depth_sigma_trans, 0, sizeof(self._depth_sigma_trans))

        self._update_valid_lengths()
        return self

    @property
    def log_scaling_base(self) -> float:
        if not self.alphabet_autoscale:
            return 0.6931471805599453

        if self._vocab_len != self._last_computed_vocab_len:
            self._last_computed_vocab_len = self._vocab_len
            self._cached_log_base = c_log(c_fmax(2.0, <double>self._vocab_len))

        return self._cached_log_base

    @property
    def node_count(self) -> int:
        """Returns the total number of allocated nodes in the reverse suffix trie."""
        return self._node_count

    @property
    def next_prune_target(self) -> int:
        """Returns the dynamic node threshold target for triggering garbage collection."""
        return self._next_prune_target

    @property
    def vocab_size(self) -> int:
        """Returns the number of unique discrete tokens in the known vocabulary."""
        return self._vocab_len

    @property
    def valid_lengths(self) -> set:
        """Returns an immutable copy of active context horizon depths."""
        return set(self.valid_lengths)

    @property
    def step(self) -> int:
        """Current global timeline step index."""
        return self.step

    @property
    def total_unigram_mass(self) -> float:
        """Cumulative unigram frequency mass."""
        return self.total_unigram_mass

    @property
    def root(self) -> TokenTrieNode:
        """Root node of the reverse suffix trie (read-only reference)."""
        return self.root

    @property
    def buffer(self) -> TokenBuffer:
        """Working memory token observation buffer."""
        return self.buffer

    @property
    def known_vocabulary(self) -> set:
        """Set copy of unique observed tokens."""
        return set(self.known_vocabulary)

    @property
    def unigram_counts(self) -> dict:
        """Dictionary copy of tracked unigram frequencies."""
        return dict(self.unigram_counts)

    cdef double _get_decay_factor(self, int delta):
        cdef double val
        if self.decay <= 0.0:
            return 0.0
        if delta in self._power_cache:
            return <double>self._power_cache[delta]

        val = self.decay ** delta
        if self._power_cache_len < self.cache_size:
            self._power_cache[delta] = val
            self._power_cache_len += 1
        return val

    cdef double _get_log_count(self, double count):
        cdef int ix
        cdef double val
        if count <= 1.0:
            return 0.0

        if count.is_integer():
            ix = <int>count
            if ix in self._int_log_cache:
                return <double>self._int_log_cache[ix]

            val = c_log(count)
            if self._log_cache_len < self.cache_size:
                self._int_log_cache[ix] = val
                self._log_cache_len += 1
            return val

        return c_log(count)

    cdef void _update_depth_stats(self, int depth, double old_mass, double new_mass) noexcept nogil:
        """Updates histogram buckets and entropy accumulators for depth order in O(1)."""
        if depth < 1 or depth >= MAX_TRACKED_DEPTH:
            return

        cdef int bin_old, bin_new
        if old_mass <= 0.0:
            self._depth_total_nodes[depth] += 1
            bin_new = _count_to_bin(new_mass)
            self._depth_hist[depth][bin_new] += 1
            self._depth_median_dirty[depth] = True
        else:
            bin_old = _count_to_bin(old_mass)
            bin_new = _count_to_bin(new_mass)
            if bin_old != bin_new:
                if self._depth_hist[depth][bin_old] > 0:
                    self._depth_hist[depth][bin_old] -= 1
                self._depth_hist[depth][bin_new] += 1
                self._depth_median_dirty[depth] = True

    cdef double _get_depth_median(self, int depth) noexcept nogil:
        """Retrieves or computes empirical median of active node counts at depth in O(1)."""
        if depth < 1 or depth >= MAX_TRACKED_DEPTH:
            return 1.0

        if not self._depth_median_dirty[depth] and self._depth_median_cache[depth] > 0.0:
            return self._depth_median_cache[depth]

        cdef int total = self._depth_total_nodes[depth]
        if total <= 0:
            self._depth_median_cache[depth] = 1.0
            self._depth_median_dirty[depth] = False
            return 1.0

        cdef int target = total // 2
        cdef int cum = 0
        cdef int b
        for b in range(1, NUM_HIST_BINS):
            cum += self._depth_hist[depth][b]
            if cum >= target:
                self._depth_median_cache[depth] = _bin_to_count(b)
                self._depth_median_dirty[depth] = False
                return self._depth_median_cache[depth]

        self._depth_median_cache[depth] = 1.0
        self._depth_median_dirty[depth] = False
        return 1.0

    cpdef double get_depth_entropy(self, int depth):
        """Computes empirical conditional Markov transition entropy at depth order in O(1)."""
        if depth < 1 or depth >= MAX_TRACKED_DEPTH:
            return 0.0
        if self._depth_total_trans[depth] <= 0.0:
            return 0.0

        cdef double h = (self._depth_sigma_node[depth] - self._depth_sigma_trans[depth]) / self._depth_total_trans[depth]
        return c_fmax(0.0, h)

    cpdef double get_depth_median(self, int depth):
        """Returns the empirical median node mass at context depth order."""
        return self._get_depth_median(depth)

    cpdef dict get_depth_stats(self):
        """Returns diagnostic statistical profile across all tracked context orders."""
        cdef dict res = {}
        cdef int d
        for d in sorted(self.valid_lengths):
            if d < MAX_TRACKED_DEPTH:
                res[d] = {
                    'median_node_mass': self.get_depth_median(d),
                    'conditional_entropy': self.get_depth_entropy(d),
                    'active_nodes': self._depth_total_nodes[d],
                    'total_transitions': self._depth_total_trans[d]
                }
        return res

    cdef list _get_context_nodes(self, str mode, tuple buffer_tuple, int max_depth):
        cdef int i, eff_len, depth, phase, match_len, beam_iters
        cdef int buf_len = len(buffer_tuple)

        if max_depth == 0 or buf_len == 0:
            return []

        cdef dict visited = {}
        cdef TokenTrieNode curr_node = self.root
        cdef object token, target_token, child

        for i in range(max_depth):
            token = buffer_tuple[buf_len - 1 - i]
            if token not in curr_node.children:
                break
            curr_node = <TokenTrieNode>curr_node.children[token]
            eff_len = i + 1
            if eff_len in self.valid_lengths:
                visited[id(curr_node)] = (curr_node, eff_len)

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

                if depth > 0 and eff_len in self.valid_lengths:
                    nid = id(curr_node)
                    if eff_len > visited.get(nid, (None, -1))[1]:
                        visited[nid] = (curr_node, eff_len)

                if depth == max_depth:
                    continue

                target_token = buffer_tuple[buf_len - 1 - depth]

                if mode == 'linear':
                    if phase == 0:
                        for token, child in curr_node.children.items():
                            queue.append((child, depth + 1, 0, eff_len))
                            if token == target_token:
                                queue.append((child, depth + 1, 1, eff_len + 1))
                    else:
                        if target_token in curr_node.children:
                            queue.append((curr_node.children[target_token], depth + 1, 1, eff_len + 1))
                else:
                    for token, child in curr_node.children.items():
                        match_len = eff_len + 1 if token == target_token else eff_len
                        queue.append((child, depth + 1, match_len))

        return list(visited.values())

    cdef void _validate_token(self, object token) except *:
        if not isinstance(token, (str, int)) or isinstance(token, bool):
            raise TypeError(f"TokenTrieModel strictly requires 'str' or 'int' tokens. Received: {type(token).__name__}")

    cpdef void update(self, object actual):
        """Ingests an observed ground-truth token and updates Markov transitions in O(1)."""
        cdef int current_step, hist_len, i, delta_uni, delta
        cdef double factor, new_val, old_mass, new_mass, old_edge, new_edge
        cdef double delta_node_s, delta_trans_s
        cdef tuple history_tuple
        cdef TokenTrieNode node
        cdef object token, t
        cdef list keys_to_remove

        self._validate_token(actual)
        self.step += 1
        current_step = self.step

        if actual not in self.known_vocabulary:
            self.known_vocabulary.add(actual)
            self._vocab_len += 1

        if not self.skip_decay:
            delta_uni = current_step - <int>self.unigram_last_update.get(actual, 0)
            if delta_uni > 0 and actual in self.unigram_counts:
                factor = self._get_decay_factor(delta_uni)
                self.unigram_counts[actual] = <double>self.unigram_counts[actual] * factor
        
        self.unigram_counts[actual] = <double>self.unigram_counts.get(actual, 0.0) + 1.0
        self.unigram_last_update[actual] = current_step
        self.total_unigram_mass += 1.0

        hist_len = self.buffer.size
        history_tuple = self.buffer.to_tuple()
        node = self.root

        for i in range(1, min(self.max_depth, hist_len) + 1):
            token = history_tuple[hist_len - i]

            if token not in node.children:
                node.children[token] = TokenTrieNode()
                self._node_count += 1
            node = <TokenTrieNode>node.children[token]

            if not self.skip_decay and node.last_visit_step != 0:
                delta = current_step - node.last_visit_step
                if delta > 0:
                    factor = self._get_decay_factor(delta)
                    keys_to_remove = []
                    node.total_mass = 0.0
                    for t, c in node.counts.items():
                        new_val = <double>c * factor
                        if new_val < self.pruning_threshold:
                            keys_to_remove.append(t)
                        else:
                            node.counts[t] = new_val
                            node.total_mass += new_val
                    for t in keys_to_remove:
                        del node.counts[t]

            node.last_visit_step = current_step

            if i in self.valid_lengths:
                old_mass = node.total_mass
                new_mass = old_mass + 1.0
                node.total_mass = new_mass

                old_edge = <double>node.counts.get(actual, 0.0)
                new_edge = old_edge + 1.0
                node.counts[actual] = new_edge

                if i < MAX_TRACKED_DEPTH:
                    self._update_depth_stats(i, old_mass, new_mass)
                    
                    # Incremental tracking of conditional Markov entropy rate
                    delta_node_s = new_mass * c_log(new_mass) - (old_mass * c_log(old_mass) if old_mass > 0.0 else 0.0)
                    delta_trans_s = new_edge * c_log(new_edge) - (old_edge * c_log(old_edge) if old_edge > 0.0 else 0.0)
                    self._depth_sigma_node[i] += delta_node_s
                    self._depth_sigma_trans[i] += delta_trans_s
                    self._depth_total_trans[i] += 1.0

        self.buffer.append(actual)

        if self.pruning_mode == 'fixed' and self.step % self.pruning_step == 0:
            self.prune_tree()
        elif self.pruning_mode == 'dynamic' and self._node_count >= self._next_prune_target:
            self.prune_tree()

    cpdef TokenTrieModel fit(self, object X, bint verbose=True):
        cdef bint is_batch = False
        cdef object first_element, iterator

        if hasattr(X, '__len__') and len(X) > 0:
            first_element = next(iter(X))
            if isinstance(first_element, (list, tuple)) or (hasattr(first_element, '__iter__') and not isinstance(first_element, (str, bytes))):
                is_batch = True

        iterator = X
        if verbose and _tqdm:
            total = len(X) if hasattr(X, '__len__') else None
            iterator = _tqdm(X, total=total, desc="TTM Fitting", unit="seq" if is_batch else "tok")

        if is_batch:
            for sequence in iterator:
                self.buffer.clear()
                for token in sequence:
                    self.update(token)
        else:
            for token in iterator:
                self.update(token)
        return self

    def predict_proba(self,
                      object temperature = "none",
                      object top_k = "none",
                      object top_p = "none",
                      str masked_mode = 'none',
                      *,
                      bint return_log_scores = False) -> Dict[Token, float]:
        """Calculates transition probability distribution using Information-Theoretic Shrinkage."""
        cdef double temp = 0.0 if temperature in (None, "none", "None") else <double>temperature
        cdef int k = 0 if top_k in (None, "none", "None") else <int>top_k
        cdef double p = 1.0 if top_p in (None, "none", "None") else <double>top_p
        if temp < 0.0: temp = 0.0
        if k < 0: k = 0
        if not (0.0 <= p <= 1.0): p = 1.0
        if masked_mode not in {'none', 'linear', 'squared'}: masked_mode = 'none'

        cdef int hist_len = self.buffer.size
        if hist_len == 0 and self._vocab_len == 0:
            return {}

        cdef dict candidate_log_scores = {}
        cdef double log_scale_base = self.log_scaling_base
        cdef double log_decay_val = self.log_decay
        cdef int current_step = self.step
        cdef int max_depth = min(self.max_depth, hist_len)
        cdef tuple buffer_tup = self.buffer.to_tuple()

        cdef list valid_nodes = self._get_context_nodes(masked_mode, buffer_tup, max_depth)
        cdef TokenTrieNode node
        cdef int length, delta
        cdef double node_factor, log_weight, count, c_val
        cdef double node_mass, sum_clogc, H_c, beta_raw, scale_l, w_node, beta_tilde, depth_multiplier
        cdef object t

        for node_tuple in valid_nodes:
            node = <TokenTrieNode>node_tuple[0]
            length = <int>node_tuple[1]

            if self.entropy_weighting:
                # 1. Compute exact Shannon Entropy H(c) and active node mass
                node_mass = 0.0
                sum_clogc = 0.0
                for t, count in node.counts.items():
                    c_val = <double>count
                    if c_val > 1e-9:
                        node_mass += c_val
                        sum_clogc += c_val * self._get_log_count(c_val)

                if node_mass > 1e-9:
                    H_c = c_log(node_mass) - (sum_clogc / node_mass)
                    if H_c < 0.0:
                        H_c = 0.0
                else:
                    H_c = 0.0

                # 2. Information Purity: Redundancy relative to maximum entropy uniform prior
                if log_scale_base > 1e-9:
                    beta_raw = 1.0 - (H_c / log_scale_base)
                    if beta_raw < 0.0:
                        beta_raw = 0.0
                    elif beta_raw > 1.0:
                        beta_raw = 1.0
                else:
                    beta_raw = 1.0

                # 3. Empirical Bayes Sample Size Support: Scaled by depth-stratified median
                scale_l = self._get_depth_median(length)
                w_node = node_mass / (node_mass + scale_l)

                # 4. Joint Depth Modulator
                beta_tilde = beta_raw * w_node
            else:
                beta_tilde = 1.0

            depth_multiplier = length * beta_tilde * log_scale_base

            if not self.skip_decay:
                delta = current_step - node.last_visit_step
                node_factor = (delta * log_decay_val) + depth_multiplier
            else:
                node_factor = depth_multiplier

            for t, count in node.counts.items():
                if count <= 1e-9:
                    continue
                log_weight = self._get_log_count(<double>count) + node_factor
                if t not in candidate_log_scores:
                    candidate_log_scores[t] = log_weight
                else:
                    candidate_log_scores[t] = _c_logsumexp(<double>candidate_log_scores[t], log_weight)

        # Unigram Smoothing Prior Fallback
        cdef double unigram_val, unigram_log
        cdef double prior = self.smoothing_prior

        if self.fallback_mode == 'interpolated_unigram' and self.unigram_counts:
            for t in self.known_vocabulary:
                unigram_val = <double>self.unigram_counts.get(t, 0.0)
                if not self.skip_decay:
                    delta = current_step - <int>self.unigram_last_update.get(t, 0)
                    if delta > 0:
                        unigram_val *= self._get_decay_factor(delta)
                unigram_log = c_log(unigram_val + prior)

                if t in candidate_log_scores:
                    candidate_log_scores[t] = _c_logsumexp(<double>candidate_log_scores[t], unigram_log)
                else:
                    candidate_log_scores[t] = unigram_log
        else:
            unigram_log = c_log(1.0 / c_fmax(1.0, <double>self._vocab_len))
            for t in self.known_vocabulary:
                if t in candidate_log_scores:
                    candidate_log_scores[t] = _c_logsumexp(<double>candidate_log_scores[t], unigram_log)
                else:
                    candidate_log_scores[t] = unigram_log

        if not candidate_log_scores:
            return {}

        # Temperature scaling
        if temp != 1.0 and temp > 1e-4:
            for t in candidate_log_scores:
                candidate_log_scores[t] = <double>candidate_log_scores[t] / temp

        if return_log_scores:
            return dict(sorted(candidate_log_scores.items(), key=_extract_item_value, reverse=True))

        cdef double max_log = -INFINITY
        for log_val in candidate_log_scores.values():
            if <double>log_val > max_log:
                max_log = <double>log_val

        cdef dict linear_scores = {}
        cdef double total_sum = 0.0
        cdef double val

        for t, log_score in candidate_log_scores.items():
            val = c_exp(<double>log_score - max_log)
            linear_scores[t] = val
            total_sum += val

        cdef dict probas = {}
        for t, val in linear_scores.items():
            probas[t] = <double>val / total_sum

        if k <= 0 and p >= 1.0:
            return dict(sorted(probas.items(), key=_extract_item_value, reverse=True))

        cdef list sorted_items = sorted(probas.items(), key=_extract_item_value, reverse=True)

        if 0 < k < len(sorted_items):
            sorted_items = sorted_items[:k]

        cdef double target_prob, cumulative_prob = 0.0
        cdef double all_prob_sum = 0.0
        cdef int idx

        if p < 1.0:
            for item in sorted_items:
                all_prob_sum += <double>item[1]
            target_prob = p * all_prob_sum

            for idx in range(len(sorted_items)):
                cumulative_prob += <double>sorted_items[idx][1]
                if cumulative_prob >= target_prob:
                    sorted_items = sorted_items[:idx + 1]
                    break

        cdef double new_total = 0.0
        for item in sorted_items:
            new_total += <double>item[1]

        if new_total > 0.0:
            probas = {}
            for item in sorted_items:
                probas[item[0]] = <double>item[1] / new_total
            return probas

        return dict(sorted_items)

    cpdef double score_transition(self, object actual):
        cdef dict probas = self.predict_proba(temperature=1.0)
        if not probas:
            return 1.0 / c_fmax(1.0, <double>(self._vocab_len + 1))
        return probas.get(actual, self.smoothing_prior / (self.total_unigram_mass + 1.0))

    cpdef double score_sequence(self, list sequence, str reduction='mean_nll'):
        if not sequence or len(sequence) < 2:
            return 0.0 if reduction in ('mean_nll', 'sum_nll') else 1.0

        cdef double min_prob = 1.0
        cdef double sum_nll = 0.0
        cdef int steps = len(sequence) - 1
        cdef double prob, p_clipped
        cdef int i

        self.reset_context()
        self.update_context(sequence[0])

        for i in range(1, len(sequence)):
            prob = self.score_transition(sequence[i])
            if prob < min_prob:
                min_prob = prob
            p_clipped = c_fmax(1e-15, prob)
            sum_nll += -c_log(p_clipped)
            self.update_context(sequence[i])

        if reduction == 'min':
            return min_prob
        elif reduction == 'mean_nll':
            return sum_nll / <double>steps
        elif reduction == 'perplexity':
            return c_exp(sum_nll / <double>steps)
        elif reduction == 'sum_nll':
            return sum_nll
        else:
            raise ValueError(f"Unknown reduction mode: '{reduction}'. Valid: 'mean_nll', 'min', 'perplexity', 'sum_nll'.")

    def predict(self,
                object temperature = "none",
                object top_k = "none",
                object top_p = "none",
                str masked_mode = 'none') -> Optional[Token]:
        cdef double temp = 0.0 if temperature in (None, "none", "None") else <double>temperature
        cdef dict probas = self.predict_proba(temperature=1.0 if temp < 1e-4 else temp,
                                              top_k=top_k,
                                              top_p=top_p,
                                              masked_mode=masked_mode)
        if not probas:
            return None

        cdef object best_token = None
        cdef double best_prob = -1.0
        cdef object t
        cdef double p_val

        if temp < 1e-4:
            for t, p_val in probas.items():
                if p_val > best_prob:
                    best_prob = p_val
                    best_token = t
            return best_token

        return random.choices(list(probas.keys()), weights=list(probas.values()), k=1)[0]

    cpdef void prune_tree(self):
        cdef int current_step = self.step
        cdef double factor, real_count, val
        cdef int delta, surviving_nodes = 0
        cdef TokenTrieNode node, child
        cdef object token
        cdef list keys_to_remove, empty_children

        cdef list stack = [(self.root, False)]
        cdef list post_order = []
        cdef bint visited

        while stack:
            node, visited = stack.pop()
            if visited:
                post_order.append(node)
            else:
                stack.append((node, True))
                for child in node.children.values():
                    stack.append((child, False))

        for node in post_order:
            keys_to_remove = []
            node.total_mass = 0.0
            if not self.skip_decay:
                delta = current_step - node.last_visit_step
                factor = self._get_decay_factor(delta) if delta > 0 else 1.0
                for token, count in node.counts.items():
                    real_count = <double>count * factor
                    if real_count < self.pruning_threshold:
                        keys_to_remove.append(token)
                    else:
                        node.counts[token] = real_count
                        node.total_mass += real_count
            else:
                for token, count in node.counts.items():
                    if <double>count < self.pruning_threshold:
                        keys_to_remove.append(token)
                    else:
                        node.total_mass += <double>count

            for token in keys_to_remove:
                del node.counts[token]

            node.last_visit_step = current_step

            empty_children = []
            for token, ch in node.children.items():
                child = <TokenTrieNode>ch
                if len(child.counts) == 0 and len(child.children) == 0:
                    empty_children.append(token)

            for token in empty_children:
                del node.children[token]

            surviving_nodes += 1

        self._node_count = surviving_nodes

        # Prune unigrams
        keys_to_remove = []
        self.total_unigram_mass = 0.0
        for token, count in self.unigram_counts.items():
            if not self.skip_decay:
                delta = current_step - <int>self.unigram_last_update.get(token, 0)
                val = <double>count * (self._get_decay_factor(delta) if delta > 0 else 1.0)
            else:
                val = <double>count

            if val < self.pruning_threshold:
                keys_to_remove.append(token)
            else:
                self.unigram_counts[token] = val
                self.unigram_last_update[token] = current_step
                self.total_unigram_mass += val

        for token in keys_to_remove:
            del self.unigram_counts[token]
            if token in self.unigram_last_update:
                del self.unigram_last_update[token]
            self.known_vocabulary.discard(token)

        self._vocab_len = len(self.known_vocabulary)
        if self.pruning_mode == 'dynamic':
            self._next_prune_target = max(self.pruning_step, int(self._node_count * 1.5))

        self._rebuild_depth_stats()

    cdef void _rebuild_depth_stats(self):
        """Rebuilds depth-stratified histograms and recalculates total node count."""
        memset(self._depth_hist, 0, sizeof(self._depth_hist))
        memset(self._depth_total_nodes, 0, sizeof(self._depth_total_nodes))
        memset(self._depth_median_cache, 0, sizeof(self._depth_median_cache))
        memset(self._depth_median_dirty, 0, sizeof(self._depth_median_dirty))

        cdef object queue = deque([(self.root, 0)])
        cdef TokenTrieNode curr_node
        cdef int d, b
        cdef object ch
        cdef int total_nodes = 0

        while queue:
            curr_node, d = queue.popleft()
            total_nodes += 1
            if 1 <= d < MAX_TRACKED_DEPTH and curr_node.total_mass > 0.0:
                self._depth_total_nodes[d] += 1
                b = _count_to_bin(curr_node.total_mass)
                self._depth_hist[d][b] += 1
                self._depth_median_dirty[d] = True

            for ch in curr_node.children.values():
                queue.append((<TokenTrieNode>ch, d + 1))

        self._node_count = total_nodes

    cdef void _truncate_depth(self, TokenTrieNode node, int depth, int max_d):
        if depth >= max_d:
            node.children.clear()
            return
        cdef object ch
        for ch in node.children.values():
            self._truncate_depth(<TokenTrieNode>ch, depth + 1, max_d)

    cpdef TokenTrieModel reconfigure(self, dict params):
        cdef double old_decay = self.decay
        cdef bint old_skip = self.skip_decay
        cdef double new_decay = params.get('decay', self.decay)
        cdef bint new_skip = (new_decay == 1.0)

        if old_decay != new_decay or old_skip != new_skip:
            self.prune_tree()

        self._power_cache.clear()
        self._power_cache_len = 0
        self._int_log_cache.clear()
        self._log_cache_len = 0
        self._last_computed_vocab_len = 0
        self._cached_log_base = 0.6931471805599453

        if 'decay' in params:
            self.decay = params['decay']
            self.skip_decay = new_skip
            self.log_decay = c_log(self.decay) if not self.skip_decay and self.decay > 0.0 else -INFINITY

        if 'entropy_weighting' in params:
            self.entropy_weighting = params['entropy_weighting']

        if 'max_depth' in params or 'min_depth' in params or 'depth_list' in params:
            self.max_depth = max(1, min(params.get('max_depth', self.max_depth), MAX_TRACKED_DEPTH - 1))
            self.min_depth = max(1, params.get('min_depth', self.min_depth))
            self.depth_list = params.get('depth_list', self.depth_list)
            self._update_valid_lengths()

            # Физически обрезаем ветки глубже нового max_depth и пересчитываем ноды
            self._truncate_depth(self.root, 0, self.max_depth)
            self.prune_tree()

            if self.buffer.maxlen != self.max_depth:
                old_items = self.buffer.to_tuple()
                new_buf = TokenBuffer(maxlen=self.max_depth)
                if len(old_items) > self.max_depth:
                    new_buf.extend(old_items[len(old_items) - self.max_depth:])
                else:
                    new_buf.extend(old_items)
                self.buffer = new_buf

        cdef list config_keys = [
            'alphabet_autoscale', 'fallback_mode', 'pruning_mode', 
            'pruning_step', 'pruning_threshold', 'max_beams', 
            'cache_size', 'smoothing_prior'
        ]
        for k in config_keys:
            if k in params:
                setattr(self, k, params[k])

        return self

    cpdef void update_context(self, object token):
        self._validate_token(token)
        self.buffer.append(token)

    cpdef void fill_context(self, object context):
        self.buffer.clear()
        for token in context:
            self._validate_token(token)
            self.buffer.append(token)

    cpdef void reset_context(self):
        self.buffer.clear()

    def set_branches(self, branches: Iterable[Tuple[Iterable[Token], Dict[Token, float]]]) -> TokenTrieModel:
        cdef int max_new_len = self.max_depth
        cdef int seq_len, i
        cdef TokenTrieNode node
        cdef object seq_list, token, target_token

        for seq, counts in branches:
            seq_list = list(seq)
            seq_len = len(seq_list)
            if seq_len > max_new_len:
                max_new_len = seq_len

        if max_new_len > self.max_depth:
            self.reconfigure({'max_depth': max_new_len})

        for seq, counts in branches:
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
                node = <TokenTrieNode>node.children[token]

            node.counts.clear()
            node.total_mass = 0.0
            for target_token, weight in counts.items():
                self._validate_token(target_token)
                self.known_vocabulary.add(target_token)
                node.counts[target_token] = float(weight)
                node.total_mass += float(weight)
                self.unigram_counts[target_token] = float(self.unigram_counts.get(target_token, 0.0) + weight)
                self.unigram_last_update[target_token] = self.step
                self.total_unigram_mass += float(weight)
            node.last_visit_step = self.step

        self._vocab_len = len(self.known_vocabulary)
        self._rebuild_depth_stats()
        return self

    def delete_branches(self, sequences: Iterable[Iterable[Token]]):
        cdef int seq_len, i
        cdef bint path_exists
        cdef TokenTrieNode node
        cdef object seq_list, token

        for seq in sequences:
            seq_list = list(seq)
            seq_len = len(seq_list)
            if seq_len == 0:
                continue

            node = self.root
            path_exists = True
            for i in range(seq_len - 1):
                token = seq_list[seq_len - 1 - i]
                if token not in node.children:
                    path_exists = False
                    break
                node = <TokenTrieNode>node.children[token]

            if path_exists:
                token = seq_list[0]
                if token in node.children:
                    del node.children[token]

        self.prune_tree()

    def merge(self, TokenTrieModel other) -> TokenTrieModel:
        if other.max_depth > self.max_depth:
            self.reconfigure({'max_depth': other.max_depth})

        self.known_vocabulary.update(other.known_vocabulary)
        self._vocab_len = len(self.known_vocabulary)

        cdef object token
        cdef double c_other
        for token, c_other in other.unigram_counts.items():
            self.unigram_counts[token] = <double>self.unigram_counts.get(token, 0.0) + c_other
            self.unigram_last_update[token] = self.step
        self.total_unigram_mass += other.total_unigram_mass

        cdef object queue = deque([(self.root, other.root)])
        cdef TokenTrieNode self_node, other_node, child_self, child_other
        cdef object target_tok

        while queue:
            self_node, other_node = queue.popleft()
            for target_tok, count in other_node.counts.items():
                self_node.counts[target_tok] = <double>self_node.counts.get(target_tok, 0.0) + <double>count

            self_node.total_mass = sum(self_node.counts.values()) if self_node.counts else 0.0
            self_node.last_visit_step = self.step

            for token, child_other in other_node.children.items():
                if token not in self_node.children:
                    self_node.children[token] = TokenTrieNode()
                    self._node_count += 1
                queue.append((self_node.children[token], child_other))

        self.prune_tree()
        return self

    def to_dict(self) -> Dict[str, Any]:
        cdef dict unigrams_dump = {}
        cdef dict updates_dump = {}
        cdef object k

        for k, v in self.unigram_counts.items():
            unigrams_dump[str(k)] = v
        for k, v in self.unigram_last_update.items():
            updates_dump[str(k)] = v

        return {
            'max_depth': self.max_depth,
            'min_depth': self.min_depth,
            'depth_list': self.depth_list,
            'decay': self.decay,
            'alphabet_autoscale': self.alphabet_autoscale,
            'fallback_mode': self.fallback_mode,
            'pruning_mode': self.pruning_mode,
            'pruning_step': self.pruning_step,
            'pruning_threshold': self.pruning_threshold,
            'max_beams': self.max_beams,
            'smoothing_prior': self.smoothing_prior,
            'entropy_weighting': self.entropy_weighting,
            'step': self.step,
            'known_vocabulary': list(self.known_vocabulary),
            'unigram_counts': unigrams_dump,
            'unigram_last_update': updates_dump,
            'total_unigram_mass': self.total_unigram_mass,
            'buffer': list(self.buffer._deque),
            'root': self.root.to_dict()
        }

    @classmethod
    def from_dict(cls, dict data) -> TokenTrieModel:
        cdef TokenTrieModel model = cls(
            max_depth=data.get('max_depth', 10),
            min_depth=data.get('min_depth', 1),
            depth_list=data.get('depth_list', None),
            decay=data.get('decay', 0.99),
            alphabet_autoscale=data.get('alphabet_autoscale', True),
            fallback_mode=data.get('fallback_mode', 'interpolated_unigram'),
            pruning_mode=data.get('pruning_mode', 'fixed'),
            pruning_step=data.get('pruning_step', 1000),
            pruning_threshold=data.get('pruning_threshold', 1e-6),
            max_beams=data.get('max_beams', 1000),
            smoothing_prior=data.get('smoothing_prior', 1e-3),
            entropy_weighting=data.get('entropy_weighting', True)
        )
        model.step = data.get('step', 0)
        model.known_vocabulary = set(data.get('known_vocabulary', []))
        model._vocab_len = len(model.known_vocabulary)

        model.unigram_counts = {}
        for k, v in data.get('unigram_counts', {}).items():
            model.unigram_counts[k] = float(v)

        model.unigram_last_update = {}
        for k, v in data.get('unigram_last_update', {}).items():
            model.unigram_last_update[k] = int(v)

        cdef double mass_sum = 0.0
        if 'total_unigram_mass' in data:
            model.total_unigram_mass = float(data['total_unigram_mass'])
        else:
            for val in model.unigram_counts.values():
                mass_sum += <double>val
            model.total_unigram_mass = mass_sum

        model.buffer.extend(data.get('buffer', []))
        model.root = TokenTrieNode.from_dict(data.get('root', {}))
        model._rebuild_depth_stats()
        return model

    def save_json(self, str filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load_json(cls, str filepath) -> TokenTrieModel:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, str filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, str filepath) -> Optional[TokenTrieModel]:
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def __getstate__(self) -> Dict[str, Any]:
        return self.to_dict()

    def __setstate__(self, dict state):
        cdef TokenTrieModel restored = TokenTrieModel.from_dict(state)
        self.max_depth = restored.max_depth
        self.min_depth = restored.min_depth
        self.depth_list = restored.depth_list
        self.decay = restored.decay
        self.skip_decay = restored.skip_decay
        self.alphabet_autoscale = restored.alphabet_autoscale
        self.fallback_mode = restored.fallback_mode
        self.pruning_mode = restored.pruning_mode
        self.pruning_step = restored.pruning_step
        self.pruning_threshold = restored.pruning_threshold
        self.max_beams = restored.max_beams
        self.cache_size = 4096
        self.smoothing_prior = restored.smoothing_prior
        self.entropy_weighting = restored.entropy_weighting

        self.root = restored.root
        self.buffer = restored.buffer
        self.step = restored.step
        self.known_vocabulary = restored.known_vocabulary
        self.unigram_counts = restored.unigram_counts
        self.unigram_last_update = restored.unigram_last_update
        self.total_unigram_mass = restored.total_unigram_mass

        self._node_count = restored._node_count
        self._next_prune_target = self.pruning_step
        self._vocab_len = len(self.known_vocabulary)
        self._last_computed_vocab_len = 0
        self._cached_log_base = 0.6931471805599453

        self._power_cache = {}
        self._power_cache_len = 0
        self.log_decay = c_log(self.decay) if not self.skip_decay and self.decay > 0.0 else -INFINITY

        self._int_log_cache = {}
        self._log_cache_len = 0

        self._rebuild_depth_stats()
        self._update_valid_lengths()