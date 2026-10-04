# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-10-04

### Added

#### Information-Theoretic Context Shrinkage
* **Information-Theoretic Redundancy ($\beta_{\text{raw}}$):** Integrated local Shannon entropy evaluation $H(c)$ per context node to measure predictability against a uniform maximum-entropy prior:
  $$\beta_{\text{raw}}(c) = \max\left(0.0,\, 1.0 - \frac{H(c)}{\ln |\mathcal{V}|}\right) = \frac{D_{\text{KL}}\left(P(\cdot \mid c) \,\|\, \mathcal{U}(\mathcal{V})\right)}{\ln |\mathcal{V}|}$$
  Ensures that deterministic transitions receive a maximal context depth boost while noisy, chaotic branches are neutralized down to baseline counts.
* **Depth-Stratified Empirical Bayes Shrinkage ($w_{\text{node}}$):** Introduced sample-size regularization scaling the depth bonus by the empirical median mass of the context order:
  $$w_{\text{node}}(c) = \frac{N_c}{N_c + \text{median}^{(\ell)}}$$
  Eliminates the small-sample bias where single-visit nodes ($N_c = 1$) falsely appear deterministic ($H=0$).
* **Joint Depth Modulator ($\tilde{\beta}(c)$):** Formulated the composite depth multiplier $\tilde{\beta}(c) = \beta_{\text{raw}}(c) \cdot w_{\text{node}}(c)$, scaling logarithmic potentials as $\ell \cdot \tilde{\beta}(c) \cdot \ln |\mathcal{V}|$.
* **Ablation Switch (`entropy_weighting`):** Added a boolean parameter to `TokenTrieModel.__init__()` and `reconfigure()` allowing instant toggle between empirical Bayes shrinkage (`True`) and the unweighted baseline (`False`) for scientific ablation studies.

#### Streaming Markov Entropy Diagnostics (`O(1)`)
* **Real-Time Conditional Entropy Tracking:** Formulated closed-form incremental tracking for the empirical conditional Markov transition entropy across context orders $\ell$:
  $$\hat{H}(X_t \mid X_{t-\ell : t-1}) = \frac{\Sigma_{\text{node}}^{(\ell)} - \Sigma_{\text{trans}}^{(\ell)}}{N^{(\ell)}}$$
  Maintained in strict $O(1)$ time per ingested token using C-level difference updates $\Delta \Sigma = (n + 1)\ln(n + 1) - n\ln n$.
* **Diagnostic Inspection API:** Added public query methods:
  * `get_depth_entropy(depth: int) -> float`: Returns the empirical conditional Markov entropy at a specific context horizon.
  * `get_depth_median(depth: int) -> float`: Returns the active median node mass for order $\ell$.
  * `get_depth_stats() -> Dict[int, Dict[str, float]]`: Provides an operational profile (median mass, entropy, active nodes, total transitions) across all tracked horizons.

#### Sequence Scoring & Likelihood Interface
* **Transition Scoring:** Added `score_transition(actual: Token) -> float` to query exact point-in-time conditional transition probabilities $P(x_t = \text{actual} \mid \mathcal{H}_t)$.
* **Sequential Health Evaluation:** Added `score_sequence(sequence: List[Token], reduction: str = "mean_nll") -> float` supporting multiple aggregation policies:
  * `'mean_nll'`: Mean Negative Log-Likelihood (Surprisal rate).
  * `'min'`: Weakest-link pointwise conditional likelihood.
  * `'perplexity'`: Sequence window perplexity ($\exp(\text{mean\_nll})$).
  * `'sum_nll'`: Cumulative negative log-likelihood.

#### Developer Experience & Static Typing
* **PEP 484 / PEP 561 Type Stubs (`tokentrie/core.pyi`):** Added complete type stub definitions covering all classes, signatures, properties, and methods, enabling full autocompletion and hover documentation in IDEs (VS Code / Pylance, PyCharm).
* **Typing Marker (`py.typed`):** Packaged PEP 561 marker file to declare official inline static typing support.
* **Read-Only Public Getters:**
  * Added `@property def node_count(self) -> int` to inspect the total allocated nodes without exposing private C structures.
  * Added `@property def vocab_size(self) -> int` as the canonical machine-learning property for vocabulary cardinality.

---

### Changed

#### Fallback Smoothing & Terminology Refinement
* **Renamed Fallback Mode (`katz_backoff` $\to$ `interpolated_unigram`):**
  * Replaced the legacy name `katz_backoff` with `interpolated_unigram` to reflect the true underlying mathematical mechanics: interpolated unigram smoothing with an additive Dirichlet prior rather than discrete Katz backoff discounting.
  * Preserved full backward compatibility by silently redirecting `'katz_backoff'` and `'unigram'` inputs to `'interpolated_unigram'`.
* **Continuous Smoothing Interpolation:** Replaced the discrete all-or-nothing fallback with continuous log-space blending:
  * Added `smoothing_prior` ($\alpha$, default: $10^{-3}$) providing an additive Dirichlet floor:
    $$P_{\text{unigram}}(y) = \frac{\text{Count}(y) + \alpha}{\sum_{y'} \text{Count}(y') + \alpha |\mathcal{V}|}$$
  * Blended unigram potentials directly into candidate distributions via LogSumExp across the full vocabulary, preventing zero-frequency boundary collapses.

#### Memory Management & Tree Operations
* **Non-Recursive Garbage Collection:** Rewrote `prune_tree()` from a recursive traversal into an iterative post-order stack algorithm, eliminating Python stack frame overhead and preventing `RecursionError` on deep context horizons.
* **Iterative State Serialization:** Replaced recursive subtree traversal in `to_dict()` and `from_dict()` with iterative breadth-first search (BFS) using double-ended queues.
* **Native C-Extension Structures:** Converted `TokenBuffer` and `TokenTrieNode` from pure Python classes (`__slots__`) to native Cython extension types (`cdef class`), giving C-level memory access and reducing attribute lookup latency.
* **$O(1)$ Histogram Median Accumulator:** Implemented a fixed 96-bin logarithmic histogram array per context horizon to track running median node mass without sorting overhead.

#### Packaging & Build Pipeline
* **PEP 517 / PEP 621 Build Configuration:** Modernized `pyproject.toml` with version `1.1.0`, enhanced scientific metadata, and optional dependency profiles (`full`, `benchmark`, `test`).
* **Compiler Optimizations:** Added `embedsignature=True` in `setup.py` compiler directives to embed C function signatures into compiled docstrings.

---

### Deprecated
* Parameter value `fallback_mode='katz_backoff'` is now deprecated in favor of `fallback_mode='interpolated_unigram'` and will be removed in a future major release.

---

### Fixed
* **Windows Subprocess Tracebacks:** Resolved Loky / Joblib core-count detection tracebacks on Windows platforms by overriding physical core detection wrappers.
* **Tqdm Widget Warnings:** Suppressed `TqdmWarning: IProgress not found` across headless environments by standardizing on direct terminal progress streaming.
* **Direct Private Variable Overwrites:** Encapsulated internal C variables (`_node_count`, `_vocab_len`) behind read-only Python properties, preventing accidental corruption of tree invariant counters.

---

## [1.0.0] - 2026-09-27

### Added

#### Core Architecture & Acceleration Engine
* **Variable Order Markov Model (VOMM):** Implemented core sequential learning engine utilizing Context Mixing and a Reverse Suffix Trie structure for $O(N)$ historical prefix matching.
* **Cython & C-Level Acceleration:**
  * Ported core algorithms to Cython (`tokentrie/core.pyx`) with direct bindings to C standard mathematical functions (`libc.math.log`, `libc.math.exp`, `libc.math.log1p`).
  * Enforced low-overhead execution via compiler directives: `boundscheck=False`, `wraparound=False`, `cdivision=True`, and `language_level=3`.
  * Introduced platform-specific aggressive compilation flags (`-O3`, `-ffast-math` on GCC/Clang; `/O2`, `/fp:fast` on MSVC).
* **Stateful Structures:**
  * `TokenTrieNode`: Memory-efficient Trie node maintaining token frequency maps (`counts`), context transitions (`children`), and timestamp tracking (`last_visit_step`).
  * `TokenBuffer`: High-performance sliding-window ring buffer wrapping `collections.deque` with an $O(1)$ manual size counter and an invalidated-on-write tuple cache (`_cache_tuple`) to eliminate allocation overhead in hot loops.
  * Strict token type-validation (`_validate_token`), constraining tokens exclusively to `str` and `int` representations.

#### Mathematical Modeling & Timeline Decay
* **Lazy Exponential Decay:**
  * Formulated time-dependent weight degradation ($w = w_0 \cdot \gamma^{\Delta t}$) computed lazily during node visits.
  * Integrated dedicated power cache (`_power_cache`) and integer logarithm cache (`_int_log_cache`) bounded by `cache_size` to bypass expensive floating-point calls.
* **Skip Decay Accelerator:** Configured automatic bypass (`skip_decay=True`) when `decay=1.0` or `decay=None`, completely eliminating temporal calculations for static models.
* **Numerically Stable LogSumExp:** Implemented pairwise C-level log-space additions to prevent underflow and overflow when combining heterogeneous context lengths.
* **Entropy-Scaled Weighting:** Added `alphabet_autoscale` to dynamically calibrate context depth scaling base ($c_0 = \ln |V|$) relative to runtime vocabulary expansion.
* **Fallback Smoothing:** Added fallback distribution baseline support (`unigram_counts` with independent timeline decays) when encountering unseen context patterns.

#### Predictive & Inference Pipeline
* **LLM-Grade Generation Controls:**
  * Added `temperature` scaling for distribution sharpening or flattening.
  * Implemented `top_k` filtering (rank truncation).
  * Implemented `top_p` (Nucleus Sampling) over dynamic cumulative probability masses.
* **Logit Extraction:** Added `return_log_scores` parameter in `predict_proba()` to retrieve pre-sorted, unnormalized Cython log-logits directly, avoiding Softmax overhead.
* **Masked Beam Search:** Added breadth-first search (`masked_mode` in `{'none', 'linear', 'squared'}`) bounded by `max_beams` to resolve contextual correlations across typos, omissions, and noise.
* **Flexible Learning Interfaces:** Added `update()` for real-time online stream learning and `fit()` supporting both flat token streams and batched sequence iterables with `tqdm` progress tracking.

#### Surgical Model Control & Introspection
* **Explicit Context Routing:** Added `depth_list` parameter to track sparse, arbitrary context horizons (e.g., depths `[2, 5, 10]`) without allocating structural Trie nodes for unrequested intermediate orders.
* **Dynamic Tree Restructuring:** Implemented `reconfigure(params)` to adjust depth bounds, decay rates, and pruning strategies on-the-fly, featuring automatic recursive subtree pruning (`_restructure_tree`) and cache invalidation.
* **Direct Knowledge Manipulation:**
  * `set_branches()`: Explicit surgical injection or overwriting of contextual rules and next-token distributions, with automatic boundary expansion (`max_depth`, `depth_list`).
  * `delete_branches()`: Surgical recursive removal of target context branches and their descendant subtrees.
* **Context State Management:** Added `update_context()`, `fill_context()`, and `reset_context()` for decoupled inference without updating transition counts.

#### Garbage Collection & Memory Management
* **Pruning Engine:**
  * Implemented recursive garbage collection (`prune_tree()`, `_prune_recursive`) that sweeps dead nodes, unigrams, and frequencies falling below `pruning_threshold`.
  * Added dual pruning policies: `'fixed'` (periodic interval tracking via `pruning_step`) and `'dynamic'` (node count proportional threshold targeting).

#### Federated Learning & Heterogeneous Merging
* **Asymmetric State Consolidation:** Implemented `merge()` to combine independent model instances with distinct timelines, hyperparameters, and decay rates.
* **Timeline Projection:** Automatically maps foreign model node weights into the local host timeline using cross-model exponential decay transformations.

#### Serialization & Persistence
* **JSON State Engine:** Implemented full structural state export and recovery via `to_dict()`, `from_dict()`, `save_json()`, and `load_json()`.
* **Pickle Serialization:** Configured `__getstate__` and `__setstate__` to exclude volatile runtime math caches from disk payloads, automatically rebuilding precomputed tables on deserialization.

#### Quality Assurance & Verification
* **Comprehensive Test Suite (`tests/`):**
  * `test_core_mechanics.py`: Verification of VOMM orders, suffix traversals, and buffer invariants.
  * `test_prediction.py`: Validation of temperature, top-k/top-p sampling, and masked beam search.
  * `test_components.py`: Unit tests for `TokenBuffer` and `TokenTrieNode`.
  * `test_serialization.py`: Exact state equality checks for JSON and Pickle round-trips.
  * `test_federated.py`: Accuracy verification of asymmetric federated merges across shifted timelines.
* **Automated CI/CD:** Integrated GitHub Actions workflow running automated testing pipelines across cross-platform Python targets (3.8–3.13).