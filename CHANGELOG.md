# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
* **Katz Backoff Smoothing:** Added graceful degradation to tracked unigram baseline distributions (`unigram_counts` with independent timeline decays) when encountering unseen context patterns.

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