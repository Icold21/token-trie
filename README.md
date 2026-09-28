# TokenTrieModel (TTM) 🧠

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://pypi.org/project/tokentrie/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Cython](https://img.shields.io/badge/Cython-Accelerated-yellow.svg)](https://cython.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI Tests](https://img.shields.io/badge/Tests-29%2F29%20Passing-brightgreen.svg)]()

**TokenTrieModel** is a high-throughput, adaptive **unsupervised sequence prediction engine** based on **Variable Order Markov Models (VOMM)** and **Context Mixing**, implemented over an unboxed **Reverse Suffix Trie** and accelerated via native **Cython / C-level primitives**.

Engineered for streaming sequential data, continuous edge learning, and high-frequency inference, TTM models arbitrary discrete token distributions ($\mathcal{V} \subset \{\text{str}, \text{int}\}$) with **$O(1)$ amortized training latency**, zero cold-start delay, lazy exponential timeline decay, and deterministic memory bounds.

---

## 📑 Table of Contents
1. [Mathematical Formulation & Problem Statement](#-1-mathematical-formulation--problem-statement)
2. [Core Architecture & Algorithmic Mechanics](#-2-core-architecture--algorithmic-mechanics)
   * [2.1 Reverse Suffix Trie Topology ($O(N)$ Traversal)](#21-reverse-suffix-trie-topology-on-traversal)
   * [2.2 Context Mixing & Dynamic Log-Scale Weighting](#22-context-mixing--dynamic-log-scale-weighting)
   * [2.3 Numerically Stable LogSumExp C-Level Normalization](#23-numerically-stable-logsumexp-c-level-normalization)
   * [2.4 Lazy Exponential Timeline Decay & Fast Path](#24-lazy-exponential-timeline-decay--fast-path)
   * [2.5 Katz-Style Unigram Smoothing Fallback](#25-katz-style-unigram-smoothing-fallback)
   * [2.6 Wildcard Context Masking via Bounded BFS](#26-wildcard-context-masking-via-bounded-bfs)
3. [Architectural Comparison: TTM vs. Neural & Classical Baselines](#-3-architectural-comparison)
4. [Empirical Benchmarks & Verification](#-4-empirical-benchmarks--verification)
   * [4.1 Real-World Showcase: T9 Keystroke Autocomplete](#41-real-world-showcase-t9-mobile-keystroke-autocomplete)
   * [4.2 Synthetic Stress Testing Suite](#42-synthetic-stress-testing-suite)
   * [4.3 Hardware-Agnostic Micro-Profiling & Latency Dispersion](#43-hardware-agnostic-micro-profiling--latency-dispersion)
5. [Installation & Build Requirements](#-5-installation--build-requirements)
6. [API Quickstart & Production Patterns](#-6-api-quickstart--production-patterns)
7. [Comprehensive Configuration Guide](#-7-comprehensive-configuration-guide)
8. [License & Citation](#-8-license--citation)

---

## 📐 1. Mathematical Formulation & Problem Statement

### 1.1 Unsupervised Streaming Sequence Modeling
Let $\mathcal{S} = (x_1, x_2, \dots, x_t, \dots)$ be an infinite non-stationary stochastic sequence of discrete categorical observations drawn from a dynamic alphabet:
$$x_t \in \mathcal{V}_t \subset \{\text{str}, \text{int}\}, \quad |\mathcal{V}_t| < \infty$$

Under causal autoregressive constraints, the predictive objective is to estimate the next-token conditional probability distribution:
$$P(x_t = y \mid \mathcal{H}_t), \quad \forall y \in \mathcal{V}_t$$
conditioned strictly on the historical working memory:
$$\mathcal{H}_t = (x_{t-k}, \dots, x_{t-2}, x_{t-1})$$
to minimize cumulative online logarithmic cross-entropy loss without offline retraining:
$$\min \mathcal{L} = -\sum_{t=1}^T \log_2 P\left(x_t = y_t^* \mid \mathcal{H}_t\right)$$

### 1.2 The Curse of Fixed-Order Markov Chains
Classical $N$-gram Markov models parameterize transitions through stationary contingency matrices:
$$P(x_t \mid x_{t-N+1}, \dots, x_{t-1})$$
* **High Bias ($N \le 2$):** Low-order chains are fundamentally blind to non-local temporal parity dependencies, failing on multi-symbol periodic trajectories and distant horizon tasks.
* **Combinatorial Explosion ($N \ge 5$):** Full transition tables scale exponentially with state space complexity $\mathcal{O}(|\mathcal{V}|^N)$. In an alphabet of size $|\mathcal{V}| = 10^4$, a 4-gram table demands $10^{16}$ parameters, rendering dense representation intractable.
* **Sample Sparsity:** In non-stationary environments, fixed-order contexts frequently encounter unobserved transitions, necessitating heuristic backoff cascades.

**The Solution:** Variable Order Markov Models (VOMM) adaptively prune the context depth per-branch, allocating memory exclusively along observed trajectory paths and interpolating active horizons via **Context Mixing**.

---

## ⚙️ 2. Core Architecture & Algorithmic Mechanics

```
               [ Root: TokenTrieNode ] (Depth 0)
                 /          |         \
               'c'         'b'        'a'      <- Suffix token x_{t-1}
              /   \         |
            'b'   'a'      'a'                 <- Suffix token x_{t-2}
           /
         'a'                                   <- Suffix token x_{t-3}
          |
     [ Node: Counts = { Target_y : Weight } ]
```

### 2.1 Reverse Suffix Trie Topology ($O(N)$ Traversal)
Conventional prefix trees store contexts chronologically ($x_{t-N} \to \dots \to x_{t-1}$). To query suffixes of varying lengths, a prefix tree requires $N$ separate traversals, yielding $\mathcal{O}(N^2)$ algorithmic complexity.

TTM stores preceding tokens in **reverse chronological order**:
$$\text{Path from Root} = (x_{t-1} \longrightarrow x_{t-2} \longrightarrow \dots \longrightarrow x_{t-k})$$
During inference, a single sequential descent matches all valid suffix context orders $\ell \in [n_{\min}, n_{\max}]$ simultaneously in strictly **$\mathcal{O}(N)$ deterministic operations**, completely eliminating string slice allocations and hash-table churn.

### 2.2 Context Mixing & Dynamic Log-Scale Weighting
Rather than executing hard decision trees, TTM blends observations across all valid context lengths $\ell \in \text{valid\_lengths}$. For each node along the matched reverse suffix path of length $\ell = |c|$, the unnormalized log-potential assigned to target candidate $y$ is defined as:
$$\phi(y \mid c) = \ln \text{Count}(c \to y) + \Delta t(c) \cdot \ln \gamma + \ell \cdot \mathcal{B}(\mathcal{V})$$
Where:
* $\text{Count}(c \to y)$ is the empirical transition frequency.
* $\gamma \in (0, 1]$ is the temporal exponential forgetting rate.
* $\Delta t(c) = t_{\text{current}} - t_{\text{last\_visit}}(c)$ is the elapsed timeline offset.
* $\mathcal{B}(\mathcal{V})$ is the **Dynamic Vocabulary Entropy Base**:
  $$\mathcal{B}(\mathcal{V}) = \begin{cases} \ln \max\left(2, |\mathcal{V}|\right), & \text{if } \texttt{alphabet\_autoscale}=\text{True} \\ \ln 2 \approx 0.69315, & \text{otherwise} \end{cases}$$

This logarithmic scaling factor ensures that longer, more specific context matches exponentially dominate shorter, ambiguous fallbacks while gracefully preserving predictive mass.

### 2.3 Numerically Stable LogSumExp C-Level Normalization
To prevent numerical underflow and precision collapse across disparate context lengths, log-potentials are accumulated via a pairwise LogSumExp reduction implemented directly in Cython over `libc.math`:
$$\text{LSE}(a, b) = \max(a, b) + \ln\left(1.0 + \exp\left(-|a - b|\right)\right) = \max(a, b) + \texttt{c\_log1p}\left(\texttt{c\_exp}\left(-|a - b|\right)\right)$$

The aggregate log-score for candidate token $y$ across all active context nodes $\mathcal{C}(\mathcal{H}_t)$ is:
$$\mathcal{S}(y) = \bigoplus_{c \in \mathcal{C}(\mathcal{H}_t)} \phi(y \mid c)$$

Normalized probability distributions under temperature scaling $\tau > 0$ are computed via shifted Softmax:
$$P(y \mid \mathcal{H}_t) = \frac{\exp\left( \frac{\mathcal{S}(y) - \mathcal{S}_{\max}}{\tau} \right)}{\sum_{y' \in \mathcal{V}} \exp\left( \frac{\mathcal{S}(y') - \mathcal{S}_{\max}}{\tau} \right)}, \quad \mathcal{S}_{\max} = \max_{y} \mathcal{S}(y)$$

### 2.4 Lazy Exponential Timeline Decay & Fast Path
In non-stationary streaming distributions, models with infinite static memory suffer from severe hysteresis. TTM implements **Lazy Exponential Decay**:
* Transition counts are stored as static floats during active updates.
* Weight degradation ($w = w_0 \cdot \gamma^{\Delta t}$) is deferred until a node is explicitly traversed.
* Mathematical evaluation is accelerated via an $O(1)$ precomputed integer power cache (`_power_cache`) and integer logarithm cache (`_int_log_cache`) bounded by `cache_size`.
* **Skip-Decay Accelerator:** When $\gamma = 1.0$ or `decay=None`, timeline delta tracking and floating-point power routines are completely bypassed (`skip_decay=True`), achieving maximum native execution speeds for stationary tasks.

### 2.5 Katz-Style Unigram Smoothing Fallback
If the working context $\mathcal{H}_t$ is unobserved in the Trie (or shorter than `min_depth`):
1. **`katz_backoff`:** Gracefully falls back to global unigram counts, with individual token frequencies independently decayed against their dedicated timeline tracking timestamps (`unigram_last_update`).
2. **`uniform`:** Allocates equal mass: $P(y) = \frac{1}{|\mathcal{V}|}$.

### 2.6 Wildcard Context Masking via Bounded BFS
For environments with sensor noise, typos, or omitted tokens, TTM implements breadth-first search (`masked_mode` in `{'linear', 'squared'}`) bounded by `max_beams`.
* **`linear`:** Explores wildcards exclusively at the suffix boundary (Phase 0), locking into strict path matching (Phase 1) upon the first structural token hit.
* **`squared`:** Explores branching wildcard substitutions across internal positions, scoring alignments by effective matching length.

---

## 📊 3. Architectural Comparison

| Architectural Dimension | **TokenTrieModel (TTM)** | **Transformers (LLMs)** | **Recurrent NNs (GRU/LSTM)** | **Classical $N$-Gram Tables** |
| :--- | :--- | :--- | :--- | :--- |
| **Learning Paradigm** | **Unsupervised / Online** | Self-Supervised (Pre-train) | Supervised / BPTT | Unsupervised / Counting |
| **Online Adaptation** | **$O(1)$ Instant Update** | Requires Fine-Tuning / LoRA | Backprop through time | $O(N)$ re-indexing |
| **Cold-Start Delay** | **Zero (Step 1 Ready)** | Heavy (Pre-training required) | High (Requires epochs) | Zero |
| **Inference Latency** | **$3–20\,\mu s$ (Microseconds)** | $10–100\,\text{ms}$ (Milliseconds) | $1–10\,\text{ms}$ | $1–5\,\mu s$ |
| **Memory Footprint** | **Sparse $\mathcal{O}(N \cdot T)$ bounded by GC** | Gigabytes / Terabytes (VRAM) | Fixed Hidden State Size | Exponential $\mathcal{O}(|\mathcal{V}|^N)$ |
| **Interpretability** | **100% Deterministic Counts** | Black-box latent weights | Black-box hidden vectors | Transparent |
| **Surgical Control** | **Inject / Delete branches** | None (Prone to hallucinations) | None | Full |
| **Hardware Target** | **Single CPU Core / Edge MCU** | High-end GPU Clusters | Edge / Server GPU / CPU | CPU RAM |

---

## 🔬 4. Empirical Benchmarks & Verification

### 4.1 Real-World Showcase: T9 Mobile Keystroke Autocomplete
To evaluate real-world sequential utility, TTM was benchmarked on an on-device **T9 Mobile Autocomplete Engine** trained on clean literary prose (Tolstoy, Dostoevsky) utilizing subword BPE tokenization.

The model preserves preceding completed words as atomic history tokens while actively typed character prefixes are compressed via BPE and prefixed with a collision-free marker:
$$\text{Context} = [w_{t-2},\, w_{t-1},\, \text{\_c}_0,\, \text{\_c}_1,\, \dots,\, \text{\_c}_k] \longrightarrow w_t$$

High-speed keystroke querying is achieved by terminating linear probability scans early over pre-sorted log-logits (`return_log_scores=True`), dropping query latency from $\sim 46,000$ iterations to just **2–5 iterations per keystroke**.

```
================================================================================
                         T9 TEST SET EVALUATION REPORT                          
================================================================================
  Training Corpus (80% Split)        : 159,174 Sentences (15.2 MB / 2.85M words)
  Training Throughput                : 230.38 sentences / second
  Total Active Trie Nodes            : 19,422,451 nodes
  Tracked Vocabulary Cardinality     : 46,644 tokens
--------------------------------------------------------------------------------
  Held-Out Evaluation Split (20%)    : 1,000 Sentences (59,815 Keystroke Steps)
  Top-1 Autocomplete Accuracy (@1)   : 48.87%
  Top-2 Suggestion Hit-Rate   (@2)   : 58.72%
  Top-3 Suggestion Hit-Rate   (@3)   : 63.53%
  Top-5 Suggestion Hit-Rate   (@5)   : 68.79%
  Mean Reciprocal Rank      (MRR@5)  : 0.5660
  Keystroke Savings Rate    (KSR %)  : 60.58%
  Incompleteness-Weighted Efficiency : 30.31%
================================================================================
```

#### Evaluation Metrics Formulation:
* **Top-$K$ Accuracy ($\text{Acc}@K$):** Percentage of keystrokes where target word $w_j \in \text{Top-}K(\mathcal{H}_j)$.
* **Keystroke Savings Rate ($\text{KSR}\%$):** Physical key presses eliminated by prompt acceptance:
  $$\text{KSR} = \frac{\sum_{w} (L_w - 1 - i_{\text{accepted}})}{\sum_{w} L_w} \times 100\% = \mathbf{60.58\%}$$
* **Incompleteness-Weighted Efficiency:** Credits early-word predictions ($i \ll L$):
  $$\text{Efficiency} = \frac{1}{N_{\text{keystrokes}}} \sum_{w} \sum_{i=0}^{L-1} \mathbb{I}(\text{hit}) \cdot \left( \frac{L - i}{L} \right) \times 100\% = \mathbf{30.31\%}$$

---

### 4.2 Synthetic Stress Testing Suite

All synthetic benchmarks are reproducible via `examples/basic_tests.ipynb` under strict zero-leakage online evaluation:

| Experiment | Generative Rule & Task | Theoretical Bound | Observed TTM Metric | Architectural Insight |
| :--- | :--- | :--- | :--- | :--- |
| **1. Discretized Sine Wave** | $V = \{0, \dots, 9\}$, Period $T = 25$ steps | $100.0\%$ (requires $N \ge 2$) | **$99.90\%$ Steady-State** (273 nodes) | Disambiguates phase transitions without state space explosion. |
| **2. Pattern Drift ($\gamma$)** | $[0, 1] \to [0, 0, 1, 1]$ at $t=500$ | $T_{\text{eff}} \approx \frac{1}{1-\gamma}$ | **Last error: $+6$ steps** ($\gamma=0.90$) vs. **$+54$ steps** ($\gamma=1.0$) | Exponential decay reduces adaptation latency by $9\times$. |
| **3. Stochastic Bernoulli** | $P(\text{'A'}) = 0.65$, i.i.d. stream | Bayes ceiling: $65.0\%$ | **$62.00\%$ Accuracy** ($97.3\%$ of sample ceiling) | Posterior probability converges to $62.97\%$ (true bias: $63.70\%$). |
| **4. Non-Local Lag Parity** | $x_t = x_{t-1} \oplus x_{t-5}$ ($k=5$) | Blind ($50\%$) for $N < 5$, $100\%$ for $N \ge 5$ | **$100.0\%$** at $n_{\max}=8$ (114 nodes vs. 511 full tree) | Resolves non-local parity with $77.7\%$ structural memory reduction. |

---

### 4.3 Hardware-Agnostic Micro-Profiling & Latency Dispersion

Benchmarked on **AMD64 (Zen 3 Architecture, CPython 3.11.3, Windows 10)** using nanosecond monotonic timing (`time.perf_counter_ns`):

#### Computational Primitives ($n_{\max} = 6$, Static Mode $\gamma = 1.0$, Active Nodes: $25,403$)
| Primitive Operation | Throughput (ops/sec) | Mean Latency ($\mu s$) | Median $P_{50}$ ($\mu s$) | Tail $P_{90}$ ($\mu s$) | Tail $P_{99}$ ($\mu s$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ingestion (`update`)** | **$75,052$** | $13.32$ | **$3.80$** | $5.50$ | $9.10$ |
| **Inference (`predict`)** | **$36,653$** | $27.28$ | **$21.20$** | $58.10$ | $73.10$ |
| **Logits Scan (`predict_proba`)** | **$58,408$** | $17.12$ | **$12.45$** | $38.80$ | $49.60$ |

#### Context Depth Horizon Scaling & Math Degradation Analysis
$$\text{Decay Penalty} = \frac{\text{Throughput}_{\text{static}} - \text{Throughput}_{\text{decay}}}{\text{Throughput}_{\text{static}}} \times 100\%$$

| Context Depth ($n_{\max}$) | Static Throughput (kOps/s) | Decaying Throughput (kOps/s) | Decay Penalty ($\%$) | Static $P_{50}$ ($\mu s$) | Static $P_{99}$ ($\mu s$) | Total Trie Nodes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2** | $368.90$ | $166.21$ | $+54.9\%$ | $1.50$ | $3.90$ | $2,545$ |
| **4** | $117.58$ | $83.12$ | $+29.3\%$ | $2.60$ | $7.00$ | $13,418$ |
| **8** | $46.90$ | $45.01$ | $+4.0\%$ | $5.00$ | $14.60$ | $37,388$ |
| **16** | $24.59$ | $24.51$ | **$+0.4\%$** | $9.40$ | $18.50$ | $85,288$ |

> **Systems Takeaway:** At shallow depths ($N \le 4$), tree traversal is so fast ($1.5\,\mu s$) that floating-point math incurs measurable overhead. At operational depths ($N \ge 8$), tree descent dominates, rendering timeline decay math **virtually zero-cost ($0.4\%$ overhead)**.

---

## 📦 5. Installation & Build Requirements

TTM compiles directly into a native C-extension via Cython and requires a standard C99 compiler (GCC, Clang, or MSVC).

```bash
# Clone the repository
git clone https://github.com/Icold21/token-trie.git
cd token-trie

# Option A: Core Engine Only
pip install -e .

# Option B: Full Development & Benchmark Suite (Jupyter, Matplotlib, Tiktoken)
pip install -e .[full]
```

### Running the Verification Suite
```bash
pytest -v
```

---

## 💻 6. API Quickstart & Production Patterns

### 6.1 Online Continuous Learning Loop
```python
from tokentrie import TokenTrieModel

# Initialize model with maximum horizon of 5 tokens and moderate decay
model = TokenTrieModel(max_depth=5, decay=0.98)

stream = ["login", "view_cart", "checkout", "login", "view_cart", "checkout"]

for token in stream:
    # 1. Query prediction prior to observation (Zero Data-Leakage)
    predicted_next = model.predict(temperature=0.0)  # Greedy argmax
    print(f"Observed: {token:<12} | Predicted Next: {predicted_next}")
    
    # 2. Ingest actual ground-truth token (O(1) amortized update)
    model.update(token)
```

### 6.2 Advanced Generation Controls (LLM-Grade Sampling)
```python
# Extract raw, unnormalized Cython log-logits (pre-sorted descending)
log_logits = model.predict_proba(return_log_scores=True)

# Temperature-scaled probabilistic inference
stochastic_token = model.predict(temperature=0.8)

# Nucleus Sampling (Top-p: limits candidates to top 90% cumulative mass)
nucleus_token = model.predict(top_p=0.90)

# Top-K Rank Filtering
top_k_token = model.predict(top_k=3)

# Typo-tolerant inference via Masked Beam Search
resilient_token = model.predict(masked_mode="linear")
```

### 6.3 Surgical Memory Manipulation
Explicitly inject deterministic business logic, overwrite rules, or prune subtrees:
```python
# Surgically inject or overwrite explicit contextual rules
model.set_branches([
    (["auth", "failure", "retry"], {"lockout": 100.0, "captcha": 10.0})
])

# Sever a contextual branch and ALL its descendant subtrees permanently
model.delete_branches([
    ["auth", "failure", "retry"]
])

# Dynamically reconfigure horizons without cold-starting
model.reconfigure({"max_depth": 3, "decay": 1.0})
```

### 6.4 Heterogeneous Asymmetric Federated Learning
Merge distributed models across heterogeneous edge timelines, depths, and decay rates:
```python
global_server = TokenTrieModel(max_depth=3, decay=0.9)
edge_client = TokenTrieModel(max_depth=8, decay=0.999).fit(["deep", "edge", "pattern"])

# Host automatically upgrades depth bounds and mathematically projects timelines
global_server.merge(edge_client)
```

---

## 🛠 7. Comprehensive Configuration Guide

| Parameter | Type | Default | Valid Range | Algorithmic Description |
| :--- | :--- | :--- | :--- | :--- |
| `max_depth` | `int` | `10` | $[1, \infty)$ | Maximum context horizon (Markov order) tracked in the Reverse Suffix Trie. |
| `min_depth` | `int` | `1` | $[1, \text{max\_depth}]$ | Minimum context length required before activating associative transitions. |
| `depth_list` | `Optional[List[int]]`| `None` | Subsets of $\mathbb{N}^+$ | Tracks sparse context horizons explicitly (e.g., `[2, 5, 8]`) without allocating intermediate nodes. |
| `decay` | `Optional[float]` | `0.99` | $[0.0, 1.0]$ | Exponential forgetting coefficient ($\gamma$). Set to `1.0` or `None` to activate fast-path. |
| `alphabet_autoscale`| `bool` | `True` | `{True, False}` | Dynamically calibrates entropy scaling base: $\ln \max(2, |\mathcal{V}|)$. |
| `fallback_mode` | `str` | `'katz_backoff'` | `{'katz_backoff', 'uniform'}` | Smoothing policy when encountering unobserved contexts. |
| `pruning_mode` | `str` | `'fixed'` | `{'fixed', 'dynamic'}` | Garbage collection strategy: interval-based (`'fixed'`) vs. node density (`'dynamic'`). |
| `pruning_step` | `int` | `1000` | $[1, \infty)$ | Step interval or target baseline for triggering tree sweeps. |
| `pruning_threshold`| `float` | `1e-6` | $[0.0, \infty)$ | Minimum transition weight below which nodes/counts are physically purged. |
| `max_beams` | `int` | `1000` | $[1, \infty)$ | Maximum queue iterations during masked beam search traversal. |
| `cache_size` | `int` | `4096` | $[1, \infty)$ | Capacity limit for LRU power and integer logarithm math caches. |

---

## 📄 8. License & Citation

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

### Citation
```bibtex
@software{kholodilo2026tokentrie,
  author       = {Ivan Kholodilo},
  title        = {TokenTrieModel: High-Performance Adaptive Variable Order Markov Models via Reverse Suffix Tries},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/Icold21/token-trie}
}
```