# TokenTrieModel (TTM) 🧠

[![Version](https://img.shields.io/badge/version-1.1.0-blue.svg)](https://pypi.org/project/tokentrie/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Cython](https://img.shields.io/badge/Cython-Accelerated-yellow.svg)](https://cython.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()

**TokenTrieModel (TTM)** is a high-throughput, adaptive **unsupervised sequence prediction engine** based on **Variable Order Markov Models (VOMM)** and **Context Mixing**, implemented over an unboxed **Reverse Suffix Trie** and accelerated via native **Cython / C-level primitives**.

Engineered for streaming sequential processes, real-time edge computing, and microsecond-scale inference, TTM models arbitrary discrete categorical distributions ($\mathcal{V} \subset \{\text{str}, \text{int}\}$) with **$O(1)$ amortized training latency**, zero cold-start delay, lazy exponential timeline decay, and deterministic memory bounds. Version 1.1.0 incorporates **Information-Theoretic Redundancy scaling**, **Depth-Stratified Empirical Bayes shrinkage**, and **$O(1)$ streaming Markov entropy rate estimation**.

---

## 📑 Table of Contents
1. [Mathematical Formulation & Problem Statement](#-1-mathematical-formulation--problem-statement)
   * [1.1 Autoregressive Online Sequence Modeling](#11-autoregressive-online-sequence-modeling)
   * [1.2 Statistical Trade-offs: Fixed-Order vs. Variable-Order Markov Models](#12-statistical-trade-offs-fixed-order-vs-variable-order-markov-models)
2. [Core Architecture & Algorithmic Foundations](#-2-core-architecture--algorithmic-foundations)
   * [2.1 Reverse Suffix Trie Topology ($O(N)$ Traversal)](#21-reverse-suffix-trie-topology-on-traversal)
   * [2.2 Context Mixing with Information-Theoretic Shrinkage](#22-context-mixing-with-information-theoretic-shrinkage)
   * [2.3 Numerically Stable LogSumExp C-Level Normalization](#23-numerically-stable-logsumexp-c-level-normalization)
   * [2.4 Streaming Markov Entropy Rate Estimation ($O(1)$ Updates)](#24-streaming-markov-entropy-rate-estimation-o1-updates)
   * [2.5 Lazy Exponential Timeline Decay & Fast Path](#25-lazy-exponential-timeline-decay--fast-path)
   * [2.6 Interpolated Unigram Fallback with Dirichlet Prior](#26-interpolated-unigram-fallback-with-dirichlet-prior)
   * [2.7 Wildcard Context Masking via Bounded BFS](#27-wildcard-context-masking-via-bounded-bfs)
3. [Theoretical & Empirical Model Comparison](#-3-theoretical--empirical-model-comparison)
4. [Empirical Benchmarks & Scientific Validation](#-4-empirical-benchmarks--scientific-validation)
   * [4.1 Real-World Showcase: T9 Keystroke Autocomplete](#41-real-world-showcase-t9-mobile-keystroke-autocomplete)
   * [4.2 Synthetic Deterministic & Stochastic Benchmark Suite](#42-synthetic-deterministic--stochastic-benchmark-suite)
   * [4.3 Hardware-Agnostic Micro-Profiling & Latency Dispersion](#43-hardware-agnostic-micro-profiling--latency-dispersion)
5. [Installation & Build Configuration](#-5-installation--build-configuration)
6. [API Quickstart & Production Recipes](#-6-api-quickstart--production-recipes)
   * [6.1 Online Continuous Learning Loop](#61-online-continuous-learning-loop)
   * [6.2 Advanced Generation Controls (LLM-Grade Sampling)](#62-advanced-generation-controls-llm-grade-sampling)
   * [6.3 Real-Time Markov Entropy Diagnostics](#63-real-time-markov-entropy-diagnostics)
   * [6.4 Surgical Memory Manipulation & Dynamic Reconfiguration](#64-surgical-memory-manipulation--dynamic-reconfiguration)
7. [Comprehensive Configuration Guide](#-7-comprehensive-configuration-guide)
8. [License & Citation](#-8-license--citation)

---

## 📐 1. Mathematical Formulation & Problem Statement

### 1.1 Autoregressive Online Sequence Modeling
Let $\mathcal{S} = (x_1, x_2, \dots, x_t, \dots)$ be a discrete non-stationary stochastic process over a dynamic alphabet:

$$
x_t \in \mathcal{V}_t \subset \{\text{str}, \text{int}\}, \quad |\mathcal{V}_t| < \infty
$$

Under causal autoregressive constraints, the goal is to recursively infer the posterior transition probability distribution of the next event $x_t$:

$$
P(x_t = y \mid \mathcal{H}_t), \quad \forall y \in \mathcal{V}_t
$$

conditioned on the sliding working history of order $k$:

$$
\mathcal{H}_t = (x_{t-k}, \dots, x_{t-2}, x_{t-1})
$$

The statistical learning objective is to minimize cumulative online logarithmic cross-entropy loss without offline batch retraining:

$$
\min \mathcal{L} = -\sum_{t=1}^T \log_2 P\left(x_t = y_t^* \mid \mathcal{H}_t\right)
$$

### 1.2 Statistical Trade-offs: Fixed-Order vs. Variable-Order Markov Models
Classical stationary $N$-gram chains parameterize transitions via contingency matrices conditioned on fixed-length context windows:

$$
P(x_t \mid x_{t-N+1}^{t-1})
$$

* **High Structural Bias ($N \le 2$):** Low-order chains cannot capture non-local temporal dependencies, periodic trajectories, or long-range state constraints.
* **Combinatorial State Space Explosion ($N \ge 5$):** Dense transition tables scale exponentially with state space complexity $\mathcal{O}(|\mathcal{V}|^N)$. In an alphabet of size $|\mathcal{V}| = 10^3$, a 5-gram matrix requires $10^{15}$ parameters, rendering dense tabular parameterization intractable.
* **Sample Sparsity & Zero-Frequency Pathology:** In non-stationary environments, fixed-order contexts frequently encounter unobserved combinations, triggering brittle, heuristic backoff cascades.

**The Solution:** Variable Order Markov Models (VOMM) allocate tree depth along observed trajectory paths and interpolate multi-scale dependencies via **Continuous Context Mixing**.

---

## ⚙️ 2. Core Architecture & Algorithmic Foundations

```
               [ Root: TokenTrieNode ] (Depth 0)
                 /          |         \
               'c'         'b'        'a'      <- Suffix token x_{t-1}
              /   \         |
            'b'   'a'      'a'                 <- Suffix token x_{t-2}
           /
         'a'                                   <- Suffix token x_{t-3}
          |
     [ Node: Counts = { Target_y : Weight }, Total Mass = N_c ]
```

### 2.1 Reverse Suffix Trie Topology ($O(N)$ Traversal)
Conventional prefix trees store contexts chronologically ($x_{t-N} \to \dots \to x_{t-1}$). To query all active suffix orders of varying lengths, a prefix tree requires $N$ distinct traversals, yielding $\mathcal{O}(N^2)$ algorithmic complexity.

TTM stores preceding tokens in **reverse chronological order**:

$$
\text{Path from Root} = (x_{t-1} \longrightarrow x_{t-2} \longrightarrow \dots \longrightarrow x_{t-k})
$$

During inference, a single sequential descent matches all valid suffix context orders $\ell \in [n_{\min}, n_{\max}]$ simultaneously in strictly **$\mathcal{O}(N)$ deterministic operations**, completely eliminating memory allocations and hash table thrashing.

---

### 2.2 Context Mixing with Information-Theoretic Shrinkage
Rather than executing hard pruning decisions, TTM blends observations across all active context orders $\ell \in \mathcal{L}_{\text{valid}}$. For each node along the matched reverse suffix path of length $\ell = |c|$, the unnormalized log-potential assigned to candidate target token $y$ is:

$$
\phi(y \mid c) = \ln \text{Count}(c \to y) + \Delta t(c) \cdot \ln \gamma + \ell \cdot \tilde{\beta}(c) \cdot \mathcal{B}(\mathcal{V})
$$

Where:
* $\text{Count}(c \to y)$ is the empirical transition frequency.
* $\gamma \in (0, 1]$ is the temporal exponential forgetting factor (`decay`).
* $\Delta t(c) = t_{\text{curr}} - t_{\text{last}}(c)$ is the elapsed timeline offset.
* $\mathcal{B}(\mathcal{V})$ is the **Logarithmic Scale Base**:

$$
\mathcal{B}(\mathcal{V}) = \begin{cases} 
\ln \max\left(2, |\mathcal{V}|\right), & \text{if } \texttt{alphabet\_autoscale=True} \\ 
\ln 2 \approx 0.69315, & \text{otherwise} 
\end{cases}
$$

* $\tilde{\beta}(c) \in [0, 1]$ is the **Joint Information-Theoretic Modulator**, defined as the product of information purity and empirical sample support:

$$
\tilde{\beta}(c) = \beta_{\text{raw}}(c) \cdot w_{\text{node}}(c)
$$

#### Component 1: Information-Theoretic Redundancy ($\beta_{\text{raw}}$)
Measures the predictability of node $c$ relative to a maximum-entropy uniform null model:

$$
\beta_{\text{raw}}(c) = \max\left(0.0,\, 1.0 - \frac{H(c)}{\mathcal{B}(\mathcal{V})}\right) = \frac{D_{\text{KL}}\left(P(\cdot \mid c) \,\|\, \mathcal{U}(\mathcal{V})\right)}{\mathcal{B}(\mathcal{V})}
$$

Where $H(c)$ is the local transition entropy computed in $O(|\text{children}|)$:

$$
H(c) = \ln N_c - \frac{1}{N_c} \sum_{y} \text{Count}(c \to y) \ln \text{Count}(c \to y), \quad N_c = \sum_{y} \text{Count}(c \to y)
$$

* **Deterministic Transitions ($H(c) \to 0$):** $\beta_{\text{raw}} \to 1.0$ (maximum contextual depth boost).
* **Pure Uniform Noise ($H(c) \to \ln |\mathcal{V}|$):** $\beta_{\text{raw}} \to 0.0$ (depth boost neutralized; reverts to flat counts).

#### Component 2: Depth-Stratified Empirical Bayes Shrinkage ($w_{\text{node}}$)
Empirical entropy alone is vulnerable to small-sample bias: a node visited once ($N_c = 1$) has $H(c) = 0$, falsely appearing deterministic. TTM scales $\beta_{\text{raw}}$ by an empirical sample support factor:

$$
w_{\text{node}}(c) = \frac{N_c}{N_c + \text{median}^{(\ell)}}
$$

Where $\text{median}^{(\ell)}$ is the empirical median node mass at Markov order $\ell$, tracked online in **$O(1)$ time** using a 96-bin logarithmic histogram accumulator.

---

### 2.3 Numerically Stable LogSumExp C-Level Normalization
To prevent floating-point underflow across disparate context horizons, log-potentials are accumulated via a pairwise LogSumExp reduction implemented directly in Cython over `libc.math`:

$$
\text{LSE}(a, b) = \max(a, b) + \ln\left(1.0 + \exp\left(-|a - b|\right)\right)
$$

The aggregate log-score for candidate token $y$ across all matched context nodes $\mathcal{C}(\mathcal{H}_t)$ is:

$$
\mathcal{S}(y) = \bigoplus_{c \in \mathcal{C}(\mathcal{H}_t)} \phi(y \mid c)
$$

Normalized probability distributions under temperature scaling $\tau > 0$ are evaluated via shifted Softmax:

$$
P(y \mid \mathcal{H}_t) = \frac{\exp\left( \frac{\mathcal{S}(y) - \mathcal{S}_{\max}}{\tau} \right)}{\sum_{y' \in \mathcal{V}} \exp\left( \frac{\mathcal{S}(y') - \mathcal{S}_{\max}}{\tau} \right)}, \quad \mathcal{S}_{\max} = \max_{y'} \mathcal{S}(y')
$$

---

### 2.4 Streaming Markov Entropy Rate Estimation ($O(1)$ Updates)
TTM tracks the empirical conditional Markov entropy for each order $\ell$ without full tree traversals:

$$
\hat{H}(X_t \mid X_{t-\ell : t-1}) = \frac{\Sigma_{\text{node}}^{(\ell)} - \Sigma_{\text{trans}}^{(\ell)}}{N^{(\ell)}}
$$

Using the incremental identity for $\Sigma = \sum n_i \ln n_i$:

$$
\Delta \Sigma = (n_k + 1)\ln(n_k + 1) - n_k \ln n_k
$$

Both $\Sigma_{\text{node}}^{(\ell)}$ and $\Sigma_{\text{trans}}^{(\ell)}$ update in $O(1)$ time per ingested token, enabling real-time monitoring of contextual entropy rates across depths.

---

### 2.5 Lazy Exponential Timeline Decay & Fast Path
In non-stationary streaming data, infinite-horizon models suffer from historical inertia. TTM implements **Lazy Exponential Decay**:
* Transition counts remain static until a node is explicitly traversed.
* Decay ($w = w_0 \cdot \gamma^{\Delta t}$) is lazily evaluated using an $O(1)$ precomputed power cache (`_power_cache`) and integer logarithm cache (`_int_log_cache`) bounded by `cache_size`.
* **Skip-Decay Fast Path:** When $\gamma = 1.0$ or `decay=None`, timeline delta arithmetic is bypassed (`skip_decay=True`), achieving maximum native execution throughput.

---

### 2.6 Interpolated Unigram Fallback with Dirichlet Prior
When context $\mathcal{H}_t$ is unobserved or shorter than `min_depth`:
1. **`interpolated_unigram` (Default):** Blends unigram probabilities with an additive Dirichlet smoothing prior $\alpha$:
   $$P_{\text{unigram}}(y) = \frac{\text{Count}(y) + \alpha}{\sum_{y'} \text{Count}(y') + \alpha |\mathcal{V}|}$$
   Individual token frequencies independently decay according to their respective update timestamps (`unigram_last_update`).
2. **`uniform`:** Assigns uniform mass: $P(y) = \frac{1}{|\mathcal{V}|}$.

---

### 2.7 Wildcard Context Masking via Bounded BFS
For noisy environments with missing tokens or typos, TTM implements queue-based breadth-first search (`masked_mode` in `{'linear', 'squared'}`) bounded by `max_beams`:
* **`linear`:** Explores wildcards exclusively at the suffix boundary (Phase 0), locking into deterministic descent (Phase 1) upon the first structural token match.
* **`squared`:** Explores branching wildcard substitutions across arbitrary internal positions, scoring alignments by effective matching length.

---

## 📊 3. Theoretical & Empirical Model Comparison

| Dimension | **TokenTrieModel (TTM)** | **Probabilistic Suffix Tree (PST)** | **DeepLog (LSTM)** | **Transformers (LLMs)** | **Fixed $N$-Gram Tables** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Learning Paradigm** | **Unsupervised / Online** | Unsupervised / Batch | Self-Supervised / Backprop | Self-Supervised / Pretrain | Unsupervised / Counting |
| **Online Adaptation** | **$O(1)$ Instant Update** | Batch Re-pruning | Slow (Backprop required) | Fine-tuning / Context window | $O(N)$ Hash updates |
| **Cold-Start Latency** | **Zero (Step 1 Ready)** | Zero | High (Requires epochs) | Extreme | Zero |
| **Inference Latency** | **$3–20\,\mu s$** | $5–25\,\mu s$ | $500–2,000\,\mu s$ | $10–100\,\text{ms}$ | $1–5\,\mu s$ |
| **Memory Footprint** | **Sparse $\mathcal{O}(N \cdot T)$ (GC)** | Pruned Suffix Graph | Dense Weights (MBs) | Gigabytes / Terabytes | Exponential $\mathcal{O}(|\mathcal{V}|^N)$ |
| **Interpretability** | **100% Deterministic Counts** | Deterministic Graph | Black-box hidden state | Black-box attention maps | Transparent |
| **Surgical Control** | **Inject / Delete branches** | Rebuild required | None | None | Full |
| **Hardware Target** | **Single CPU Core / Edge MCU** | CPU RAM | GPU / Server CPU | High-end GPU Clusters | CPU RAM |

---

## 🔬 4. Empirical Benchmarks & Scientific Validation

### 4.1 Real-World Showcase: T9 Keystroke Autocomplete
TTM was evaluated on a standalone **T9 Mobile Keystroke Autocomplete Engine** trained on clean literary prose utilizing subword Byte-Pair Encoding (BPE).

Completed history words are retained as context tokens, while actively typed character prefixes are compressed via subwords and queried sequentially:

$$
\text{Context} = [w_{t-2},\, w_{t-1},\, c_0,\, c_1,\, \dots,\, c_k] \longrightarrow w_t
$$

High-speed completion queries scan sorted log-potentials early (`return_log_scores=True`), reducing inference sweeps to **2–5 iterations per keystroke**.

```
================================================================================
                         T9 TEST SET EVALUATION REPORT                          
================================================================================
  Training Corpus (80% Split)        : 159,174 Sentences (198,968 Clean Retained)
  Training Throughput                : 1,836.70 sentences / second (01:26 total)
  Total Active Trie Nodes            : 19,422,451 nodes
  Tracked Vocabulary Cardinality     : 46,644 tokens
--------------------------------------------------------------------------------
  Evaluated Keystroke Steps          : 17,967 Keystrokes (300 Test Sentences)
  Top-1 Autocomplete Accuracy (@1)   : 48.32%
  Top-2 Suggestion Hit-Rate   (@2)   : 59.04%
  Top-3 Suggestion Hit-Rate   (@3)   : 63.67%
  Top-5 Suggestion Hit-Rate   (@5)   : 68.98%
  Mean Reciprocal Rank      (MRR@5)  : 0.5644
  Keystroke Savings Rate    (KSR %)  : 57.43%
  Incompleteness-Weighted Efficiency : 30.38%
================================================================================
```

#### Metric Definitions:
* **Top-$K$ Accuracy ($\text{Acc}@K$):** Percentage of keystroke queries where target word $w^* \in \text{Top-}K(\mathcal{H}_j)$.
* **Keystroke Savings Rate ($\text{KSR}\%$):** Physical keystrokes eliminated via candidate acceptance:
  $$\text{KSR} = \frac{\sum_{w} (L_w - 1 - i_{\text{accepted}})}{\sum_{w} L_w} \times 100\% = \mathbf{57.43\%}$$
* **Incompleteness-Weighted Efficiency:** Credits early-prefix predictive hits ($i \ll L$):
  $$\text{Efficiency} = \frac{1}{N_{\text{keystrokes}}} \sum_{w} \sum_{i=0}^{L-1} \mathbb{I}(\text{hit}) \cdot \left( \frac{L - i}{L} \right) \times 100\% = \mathbf{30.38\%}$$

---

### 4.2 Synthetic Deterministic & Stochastic Benchmark Suite
All synthetic benchmarks are reproducible via `examples/basic_tests.ipynb` under zero-leakage online evaluation:

| Experiment | Generative Rule & Task | Theoretical Bound | Observed TTM Metric | Architectural Insight |
| :--- | :--- | :--- | :--- | :--- |
| **1. Discretized Sine Wave** | $V = \{0, \dots, 9\}$, Period $T = 25$ steps | $100.0\%$ (requires $N \ge 2$) | **$99.90\%$ Steady-State** (273 nodes) | Disambiguates phase transitions without state space explosion. |
| **2. Pattern Drift ($\gamma$)** | $[0, 1] \to [0, 0, 1, 1]$ at $t=500$ | $T_{\text{eff}} \approx \frac{1}{1-\gamma}$ | **Last error: $+6$ steps** ($\gamma=0.90$) vs. **$+54$ steps** ($\gamma=1.0$) | Exponential decay reduces adaptation latency by $9\times$. |
| **3. Stochastic Bernoulli** | $P(\text{'A'}) = 0.65$, i.i.d. stream | Bayes ceiling: $65.0\%$ | **$62.00\%$ Accuracy** ($97.3\%$ of ceiling) | Posterior probability converges to $62.97\%$ (true bias: $63.70\%$). |
| **4. Non-Local Lag Parity** | $x_t = x_{t-1} \oplus x_{t-5}$ ($k=5$) | Blind ($50\%$) for $N < 5$, $100\%$ for $N \ge 5$ | **$100.0\%$** at $n_{\max}=8$ (114 nodes vs. 511 full) | Resolves non-local parity with $77.7\%$ structural memory reduction. |

---

### 4.3 Hardware-Agnostic Micro-Profiling & Latency Dispersion

#### Execution Topology & Hardware Platform
* **Architecture:** AMD64 (Zen 3 Architecture — AMD Family 25, Model 80, Stepping 0)
* **Operating System:** Windows 10 (CPython 3.11.3, Monotonic Timer Resolution: $100.00\,\text{ns}$)

#### Core Computational Primitives ($n_{\max} = 6$, Static Mode $\gamma = 1.0$, Active Nodes: $25,403$)
| Primitive Operation | Throughput (ops/sec) | Mean Latency ($\mu s$) | Median $P_{50}$ ($\mu s$) | Tail $P_{90}$ ($\mu s$) | Tail $P_{99}$ ($\mu s$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ingestion (`update`)** | **$80,582$** | $12.41$ | **$1.60$** | $3.00$ | $5.30$ |
| **Inference (`predict`)** | **$24,746$** | $40.41$ | **$38.00$** | $53.10$ | $64.40$ |
| **Logits Scan (`predict_proba`)** | **$40,029$** | $24.98$ | **$22.40$** | $36.80$ | $46.40$ |

#### Context Depth Horizon Scaling & Math Degradation Analysis

$$
\text{Decay Penalty} = \frac{\text{Throughput}_{\text{static}} - \text{Throughput}_{\text{decay}}}{\text{Throughput}_{\text{static}}} \times 100\%
$$

| Context Depth ($n_{\max}$) | Static Throughput (kOps/s) | Decaying Throughput (kOps/s) | Decay Penalty ($\%$) | Static $P_{50}$ ($\mu s$) | Static $P_{99}$ ($\mu s$) | Total Trie Nodes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2** | $466.10$ | $329.78$ | $+29.2\%$ | $0.80$ | $2.50$ | $2,545$ |
| **4** | $119.89$ | $131.23$ | $-9.5\%$ | $1.20$ | $3.60$ | $13,418$ |
| **8** | $38.63$ | $66.62$ | $-72.5\%$ | $2.10$ | $7.70$ | $37,388$ |
| **16** | $22.84$ | $34.87$ | **$-52.7\%$** | $4.10$ | $9.40$ | $85,288$ |

> **Systems Takeaway:** At shallow context horizons ($N \le 2$), static ingestion is faster because tree traversal takes under $1\,\mu s$ and floating-point power decay arithmetic adds measurable overhead. However, at deeper operational horizons ($N \ge 4$), **decaying mode achieves significantly higher throughput (up to $1.72\times$ faster)**. Exponential weight decay and garbage collection prune inactive branches, preserving L1/L2 cache locality and dramatically reducing tree traversal depth.

---

## 📦 5. Installation & Build Configuration

TTM compiles directly into a native C-extension via Cython and requires a standard C99 compiler (GCC, Clang, or MSVC).

```bash
# Clone repository
git clone https://github.com/Icold21/token-trie.git
cd token-trie

# Option A: Core Engine Only
pip install -e . --no-build-isolation

# Option B: Full Development & Benchmark Suite (Jupyter, PyTorch, CatBoost, Optuna)
pip install -e ".[full]" --no-build-isolation
```

### Verification Test Suite
```bash
pytest -v
```

---

## 💻 6. API Quickstart & Production Recipes

### 6.1 Online Continuous Learning Loop
```python
from tokentrie import TokenTrieModel

# Initialize model with maximum horizon of 5 tokens, moderate decay, and shrinkage
model = TokenTrieModel(max_depth=5, decay=0.98, entropy_weighting=True)

stream = ["login", "view_cart", "checkout", "login", "view_cart", "checkout"]

for token in stream:
    # 1. Query prediction prior to observation (zero data leakage)
    predicted_next = model.predict(temperature=0.0)  # Greedy argmax
    print(f"Observed: {token:<12} | Predicted Next: {predicted_next}")
    
    # 2. Ingest actual ground-truth token (O(1) amortized update)
    model.update(token)
```

### 6.2 Advanced Generation Controls (LLM-Grade Sampling)
```python
# Extract raw, unnormalized Cython log-potentials (sorted descending)
log_potentials = model.predict_proba(return_log_scores=True)

# Temperature-scaled probabilistic inference
stochastic_token = model.predict(temperature=0.8)

# Nucleus Sampling (Top-p: restricts candidates to top 90% cumulative mass)
nucleus_token = model.predict(top_p=0.90)

# Top-K Rank Filtering
top_k_token = model.predict(top_k=3)

# Typo-tolerant inference via Masked Beam Search
resilient_token = model.predict(masked_mode="linear")
```

### 6.3 Real-Time Markov Entropy Diagnostics
```python
# Inspect empirical statistical profiles across context depths in O(1)
depth_diagnostics = model.get_depth_stats()

for depth, stats in depth_diagnostics.items():
    print(f"Depth {depth:02d} | "
          f"Entropy: {stats['conditional_entropy']:.4f} nats | "
          f"Median Mass: {stats['median_node_mass']:.1f} | "
          f"Active Nodes: {stats['active_nodes']}")
```

### 6.4 Surgical Memory Manipulation & Dynamic Reconfiguration
```python
# Surgically inject or overwrite explicit deterministic transitions
model.set_branches([
    (["auth", "failure", "retry"], {"lockout": 100.0, "captcha": 10.0})
])

# Sever a contextual branch and all its descendant subtrees permanently
model.delete_branches([
    ["auth", "failure", "retry"]
])

# Access read-only model diagnostics
print(f"Total Active Nodes: {model.node_count}")
print(f"Known Vocabulary: {model.vocab_size}")

# Dynamically reconfigure horizons without cold-starting
model.reconfigure({"max_depth": 3, "decay": 1.0, "entropy_weighting": True})
```

---

## 🛠 7. Comprehensive Configuration Guide

| Parameter | Type | Default | Valid Range | Algorithmic Description |
| :--- | :--- | :--- | :--- | :--- |
| `max_depth` | `int` | `10` | `[1, 63]` | Maximum Markov order context horizon tracked in the Reverse Suffix Trie. |
| `min_depth` | `int` | `1` | `[1, max_depth]` | Minimum context length required before activating associative transitions. |
| `depth_list` | `Optional[List[int]]`| `None` | Subsets of positive integers | Tracks sparse context horizons explicitly (e.g., `[2, 5, 8]`) without intermediate node overhead. |
| `decay` | `Optional[float]` | `0.99` | `[0.0, 1.0]` | Exponential forgetting coefficient ($\gamma$). Set to `1.0` or `None` to enable the skip-decay fast path. |
| `entropy_weighting`| `bool` | `True` | `{True, False}` | Enables Information-Theoretic Redundancy ($\beta_{\text{raw}}$) and Empirical Bayes median shrinkage ($w_{\text{node}}$). |
| `alphabet_autoscale`| `bool` | `True` | `{True, False}` | Scales log base by vocabulary cardinality: $\ln \max(2, \|\mathcal{V}\|)$. Set to `False` ($\ln 2$) for massive subword vocabularies. |
| `fallback_mode` | `str` | `'interpolated_unigram'` | `{'interpolated_unigram', 'uniform'}` | Smoothing policy when encountering unobserved contexts. |
| `smoothing_prior` | `float` | `1e-3` | `(0.0, inf)` | Dirichlet additive pseudo-count prior ($\alpha$) protecting unigram fallbacks. |
| `pruning_mode` | `str` | `'fixed'` | `{'fixed', 'dynamic'}` | Garbage collection strategy: interval-based (`'fixed'`) vs. node density thresholding (`'dynamic'`). |
| `pruning_step` | `int` | `1000` | `[1, inf)` | Step interval or target baseline for triggering tree sweeps. |
| `pruning_threshold`| `float` | `1e-6` | `[0.0, inf)` | Minimum transition weight below which nodes/counts are physically purged. |
| `max_beams` | `int` | `1000` | `[1, inf)` | Maximum queue exploration budget during masked beam search traversal. |
| `cache_size` | `int` | `4096` | `[1, inf)` | Capacity limit for integer power and integer logarithm math caches. |

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