# EXP--TRJ001 Final Research Paper / Technical Report

## Preserving Task-Relevant Geometry in LLM Hidden States: Evidence That Variance-Maximizing PCA Can Obscure Hallucination Trajectory Signals

> **Core Thesis**  
> *The problem is not dimensionality reduction itself; it is reducing the wrong geometry. Variance-preserving dimensionality reduction is not necessarily signal-preserving dimensionality reduction for LLM internal-state trajectory analysis.*

**Experiment ID:** EXP--TRJ001 / EXP-2026-NVS-001  
**Status:** Draft for final paper/technical report  
**Primary model:** `Qwen/Qwen2.5-7B-Instruct`  
**Primary evaluated layers:** Layer 8, Layer 16, Layer 24  
**Hidden-state dimensionality:** 3584  
**Primary confirmatory representation:** 64D PCA projection  
**Primary exploratory ablation:** Raw 3584D-derived trajectory features  
**Secondary ablation:** PC4--67  
**Author:** Tomohiko Nakamura / Gemmina Intelligence LLC

---

# Abstract

This study evaluates whether token-level hidden-state trajectories from selected layers of a large language model (LLM) can provide non-destructive signals for detecting response reliability and hallucination-related behavior. The study was conducted under a pre-registered protocol in which the primary confirmatory analysis used a 64-dimensional principal component analysis (PCA) projection of the original 3584-dimensional hidden states.

Under the pre-registered PCA condition, the trajectory-derived PROP-$\kappa$ representation at Layer 16 achieved a test AUROC of 0.7806, compared with 0.8077 for the response-length baseline (BL-LEN). The AUROC difference was $\Delta=-0.0271$ with $p=0.8685$, and the pre-registered Gate 3 criterion for exceeding the response-length baseline was therefore not met. The overall Gate 3 outcome was FAIL.

As a post-preregistered exploratory analysis, we evaluated trajectory features derived without dimensionality reduction from the original 3584-dimensional hidden states. The Raw-space trajectory representation achieved an AUROC of 0.8472 and exceeded BL-LEN by $\Delta=+0.0395$ (95% CI [0.0114, 0.0726], $p=0.004$). Because this analysis differed from the confirmatory pipeline in one threshold-calibration detail, it is treated as exploratory rather than confirmatory evidence.

Layer comparison further showed that, among the evaluated layers, Layer 16 exhibited the strongest trajectory-based discrimination: AUROC 0.7090 at Layer 8, 0.7806 at Layer 16, and 0.7167 at Layer 24. Pairwise comparisons favored Layer 16 over Layers 8 and 24 after Holm correction ($p<0.001$ for both comparisons). A secondary PCA subspace ablation using PC4--67 achieved AUROC 0.7969 and did not significantly exceed BL-LEN ($\Delta=-0.0108$, 95% CI [-0.0490, 0.0271], $p=0.6935$).

Taken together, the results provide evidence that the evaluated PCA projection does not preserve all task-relevant discriminative information present in the raw hidden-state trajectory representation. The findings support a distinction between **variance-preserving representation** and **task-signal-preserving representation** and motivate geometry-aware projection methods for LLM internal-state analysis. The study does not establish that hallucination signals necessarily occupy a specific low-variance subspace; rather, it identifies this and related geometric explanations as hypotheses for subsequent investigation.

---

# 1. Introduction

## 1.1 Background and Motivation

Large language models have demonstrated strong performance across a broad range of language and reasoning tasks, yet reliable estimation of whether a generated response is correct, supported, or hallucinated remains difficult. Conventional evaluation often observes only the final generated text. Such output-level evaluation is useful, but it provides limited access to the internal dynamics by which a model arrives at an answer.

Hidden states provide a potential non-destructive observation channel into these internal dynamics. During autoregressive generation, the model produces a sequence of internal vector states associated with successive generated tokens. Instead of treating hidden states only as static feature vectors, they can be treated as a trajectory in a high-dimensional representational space.

This study focuses on that trajectory perspective. The central question is not merely whether a hidden-state vector contains information related to response reliability, but whether **the geometry and dynamics of the hidden-state trajectory** contain useful predictive information.

A practical obstacle arises immediately. Hidden states of modern LLMs are high dimensional. The model used in this study has a hidden dimension of 3584. Storing, transporting, and repeatedly processing full trajectories can be computationally expensive. Dimensionality reduction is therefore attractive as an engineering strategy.

However, dimensionality reduction introduces a scientific question that is often treated as an implementation detail:

> **Which geometry should be preserved?**

PCA is a standard answer when the objective is to preserve global variance. But the geometry with the largest variance is not necessarily the geometry that carries the strongest task-relevant signal.

This study examines that distinction directly.

## 1.2 The Dimensionality-Reduction Problem

The primary confirmatory protocol specified a 64-dimensional PCA projection. This creates a controlled comparison between:

1. a high-dimensional hidden-state representation,
2. a variance-maximizing lower-dimensional representation, and
3. a simple output-level baseline based on response length.

The underlying concern is that the principal directions that explain the greatest proportion of global variance may preferentially represent broad properties of language generation, while reliability-related dynamics may appear as smaller, localized, anisotropic, or otherwise non-dominant structures.

The point of the present study is not to argue that PCA is intrinsically defective. Rather, it asks whether **variance preservation is sufficient for preserving the particular trajectory signal under study**.

## 1.3 Research Questions

### RQ1 — Signal Existence and Localization

Can token-level hidden-state trajectories from the evaluated layers predict response reliability, and does their discriminative strength vary systematically across the evaluated layers?

### RQ2 — PCA Signal Preservation

Does a variance-maximizing linear PCA projection to 64 dimensions preserve trajectory-based discriminative signal sufficient to outperform the response-length baseline?

### RQ3 — High-Dimensional Recovery

Can trajectory features derived from the uncompressed 3584-dimensional hidden states recover predictive performance beyond the response-length baseline?

## 1.4 Contributions

This study makes four primary contributions.

1. **Transparent confirmatory negative result.**  
   The pre-registered 64D PCA condition did not satisfy the primary comparative Gate 3 criterion. The failure is reported directly rather than being replaced by a post hoc successful configuration.

2. **Layer-localized discrimination.**  
   Among the three evaluated layers, Layer 16 provided the strongest trajectory-based discrimination, significantly exceeding the corresponding scores at Layers 8 and 24 after Holm correction.

3. **Exploratory high-dimensional signal recovery.**  
   Raw 3584D hidden states yielded trajectory features with AUROC 0.8472, significantly exceeding BL-LEN in the exploratory comparison.

4. **A distinction between variance preservation and signal preservation.**  
   The results motivate the proposition that dimensionality reduction should be evaluated not only by explained variance, but also by preservation of task-relevant geometry.

---

# 2. Background and Hypotheses

## 2.1 Hidden-State Geometry and Trajectory Representation

For an autoregressive generation sequence of length $T$, let the hidden state at a selected layer and token position $t$ be

$$
\mathbf{h}_t \in \mathbb{R}^{3584}.
$$

The generated response therefore induces a sequence

$$
\mathcal{H}=\{\mathbf{h}_1,\mathbf{h}_2,\ldots,\mathbf{h}_T\}.
$$

Rather than using only individual states, this work examines statistical properties of transitions along the sequence. The trajectory representation is summarized using $\kappa$-derived statistics intended to capture aspects of the dynamics of the hidden-state path.

The feature vector used for classification includes aggregated quantities such as:

- `kappa_mean`
- `kappa_max`
- `kappa_p95`
- `kappa_std`
- `kappa_auc_density`

The analytical pipeline is therefore:

$$
\text{Hidden-State Trajectory}
\rightarrow
\kappa\text{-derived trajectory features}
\rightarrow
\text{Classifier}
\rightarrow
\text{Test AUROC}.
$$

Two trajectory representation conditions are compared:

- **PCA condition:** the original hidden states are projected into the evaluated lower-dimensional PCA representation before trajectory-feature extraction.
- **Raw condition:** trajectory features are derived without the PCA dimensionality reduction.

This distinction is central to the interpretation of the results.

## 2.2 PCA and Variance Maximization

Given centered observations in $\mathbb{R}^{d}$, PCA constructs an orthogonal basis whose leading directions maximize the variance captured by the corresponding lower-dimensional subspace.

For a rank-$k$ projection matrix $P_k$, the classical objective can be written conceptually as

$$
\max_{P_k}
\operatorname{Var}(P_k\mathbf{x}),
$$

subject to the appropriate orthogonality constraints.

The optimization target is therefore global variance preservation. It is not directly the predictive discrimination of hallucination, reliability, or any other downstream task.

This motivates the distinction:

$$
\boxed{
\text{High Explained Variance}
\neq
\text{High Task-Relevant Information}
}
$$

The study tests this distinction empirically within the specific PCA configuration defined by the experimental protocol.

## 2.3 Empirical Framework: Observation, Interpretation, Hypothesis

To avoid conflating measured results with mechanistic explanations, the study adopts a three-stage framework.

### Observation

The raw 3584D-derived trajectory representation produces higher discrimination than the evaluated PCA 64D representation at Layer 16:

$$
\text{AUROC}_{\text{Raw}}=0.8472
>
\text{AUROC}_{\text{PCA64}}=0.7806.
$$

The confirmatory PCA condition does not exceed BL-LEN, while the exploratory raw-space condition does.

### Interpretation

The evaluated PCA projection does not preserve all of the discriminative information present in the raw hidden-state trajectory representation.

This is supported by the following comparisons:

$$
\Delta_{\text{PCA64 vs BL-LEN}}=-0.0271,\quad p=0.8685,
$$

versus

$$
\Delta_{\text{Raw vs BL-LEN}}=+0.0395,\quad p=0.004.
$$

### Hypothesis

A possible mechanism is that task-relevant reliability information is not aligned with the directions selected by global variance maximization. Such information could be associated with low-variance directions, local anisotropy, nonlinear structure, or other geometry not adequately preserved by the evaluated PCA subspace.

This mechanism remains a hypothesis and is not directly established by the present experiment.

The PC4--67 ablation is consistent with rejecting the simplest explanation that only PC1--3 were undesirable, but it does not by itself identify the precise location or form of the missing signal.

---

# 3. Experimental Protocol

## 3.1 Model and Inference Environment

The frozen experimental environment records the following configuration:

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Model commit | `a09a35458c702b33eeacc393d103063234e8bc28` |
| vLLM | `0.29.0` |
| CUDA / Driver | CUDA 13.0 |
| Torch | `2.13.0+cu130` |
| GPU | NVIDIA A40 |
| Python | 3.11.10 |
| Hidden dimension | 3584 |

These values are recorded in the experimental freeze manifest. The model entry explicitly records the migration to Qwen from the previously considered Llama configuration. 

## 3.2 Dataset Construction and Filtering

The experiment uses Type A, Type B, and calibration prompt groups. Phase 1 included 1,000 Type A prompts and 8 samples per prompt, producing 8,000 generations in the pilot stage. The manifest records a mixed-prompt rate of 0.599 and an observed mixed-positive rate of 0.471. 

The Phase 2 collection consisted of:

$$
493\ \text{prompts}\times 8\ \text{samples}=3,944\ \text{generations}.
$$

The recorded label composition was:

| Label | Count |
|---|---:|
| `type_a_positive` | 2,191 |
| `type_d_hedge` | 1,616 |
| `type_c_abstain` | 137 |
| **Total** | **3,944** |

After the mixed-prompt filtering stage, 299 prompts and 2,392 samples remained for the Phase 3 evaluation. 

The protocol also includes automated filtering based on the specified Wikipedia conflict-check procedure. The exact operational definition, matching rules, and thresholds should be reproduced in Appendix B from the frozen source files rather than reconstructed here.

## 3.3 Hidden-State Extraction

Hidden states are collected at the evaluated layers:

- Layer 8
- Layer 16
- Layer 24

Each token-level state has dimensionality 3584 before projection.

The experiment retains both the lower-dimensional PCA trajectory representation and, for the designated ablation subset, the original high-dimensional hidden states.

The manifest records 98 of 200 originally targeted prompts as the available ablation subset and explicitly marks this shortfall rather than silently replacing it. 

## 3.4 PCA Configuration

The frozen artifact set contains a PCA projection matrix with shape:

$$
3584\times128.
$$

The PCA diagnostics recorded:

| Diagnostic | Value |
|---|---:|
| PC1--64 cumulative explained variance | 0.99992 |
| PC1--128 cumulative explained variance | 0.99996 |
| PC1--3 cumulative explained variance | 0.99933 |
| Number of calibration tokens | 8,357 |

These values are important because they show that the evaluated PCA representation retains extremely high global explained variance while still failing to demonstrate superiority over the response-length baseline in the primary comparative test. 

The present study therefore treats explained variance as a representation diagnostic rather than as evidence of downstream task-signal preservation.

## 3.5 Trajectory Feature Construction

The trajectory feature pipeline is:

$$
\text{Hidden State Trajectory}
\rightarrow
\kappa
\rightarrow
\{
\text{kappa\_mean},
\text{kappa\_max},
\text{kappa\_p95},
\text{kappa\_std},
\text{kappa\_auc\_density}
\}
\rightarrow
\text{Classifier}.
$$

The classifier produces a test score evaluated by AUROC.

The exact estimator, calibration strategy, and feature-standardization procedure should be reproduced from the frozen implementation artifacts in the final reproducibility appendix.

## 3.6 Baselines

### BL-LEN

BL-LEN uses response sequence length as a simple output-level baseline.

### BL-NORM

BL-NORM is also recorded in the frozen metrics but is not the principal comparative baseline for the central research claim.

At Layer 16, BL-NORM achieved AUROC 0.8588, while BL-LEN achieved 0.8077. The current paper therefore centers the pre-registered Gate 3 comparison on BL-LEN as defined by the protocol.

## 3.7 Statistical Evaluation

The primary discrimination metric is test AUROC.

For comparisons against BL-LEN, the analysis reports:

- AUROC difference $\Delta$
- bootstrap 95% confidence interval
- the corresponding empirical comparison $p$-value reported by the frozen analysis

Layer-family comparisons use Holm correction across the specified pairwise comparisons.

The study maintains a distinction between:

1. **Confirmatory analysis**, defined by the pre-registered protocol.
2. **Exploratory analysis**, performed after the pre-registered outcome was obtained.

This distinction is maintained throughout the Results and Discussion sections.

## 3.8 Pre-registration and Freeze Protocol

The freeze manifest defines a strict rule:

> Data generated before the freeze is not to be adopted as an experimental result.

The manifest also lists hashes for the experimental plan, PCA matrix, centering vector, prompt inventories, label logic, generation configuration, masking procedure, and Type B reference assets.

The current uploaded manifest still shows the freeze timestamp, freeze commit, and approver fields as blank placeholders. The document also specifies that the freeze becomes formally established when all hash fields are complete and the `v1.4-frozen` tag has been applied. This administrative state must be checked and finalized before submission as a formally frozen study.

---

# 4. Results — The Five-Act Narrative

## 4.1 Act I — The Preregistered PCA Result (RQ2)

The primary confirmatory evaluation used the designated PCA representation.

At Layer 16:

$$
AUROC_{\text{PROP-}\kappa}=0.7806
$$

while

$$
AUROC_{\text{BL-LEN}}=0.8077.
$$

The difference was:

$$
\Delta=-0.0271
$$

with

$$
95\%\,CI=[-0.0727,\ 0.0190]
$$

and

$$
p=0.8685.
$$

The first Gate 3 condition, requiring AUROC to exceed 0.5, passed. The second condition, requiring the trajectory representation to exceed BL-LEN, failed. Consequently:

$$
\boxed{\text{Gate 3 = FAIL}}
$$

The negative result is the principal confirmatory finding of the study.

It is important to distinguish this from a conclusion that hidden-state trajectories contain no useful information. The confirmatory result establishes only that the **evaluated PCA trajectory representation did not demonstrate superiority over BL-LEN under the specified Gate 3 criterion**.

## 4.2 Act II — Layer Localization (RQ1)

The three evaluated layers showed distinct trajectory-discrimination performance:

| Layer | PROP-$\kappa$ AUROC | 95% CI |
|---|---:|---|
| 8 | 0.7090 | [0.6554, 0.7599] |
| 16 | **0.7806** | [0.7298, 0.8232] |
| 24 | 0.7167 | [0.6570, 0.7693] |

Layer 16 was the strongest among the evaluated layers.

Pairwise comparisons showed:

- Layer 16 vs Layer 8: $\Delta=+0.0717$, 95% CI [0.0359, 0.1080], Holm-adjusted $p<0.001$.
- Layer 16 vs Layer 24: $\Delta=+0.0639$, 95% CI [0.0368, 0.0944], Holm-adjusted $p<0.001$.

Thus, within the limited set of evaluated layers, Layer 16 exhibits a statistically stronger trajectory-based discrimination signal.

This result should be interpreted as **localization within the evaluated layers**, not as evidence that Layer 16 is globally optimal across all layers of the model.

## 4.3 Act III — High-Dimensional Raw-Space Recovery (RQ3)

The exploratory raw-space ablation removed the PCA dimensionality reduction and derived trajectory features directly from the original 3584-dimensional hidden states.

The resulting performance was:

$$
AUROC_{\text{Raw}}=0.8472
$$

with

$$
95\%\,CI=[0.8060,\ 0.8824].
$$

Relative to BL-LEN:

$$
\Delta=+0.0395
$$

with

$$
95\%\,CI=[0.0114,\ 0.0726]
$$

and

$$
p=0.004.
$$

This provides exploratory evidence that the raw hidden-state trajectory contains discriminative information that is not captured adequately by the response-length baseline.

Importantly, this result is not treated as a confirmatory success of the original protocol. It is an exploratory post-preregistered analysis.

## 4.4 Act IV — PCA Subspace Ablation (RQ2 Supplement)

The PC4--67 ablation achieved:

$$
AUROC_{PC4\text{--}67}=0.7969
$$

with

$$
95\%\,CI=[0.7498,\ 0.8390].
$$

Relative to BL-LEN:

$$
\Delta=-0.0108
$$

with

$$
95\%\,CI=[-0.0490,\ 0.0271]
$$

and

$$
p=0.6935.
$$

This result argues against a simplistic interpretation in which only PC1--3 were harmful noise dimensions.

However, it does not identify the precise missing geometry. Rather, it supports the narrower conclusion that the specific evaluated PCA subspace did not preserve the task-relevant trajectory signal sufficiently to exceed BL-LEN.

## 4.5 Act V — Synthesis

The four principal representations give the following compact comparison:

| Representation | AUROC | Relation to BL-LEN |
|---|---:|---|
| PCA 64D | 0.7806 | $\Delta=-0.0271,\ p=0.8685$ |
| PC4--67 | 0.7969 | $\Delta=-0.0108,\ p=0.6935$ |
| BL-LEN | 0.8077 | Reference |
| Raw 3584D-derived trajectory features | **0.8472** | $\Delta=+0.0395,\ p=0.004$ |

The central empirical pattern is therefore:

$$
\boxed{
\text{Raw 3584D-derived trajectory}
>
\text{BL-LEN}
>
\text{PCA 64D}
}
$$

with the important qualification that the raw-space result is exploratory.

---

# 5. Discussion

## 5.1 Explained Variance vs. Task-Relevant Information

The most direct interpretation of the experiment is not that PCA is inherently unsuitable for LLM representation analysis.

Rather, the result suggests a mismatch between two different optimization objectives:

### PCA objective

Preserve global variance and reconstruction fidelity.

### Reliability-analysis objective

Preserve the geometry that is informative for discriminating response reliability.

These objectives need not select the same directions.

The PCA diagnostics make this distinction especially clear. The first 64 principal components explain 99.992% of the measured variance, and the first 128 explain 99.996%. Yet the evaluated 64D trajectory representation does not outperform BL-LEN at Layer 16.

The relevant logical point is therefore:

$$
\boxed{
\text{High explained variance}
\not\Rightarrow
\text{high task-relevant information}
}
$$

This should not be interpreted as evidence that the discarded 0.008% of variance necessarily contains the entire signal. The current experiment does not establish such a quantitative localization.

Instead, the results show that the tested variance-maximizing projection does not preserve the task-relevant trajectory signal sufficiently for the primary comparative criterion.

## 5.2 Layer 16 as a Localized Signal Region

Layer 16 is the strongest of the evaluated layers.

One possible interpretation is that intermediate representations may contain a particularly useful mixture of contextual integration and emerging output structure. However, this interpretation remains speculative within the current evidence because only Layers 8, 16, and 24 were evaluated.

The empirical claim that can be made without overreach is:

> Layer 16 exhibited the strongest trajectory-based discrimination among the evaluated layers.

The broader mechanistic question of why this occurs should be tested in a future experiment involving denser layer sampling and, ideally, independent datasets.

## 5.3 What ABL-2 Does and Does Not Establish

ABL-2 is scientifically useful because it tests a specific alternative explanation.

One possible story would have been:

> PC1--3 contain dominant nuisance variation, and removing those components should restore the task signal.

That does not occur. PC4--67 still fails to outperform BL-LEN.

This reduces the plausibility of a narrowly targeted explanation centered only on the first three components.

However, the experiment does not yet distinguish among:

- informative low-variance directions,
- nonlinear structure,
- local anisotropy,
- trajectory-specific interactions,
- information distributed across many weak dimensions,
- or other representation effects.

That uncertainty is a feature of the research result, not a defect that should be hidden.

## 5.4 The Central Thesis

The study therefore proposes the following design principle:

> **The problem is not dimensionality reduction itself; it is reducing the wrong geometry.**

A practical consequence is that future projection methods should be evaluated against two different criteria:

1. **Representation preservation**
2. **Task-signal preservation**

A projection can score extremely well on the first and poorly on the second.

The present study provides evidence that this distinction matters for LLM hidden-state trajectory analysis.

## 5.5 Engineering Implications

The results matter beyond this particular experiment because high-dimensional hidden states are expensive to retain and process.

A future real-time system may need to transform:

$$
3584D
\rightarrow
16D\text{--}64D
$$

while preserving the trajectory signal relevant to confidence or reliability estimation.

The implication is not that raw 3584D representations should always be retained in production. The more useful engineering question is:

> **Can a compact learned geometry preserve the same task-relevant structure that the raw representation contains?**

This motivates task-aware projection rather than variance-only compression.

## 5.6 Limitations

### 5.6.1 Exploratory Status of ABL-1

ABL-1 was conducted after the confirmatory outcome and therefore must be interpreted as exploratory.

### 5.6.2 Threshold-Calibration Difference

The raw-space ablation could not calibrate the `spike_rate` threshold using the same train-split raw-state data as the main pipeline. The frozen metrics explicitly note this condition difference.

Consequently, ABL-1 should not be interpreted as a perfectly matched replication of the confirmatory PCA analysis.

### 5.6.3 Limited Layer Sampling

Only Layers 8, 16, and 24 were evaluated for the reported layer-localization analysis. The result therefore supports a statement about localization **within the evaluated layer set**, not across the full network.

### 5.6.4 Limited Ablation Inventory

The manifest records a shortfall in the designated ablation prompt inventory: 98 of the nominal 200 prompts were available. This must be preserved as a limitation rather than being silently generalized away.

### 5.6.5 Single-Model Evaluation

The current study uses `Qwen/Qwen2.5-7B-Instruct`. Generalization to other model families, parameter scales, and training procedures remains untested.

### 5.6.6 Mechanism Not Identified

The study demonstrates a performance difference between representations. It does not directly identify the geometric mechanism responsible for that difference.

In particular, the hypothesis that informative structure resides specifically in low-variance directions is not established by the present data.

---

# 6. Conclusions and Future Work

## 6.1 Conclusion

This study began with a pre-registered hypothesis concerning whether a 64D PCA representation of LLM hidden-state trajectories could exceed a response-length baseline in hallucination/reliability discrimination.

The primary confirmatory analysis did not meet the Gate 3 criterion:

$$
AUROC_{PCA64}=0.7806
<
AUROC_{BL-LEN}=0.8077.
$$

The confirmatory comparison was not statistically superior to BL-LEN:

$$
\Delta=-0.0271,\quad p=0.8685.
$$

Rather than treating this negative result as evidence that hidden-state trajectories contain no useful information, the study examined the role of the representation itself.

Two important patterns emerged.

First, among the evaluated layers, Layer 16 showed the strongest trajectory-based discrimination.

Second, the exploratory raw-space analysis recovered stronger performance:

$$
AUROC_{Raw}=0.8472,
$$

which exceeded BL-LEN by

$$
\Delta=+0.0395
$$

with

$$
p=0.004.
$$

The PC4--67 ablation did not reproduce this superiority, suggesting that the phenomenon is not adequately explained by simply removing a small number of leading principal components.

The resulting scientific interpretation is deliberately narrower than a claim that PCA is universally harmful:

> **The evaluated variance-maximizing PCA projection did not preserve all of the task-relevant trajectory signal available in the raw hidden-state representation.**

This supports the broader design principle:

$$
\boxed{
\text{Variance-preserving reduction}
\neq
\text{Signal-preserving reduction}
}
$$

and motivates the search for projection methods that explicitly preserve task-relevant geometry.

## 6.2 Future Directions

### 6.2.1 Geometry-Preserving Supervised Projection

The next experimental stage should construct a learned projection

$$
\mathbb{R}^{3584}
\rightarrow
\mathbb{R}^{16\text{--}64}
$$

that optimizes task-relevant structure rather than global variance alone.

Candidate approaches include:

- supervised linear projection,
- supervised autoencoder,
- metric learning,
- contrastive projection,
- low-rank learned projection.

The evaluation should compare both predictive performance and geometric distortion.

### 6.2.2 Dense Layer Sweep

Instead of testing only Layers 8, 16, and 24, future work should evaluate a substantially denser set of layers to determine whether Layer 16 represents a genuine intermediate-layer maximum or merely a local maximum within the sampled points.

### 6.2.3 Direct Geometry Diagnostics

Future experiments should explicitly test candidate mechanisms rather than infer them indirectly.

Useful diagnostics include:

- variance rank of task-discriminative directions,
- Fisher or between-class / within-class geometry,
- local neighborhood preservation,
- curvature and trajectory-shape preservation,
- nonlinear separability before and after projection.

### 6.2.4 Cross-Model and Cross-Dataset Validation

The strongest next validation would repeat the experiment across:

- multiple LLM families,
- multiple parameter scales,
- independent prompt inventories,
- and independently curated reliability/hallucination datasets.

### 6.2.5 Real-Time Integration

A successful compact geometry-preserving projection could eventually serve as a low-latency internal-state signal for real-time control systems such as Project L-Agent and Project V-Agent.

The target architecture is conceptually:

$$
\text{LLM Hidden State}
\rightarrow
\text{Geometry-Aware Projection}
\rightarrow
\text{Trajectory Reliability Signal}
\rightarrow
\text{Confidence Feedback / Control}
$$

The key requirement is not merely low dimensionality. It is preservation of the information that makes the signal useful.

---

# 7. Figure Plans

## Figure 1 — Experimental Pipeline & Feature Extraction Architecture

Recommended layout:

```text
Prompt
  |
  v
LLM Generation
  |
  v
Token-level Hidden-State Trajectory
  |
  +-----------------------------+
  |                             |
  v                             v
Raw 3584D                    PCA Projection
  |                             |
  |                             v
  |                          64D / PC subset
  |                             |
  +-------------+---------------+
                |
                v
        κ-derived trajectory
             features
                |
                v
            Classifier
                |
                v
           Test AUROC
                |
                v
        Comparison to BL-LEN
```

The figure should visually distinguish confirmatory and exploratory branches.

## Figure 2 — Layer Localization

Display the AUROC values for Layers 8, 16, and 24 with confidence intervals.

The key visual message is the Layer 16 peak among the evaluated layers.

## Figure 3 — Central Result Comparison

Display:

- PCA 64D: 0.7806
- PC4--67: 0.7969
- BL-LEN: 0.8077
- Raw 3584D-derived trajectory features: 0.8472

The figure should mark the Raw-vs-BL-LEN comparison as exploratory and statistically significant.

## Figure 4 — Conceptual Geometry

Illustrate the distinction between:

- global variance-maximizing PCA directions, and
- possible task-relevant geometric structure.

The diagram must use language such as:

> **Possible locations of task-relevant structure**

rather than asserting that the task signal has been proven to occupy a low-variance subspace.

---

# 8. Reproducibility and Frozen Artifacts

The frozen artifact inventory includes hashes for:

- experimental plan,
- PCA projection matrix,
- centering vector,
- calibration prompts,
- Type A prompts,
- refusal/hedging patterns,
- Type B reference set,
- $\kappa$ computation module,
- hallucination label module,
- generation configuration,
- masking protocol,
- Type B prompts.

The freeze manifest records the primary Phase 1, Phase 2, and Phase 3 counts, together with the major evaluation metrics and artifact locations.

### Current administrative verification

Before submission, verify and complete:

- freeze timestamp,
- freeze commit,
- approver,
- actual `v1.4-frozen` tag state,
- exact repository commit associated with the artifact hashes.

The current manifest contains placeholders for the first three fields, so the paper should not claim administrative freeze completion beyond what has actually been verified.

---

# 9. Appendix

## Appendix A — Freeze Manifest

Source artifact:

`FREEZE_MANIFEST.md`

The manifest should be included or cited as the authoritative record for:

- environment,
- artifact hashes,
- sample counts,
- data-split definitions,
- PCA diagnostics,
- Phase 1--3 outcomes,
- Gate 3 status,
- ABL-1 and ABL-2 outcomes.

#
# Appendix B — Prompt Construction and Wikipedia-Based Filtering Protocol

## B.1 Scope and Role of This Appendix

This appendix specifies the prompt-construction and factual-conflict filtering procedure used in EXP--TRJ001 to the extent supported by the frozen experimental record.

The purpose of this appendix is to make explicit:

1. the relationship among the Type A, Type B, and calibration prompt inventories;
2. the generation and sampling structure used in the experiment;
3. the transition from the Phase 1 prompt inventory to the Phase 2 evaluation pool;
4. the Wikipedia-based factual-conflict filtering stage; and
5. the exact provenance artifacts that determine which prompts and generated samples entered the reported evaluation.

Where the currently available frozen record does not expose an implementation-level threshold, matching rule, or prompt template verbatim, this appendix labels that item as **SOURCE-TEXT REQUIRED** rather than reconstructing it from memory or inference.

This distinction is important because prompt construction and filtering are part of the experimental condition. A post hoc reconstruction would not be equivalent to the frozen protocol.

---

## B.2 Prompt Inventory Structure

The experiment uses three distinct prompt-related resources:

| Resource | Role | Frozen artifact / record |
|---|---|---|
| Calibration prompts | Calibration of the representation/projection procedure | `configs/calibration_prompts_v1.json` |
| Type A prompts | Primary prompt inventory used for the main data-generation workflow | `configs/prompts_v1.json` |
| Type A Phase 2 prompts | Independent inventory used to satisfy the Phase 2 requirement | `configs/prompts_v1_phase2.json` |
| Type B prompts | Reference/verification condition | `configs/prompts_typeB_v1.json` |
| Type B reference set | Reference answers / matching rules | `configs/typeB_reference.json` |
| Refusal / hedging patterns | Label-related structural patterns | `configs/refusal_patterns_en.json` |

The freeze manifest records SHA-256 hashes for these assets, together with the implementation modules responsible for $\kappa$ computation and hallucination labeling. These hashes constitute the provenance boundary for the reported study.

### B.2.1 Calibration Inventory

The calibration inventory contains 200 prompts:

`configs/calibration_prompts_v1.json`

The exact prompt-generation template, semantic balancing criteria, and selection procedure are **SOURCE-TEXT REQUIRED** for a fully self-contained reproduction.

### B.2.2 Type A Inventory

The primary Type A inventory contains 1,000 prompts:

`configs/prompts_v1.json`

The experiment generated 8 samples per prompt during the Phase 1 pilot stage:

$$
1,000\times8=8,000\ \text{generations}.
$$

The frozen record reports a mixed-prompt rate of:

$$
599/1,000=0.599.
$$

Within the mixed subset, the observed positive rate is:

$$
0.471.
$$

These values were used to determine the Phase 2 inventory requirement.

### B.2.3 Type A Phase 2 Inventory

The Phase 2 inventory contains 493 prompts:

`configs/prompts_v1_phase2.json`

The freeze record states that this inventory was newly constructed as an independent pool and that the calculated requirement was:

$$
M=444.1\rightarrow445.
$$

A total of 493 prompts was therefore retained, providing a margin above the calculated requirement without additional post hoc prompt generation.

The Phase 2 collection generated:

$$
493\times8=3,944\ \text{samples}.
$$

### B.2.4 Type B Inventory

The Type B condition contains 200 prompts:

`configs/prompts_typeB_v1.json`

and is accompanied by a frozen reference set:

`configs/typeB_reference.json`.

The exact semantic construction and reference-answer generation procedure must be inserted from the frozen source protocol.

---

## B.3 Sampling and Phase Transition

The experiment proceeds through a staged pipeline.

### B.3.1 Phase 1

Phase 1 begins with the 1,000-prompt Type A inventory.

For every prompt, 8 generated samples are collected:

$$
N_{\text{Phase1}}=1,000\times8=8,000.
$$

The resulting pilot set is stored as:

`results/phase1_pilot_trajectories.json`

and the freeze record indicates that the corresponding generation rule passed the specified Rule 3 check.

### B.3.2 Estimation of the Phase 2 Requirement

The freeze record reports:

- mixed prompt rate = 0.599;
- mixed positive rate = 0.471;
- calculated requirement $M=444.1$;
- rounded requirement = 445;
- independent Phase 2 inventory = 493 prompts.

The Phase 2 inventory was therefore fixed at 493 prompts before Phase 2 generation proceeded.

### B.3.3 Phase 2 Generation

The complete Phase 2 generation consists of:

$$
493\ \text{prompts}\times8\ \text{samples}=3,944\ \text{generated samples}.
$$

The recorded sample labels are:

| Label | Number |
|---|---:|
| `type_a_positive` | 2,191 |
| `type_d_hedge` | 1,616 |
| `type_c_abstain` | 137 |
| **Total** | **3,944** |

The generated trajectory metadata is stored in:

`results/phase2_trajectories.json`

with projected layer-wise trajectory data stored in:

`results/phase2_projected/*.npz`

and the designated raw hidden-state ablation subset stored in:

`results/ablation_raw_hidden_states/*.npy`.

---

## B.4 Wikipedia-Based Factual-Conflict Filtering

### B.4.1 Purpose

The Wikipedia-based filtering stage is intended to reduce contamination or ambiguity arising when a generated response conflicts with externally verifiable factual content.

The filtering stage is applied before the final Phase 3 mixed-prompt evaluation set is established.

The reported Phase 3 result states that, after the specified mixed-prompt filtering procedure:

$$
299\ \text{prompts}
$$

and

$$
2,392\ \text{samples}
$$

remained.

Thus the retained sample count is consistent with:

$$
299\times8=2,392.
$$

### B.4.2 Conceptual Procedure

At the level supported by the present record, the filtering process can be represented as:

```text
Generated response
      |
      v
Identify factual / reference-bearing content
      |
      v
Wikipedia conflict check
      |
      v
Apply frozen filtering / exclusion rules
      |
      v
Retain eligible mixed prompts / samples
      |
      v
Phase 3 evaluation set
```

The exact implementation details are normative experimental content and must be taken from the frozen filtering source.

### B.4.3 Required Implementation-Level Details

The following fields are required for a completely reproducible Appendix B but are not exposed in the currently available manuscript record.

**SOURCE-TEXT REQUIRED:**

1. The exact Wikipedia retrieval/query rule.
2. Whether lookup is page-based, title-based, keyword-based, entity-based, or another matching mechanism.
3. The exact normalization applied before comparison.
4. The exact text fields extracted from Wikipedia.
5. The exact definition of a factual conflict.
6. The matching criterion or similarity threshold.
7. Handling of aliases, punctuation, capitalization, dates, numbers, and named entities.
8. Handling of multiple candidate Wikipedia pages.
9. Treatment of ambiguous or missing Wikipedia entries.
10. Whether a conflict is evaluated per token, per sentence, per response, or per prompt.
11. Whether one conflict causes sample-level exclusion, prompt-level exclusion, or both.
12. The precise order of operations relative to label assignment.
13. The exact code path / function implementing the filter.
14. The version or retrieval snapshot of the external Wikipedia content, if the protocol defines one.

These items should be copied from the frozen protocol and implementation without reinterpretation.

---

## B.5 Mixed-Prompt Filtering

The Phase 3 record reports:

- Phase 2 prompt pool: 493 prompts;
- Phase 2 generated samples: 3,944;
- retained mixed prompts after filtering: 299;
- retained mixed samples: 2,392.

The reduction can be expressed as:

$$
493\rightarrow299\ \text{prompts}
$$

and

$$
3,944\rightarrow2,392\ \text{samples}.
$$

The corresponding prompt-retention fraction is:

$$
\frac{299}{493}\approx0.6065.
$$

Because the available record identifies the result of the filtering stage but does not expose all implementation rules, this fraction should be treated as an empirical outcome, not as a design target.

---

## B.6 Prompt Identity, Determinism, and Provenance

Prompt identities are part of the experiment's reproducibility boundary.

The freeze manifest records explicit hashes for:

- `configs/prompts_v1.json`;
- `configs/prompts_v1_phase2.json`;
- `configs/calibration_prompts_v1.json`;
- `configs/prompts_typeB_v1.json`;
- `configs/typeB_reference.json`;
- `configs/refusal_patterns_en.json`.

The data split itself is also hashed:

`configs/data_splits.json`

with the recorded Phase 2 split:

- train: 296;
- validation: 99;
- test: 98.

The designated ablation prompt inventory is separately hashed:

`configs/ablation_prompt_ids.json`

and the record explicitly states that 98/200 designated ablation prompts were available, with the shortfall retained in the file's `shortfall` field.

For a future exact reproduction, the correct procedure is therefore:

1. obtain the frozen prompt assets;
2. verify their SHA-256 hashes;
3. reproduce the prompt IDs and split assignments;
4. apply the frozen filtering implementation;
5. verify the retained counts before executing any downstream hidden-state analysis.

---

## B.7 Relationship Between Prompt-Level and Sample-Level Filtering

The experiment distinguishes prompt inventories from generated samples.

A single prompt generates multiple independent samples:

$$
p_i\rightarrow
\{y_{i,1},y_{i,2},\ldots,y_{i,8}\}.
$$

Consequently, filtering must be represented carefully.

A prompt-level decision and a sample-level decision are not interchangeable:

```text
Prompt
  |
  +--> Sample 1
  +--> Sample 2
  +--> ...
  +--> Sample 8
```

The reported Phase 3 record retains 299 prompts and 2,392 samples, exactly corresponding to 8 samples per retained prompt.

This indicates that the reported final mixed-prompt set is represented at the prompt × 8-sample unit. The exact rule determining when a prompt becomes eligible for retention must nevertheless be taken from the frozen filtering implementation.

**SOURCE-TEXT REQUIRED:** insert the exact prompt-level/sample-level eligibility rule here.

---

## B.8 Label Assignment and Filtering Order

The frozen record separately identifies:

- generation configuration;
- refusal / hedging patterns;
- hallucination label logic;
- Type B reference rules;
- data filtering.

For reproducibility, the final manuscript should explicitly state the order in which these operations occur.

The manuscript currently supports the following high-level dependency graph:

```text
Prompt inventory
      |
      v
Generation
      |
      v
Generated samples
      |
      +--> factual / Wikipedia filtering
      |
      +--> label determination
      |
      +--> hidden-state trajectory extraction
      |
      v
Phase 3 eligibility / evaluation set
```

However, the exact ordering of filtering relative to labeling and trajectory extraction is not fully specified by the current manuscript record.

**SOURCE-TEXT REQUIRED:** replace the above conceptual graph with the exact frozen execution order once the source implementation is available.

---

## B.9 Required Reproducibility Table

The final published version should include a table of the following form.

| Protocol component | Frozen source | SHA-256 / version | Status |
|---|---|---|---|
| Calibration prompts | `configs/calibration_prompts_v1.json` | Recorded in freeze manifest | Verified |
| Type A Phase 1 prompts | `configs/prompts_v1.json` | Recorded in freeze manifest | Verified |
| Type A Phase 2 prompts | `configs/prompts_v1_phase2.json` | Recorded in freeze manifest | Verified |
| Type B prompts | `configs/prompts_typeB_v1.json` | Recorded in freeze manifest | Verified |
| Type B references | `configs/typeB_reference.json` | Recorded in freeze manifest | Verified |
| Refusal / hedging patterns | `configs/refusal_patterns_en.json` | Recorded in freeze manifest | Verified |
| Split assignment | `configs/data_splits.json` | Recorded in freeze manifest | Verified |
| Ablation prompt IDs | `configs/ablation_prompt_ids.json` | Recorded in freeze manifest | Verified with shortfall |
| Wikipedia filtering implementation | **SOURCE-TEXT REQUIRED** | **SOURCE-TEXT REQUIRED** | Pending insertion |
| Prompt construction implementation | **SOURCE-TEXT REQUIRED** | **SOURCE-TEXT REQUIRED** | Pending insertion |

---

## B.10 Audit Trail

The filtering and prompt-construction appendix should be auditable from the reported counts.

At minimum, the final audit trail should permit the reader to verify:

$$
1,000\times8=8,000
$$

for the Phase 1 pilot,

$$
493\times8=3,944
$$

for Phase 2,

and

$$
299\times8=2,392
$$

for the final Phase 3 mixed set.

The published record should also connect each transition to a concrete artifact hash and, where applicable, to a deterministic script or implementation module.

---

## B.11 Scientific Interpretation Boundary

The Wikipedia filtering stage is a **data-quality and eligibility-control mechanism**, not itself evidence that the resulting labels are universally correct.

Accordingly:

- the filtering procedure defines which observations enter the analysis;
- the label procedure defines how retained observations are categorized;
- the hidden-state trajectory procedure defines which internal representation is measured;
- the classifier defines how trajectory-derived features are converted into predictions.

These are separate methodological layers and should remain separate in the paper.

This separation is particularly important because the central scientific claim of EXP--TRJ001 concerns the geometry and predictive utility of hidden-state trajectories, not the validity of Wikipedia as a universal ground-truth source.

---

## B.12 Appendix B Completion Checklist

Before submission, replace every **SOURCE-TEXT REQUIRED** marker with the exact content from the frozen protocol and implementation.

### Prompt construction

- [ ] Exact Type A template
- [ ] Exact Type B template
- [ ] Exact calibration prompt construction
- [ ] Prompt generation / sampling algorithm
- [ ] Prompt balancing constraints
- [ ] Prompt ID generation rule

### Wikipedia filtering

- [ ] Retrieval mechanism
- [ ] Query construction
- [ ] Text normalization
- [ ] Matching rule
- [ ] Similarity / conflict threshold
- [ ] Entity / alias handling
- [ ] Missing-page handling
- [ ] Ambiguity handling
- [ ] Prompt-level vs sample-level exclusion rule
- [ ] Exact execution order
- [ ] Exact implementation path
- [ ] External-content version / snapshot, if applicable

### Provenance

- [ ] Hash verification
- [ ] Split verification
- [ ] Final retained prompt count = 299
- [ ] Final retained sample count = 2,392
- [ ] Reproduction of Phase 1 = 8,000 generations
- [ ] Reproduction of Phase 2 = 3,944 generations

---

## B.13 Canonical Statement for the Main Paper

The following sentence can be used in the Methods section after the exact source-text details are inserted:

> **Prompts were constructed from frozen Type A, Type B, and calibration inventories; each Phase 2 prompt was sampled eight times, and generated samples were subjected to the pre-specified factual-conflict filtering procedure based on Wikipedia references before the final mixed-prompt evaluation set was established. All prompt inventories, split assignments, reference sets, and filtering-related artifacts were hash-recorded as part of the experimental freeze.**

The statement should only be considered final once the exact Wikipedia filtering implementation has been inserted and independently checked against the frozen source.

# Appendix C — Detailed AUROC Tables

### C.1 Layer-level results

| Layer | BL-LEN | PROP-$\kappa$ | PROP-$\kappa$ + LEN |
|---|---:|---:|---:|
| 8 | 0.8077 | 0.7090 | 0.8083 |
| 16 | 0.8077 | **0.7806** | 0.8293 |
| 24 | 0.8077 | 0.7167 | 0.8314 |

### C.2 Raw and PCA ablations

| Condition | AUROC | Δ vs BL-LEN | p |
|---|---:|---:|---:|
| PCA 64D, Layer 16 | 0.7806 | -0.0271 | 0.8685 |
| Raw 3584D-derived features | **0.8472** | **+0.0395** | **0.004** |
| PC4--67 | 0.7969 | -0.0108 | 0.6935 |

### C.3 Confidence intervals

| Condition | AUROC | 95% CI |
|---|---:|---|
| PCA 64D, Layer 16 | 0.7806 | [0.7298, 0.8232] |
| BL-LEN | 0.8077 | [0.7632, 0.8475] |
| Raw 3584D-derived features | 0.8472 | [0.8060, 0.8824] |
| PC4--67 | 0.7969 | [0.7498, 0.8390] |

---

# 10. Data and Analysis Status

This report distinguishes three categories of statements:

### Confirmatory

Results defined by the pre-registered protocol, especially the 64D PCA condition and Gate 3 decision.

### Exploratory

ABL-1 raw-space analysis and any interpretation derived after observing the confirmatory result.

### Hypothesis

Mechanistic explanations for why the evaluated PCA representation fails to retain sufficient task-relevant signal.

This separation is essential to preserve the scientific meaning of the negative result and to prevent exploratory recovery from being misrepresented as pre-registered confirmation.

---

# 11. Final Scientific Statement

The principal result of EXP--TRJ001 is not that PCA is invalid, nor that 3584 dimensions are intrinsically required.

The principal result is more specific:

> **For the evaluated LLM, layers, trajectory representation, and experimental protocol, a variance-maximizing PCA projection retained very high global explained variance but did not preserve sufficient trajectory-based discriminative information to exceed the response-length baseline. An exploratory analysis of raw 3584D hidden-state trajectories recovered a stronger signal that exceeded that baseline.**

This leads to a practical and testable research principle:

$$
\boxed{
\text{The problem is not dimensionality reduction itself;}
\quad
\text{it is reducing the wrong geometry.}
}
$$

The next scientific question is therefore not:

> *Can the hidden state be compressed?*

but:

> *Can it be compressed while preserving the geometry that matters for the task?*

That question defines the next experimental stage of EXP--TRJ001.
