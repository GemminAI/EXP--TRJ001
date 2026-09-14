# Preserving Task-Relevant Geometry in LLM Hidden States: Evidence That Variance-Maximizing PCA Can Obscure Hallucination Trajectory Signals

> **Core Thesis**
>
> _The problem is not dimensionality reduction itself; it is reducing the wrong geometry. Variance-preserving dimensionality reduction is not necessarily signal-preserving dimensionality reduction for LLM internal-state trajectory analysis._

**Experiment ID:** EXP--TRJ001 / EXP-2026-NVS-001

**Status:** Manuscript v3.4 (reviewer tasks A–F; v3.4 revises v3.3's confirmatory framing — see §3.1a)

**Primary Model:** `Qwen/Qwen2.5-7B-Instruct`

**Evaluated Layers:** Layer 8, Layer 16, Layer 24

**Hidden Dimension ($D$):** 3584

**Primary Representation:** 64D PCA projection — **exploratory, not confirmatory** (§3.1a: the plan revision to this model/dimension, `docs/EXP-2026-NVS-001_v1.4.md`, was written 2026-09-14, after Phase 0–3 execution on 2026-09-13, so it cannot satisfy the pre-registration guarantee described in §1 of that plan)

**Paired test inventory:** 98 prompts $\times$ 8 generations $= 784$; Class E ($T<20$) excludes $288$; **$n=496$ eligible generations**. Raw 3584D, PCA 64D, PC4--67, BL-LEN, and BL-NORM are scored on this **identical** set.

**Author:** Tomohiko Nakamura / Gemmina Intelligence LLC

**Reanalysis script:** `scripts/run_trj001_reanalysis.py` → `results/trj001_reanalysis_metrics.json`

## Abstract

This study evaluates whether token-level hidden-state trajectories from selected layers of a large language model (LLM) can provide non-destructive signals for detecting response reliability and hedging/hallucination behavior. The evaluation instrument (Gram-determinant curvature $\kappa$, Gate 3 statistical procedure, bootstrap/Holm methodology) was pre-registered in `docs/EXP-2026-NVS-001_v1.3.md` for a different model (`meta-llama/Meta-Llama-3.1-8B-Instruct`, $D=4096$). Execution instead used `Qwen/Qwen2.5-7B-Instruct` ($D=3584$); the plan revision documenting that change, `docs/EXP-2026-NVS-001_v1.4.md`, was written 2026-09-14, **after** Phase 0–3 execution completed on 2026-09-13. Every result below that depends on the model/dimension actually used is therefore reported as **exploratory**, not confirmatory (§3.1a) — the pre-registration guarantee (freezing hypotheses and criteria before seeing data, to prevent p-hacking) does not hold for this execution, even though the statistical machinery itself was specified in advance for a different model.

All five Layer 16 representations—PCA 64D (0.7806), PC4--67 (0.7969), BL-LEN (0.8077), Raw 3584D PROP-$\kappa$ (0.8472), and BL-NORM (0.8588)—are compared as **paired** evaluations on the same 98-prompt test inventory ($n=496$ after Class E). BL-LEN and BL-NORM are not scored on a different sample: they occupy the same $496$ rows as the trajectory methods. Under the PCA condition, PROP-$\kappa$ failed Gate 3 versus BL-LEN ($\Delta = -0.0271$, $p = 0.8685$; Gate 3: **FAIL**; exploratory result, §3.1a). Removing compression recovered AUROC $0.8472$ versus BL-LEN ($\Delta = +0.0395$, $95\%$ CI $[0.0114, 0.0726]$, $p = 0.004$; exploratory). Discarding PC1--3 (PC4--67) raised AUROC only to $0.7969$ ($\Delta = -0.0108$ vs BL-LEN, $p = 0.6935$; **$+0.0163$ vs PCA 64D**). The hidden-state $L_2$-norm baseline BL-NORM is the strongest single predictor (AUROC $0.8588$, $95\%$ CI $[0.8239, 0.8903]$).

PCA diagnostics show that PC1--3 capture $99.933\%$ of calibration variance, consistent with massive-activation dimensions monopolizing global variance in modern LLMs (Sun et al., 2024). Variance-maximizing PCA therefore distorts relative trajectory configuration rather than merely “running out of dimensions”: discrete curvature $\kappa$ lives in a 2-dimensional osculating plane and does not require ambient $D=3584$. The findings **indicate** that variance-preserving representation is not equivalent to signal-preserving representation. The load-bearing paired contrast on this split is PC4--67 versus PCA 64D ($\Delta=+0.0163$).

## 1. Introduction

### 1.1 Background and Motivation

Evaluating the reliability of autoregressive generations from Large Language Models (LLMs) remains a critical challenge. Standard evaluation protocols typically observe external text outputs, such as verbalized confidence scores or response lengths. While useful, output-level signals offer limited visibility into the internal computational dynamics that precede token emission.

Internal hidden states present a non-destructive observation channel. As an LLM generates tokens autoregressively, it constructs a continuous sequence of high-dimensional vectors. Rather than treating hidden states as isolated static snapshots, examining the **trajectory dynamics** across generated sequences can reveal internal instability, uncertainty, or hedging behavior (Farquhar et al., 2024; Azaria & Mitchell, 2023).

However, modern open-weight LLMs possess high-dimensional hidden spaces ($D = 3584$ for `Qwen2.5-7B`). Storing and analyzing full high-dimensional trajectories incurs prohibitive computational and memory overhead. Dimensionality reduction is thus standard engineering practice.

This necessity introduces a crucial representational question:

$$\text{\textbf{Which internal geometry should dimensionality reduction preserve?}}$$

Principal Component Analysis (PCA) is the default choice for linear compression because it maximizes global explained variance. However, global variance in LLMs is often dominated by task-agnostic features or massive activation outliers (Sun et al., 2024). Preserving global variance does not guarantee preserving the geometric structures that encode response reliability.

### 1.2 Research Questions

- **RQ1 (Signal Existence & Layer Localization):** Do token-level hidden-state trajectories contain discriminative signals for response reliability, and how does performance vary across network layers (Layers 8, 16, and 24)? Intermediate-layer superiority is treated as a **reconfirmation** of a known probing pattern via trajectory curvature, not as a novel localization discovery.

- **RQ2 (PCA Signal Preservation):** Does a 64-dimensional variance-maximizing PCA projection preserve sufficient trajectory geometry to outperform simple output-level baselines (BL-LEN)?

- **RQ3 (High-Dimensional Signal Recovery):** Do trajectory features extracted directly from uncompressed 3584-dimensional hidden states recover discriminative performance beyond output-level baselines?

### 1.3 Contributions

This study makes **three** primary contributions.

1. **Transparent negative result (exploratory, §3.1a).** The 64D PCA condition did not satisfy Gate 3 versus BL-LEN. The failure is reported directly. This is not a confirmatory pre-registered result for the model actually executed — see §3.1a.

2. **Paired recovery without variance-max PCA.** On the same 98-prompt inventory, raw 3584D PROP-$\kappa$ reached 0.8472 ($p=0.004$ vs BL-LEN; exploratory). PC4--67 improved on PCA 64D by only $+0.0163$ and remained below BL-LEN. BL-NORM (0.8588) is reported in full on the identical $n=496$.

3. **Massive-activation account of variance collapse.** PC1--3 occupy $99.933\%$ of calibration variance (Sun et al., 2024). The implied mechanism is distortion of relative token placement under a variance-max objective, not a missing-dimension budget.

## 2. Background and Related Work

### 2.1 Internal State Probing and Uncertainty

Probing internal representations for truthfulness and uncertainty has gained significant traction. Azaria & Mitchell (2023) demonstrated that linear classifiers on hidden states can detect statement veracity. Farquhar et al. (2024) introduced Semantic Entropy to capture semantic-level uncertainty. Related internal detectors such as INSIDE (Chen et al., 2024) measure structural properties of intermediate activation spaces. Our work builds upon this foundation by evaluating the **time-series trajectory dynamics ($\kappa$)** of hidden states across autoregressive generation steps.

### 2.2 Massive Activations and Variance Dominance

Sun et al. (2024) identified that modern LLMs exhibit *massive activations*—a small subset of hidden dimensions that systematically register extremely large magnitude values. In linear variance-based methods like PCA, these dimensions dominate the principal components. On `Qwen2.5-7B-Instruct`, PC1--3 capture **$99.933\%$** of calibration variance (PC1--64: $99.992\%$). We interpret this as massive-activation dimensions monopolizing global variance, so that a 64D PCA projection collapses $D=3584$ into an extremely low effective rank governed by magnitude rather than directional trajectory transitions.

### 2.3 Mathematical Formulation of Trajectory $\kappa$

Let $\mathbf{z}_t \in \mathbb{R}^{D}$ be the (possibly PCA-projected) hidden state at generated-token index $t$ and selected layer $\ell$. Write $\hat{\mathbf{z}}_t = \mathbf{z}_t / \lVert\mathbf{z}_t\rVert_2$.

**Eq. (1) — scale-invariant directional statistic.** The one-step cosine change is

$$\kappa^{\mathrm{cos}}_{t} = 1 - \hat{\mathbf{z}}_t^{\top}\hat{\mathbf{z}}_{t+1} = 1 - \frac{\mathbf{z}_t^{\top} \mathbf{z}_{t+1}}{\lVert\mathbf{z}_t\rVert_2 \,\lVert\mathbf{z}_{t+1}\rVert_2},\qquad t=1,\ldots,T-1.$$

This quantity is invariant under any per-token rescaling $\mathbf{z}_t \mapsto \alpha_t\mathbf{z}_t$ with $\alpha_t>0$. It is the diagnostic that isolates *turning* from *magnitude*. Confirmatory PROP-$\kappa$ is **not** $\kappa^{\mathrm{cos}}$; it remains the frozen Gram estimator below. Task B of the reanalysis (`scripts/run_trj001_reanalysis.py`) measures (i) Spearman correlation of `kappa_mean` with mean $\lVert\mathbf{z}_t\rVert_2$, (ii) AUROC of $\kappa^{\mathrm{cos}}$ features computed on $L_2$-normalized trajectories, and (iii) residual discriminability of $\kappa$ after regression on norm.

**Frozen confirmatory curvature.** Reported PROP-$\kappa$ scores use the pre-registered discrete curvature (`src/compute_kappa.py`). Parameterization is by token index, not arc length. Central differences are

$$\mathbf{v}_t = \frac{\mathbf{z}_{t+1}-\mathbf{z}_{t-1}}{2},\qquad \mathbf{a}_t = \mathbf{z}_{t+1}-2\mathbf{z}_t+\mathbf{z}_{t-1}.$$

With Gram matrix $G_t$ of $(\mathbf{v}_t,\mathbf{a}_t)$,

$$\det G_t = \lVert\mathbf{v}_t\rVert_2^2 \lVert\mathbf{a}_t\rVert_2^2 - (\mathbf{v}_t^{\top}\mathbf{a}_t)^2,\qquad \kappa_t = \frac{\sqrt{\det G_t}}{\lVert\mathbf{v}_t\rVert_2^3 + \varepsilon},$$

where $\varepsilon = 10^{-8}$ is frozen. Endpoints $t=1,T$ are undefined. Sequences with $T<20$ are Class E and excluded.

**Intrinsic dimensionality of $\kappa$.** $\det G_t$ is the squared area of the parallelogram spanned by $(\mathbf{v}_t,\mathbf{a}_t)$. Thus $\kappa_t$ is ordinary curvature of a discrete space curve: it is a property of the **osculating plane** and is unchanged if the trajectory is isometrically embedded in any ambient dimension $D'\ge 2$. Extra coordinates that are constant, or that merely rescale the same plane, cannot increase intrinsic curvature. A 64-dimensional representation is therefore not “too small” for $\kappa$ as a geometric object. What variance-maximizing PCA *can* do is rotate and anisotropically rescale relative token placement so that the projected curve is a distorted image of the original. The paired PC4--67 increment of only $+0.0163$ over PCA 64D is consistent with that distortion account and inconsistent with a simple missing-dimension story.

From valid $\{\kappa_t\}$, five length-normalized summaries form $\mathbf{x}_{\kappa}\in\mathbb{R}^{5}$: `kappa_mean`, `kappa_max`, `kappa_p95`, `kappa_std`, `kappa_auc_density`. The confirmatory set also includes `spike_rate` with $\theta$ the train-split 90th percentile of $\kappa_t$. Raw 3584D dropped `spike_rate` (no train-split raw $\theta$), which is why ABL-1 is exploratory. Task D recomputes PCA 64D under the same 5-feature set so that Raw vs PCA is not confounded by feature cardinality.

### 2.4 Classifier Specification

Every representation (PROP-$\kappa$, PROP-$\kappa$+LEN, BL-LEN, BL-NORM) is scored by $L_2$-regularized logistic regression. Features are standardized with `StandardScaler` fit on the training split only. Inverse regularization $C \in \{10^{-3}, 10^{-2}, 10^{-1}, 1, 10, 100\}$ is selected by validation AUROC. On Layer 16 the freeze record selected $C=0.1$ (PROP-$\kappa$), $C=0.001$ (BL-LEN), and $C=100$ (BL-NORM). Test AUROC uses prompt-clustered bootstrap (2,000 resamples); DeLong tests are not used because the eight samples of one prompt are not independent.

## 3. Experimental Protocol

### 3.1 Model and Execution Environment

Values below are the freeze record (`docs/Paper/FREEZE_MANIFEST.md`). `results/env_info.json` in this clone is reconstructed from that record; the GPU-side original file was not present locally. `configs/generation_config.json` still names the pre-U8 Llama draft and is **not** the execution environment: the freeze (Qwen2.5-7B-Instruct, vLLM 0.29.0, CUDA 13.0, A40) is authoritative.

### 3.1a Pre-registration status of this execution (added 2026-09-14, v3.4)

`docs/EXP-2026-NVS-001_v1.3.md` pre-registers the evaluation instrument (§3: Gram-determinant $\kappa$; §6: Gate 3 procedure; §6.1–6.3: bootstrap/Holm methodology) for `meta-llama/Meta-Llama-3.1-8B-Instruct` ($D=4096$). The model actually executed, `Qwen/Qwen2.5-7B-Instruct` ($D=3584$), is a deviation from that pre-registration (tracked as update U8). A revised plan describing this and other deviations (U1, U4, U8, U9 — U1–U3 and U5–U7 have no surviving documentation), `docs/EXP-2026-NVS-001_v1.4.md`, exists only as of 2026-09-14 — written *after* Phase 0–3 execution (2026-09-13) and after this manuscript's numeric results were already known.

Per the plan's own stated purpose (v1.3 §1: freezing hypotheses and criteria before seeing data "to prevent post-hoc reinterpretation and metric-selection bias, i.e. p-hacking"), a plan revision written after the results are known cannot retroactively confer confirmatory status on those results, regardless of how faithfully the pre-specified statistical *procedure* (bootstrap CIs, Holm correction, Gate 3's two-condition rule) was otherwise followed. Consequently:

- Every result in this manuscript that depends on the Qwen2.5-7B/3584D execution (Gate 3 verdict, all AUROC/CI/$p$ values in §4–§5) is **exploratory**, not confirmatory — despite some being labeled "confirmatory" or "pre-registered" elsewhere in earlier drafts of this text.
- This does not mean the numbers are wrong or fabricated — `docs/Paper/FREEZE_MANIFEST.md` records real, hash-verified artifacts and measured statistics. It means the numbers cannot claim the specific evidentiary weight a true pre-registered confirmatory test provides (protection against p-hacking via advance commitment).
- A genuine confirmatory test of H0/Gate 3 for the model actually pre-registered in v1.3 (Llama-3.1-8B-Instruct) has not been run.
- `docs/Paper/FREEZE_MANIFEST.md`'s own `v1.4-frozen` tag has not been applied (its 凍結日時/凍結コミット fields are blank), which is a separate, additional reason this data does not yet meet the plan's own definition of a completed freeze.

| Parameter | Specification |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Model revision | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Inference | vLLM `0.29.0` (PyTorch `2.13.0+cu130`) |
| Hardware | NVIDIA A40 (CUDA 13.0) |
| Python | 3.11.10 |
| Target layers | 8, 16, 24 |
| Hidden dimension $D$ | 3584 |

### 3.2 Dataset, Filtering, and Paired Split

- **Phase 1:** 1,000 Type A prompts $\times$ 8 generations = 8,000 samples.
- **Phase 2:** 493 independent prompts $\times$ 8 = 3,944 samples. Type A entities are collision-checked with the Wikipedia MediaWiki `action=query` API (`missing` flag). No second web-search backend is used.
- **Phase 3:** 299 mixed prompts (2,392 samples). Eligibility is **prompt-cluster** level: a prompt is retained iff its eight generations are mixed (not all-positive and not all-negative). Split: train 296 / val 99 / **test 98**.

**Paired evaluation set (Task A-1-3).** All Layer 16 comparisons among Raw 3584D, PCA 64D, PC4--67, BL-LEN, and BL-NORM use this **same 98-prompt test inventory**. Arithmetic: $98\times 8=784$ generations; freeze $n_{\mathrm{test}}=496$ after Class E ($T<20$); **288 excluded**. BL-LEN and BL-NORM are strictly paired on these $496$ eligible rows—they are not evaluated on the full 784 or on a different prompt list. Inventory composition is therefore not a confounder.

Binary targets: `type_a_positive` $\mapsto y=1$; `type_d_hedge` and `type_c_abstain` $\mapsto y=0$. The task is reliability versus hedging propensity, not unhedged factual error alone.

### 3.3 Five Representations (First-Class Comparators)

1. **PCA 64D (exploratory — see §3.1a; not confirmatory):** Center, project onto PC1--64 (8,357 calibration tokens), then $\kappa$ (6 features including `spike_rate`).
2. **PC4--67 (subspace ablation):** Components 4--67; discards PC1--3.
3. **Raw 3584D (exploratory):** $\kappa$ from uncompressed states; `spike_rate` omitted (5 features).
4. **BL-LEN (output baseline):** Sequence length $T$. Pre-registered Gate 3 comparator.
5. **BL-NORM (magnitude baseline):** Statistics of $\lVert\mathbf{z}_t\rVert_2$ (mean, max, std). Reported in full; not a suppressed comparator.

## 4. Results

### 4.1 Gate 3 (RQ2) — exploratory, see §3.1a

At Layer 16 on the paired 98-prompt set, PROP-$\kappa$ AUROC $= 0.7806$ versus BL-LEN $0.8077$ ($\Delta=-0.0271$, $p=0.8685$). Gate 3 condition 1 (AUROC $>0.5$) passed; condition 2 (exceed BL-LEN) failed. **Gate 3 = FAIL.** This does not imply that trajectories contain no signal—only that the evaluated PCA representation did not beat BL-LEN under the Gate 3 criterion, applied here in an exploratory (post-hoc-plan) capacity rather than as a confirmatory pre-registered test (§3.1a).

### 4.2 Layer Localization (RQ1)

| Layer | PROP-$\kappa$ AUROC | 95% CI |
|---|---:|---|
| 8 | 0.7090 | [0.6554, 0.7599] |
| 16 | **0.7806** | [0.7298, 0.8232] |
| 24 | 0.7167 | [0.6570, 0.7693] |

Holm-corrected: Layer 16 vs 8, $\Delta=+0.0717$, $p<0.001$; Layer 16 vs 24, $\Delta=+0.0639$, $p<0.001$. This is a trajectory-curvature **reconfirmation** of intermediate-layer probing strength within the evaluated set, not a claim that Layer 16 is globally optimal. PROP-$\kappa$+LEN reaches 0.8314 at Layer 24 vs 0.8293 at Layer 16.

### 4.3 Paired Five-Way Comparison (RQ3)

No comparator is omitted. All rows are the same 98-prompt test set ($n=496$).

| Representation | Test AUROC | $\Delta$ vs BL-LEN | $p$ | 95% CI |
|---|---:|---:|---:|---|
| PCA 64D (exploratory, §3.1a) | 0.7806 | $-0.0271$ | 0.8685 | [0.7298, 0.8232] |
| PC4--67 (ablation) | 0.7969 | $-0.0108$ | 0.6935 | [0.7498, 0.8390] |
| BL-LEN (output baseline) | 0.8077 | — | — | [0.7632, 0.8475] |
| Raw 3584D PROP-$\kappa$ (exploratory) | 0.8472 | $+0.0395$ | 0.004 | [0.8060, 0.8824] |
| **BL-NORM (magnitude)** | **0.8588** | $+0.0511$ | — | [0.8239, 0.8903] |

BL-NORM $\Delta=+0.0511$ is the difference of point estimates; its AUROC CI lies entirely above the BL-LEN point estimate. Frozen paired bootstrap $p$ for **BL-NORM vs BL-LEN** and **Raw vs BL-NORM** on the same $n=496$ is Task C of `scripts/run_trj001_reanalysis.py` (requires freeze trajectories; not invented here).

Direct PCA-family pairing:

$$\mathrm{AUROC}_{\mathrm{Raw}\ (0.8472)} > \mathrm{AUROC}_{\mathrm{PC4--67}\ (0.7969)} > \mathrm{AUROC}_{\mathrm{PCA64}\ (0.7806)}$$

PC4--67 vs PCA 64D is only $\Delta=+0.0163$. Removing PC1--3 is not a sufficient fix. Full ordering on this split:

$$\mathrm{BL\text{-}NORM}\ \ge\ \mathrm{Raw}\ >\ \mathrm{BL\text{-}LEN}\ >\ \mathrm{PC4--67}\ >\ \mathrm{PCA\ 64D}.$$

### 4.4 Class E exclusion (Task A)

| Quantity | Value |
|---|---|
| Test prompts | 98 |
| Generations | $98\times 8=784$ |
| Eligible ($T\ge 20$) | $496$ |
| Class E excluded ($T<20$) | $288$ ($36.7\%$) |

Label counts ($y=1$ vs $y=0$) among the 288 excluded generations, positivity rates before vs after exclusion, and AUROC rank stability at $T\ge 10,15,20,30$ for Raw 3584D, PCA 64D, BL-LEN, and BL-NORM are defined in the reanalysis script (Tasks A-1, A-2). They require `results/phase2_trajectories.json` (and projected/raw arrays). Those artifacts are not in this git clone; the script writes them into `results/trj001_reanalysis_metrics.json` when `--data-root` points at the freeze store.

### 4.5 Scale-invariant $\kappa^{\mathrm{cos}}$ and feature-count control (Tasks B, D)

Task B: Spearman($\texttt{kappa\_mean}$, mean $\lVert\mathbf{z}_t\rVert_2$); AUROC of $\kappa^{\mathrm{cos}}$ on $L_2$-normalized trajectories; incremental AUROC of $\kappa$+NORM versus NORM; residual of $\kappa$ after regression on norm.

Task D: 5-feature PCA 64D (`spike_rate` dropped, matching Raw) so that $\Delta(\mathrm{Raw}-\mathrm{PCA})$ is not confounded by having six versus five logistic features.

Both tasks are implemented in `scripts/run_trj001_reanalysis.py`. Numeric fills are the JSON file after a freeze-data run, not the `--self-test` synthetic path.

## 5. Discussion

### 5.1 Massive Activation Dominance — not a missing-dimension story

PC1--3 cumulative explained variance: **99.933%**. PC1--64: **99.992%**. Following Sun et al. (2024), we interpret the Qwen spectrum as massive-activation dimensions monopolizing global variance. Variance-maximizing PCA then collapses $D=3584$ into a near-rank-3 magnitude subspace and can distort the relative configuration that Gram $\kappa$ reads from the osculating plane (§2.3). This is a mechanism hypothesis: the experiment shows signal loss under this projection, not a unique localization of the lost bits in the discarded $0.008\%$ of variance.

The same magnitude geometry is layer-dependent as a predictor: **BL-NORM falls from 0.8588 at Layer 16 to 0.7158 at Layer 24**, consistent with massive-activation structure changing near the output.

The load-bearing paired number for “discarding PC1--3 is not enough” remains **PC4--67 − PCA 64D $= +0.0163$**.

$$\boxed{\text{High Explained Variance} \neq \text{High Task-Relevant Information}}$$

### 5.2 Facing BL-NORM

BL-NORM is the strongest individual score (0.8588). Hiding it would invite the charge that a stronger internal baseline was suppressed. The result is informative: hedging/reliability is tightly coupled to activation *magnitude*—the same geometry that massive activations and leading PCs occupy. Raw $\kappa$ (0.8472) is directional; BL-NORM is magnitude. They are not interchangeable. Gate 3 remains defined against BL-LEN because that was the pre-registered output-level comparator. Paired $\Delta$, 95% CI, and $p$ for BL-NORM vs BL-LEN and Raw vs BL-NORM on $n=496$ are Task C (script); they are not invented from the existing Phase 3 JSON, which lacks those two pairwise tests.

### 5.3 Facing PC4--67

PC4--67 tests the story that PC1--3 are the only nuisance directions. The paired gain over PCA 64D is small ($+0.0163$) and PC4--67 still fails to beat BL-LEN. Removing massive-activation axes is not sufficient to recover the raw-space $\kappa$ signal.

### 5.4 Limitations

1. **All results depending on the executed model are exploratory, not confirmatory (§3.1a).** The plan revision to Qwen2.5-7B-Instruct/3584D (`docs/EXP-2026-NVS-001_v1.4.md`) was written after Phase 0–3 execution and results were known; it cannot satisfy the pre-registration guarantee v1.3 §1 describes. This is the most significant limitation of the study as currently reported and is not merely a documentation formality.
2. Raw 3584D is post-preregistration and dropped `spike_rate` (Task D is the 5-feature control).
3. Labels map factual-positive vs hedge/abstain, not unhedged factual hallucination alone.
4. Only three layers were sampled. Layer-16 superiority is a reconfirmation within that set.
5. Ablation inventory: 98/200 prompts available.
6. Single model (`Qwen2.5-7B-Instruct`), which is itself a deviation from the model actually pre-registered before any data existed (Llama-3.1-8B-Instruct, v1.3).
7. Freeze trajectory arrays (`phase2_trajectories.json`, `phase2_projected/`, `ablation_raw_hidden_states/`) are gitignored and were not present in this clone; Tasks A-1 labels, A-2, B, C, D numbers require mounting that store.
8. `docs/Paper/FREEZE_MANIFEST.md`'s `v1.4-frozen` tag has not been applied (freeze date/commit fields blank), a separate gap from the post-hoc plan-document issue in item 1.

## 6. Conclusions and Future Work

EXP--TRJ001 shows that variance-maximizing PCA can obscure task-critical trajectory signals because massive activations monopolize global variance. 64D PCA failed Gate 3 versus BL-LEN (exploratory result for the executed model, Qwen2.5-7B-Instruct — see §3.1a; a confirmatory test of the originally pre-registered model, Llama-3.1-8B-Instruct, has not been run). On the same 98-prompt set ($n=496$), PC4--67 reached only 0.7969 ($+0.0163$ vs PCA 64D), raw 3584D PROP-$\kappa$ reached 0.8472 ($p=0.004$ vs BL-LEN; exploratory), and BL-NORM reached 0.8588. The findings **indicate**—they do not “establish” as a universal law—that variance-preserving reduction is not signal-preserving reduction for this trajectory probe.

Future work (TRJ002a) will construct task-aware supervised projections $\mathbb{R}^{3584}\to\mathbb{R}^{16\text{--}64}$ that preserve discriminative trajectory geometry rather than global variance alone.

$$\boxed{\text{The problem is not dimensionality reduction itself; it is reducing the wrong geometry.}}$$

## References

- Azaria, A., & Mitchell, T. (2023). The Internal State of an LLM Knows When It's Lying.
- Chen, C., et al. (2024). INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection.
- Farquhar, S., et al. (2024). Detecting Hallucinations in Large Language Models Using Semantic Entropy. *Nature*.
- Qwen Team (2024). Qwen2.5 Technical Report. arXiv:2412.15115.
- Sun, M., Chen, X., Kolter, J. Z., & Liu, Z. (2024). Massive Activations in Large Language Models. COLM 2024. arXiv:2402.17762.

## Appendix. Reanalysis reproduction

```
python scripts/run_trj001_reanalysis.py \
  --data-root /path/to/freeze/results \
  --splits configs/data_splits.json \
  --out results/trj001_reanalysis_metrics.json
```

Expected freeze files: `phase2_trajectories.json`, `phase2_projected/*.npz`, `ablation_raw_hidden_states/*.npy`, `configs/data_splits.json`, optional `configs/W_pca128.npy` / `mu_pca.npy`. Do not pass `--self-test` for paper numbers. `--self-test` writes `results/trj001_reanalysis_metrics.selftest.json` only.
