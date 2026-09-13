"""Phase 3 analysis for EXP-2026-NVS-001 (plan v1.4 §6): raw-trajectory
hallucination detection, Gate 3 (H0) verdict, F-Layer exploratory comparison,
and the ABL-1/ABL-2 ablations.

Reads Phase 2's saved collection (`results/phase2_trajectories.json` +
per-sample `.npz` projected trajectories + ablation `.npy` raw hidden
states), builds the `PROP-κ` / `BL-LEN` / `BL-NORM` / `PROP-κ+LEN` feature
sets per §4, fits Logistic Regression per §6.1, and evaluates via prompt-unit
cluster bootstrap (2,000 resamples, §6.1 -- explicitly NOT DeLong, since the
8-samples-per-prompt clustering breaks DeLong's independence assumption;
this is a frozen v1.3 decision, unchanged in v1.4).

Ground-truth label (§5.2/§5.5): a sample is `positive` if
`type_a_positive`, `negative` if `type_c_abstain` or `type_d_hedge`.
`class_e_indeterminate` samples are dropped entirely (no label). A prompt
whose remaining (non-Class-E) samples are all-positive or all-negative is
excluded from the main analysis (§5.5 -- confound removal); this module
therefore only ever analyzes MIXED prompts.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# This sklearn version (1.9.1) warns that explicit penalty="l2" is deprecated
# in favor of l1_ratio -- cosmetic only (behavior is unchanged; C-based L2
# regularization still works exactly as passed), but noisy across the ~80
# LogisticRegression fits this module runs per real invocation.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model._logistic")

from kappa_features import KappaFeatures, compute_theta, kappa_features, norm_features

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
RESULTS = REPO_ROOT / "results"

C_GRID = [1e-3, 1e-2, 1e-1, 1, 10, 100]  # §6.1
N_BOOTSTRAP = 2000  # §6.1, frozen (v1.3; unchanged by v1.4 -- see docs/EXP-2026-NVS-001_v1.4.md §6.1 note)
BOOTSTRAP_SEED = 20260912
GATE3_P1_ALPHA = 0.01  # §6.2 condition 1
GATE3_P2_ALPHA = 0.05  # §6.2 condition 2
LAYERS = (8, 16, 24)
PRIMARY_LAYER = 16
POSITIVE_LABEL = "type_a_positive"
NEGATIVE_LABELS = ("type_c_abstain", "type_d_hedge")


# --------------------------------------------------------------------------
# Loading + filtering
# --------------------------------------------------------------------------


def load_phase2_records(path: Path = RESULTS / "phase2_trajectories.json") -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def load_splits(path: Path = CONFIGS / "data_splits.json") -> dict[str, list[str]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {"train": d["train"], "val": d["val"], "test": d["test"]}


def _sample_label(record: dict) -> Optional[int]:
    """1 = positive (hallucination), 0 = negative, None = Class E (dropped)."""
    if record["label"] == POSITIVE_LABEL:
        return 1
    if record["label"] in NEGATIVE_LABELS:
        return 0
    return None


def filter_mixed_prompts(records: list[dict]) -> list[dict]:
    """§5.5: drop Class E samples, then drop every sample belonging to a
    prompt whose remaining samples are all-positive or all-negative."""
    by_prompt: dict[str, list[dict]] = {}
    for r in records:
        label = _sample_label(r)
        if label is None:
            continue
        by_prompt.setdefault(r["prompt_id"], []).append(r)

    kept: list[dict] = []
    for prompt_id, samples in by_prompt.items():
        labels = [_sample_label(s) for s in samples]
        if len(set(labels)) < 2:
            continue  # all-positive or all-negative -- excluded (§5.5)
        kept.extend(samples)
    return kept


# --------------------------------------------------------------------------
# Per-sample data access (projected trajectories / raw ablation states)
# --------------------------------------------------------------------------


@dataclass
class SampleData:
    prompt_id: str
    sample_index: int
    label: int  # 1 positive, 0 negative
    token_count: int
    split: str
    kappa_by_layer: dict[int, np.ndarray]  # layer -> (T,)
    valid_mask_by_layer: dict[int, np.ndarray]  # layer -> (T,) bool
    proj128_by_layer: dict[int, np.ndarray]  # layer -> (T, 128)


def _prompt_split(prompt_id: str, splits: dict[str, list[str]]) -> Optional[str]:
    for name, ids in splits.items():
        if prompt_id in ids:
            return name
    return None


def load_sample_data(records: list[dict], splits: dict[str, list[str]]) -> list[SampleData]:
    """Loads each mixed-prompt sample's per-layer projected trajectory
    (from its .npz) and computes kappa(t) for Layers 8/16/24 uniformly
    (collection time only computed it for the primary layer; §6.1's F-Layer
    comparison needs all three, so it is derived here from the saved 128D
    projections -- same formula, same PC1-64 slice, just applied per-layer)."""
    from compute_kappa import compute_kappa

    out: list[SampleData] = []
    for r in records:
        split = _prompt_split(r["prompt_id"], splits)
        if split is None:
            continue  # not part of the Phase 2 split pool (shouldn't happen)
        label = _sample_label(r)
        if label is None:
            continue

        npz = np.load(REPO_ROOT / r["projected_path"])
        kappa_by_layer: dict[int, np.ndarray] = {}
        valid_by_layer: dict[int, np.ndarray] = {}
        proj_by_layer: dict[int, np.ndarray] = {}
        skip = False
        for layer in LAYERS:
            proj = npz[f"layer_{layer}_128d"].astype(np.float64)
            proj_by_layer[layer] = proj
            if r["token_count"] < 20:
                skip = True
                break
            kr = compute_kappa(proj[:, :64])
            kappa_by_layer[layer] = kr.kappa
            valid_by_layer[layer] = kr.valid_mask
        if skip:
            continue  # Class-E-length sequences shouldn't reach here (already
            # filtered by label), but guard against T<20 edge cases defensively

        out.append(
            SampleData(
                prompt_id=r["prompt_id"],
                sample_index=r["sample_index"],
                label=label,
                token_count=r["token_count"],
                split=split,
                kappa_by_layer=kappa_by_layer,
                valid_mask_by_layer=valid_by_layer,
                proj128_by_layer=proj_by_layer,
            )
        )
    return out


# --------------------------------------------------------------------------
# Feature matrices
# --------------------------------------------------------------------------


def build_features(
    samples: list[SampleData], layer: int, theta: float, feature_set: str
) -> np.ndarray:
    """feature_set in {"BL-LEN", "BL-NORM", "PROP-kappa", "PROP-kappa+LEN"}."""
    rows = []
    for s in samples:
        if feature_set == "BL-LEN":
            rows.append([s.token_count])
        elif feature_set == "BL-NORM":
            rows.append(list(norm_features(s.proj128_by_layer[layer][:, :64])))
        elif feature_set == "PROP-kappa":
            kf = kappa_features(s.kappa_by_layer[layer], s.valid_mask_by_layer[layer], theta)
            rows.append(list(kf.to_array()))
        elif feature_set == "PROP-kappa+LEN":
            kf = kappa_features(s.kappa_by_layer[layer], s.valid_mask_by_layer[layer], theta)
            rows.append(list(kf.to_array()) + [s.token_count])
        else:
            raise ValueError(feature_set)
    return np.array(rows, dtype=np.float64)


def fit_select_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, float]:
    """§6.1: StandardScaler + L2 LogisticRegression; C selected on Val AUROC.
    Returns (test_scores, best_C)."""
    scaler = StandardScaler().fit(X_train)
    Xtr, Xv, Xte = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

    if len(np.unique(y_val)) < 2:
        raise ValueError("Validation split has only one class -- cannot select C via Val AUROC (§6.1)")

    best_c, best_auroc, best_model = C_GRID[0], -np.inf, None
    for c in C_GRID:
        model = LogisticRegression(C=c, penalty="l2", max_iter=2000)
        model.fit(Xtr, y_train)
        val_scores = model.predict_proba(Xv)[:, 1]
        auroc = roc_auc_score(y_val, val_scores)
        if auroc > best_auroc:
            best_auroc, best_c, best_model = auroc, c, model

    test_scores = best_model.predict_proba(Xte)[:, 1]
    return test_scores, best_c


# --------------------------------------------------------------------------
# Cluster bootstrap (prompt-unit, §6.1)
# --------------------------------------------------------------------------


def cluster_bootstrap_auroc(
    y_true: np.ndarray, y_score: np.ndarray, groups: np.ndarray, n_resamples: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    unique_groups = np.unique(groups)
    group_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}
    rng = np.random.default_rng(seed)

    point = roc_auc_score(y_true, y_score)
    boot_aurocs = []
    for _ in range(n_resamples):
        chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_to_idx[g] for g in chosen])
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_aurocs.append(roc_auc_score(yt, y_score[idx]))
    boot_aurocs = np.array(boot_aurocs)

    ci_lo, ci_hi = (float(x) for x in np.percentile(boot_aurocs, [2.5, 97.5]))
    p_le_half = float(np.mean(boot_aurocs <= 0.5))
    return {
        "point_estimate": float(point),
        "ci95_lower": ci_lo,
        "ci95_upper": ci_hi,
        "p_value_le_0.5": p_le_half,
        "n_valid_resamples": int(boot_aurocs.size),
        "n_resamples": n_resamples,
    }


def cluster_bootstrap_delta_auroc(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    groups: np.ndarray,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Delta = AUROC(a) - AUROC(b), same resample indices for both (paired)."""
    unique_groups = np.unique(groups)
    group_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}
    rng = np.random.default_rng(seed)

    point = roc_auc_score(y_true, y_score_a) - roc_auc_score(y_true, y_score_b)
    deltas = []
    for _ in range(n_resamples):
        chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_to_idx[g] for g in chosen])
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        deltas.append(roc_auc_score(yt, y_score_a[idx]) - roc_auc_score(yt, y_score_b[idx]))
    deltas = np.array(deltas)

    ci_lo, ci_hi = (float(x) for x in np.percentile(deltas, [2.5, 97.5]))
    p_le_zero = float(np.mean(deltas <= 0.0))
    return {
        "point_estimate": float(point),
        "ci95_lower": ci_lo,
        "ci95_upper": ci_hi,
        "p_value_le_0": p_le_zero,
        "n_valid_resamples": int(deltas.size),
        "n_resamples": n_resamples,
    }


def holm_correct(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in
    the ORIGINAL input order."""
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p_values[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


# --------------------------------------------------------------------------
# Per-layer AUROC / Gate 3 / F-Layer orchestration
# --------------------------------------------------------------------------


def evaluate_layer(samples: list[SampleData], layer: int) -> dict[str, Any]:
    """Fits BL-LEN, BL-NORM, PROP-kappa, PROP-kappa+LEN for one layer;
    returns Test-split AUROC (point + cluster-bootstrap CI/p) for each, plus
    PROP-kappa vs BL-LEN delta (used directly by Gate 3 for layer 16, and
    reported for 8/24 too as supporting context)."""
    train = [s for s in samples if s.split == "train"]
    val = [s for s in samples if s.split == "val"]
    test = [s for s in samples if s.split == "test"]

    theta = compute_theta(
        [s.kappa_by_layer[layer] for s in train],
        [s.valid_mask_by_layer[layer] for s in train],
    )

    y_train = np.array([s.label for s in train])
    y_val = np.array([s.label for s in val])
    y_test = np.array([s.label for s in test])
    test_groups = np.array([s.prompt_id for s in test])

    results: dict[str, Any] = {"theta": theta, "n_train": len(train), "n_val": len(val), "n_test": len(test)}
    test_scores: dict[str, np.ndarray] = {}
    for feature_set in ("BL-LEN", "BL-NORM", "PROP-kappa", "PROP-kappa+LEN"):
        X_train = build_features(train, layer, theta, feature_set)
        X_val = build_features(val, layer, theta, feature_set)
        X_test = build_features(test, layer, theta, feature_set)
        scores, best_c = fit_select_predict(X_train, y_train, X_val, y_val, X_test)
        test_scores[feature_set] = scores
        boot = cluster_bootstrap_auroc(y_test, scores, test_groups)
        results[feature_set] = {**boot, "selected_C": best_c}

    delta = cluster_bootstrap_delta_auroc(y_test, test_scores["PROP-kappa"], test_scores["BL-LEN"], test_groups)
    results["PROP-kappa_vs_BL-LEN_delta"] = delta
    return results


def gate3_verdict(layer16_results: dict[str, Any]) -> dict[str, Any]:
    """§6.2: both conditions must hold."""
    prop_kappa = layer16_results["PROP-kappa"]
    delta = layer16_results["PROP-kappa_vs_BL-LEN_delta"]

    cond1 = prop_kappa["ci95_lower"] > 0.5 and prop_kappa["p_value_le_0.5"] < GATE3_P1_ALPHA
    cond2 = delta["ci95_lower"] > 0.0 and delta["p_value_le_0"] < GATE3_P2_ALPHA
    return {
        "condition_1_auroc_gt_half": {
            "pass": bool(cond1),
            "ci95_lower": prop_kappa["ci95_lower"],
            "p_value_le_0.5": prop_kappa["p_value_le_0.5"],
            "alpha": GATE3_P1_ALPHA,
        },
        "condition_2_delta_gt_zero": {
            "pass": bool(cond2),
            "ci95_lower": delta["ci95_lower"],
            "p_value_le_0": delta["p_value_le_0"],
            "alpha": GATE3_P2_ALPHA,
        },
        "gate3_pass": bool(cond1 and cond2),
    }


def run_ablation_1(
    samples_test_mixed: list[SampleData],
    ablation_ids: set[str],
    ablation_raw_dir: Path,
) -> Optional[dict[str, Any]]:
    """§6.3 ABL-1: kappa computed directly on the RAW (unprojected) Layer-16
    trajectory, for the ablation-flagged Test-split prompts only (raw
    hidden states were saved only for those -- see execute_phase2_collection).

    Deviation (documented, not silently absorbed): theta normally comes from
    a Train-split pooled distribution (§4.1), but no raw hidden states were
    collected for Train prompts at all (ablation flagging is Test-split-only
    by design, §2.3). `spike_rate` (the only theta-dependent feature) is
    therefore DROPPED for this ablation; the other 5 kappa features, which
    need no threshold, are used as-is.
    """
    from compute_kappa import compute_kappa

    subset = [s for s in samples_test_mixed if s.prompt_id in ablation_ids]
    available = [s for s in subset if (ablation_raw_dir / f"{s.prompt_id}__{s.sample_index}.npy").exists()]
    if len(available) < 10:
        return None

    feats, labels, prompt_ids, token_counts = [], [], [], []
    for s in available:
        raw = np.load(ablation_raw_dir / f"{s.prompt_id}__{s.sample_index}.npy").astype(np.float64)
        kr = compute_kappa(raw)
        kf = kappa_features(kr.kappa, kr.valid_mask, theta=np.inf)  # theta=inf -> spike_rate always 0, dropped below
        feats.append([kf.kappa_mean, kf.kappa_max, kf.kappa_p95, kf.kappa_std, kf.kappa_auc_density])
        labels.append(s.label)
        prompt_ids.append(s.prompt_id)
        token_counts.append(s.token_count)

    if len(set(labels)) < 2:
        return None

    X = np.array(feats)
    y = np.array(labels)
    groups = np.array(prompt_ids)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = LogisticRegression(penalty="l2", C=1.0, max_iter=2000).fit(Xs, y)
    scores = model.predict_proba(Xs)[:, 1]

    boot = cluster_bootstrap_auroc(y, scores, groups)
    len_scores = np.array(token_counts, dtype=np.float64)
    delta = cluster_bootstrap_delta_auroc(y, scores, len_scores, groups)
    return {
        "n_samples": int(X.shape[0]),
        "features_used": ["kappa_mean", "kappa_max", "kappa_p95", "kappa_std", "kappa_auc_density"],
        "note": "spike_rate dropped (no Train-split raw data to calibrate theta -- see docstring)",
        "raw_kappa_auroc": boot,
        "raw_kappa_vs_BL-LEN_delta": delta,
    }


def run_ablation_2(samples: list[SampleData], layer: int = PRIMARY_LAYER) -> dict[str, Any]:
    """§6.3 ABL-2: PC4-67 (0-indexed [3:67], 64 dims) instead of PC1-64.
    Uses the full Test split (no raw data needed -- just a different slice
    of the already-saved 128D projection), unlike ABL-1."""
    from compute_kappa import compute_kappa

    train = [s for s in samples if s.split == "train"]
    test = [s for s in samples if s.split == "test"]

    def pc4_67_kappa(s: SampleData):
        proj = s.proj128_by_layer[layer][:, 3:67]
        return compute_kappa(proj)

    train_kr = [pc4_67_kappa(s) for s in train]
    theta = compute_theta([kr.kappa for kr in train_kr], [kr.valid_mask for kr in train_kr])

    test_kr = [pc4_67_kappa(s) for s in test]
    X_test = np.array([kappa_features(kr.kappa, kr.valid_mask, theta).to_array() for kr in test_kr])
    y_test = np.array([s.label for s in test])
    groups = np.array([s.prompt_id for s in test])

    train_X = np.array([kappa_features(kr.kappa, kr.valid_mask, theta).to_array() for kr in train_kr])
    y_train = np.array([s.label for s in train])
    scaler = StandardScaler().fit(train_X)
    model = LogisticRegression(penalty="l2", C=1.0, max_iter=2000).fit(scaler.transform(train_X), y_train)
    scores = model.predict_proba(scaler.transform(X_test))[:, 1]

    boot = cluster_bootstrap_auroc(y_test, scores, groups)
    len_scores = np.array([s.token_count for s in test], dtype=np.float64)
    delta = cluster_bootstrap_delta_auroc(y_test, scores, len_scores, groups)
    return {"theta": theta, "n_test": len(test), "pc4_67_kappa_auroc": boot, "pc4_67_kappa_vs_BL-LEN_delta": delta}


# --------------------------------------------------------------------------
# Top-level runner
# --------------------------------------------------------------------------


def run_phase3_analysis(
    phase2_path: Path = RESULTS / "phase2_trajectories.json",
    splits_path: Path = CONFIGS / "data_splits.json",
    ablation_ids_path: Path = CONFIGS / "ablation_prompt_ids.json",
    ablation_raw_dir: Path = RESULTS / "ablation_raw_hidden_states",
    output_path: Path = RESULTS / "phase3_metrics.json",
) -> Path:
    print("=== Phase 3: raw-trajectory detector evaluation (Gate 3) ===")
    records = load_phase2_records(phase2_path)
    print(f"[+] loaded {len(records)} Phase 2 records")

    mixed_records = filter_mixed_prompts(records)
    n_mixed_prompts = len({r["prompt_id"] for r in mixed_records})
    print(f"[+] {len(mixed_records)} samples across {n_mixed_prompts} mixed prompts (§5.5 filter applied)")

    splits = load_splits(splits_path)
    samples = load_sample_data(mixed_records, splits)
    print(f"[+] loaded per-sample trajectory data for {len(samples)} samples")

    per_layer: dict[int, Any] = {}
    for layer in LAYERS:
        print(f"[+] evaluating layer {layer}...")
        per_layer[layer] = evaluate_layer(samples, layer)

    gate3 = gate3_verdict(per_layer[PRIMARY_LAYER])
    print(f"[+] Gate 3 verdict: {'PASS' if gate3['gate3_pass'] else 'FAIL'}")

    # F-Layer family (exploratory, Holm-corrected, §7.3)
    train = [s for s in samples if s.split == "train"]
    val = [s for s in samples if s.split == "val"]
    test = [s for s in samples if s.split == "test"]
    y_test = np.array([s.label for s in test])
    test_groups = np.array([s.prompt_id for s in test])

    layer_scores: dict[int, np.ndarray] = {}
    for layer in LAYERS:
        theta = per_layer[layer]["theta"]
        X_train = build_features(train, layer, theta, "PROP-kappa")
        X_val = build_features(val, layer, theta, "PROP-kappa")
        X_test = build_features(test, layer, theta, "PROP-kappa")
        y_train = np.array([s.label for s in train])
        y_val = np.array([s.label for s in val])
        scores, _ = fit_select_predict(X_train, y_train, X_val, y_val, X_test)
        layer_scores[layer] = scores

    f_layer_16_vs_8 = cluster_bootstrap_delta_auroc(y_test, layer_scores[16], layer_scores[8], test_groups)
    f_layer_16_vs_24 = cluster_bootstrap_delta_auroc(y_test, layer_scores[16], layer_scores[24], test_groups)
    raw_p = [f_layer_16_vs_8["p_value_le_0"], f_layer_16_vs_24["p_value_le_0"]]
    adj_p = holm_correct(raw_p)
    f_layer_family = {
        "layer16_vs_layer8": {**f_layer_16_vs_8, "p_holm": adj_p[0]},
        "layer16_vs_layer24": {**f_layer_16_vs_24, "p_holm": adj_p[1]},
    }

    # Ablations (§6.3)
    ablation_ids = set(json.loads(ablation_ids_path.read_text(encoding="utf-8"))["prompt_ids"])
    abl1 = run_ablation_1(test, ablation_ids, ablation_raw_dir)
    abl2 = run_ablation_2(samples)

    out = {
        "n_phase2_records": len(records),
        "n_mixed_prompts": n_mixed_prompts,
        "n_mixed_samples": len(mixed_records),
        "per_layer": per_layer,
        "gate3": gate3,
        "f_layer_family_holm": f_layer_family,
        "ablation_1_raw_kappa": abl1,
        "ablation_2_pc4_67": abl2,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8")
    print(f"[+] saved: {output_path}")
    return output_path


if __name__ == "__main__":
    run_phase3_analysis()
