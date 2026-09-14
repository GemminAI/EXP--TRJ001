"""Shared utilities for EXP-TRJ002a: data loading, kappa features, cluster
bootstrap AUROC/CI/p-value. Reuses src/compute_kappa.py (the frozen kappa
definition from EXP--TRJ001 plan v1.3 S3) and follows the classifier /
evaluation procedure of plan v1.3 S6.1-6.2 (Phase 3 / Gate 3), applied here
to TRJ002a's 6 representation conditions instead of the single PROP-kappa
condition Gate 3 was written for.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from compute_kappa import compute_kappa, MIN_SEQUENCE_LENGTH  # noqa: E402

C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def load_split_records(data_dir: Path, prompt_ids: list[str]) -> list[dict]:
    """Load all saved (T,3584) trajectories for the given prompt_ids."""
    records = []
    for prompt_id in prompt_ids:
        pdir = data_dir / prompt_id
        if not pdir.exists():
            continue
        for npz_path in sorted(pdir.glob("*.npz")):
            d = np.load(npz_path)
            records.append(
                {
                    "prompt_id": prompt_id,
                    "sample_index": int(npz_path.stem),
                    "hidden_states": d["hidden_states"].astype(np.float64),
                    "label": int(d["label"]),
                    "token_count": int(d["token_count"]),
                }
            )
    return records


def filter_mixed_prompts(records: list[dict]) -> list[dict]:
    """Plan v1.3 S5.5: main analysis uses only prompts whose 8 samples are
    not all-positive or all-negative."""
    by_prompt: dict[str, list[int]] = {}
    for r in records:
        by_prompt.setdefault(r["prompt_id"], []).append(r["label"])
    mixed_prompts = {
        pid for pid, labels in by_prompt.items() if len(set(labels)) > 1
    }
    return [r for r in records if r["prompt_id"] in mixed_prompts]


def kappa_theta_from_train(records: list[dict], project_fn) -> float:
    """90th percentile of kappa(t) pooled over the Train split, in the given
    representation space (plan v1.3 S4.1: theta is fixed from Train, applied
    to Val/Test unchanged)."""
    all_kappa = []
    for r in records:
        z = project_fn(r["hidden_states"])
        if z.shape[0] < 3:
            continue
        res = compute_kappa(z)
        vals = res.kappa[res.valid_mask]
        vals = vals[~np.isnan(vals)]
        all_kappa.append(vals)
    pooled = np.concatenate(all_kappa) if all_kappa else np.array([0.0])
    return float(np.percentile(pooled, 90))


def kappa_features(z: np.ndarray, theta: float) -> np.ndarray:
    """Plan v1.3 S4.1's 6 kappa-derived features, length-normalized."""
    res = compute_kappa(z)
    vals = res.kappa[res.valid_mask]
    vals = vals[~np.isnan(vals)]
    T = z.shape[0]
    if vals.size == 0:
        return np.zeros(6)
    kappa_mean = float(np.mean(vals))
    kappa_max = float(np.max(vals))
    kappa_p95 = float(np.percentile(vals, 95))
    kappa_std = float(np.std(vals))
    spike_rate = float(np.mean(vals > theta)) if (T - 2) > 0 else 0.0
    kappa_auc_density = float(np.sum(vals) / max(T - 2, 1))
    return np.array(
        [kappa_mean, kappa_max, kappa_p95, kappa_std, spike_rate, kappa_auc_density]
    )


def build_feature_matrix(
    records: list[dict], project_fn, theta: float
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X, y, pids = [], [], []
    for r in records:
        z = project_fn(r["hidden_states"])
        X.append(kappa_features(z, theta))
        y.append(r["label"])
        pids.append(r["prompt_id"])
    return np.array(X), np.array(y), pids


def select_C_and_fit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[LogisticRegression, StandardScaler, float, float]:
    """Plan v1.3 S6.1 step 2: StandardScaler + L2 logistic regression, C
    selected on Val. Returns (final model fit on train+val at best C,
    scaler fit on train+val, best_C, val_auroc_at_best_C)."""
    scaler = StandardScaler().fit(X_train)
    Xs_train = scaler.transform(X_train)
    Xs_val = scaler.transform(X_val)

    best_c, best_auroc = None, -1.0
    for c in C_GRID:
        clf = LogisticRegression(C=c, penalty="l2", max_iter=2000)
        if len(set(y_train.tolist())) < 2:
            continue
        clf.fit(Xs_train, y_train)
        if len(set(y_val.tolist())) < 2:
            auroc = float("nan")
        else:
            score = clf.predict_proba(Xs_val)[:, 1]
            auroc = roc_auc_score(y_val, score)
        if auroc is not None and not np.isnan(auroc) and auroc > best_auroc:
            best_auroc, best_c = auroc, c

    if best_c is None:
        best_c = 1.0

    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    final_scaler = StandardScaler().fit(X_all)
    final_clf = LogisticRegression(C=best_c, penalty="l2", max_iter=2000)
    final_clf.fit(final_scaler.transform(X_all), y_all)
    return final_clf, final_scaler, best_c, best_auroc


def cluster_bootstrap_auroc(
    y: np.ndarray,
    score: np.ndarray,
    prompt_ids: list[str],
    n_boot: int = 2000,
    seed: int = 20260914,
) -> dict:
    """Plan v1.3 S6.1 step 3: prompt-level cluster bootstrap (resample whole
    prompts with replacement, not individual samples) -> 95% CI for AUROC."""
    rng = np.random.default_rng(seed)
    unique_prompts = np.array(sorted(set(prompt_ids)))
    pid_arr = np.array(prompt_ids)
    point_auroc = roc_auc_score(y, score)

    boot_aurocs = []
    for _ in range(n_boot):
        sampled_prompts = rng.choice(unique_prompts, size=len(unique_prompts), replace=True)
        idx = np.concatenate([np.where(pid_arr == p)[0] for p in sampled_prompts])
        y_b, s_b = y[idx], score[idx]
        if len(set(y_b.tolist())) < 2:
            continue
        boot_aurocs.append(roc_auc_score(y_b, s_b))

    boot_aurocs = np.array(boot_aurocs)
    ci_lo, ci_hi = np.percentile(boot_aurocs, [2.5, 97.5]) if len(boot_aurocs) else (np.nan, np.nan)
    return {
        "auroc": float(point_auroc),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "n_boot_valid": int(len(boot_aurocs)),
    }


def cluster_bootstrap_pvalue_diff(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    prompt_ids: list[str],
    n_boot: int = 2000,
    seed: int = 20260914,
) -> dict:
    """Two-sided bootstrap p-value + 95% CI for AUROC(a) - AUROC(b), using
    the same prompt-level resample for both scores each iteration (paired)."""
    rng = np.random.default_rng(seed)
    unique_prompts = np.array(sorted(set(prompt_ids)))
    pid_arr = np.array(prompt_ids)
    point_diff = roc_auc_score(y, score_a) - roc_auc_score(y, score_b)

    diffs = []
    for _ in range(n_boot):
        sampled_prompts = rng.choice(unique_prompts, size=len(unique_prompts), replace=True)
        idx = np.concatenate([np.where(pid_arr == p)[0] for p in sampled_prompts])
        y_b = y[idx]
        if len(set(y_b.tolist())) < 2:
            continue
        auroc_a = roc_auc_score(y_b, score_a[idx])
        auroc_b = roc_auc_score(y_b, score_b[idx])
        diffs.append(auroc_a - auroc_b)

    diffs = np.array(diffs)
    if len(diffs) == 0:
        return {"diff": float(point_diff), "ci_lo": float("nan"), "ci_hi": float("nan"), "p_value": float("nan")}
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided p-value: proportion of bootstrap diffs on the other side of 0
    # from the point estimate, doubled (standard percentile-bootstrap test).
    if point_diff >= 0:
        p = 2 * np.mean(diffs <= 0)
    else:
        p = 2 * np.mean(diffs >= 0)
    p = float(min(1.0, p))
    return {"diff": float(point_diff), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi), "p_value": p}
