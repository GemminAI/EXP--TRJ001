"""Phase 4.2 / TRJ002b: evaluate M4 (GP), M5 (Neural ODE), and their
Token-Shuffled Controls on the Test split, plus BL-LEN / BL-NORM, and apply
the pre-registered decision matrix (spec S3).

For each condition (M4, M5) x (genuine, shuffled), a per-trajectory feature
vector is built (M4: 5 GP-derived features; M5: 3 ODE-derived features),
then the same StandardScaler + L2-logistic-regression + prompt-level
cluster-bootstrap pipeline from TRJ002a (trj002a_common.py) is reused to get
Test AUROC, 95% CI, and p-values -- consistently with how TRJ002a's own
benchmark was scored, rather than inventing a second evaluation convention.

Validation Freeze Rule: classifier hyperparameters (C) are selected on Val;
Test is touched only for the final AUROC/CI numbers.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from trj002a_common import (
    build_feature_matrix,
    cluster_bootstrap_auroc,
    cluster_bootstrap_pvalue_diff,
    filter_mixed_prompts,
    kappa_theta_from_train,
    load_split_records,
    select_C_and_fit,
)
from trj002b_common import (
    DEVICE,
    bl_norm_features,
    gp_m4_features,
    load_m5,
    load_p_opt,
    m5_features,
    token_shuffle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

EXTERNAL_BL_LEN = 0.8077
EXTERNAL_BL_NORM = 0.8588
OUTCOME_A_MARGIN = 0.02
OUTCOME_B_MARGIN = 0.02


def project_records(records, project):
    out = []
    for r in records:
        z = project(r["hidden_states"].astype(np.float32))
        out.append({**r, "z": z})
    return out


def build_m4_features(records, project, shuffled: bool):
    X, y, pids = [], [], []
    for r in records:
        z = r["z"]
        if shuffled:
            z = token_shuffle(z, r["prompt_id"], r["sample_index"])
        X.append(gp_m4_features(z))
        y.append(r["label"])
        pids.append(r["prompt_id"])
    return np.array(X), np.array(y), pids


def build_m5_features(records, project, func, odeint, shuffled: bool):
    X, y, pids = [], [], []
    for r in records:
        z = r["z"]
        if shuffled:
            z = token_shuffle(z, r["prompt_id"], r["sample_index"])
        X.append(m5_features(func, odeint, z))
        y.append(r["label"])
        pids.append(r["prompt_id"])
    return np.array(X), np.array(y), pids


def build_simple_features(records, kind: str):
    X, y, pids = [], [], []
    for r in records:
        if kind == "bl_len":
            X.append([r["token_count"]])
        elif kind == "bl_norm":
            X.append(bl_norm_features(r["z"]))
        y.append(r["label"])
        pids.append(r["prompt_id"])
    return np.array(X), np.array(y), pids


def fit_eval(X_tr, y_tr, X_val, y_val, X_test, y_test, pid_test):
    clf, scaler, best_c, val_auroc = select_C_and_fit(X_tr, y_tr, X_val, y_val)
    score_test = clf.predict_proba(scaler.transform(X_test))[:, 1]
    boot = cluster_bootstrap_auroc(y_test, score_test, pid_test)
    return {
        "best_C": best_c,
        "val_auroc": val_auroc,
        "test_auroc": boot["auroc"],
        "test_ci_lo": boot["ci_lo"],
        "test_ci_hi": boot["ci_hi"],
        "_score_test": score_test,
        "_y_test": y_test,
        "_pid_test": pid_test,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "trajectories"))
    ap.add_argument("--splits", default=str(REPO_ROOT / "configs" / "data_splits.json"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))

    print("loading records + projecting via P_opt ...", flush=True)
    project = load_p_opt()
    train_records = filter_mixed_prompts(load_split_records(data_dir, splits["splits"]["train"]))
    val_records = filter_mixed_prompts(load_split_records(data_dir, splits["splits"]["val"]))
    test_records = filter_mixed_prompts(load_split_records(data_dir, splits["splits"]["test"]))
    train_records = project_records(train_records, project)
    val_records = project_records(val_records, project)
    test_records = project_records(test_records, project)
    n_test_prompts = len(set(r["prompt_id"] for r in test_records))
    print(
        f"mixed-prompt filtered: train={len(train_records)} val={len(val_records)} "
        f"test={len(test_records)} ({n_test_prompts} prompts)",
        flush=True,
    )

    func, odeint = load_m5()

    results = {}
    t0 = time.time()

    # --- M4 GP: genuine + shuffled ---
    for tag, shuffled in [("m4_genuine", False), ("m4_shuffled", True)]:
        print(f"building {tag} features ...", flush=True)
        X_tr, y_tr, _ = build_m4_features(train_records, project, shuffled)
        X_val, y_val, _ = build_m4_features(val_records, project, shuffled)
        X_test, y_test, pid_test = build_m4_features(test_records, project, shuffled)
        results[tag] = fit_eval(X_tr, y_tr, X_val, y_val, X_test, y_test, pid_test)
        print(f"  {tag}: test_auroc={results[tag]['test_auroc']:.4f} elapsed={(time.time()-t0)/60:.1f}min", flush=True)

    # --- M5 Neural ODE: genuine + shuffled ---
    for tag, shuffled in [("m5_genuine", False), ("m5_shuffled", True)]:
        print(f"building {tag} features ...", flush=True)
        X_tr, y_tr, _ = build_m5_features(train_records, project, func, odeint, shuffled)
        X_val, y_val, _ = build_m5_features(val_records, project, func, odeint, shuffled)
        X_test, y_test, pid_test = build_m5_features(test_records, project, func, odeint, shuffled)
        results[tag] = fit_eval(X_tr, y_tr, X_val, y_val, X_test, y_test, pid_test)
        print(f"  {tag}: test_auroc={results[tag]['test_auroc']:.4f} elapsed={(time.time()-t0)/60:.1f}min", flush=True)

    # --- BL-LEN / BL-NORM ---
    for tag in ["bl_len", "bl_norm"]:
        print(f"building {tag} features ...", flush=True)
        X_tr, y_tr, _ = build_simple_features(train_records, tag)
        X_val, y_val, _ = build_simple_features(val_records, tag)
        X_test, y_test, pid_test = build_simple_features(test_records, tag)
        results[tag] = fit_eval(X_tr, y_tr, X_val, y_val, X_test, y_test, pid_test)
        print(f"  {tag}: test_auroc={results[tag]['test_auroc']:.4f}", flush=True)

    # --- paired significance tests ---
    def diff(a, b):
        ra, rb = results[a], results[b]
        d = cluster_bootstrap_pvalue_diff(ra["_y_test"], ra["_score_test"], rb["_score_test"], ra["_pid_test"])
        return d

    sig = {
        "m5_vs_m4": diff("m5_genuine", "m4_genuine"),
        "m4_vs_bl_len": diff("m4_genuine", "bl_len"),
        "m5_vs_bl_len": diff("m5_genuine", "bl_len"),
        "m4_genuine_vs_shuffled": diff("m4_genuine", "m4_shuffled"),
        "m5_genuine_vs_shuffled": diff("m5_genuine", "m5_shuffled"),
    }

    auroc_m4 = results["m4_genuine"]["test_auroc"]
    auroc_m5 = results["m5_genuine"]["test_auroc"]
    self_bl_len = results["bl_len"]["test_auroc"]
    self_bl_norm = results["bl_norm"]["test_auroc"]

    # Decision matrix (spec S3), evaluated against BOTH this run's own
    # measured BL-LEN/BL-NORM and the external spec figures (0.8077/0.8588),
    # exactly as TRJ002a's acceptance criterion reported both -- these two
    # baselines were never reproduced on this repo/model either.
    def decide(bl_len, bl_norm, label):
        outcome_a = (auroc_m5 > auroc_m4 + OUTCOME_A_MARGIN) and (sig["m5_vs_m4"]["p_value"] < 0.05)
        outcome_b = (auroc_m4 >= auroc_m5 - OUTCOME_B_MARGIN) and (auroc_m4 > bl_len)
        outcome_c = max(auroc_m4, auroc_m5) < bl_norm
        return {
            "baseline_source": label,
            "bl_len": bl_len,
            "bl_norm": bl_norm,
            "outcome_A_neural_ode_superior": bool(outcome_a),
            "outcome_B_gp_equivalent_or_better": bool(outcome_b),
            "outcome_C_retract_hypothesis": bool(outcome_c),
        }

    decision_external = decide(EXTERNAL_BL_LEN, EXTERNAL_BL_NORM, "external_spec_TRJ001")
    decision_self = decide(self_bl_len, self_bl_norm, "self_measured_this_run")

    clean_results = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()}

    out = {
        "note": (
            "BL-LEN (0.8077) and BL-NORM (0.8588) external figures are taken "
            "from the TRJ002 spec, not reproduced by this repo (same caveat "
            "as TRJ002a's external Raw-3584D figure). This run's own "
            "measured bl_len/bl_norm Test AUROC are reported under 'results' "
            "and used for a self-consistent second decision-matrix reading "
            "('decision_self_measured')."
        ),
        "auroc_m4": auroc_m4,
        "auroc_m5": auroc_m5,
        "significance_tests": sig,
        "decision_external_baselines": decision_external,
        "decision_self_measured": decision_self,
        "results": clean_results,
        "n_test_prompts_mixed": n_test_prompts,
    }

    out_path = REPO_ROOT / "results" / "trj002b_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}", flush=True)

    print("\n=== SUMMARY ===")
    for name, r in clean_results.items():
        print(f"{name:16s} AUROC={r['test_auroc']:.4f} [{r['test_ci_lo']:.4f}, {r['test_ci_hi']:.4f}]")
    print(f"\nM5 vs M4 diff={sig['m5_vs_m4']['diff']:.4f} p={sig['m5_vs_m4']['p_value']}")
    print(f"M4 genuine vs shuffled diff={sig['m4_genuine_vs_shuffled']['diff']:.4f} p={sig['m4_genuine_vs_shuffled']['p_value']}")
    print(f"M5 genuine vs shuffled diff={sig['m5_genuine_vs_shuffled']['diff']:.4f} p={sig['m5_genuine_vs_shuffled']['p_value']}")
    print(f"\nDecision (external baselines): {decision_external}")
    print(f"Decision (self-measured baselines): {decision_self}")


if __name__ == "__main__":
    main()
