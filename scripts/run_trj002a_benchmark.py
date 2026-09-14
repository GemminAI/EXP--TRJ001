"""Step 3+4 (TRJ002a): evaluate all 6 conditions on the Test split, judge
the acceptance criterion, and write results/trj002a_metrics.json.

Conditions:
  1. raw_3584d                  - identity (no projection)
  2. supervised_autoencoder_64d - configs/candidates/supervised_autoencoder_64d.npz
  3. supervised_linear_64d      - configs/candidates/supervised_linear_64d.npz
  4. random_projection_64d      - fresh seeded W ~ N(0, 1/64), 3584x64 (JL baseline)
  5. pca_64d                    - configs/candidates/pca_64d.npz
  6. bl_len                     - response length T only (1D, not a kappa condition)

Procedure (plan v1.3 S6.1, applied per condition):
  - kappa-derived 6 features (theta from Train, in that condition's space),
    except BL-LEN which is just T.
  - StandardScaler + L2 logistic regression, C selected on Val, refit on
    Train+Val at best C.
  - Test AUROC via prompt-level cluster bootstrap (2000 resamples) -> 95% CI.
  - p-value vs PCA 64D: two-sided cluster-bootstrap test on the paired AUROC
    difference (same resamples for both scores each iteration).

Acceptance criterion (TRJ002a spec S1.2):
  AUROC(P_opt) >= AUROC_Raw(0.8472) - 0.015 = 0.8322, AND p < 0.05 vs PCA 64D.
  AUROC_Raw(0.8472) is taken as given per user confirmation (2026-09-14) --
  it is NOT reproduced by this script, since this repo never ran EXP--TRJ001
  Phase 3 itself (see configs/data_splits.json _notice and this run's own
  README caveats). This script reports this run's OWN measured Raw 3584D
  Test AUROC alongside the external 0.8472 figure so the two are never
  conflated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trj002a_common import (
    REPO_ROOT,
    build_feature_matrix,
    cluster_bootstrap_auroc,
    cluster_bootstrap_pvalue_diff,
    filter_mixed_prompts,
    kappa_theta_from_train,
    load_split_records,
    select_C_and_fit,
)

EXTERNAL_RAW_AUROC = 0.8472  # TRJ001 "established" figure, per spec S1.1/2026-09-14 confirmation
ACCEPTANCE_MARGIN = 0.015
ACCEPTANCE_THRESHOLD = EXTERNAL_RAW_AUROC - ACCEPTANCE_MARGIN  # 0.8322
BL_LEN_EXTERNAL = 0.8077  # spec S1.1 decision-matrix baseline, not reproduced here


def load_npz_projector(path: Path):
    d = np.load(path, allow_pickle=True)
    kind = str(d["type"])
    if kind == "pca":
        components, mean = d["components"], d["mean"]

        def fn(z):
            return (z - mean) @ components.T
        return fn
    if kind == "linear":
        W = d["W"]  # (64, 3584)

        def fn(z):
            return z @ W.T
        return fn
    if kind == "autoencoder":
        W0, b0, W2, b2 = d["W0"], d["b0"], d["W2"], d["b2"]

        def fn(z):
            h = np.maximum(z @ W0.T + b0, 0.0)
            return h @ W2.T + b2
        return fn
    raise ValueError(f"unknown projector type {kind}")


def make_random_projection(seed: int = 20260914, d_in=3584, d_out=64):
    rng = np.random.default_rng(seed)
    W = rng.normal(loc=0.0, scale=1.0 / np.sqrt(d_out), size=(d_in, d_out))

    def fn(z):
        return z @ W
    return fn


def identity_fn(z):
    return z


def evaluate_condition(name, project_fn, train_records, val_records, test_records):
    theta = kappa_theta_from_train(train_records, project_fn)
    X_tr, y_tr, _ = build_feature_matrix(train_records, project_fn, theta)
    X_val, y_val, _ = build_feature_matrix(val_records, project_fn, theta)
    X_test, y_test, pid_test = build_feature_matrix(test_records, project_fn, theta)

    clf, scaler, best_c, val_auroc = select_C_and_fit(X_tr, y_tr, X_val, y_val)
    score_test = clf.predict_proba(scaler.transform(X_test))[:, 1]

    boot = cluster_bootstrap_auroc(y_test, score_test, pid_test)
    return {
        "name": name,
        "best_C": best_c,
        "val_auroc": val_auroc,
        "test_auroc": boot["auroc"],
        "test_ci_lo": boot["ci_lo"],
        "test_ci_hi": boot["ci_hi"],
        "n_boot_valid": boot["n_boot_valid"],
        "_score_test": score_test,
        "_y_test": y_test,
        "_pid_test": pid_test,
    }


def evaluate_bl_len(train_records, val_records, test_records):
    def feat(records):
        X = np.array([[r["token_count"]] for r in records], dtype=np.float64)
        y = np.array([r["label"] for r in records])
        pids = [r["prompt_id"] for r in records]
        return X, y, pids

    X_tr, y_tr, _ = feat(train_records)
    X_val, y_val, _ = feat(val_records)
    X_test, y_test, pid_test = feat(test_records)
    clf, scaler, best_c, val_auroc = select_C_and_fit(X_tr, y_tr, X_val, y_val)
    score_test = clf.predict_proba(scaler.transform(X_test))[:, 1]
    boot = cluster_bootstrap_auroc(y_test, score_test, pid_test)
    return {
        "name": "bl_len",
        "best_C": best_c,
        "val_auroc": val_auroc,
        "test_auroc": boot["auroc"],
        "test_ci_lo": boot["ci_lo"],
        "test_ci_hi": boot["ci_hi"],
        "n_boot_valid": boot["n_boot_valid"],
        "_score_test": score_test,
        "_y_test": y_test,
        "_pid_test": pid_test,
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "trajectories"))
    ap.add_argument("--splits", default=str(REPO_ROOT / "configs" / "data_splits.json"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))

    train_records = filter_mixed_prompts(load_split_records(data_dir, splits["splits"]["train"]))
    val_records = filter_mixed_prompts(load_split_records(data_dir, splits["splits"]["val"]))
    test_records = filter_mixed_prompts(load_split_records(data_dir, splits["splits"]["test"]))
    print(
        f"mixed-prompt filtered counts: train={len(train_records)} "
        f"val={len(val_records)} test={len(test_records)}",
        flush=True,
    )
    n_test_prompts = len(set(r["prompt_id"] for r in test_records))
    print(f"test split: {n_test_prompts} mixed prompts", flush=True)

    cand_dir = REPO_ROOT / "configs" / "candidates"
    conditions = {
        "raw_3584d": identity_fn,
        "supervised_autoencoder_64d": load_npz_projector(cand_dir / "supervised_autoencoder_64d.npz"),
        "supervised_linear_64d": load_npz_projector(cand_dir / "supervised_linear_64d.npz"),
        "random_projection_64d": make_random_projection(),
        "pca_64d": load_npz_projector(cand_dir / "pca_64d.npz"),
    }

    results = {}
    for name, fn in conditions.items():
        print(f"evaluating {name} ...", flush=True)
        results[name] = evaluate_condition(name, fn, train_records, val_records, test_records)

    print("evaluating bl_len ...", flush=True)
    results["bl_len"] = evaluate_bl_len(train_records, val_records, test_records)

    # p-value / delta vs PCA 64D for every condition
    pca_score = results["pca_64d"]["_score_test"]
    pca_y = results["pca_64d"]["_y_test"]
    pca_pid = results["pca_64d"]["_pid_test"]
    for name, r in results.items():
        if name == "pca_64d":
            r["delta_vs_pca64"] = 0.0
            r["p_value_vs_pca64"] = None
            continue
        # PCA and this condition share the same underlying test records/order
        # (same filtered split, same iteration order), so scores align 1:1.
        diff = cluster_bootstrap_pvalue_diff(pca_y, r["_score_test"], pca_score, pca_pid)
        r["delta_vs_pca64"] = diff["diff"]
        r["p_value_vs_pca64"] = diff["p_value"]
        r["delta_ci_lo"] = diff["ci_lo"]
        r["delta_ci_hi"] = diff["ci_hi"]

    # P_opt = whichever candidate won Val AUROC in Step 2
    manifest_path = REPO_ROOT / "configs" / "P_opt_64d_manifest.json"
    p_opt_name = None
    if manifest_path.exists():
        p_opt_name = json.loads(manifest_path.read_text())["winner"]

    acceptance = None
    if p_opt_name and p_opt_name in results:
        r = results[p_opt_name]
        meets_margin = r["test_auroc"] >= ACCEPTANCE_THRESHOLD
        p_sig = (r["p_value_vs_pca64"] is not None) and (r["p_value_vs_pca64"] < 0.05)
        acceptance = {
            "p_opt": p_opt_name,
            "p_opt_test_auroc": r["test_auroc"],
            "threshold": ACCEPTANCE_THRESHOLD,
            "meets_margin_vs_external_raw_0_8472": meets_margin,
            "p_value_vs_pca64": r["p_value_vs_pca64"],
            "significant_vs_pca64_p_lt_0_05": p_sig,
            "ACCEPTED": bool(meets_margin and p_sig),
        }

    # strip internal arrays before serializing
    clean_results = {}
    for name, r in results.items():
        clean_results[name] = {k: v for k, v in r.items() if not k.startswith("_")}

    out = {
        "external_raw_3584d_auroc_TRJ001": EXTERNAL_RAW_AUROC,
        "external_bl_len_auroc_TRJ001": BL_LEN_EXTERNAL,
        "note_on_external_figures": (
            "These two figures are taken from the TRJ002a spec / user "
            "confirmation, not reproduced by this repo (EXP--TRJ001 here "
            "never ran Phase 1-3; see configs/data_splits.json). This run's "
            "OWN measured raw_3584d and bl_len Test AUROC are reported "
            "below under 'results' for direct, apples-to-apples comparison "
            "against this run's own pca_64d / supervised_* numbers."
        ),
        "acceptance_criterion": acceptance,
        "results": clean_results,
        "n_test_prompts_mixed": n_test_prompts,
    }

    out_path = REPO_ROOT / "results" / "trj002a_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)

    print("\n=== SUMMARY (this run's own measured Test AUROC) ===")
    for name, r in clean_results.items():
        print(
            f"{name:28s} AUROC={r['test_auroc']:.4f} "
            f"[{r['test_ci_lo']:.4f}, {r['test_ci_hi']:.4f}] "
            f"delta_vs_pca64={r.get('delta_vs_pca64', 0):.4f} "
            f"p={r.get('p_value_vs_pca64')}"
        )
    if acceptance:
        print(f"\nAcceptance criterion (P_opt={acceptance['p_opt']}): ACCEPTED={acceptance['ACCEPTED']}")


if __name__ == "__main__":
    main()
