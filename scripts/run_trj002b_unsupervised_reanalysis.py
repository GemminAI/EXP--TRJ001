"""ANALYSIS-TRJ002B-STRATEGY-001 Step 1: Norm-Free re-analysis on the
UNSUPERVISED PCA 64D projection instead of P_opt (Supervised Autoencoder).

Purpose: the first Norm-Free re-analysis (run_trj002b_temporal_reanalysis.py)
showed M5-NF passing its token-shuffled validity gate at p=0.0, but BL-NORM
(pure ||z(t)|| statistics) still scored 0.9901 -- almost certainly because
P_opt's classification loss (applied to the mean-pooled 64D encoding) baked
label information directly into the latent norm. PCA 64D (trained
unsupervised in TRJ002a, no classification signal at all) has no such
mechanism to shape its norm around the label, so this script re-runs the
identical Norm-Free pipeline on PCA space to check whether:
  (a) BL-NORM's AUROC collapses toward a much weaker, more "generic"
      baseline once the supervised norm-shaping is removed entirely, and
  (b) M5-NF still beats its shuffled control at p<0.01 in this unsupervised
      space -- i.e. the temporal-dynamics signal found in the P_opt space
      was not itself somehow an artifact of P_opt's supervision either.

Everything else (decomposition, kappa_cos, M4-NF GP, M5-NF Neural ODE,
token-shuffle, evaluation/bootstrap pipeline, decision matrix) is identical
to run_trj002b_temporal_reanalysis.py -- only the projection and the M5-NF
checkpoint path change, so this run is directly comparable to that one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from trj002a_common import (
    cluster_bootstrap_auroc,
    cluster_bootstrap_pvalue_diff,
    filter_mixed_prompts,
    load_split_records,
    select_C_and_fit,
)
from trj002b_common import (
    DEVICE,
    ODEFunc,
    SOLVER_KW,
    bl_norm_features,
    gp_m4_nf_features,
    load_pca,
    m5_features,
    normalized_time,
    token_shuffle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from features.norm_free_trajectory import (  # noqa: E402
    decompose,
    direction_arc_length,
    kappa_cos_features,
    kappa_cos_theta_from_train,
)

ODE_NF_PCA_STATE_PATH = REPO_ROOT / "configs" / "M5_NF_PCA_ode_state.pt"

EXTERNAL_BL_LEN = 0.8077
EXTERNAL_BL_NORM = 0.8588
OUTCOME_A_MARGIN = 0.02
OUTCOME_B_MARGIN = 0.02
SHUFFLE_VALIDITY_P = 0.01


def train_m5_nf(train_dirs, val_dirs, epochs=4, lr=1e-3, seed=20260915):
    torch.manual_seed(seed)
    func = ODEFunc().to(DEVICE)
    opt = torch.optim.Adam(func.parameters(), lr=lr)
    from torchdiffeq import odeint

    def traj_loss(direction):
        T = direction.shape[0]
        z_obs = torch.tensor(direction, dtype=torch.float32, device=DEVICE)
        t_eval = torch.tensor(normalized_time(T), dtype=torch.float32, device=DEVICE)
        zhat = odeint(func, z_obs[0], t_eval, **SOLVER_KW)
        return ((zhat - z_obs) ** 2).mean()

    rng = np.random.default_rng(seed)
    idx_order = np.arange(len(train_dirs))
    best_val = float("inf")
    t0 = time.time()
    for epoch in range(epochs):
        rng.shuffle(idx_order)
        total = 0.0
        for count, i in enumerate(idx_order, 1):
            opt.zero_grad()
            loss = traj_loss(train_dirs[i])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(func.parameters(), 5.0)
            opt.step()
            total += loss.item()
            if count % 300 == 0:
                print(
                    f"  [M5-NF/PCA] epoch {epoch+1}/{epochs} [{count}/{len(train_dirs)}] "
                    f"loss={total/count:.5f} elapsed={(time.time()-t0)/60:.1f}min",
                    flush=True,
                )
        func.eval()
        with torch.no_grad():
            val_loss = float(np.mean([traj_loss(d).item() for d in val_dirs]))
        func.train()
        print(
            f"[M5-NF/PCA] epoch {epoch+1}/{epochs} DONE train_loss={total/len(train_dirs):.5f} "
            f"val_loss={val_loss:.5f} elapsed={(time.time()-t0)/60:.1f}min",
            flush=True,
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(func.state_dict(), ODE_NF_PCA_STATE_PATH)
            print(f"  -> saved new best M5-NF/PCA (val_loss={val_loss:.5f})", flush=True)

    func.eval()
    func.load_state_dict(torch.load(ODE_NF_PCA_STATE_PATH, map_location=DEVICE))
    from torchdiffeq import odeint as odeint2

    return func, odeint2


def maybe_shuffle(direction, r, shuffled):
    return token_shuffle(direction, r["prompt_id"], r["sample_index"]) if shuffled else direction


def build_kappa_cos(records, theta, shuffled):
    X, y, pids = [], [], []
    for r in records:
        d = maybe_shuffle(r["direction"], r, shuffled)
        X.append(kappa_cos_features(d, theta))
        y.append(r["label"])
        pids.append(r["prompt_id"])
    return np.array(X), np.array(y), pids


def build_m4_nf(records, shuffled):
    X, y, pids = [], [], []
    for r in records:
        d = maybe_shuffle(r["direction"], r, shuffled)
        X.append(gp_m4_nf_features(direction_arc_length(d)))
        y.append(r["label"])
        pids.append(r["prompt_id"])
    return np.array(X), np.array(y), pids


def build_m5_nf(records, func, odeint, shuffled):
    X, y, pids = [], [], []
    for r in records:
        d = maybe_shuffle(r["direction"], r, shuffled)
        X.append(m5_features(func, odeint, d))
        y.append(r["label"])
        pids.append(r["prompt_id"])
    return np.array(X), np.array(y), pids


def build_bl_norm(records):
    X, y, pids = [], [], []
    for r in records:
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
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))

    print("loading records + projecting via UNSUPERVISED PCA 64D + decomposing norm/direction ...", flush=True)
    project = load_pca()
    train_records = filter_mixed_prompts(load_split_records(data_dir, splits["splits"]["train"]))
    val_records = filter_mixed_prompts(load_split_records(data_dir, splits["splits"]["val"]))
    test_records = filter_mixed_prompts(load_split_records(data_dir, splits["splits"]["test"]))

    for recs in (train_records, val_records, test_records):
        for r in recs:
            z = project(r["hidden_states"].astype(np.float32))
            magnitude, direction = decompose(z)
            r["z"] = z
            r["magnitude"] = magnitude
            r["direction"] = direction

    n_test_prompts = len(set(r["prompt_id"] for r in test_records))
    print(
        f"mixed-prompt filtered: train={len(train_records)} val={len(val_records)} "
        f"test={len(test_records)} ({n_test_prompts} prompts)",
        flush=True,
    )

    print("training M5-NF/PCA (Neural ODE on PCA direction trajectory) ...", flush=True)
    train_dirs = [r["direction"] for r in train_records]
    val_dirs = [r["direction"] for r in val_records]
    func, odeint = train_m5_nf(train_dirs, val_dirs, epochs=args.epochs)

    theta = kappa_cos_theta_from_train(train_dirs)
    print(f"kappa_cos theta (90th pct, Train, PCA space) = {theta:.4g}", flush=True)

    results = {}
    t0 = time.time()

    stages = [
        ("kappa_cos_NF_genuine", lambda recs, sh: build_kappa_cos(recs, theta, sh), False),
        ("kappa_cos_NF_shuffled", lambda recs, sh: build_kappa_cos(recs, theta, sh), True),
        ("m4_NF_genuine", lambda recs, sh: build_m4_nf(recs, sh), False),
        ("m4_NF_shuffled", lambda recs, sh: build_m4_nf(recs, sh), True),
        ("m5_NF_genuine", lambda recs, sh: build_m5_nf(recs, func, odeint, sh), False),
        ("m5_NF_shuffled", lambda recs, sh: build_m5_nf(recs, func, odeint, sh), True),
    ]
    for tag, builder, shuffled in stages:
        print(f"building {tag} features ...", flush=True)
        X_tr, y_tr, _ = builder(train_records, shuffled)
        X_val, y_val, _ = builder(val_records, shuffled)
        X_test, y_test, pid_test = builder(test_records, shuffled)
        results[tag] = fit_eval(X_tr, y_tr, X_val, y_val, X_test, y_test, pid_test)
        print(f"  {tag}: test_auroc={results[tag]['test_auroc']:.4f} elapsed={(time.time()-t0)/60:.1f}min", flush=True)

    print("building bl_norm (PCA-space magnitude-only control) features ...", flush=True)
    X_tr, y_tr, _ = build_bl_norm(train_records)
    X_val, y_val, _ = build_bl_norm(val_records)
    X_test, y_test, pid_test = build_bl_norm(test_records)
    results["bl_norm"] = fit_eval(X_tr, y_tr, X_val, y_val, X_test, y_test, pid_test)
    print(f"  bl_norm: test_auroc={results['bl_norm']['test_auroc']:.4f}", flush=True)

    def diff(a, b):
        ra, rb = results[a], results[b]
        return cluster_bootstrap_pvalue_diff(ra["_y_test"], ra["_score_test"], rb["_score_test"], ra["_pid_test"])

    sig = {
        "kappa_cos_NF_genuine_vs_shuffled": diff("kappa_cos_NF_genuine", "kappa_cos_NF_shuffled"),
        "m4_NF_genuine_vs_shuffled": diff("m4_NF_genuine", "m4_NF_shuffled"),
        "m5_NF_genuine_vs_shuffled": diff("m5_NF_genuine", "m5_NF_shuffled"),
        "m5_NF_vs_m4_NF": diff("m5_NF_genuine", "m4_NF_genuine"),
        "m4_NF_vs_bl_norm": diff("m4_NF_genuine", "bl_norm"),
        "m5_NF_vs_bl_norm": diff("m5_NF_genuine", "bl_norm"),
        "kappa_cos_NF_vs_bl_norm": diff("kappa_cos_NF_genuine", "bl_norm"),
    }

    auroc_m4nf = results["m4_NF_genuine"]["test_auroc"]
    auroc_m5nf = results["m5_NF_genuine"]["test_auroc"]
    self_bl_norm = results["bl_norm"]["test_auroc"]

    m4_valid = sig["m4_NF_genuine_vs_shuffled"]["p_value"] < SHUFFLE_VALIDITY_P and sig["m4_NF_genuine_vs_shuffled"]["diff"] > 0
    m5_valid = sig["m5_NF_genuine_vs_shuffled"]["p_value"] < SHUFFLE_VALIDITY_P and sig["m5_NF_genuine_vs_shuffled"]["diff"] > 0

    def decide(bl_len, bl_norm, label):
        outcome_a = (auroc_m5nf > auroc_m4nf + OUTCOME_A_MARGIN) and (sig["m5_NF_vs_m4_NF"]["p_value"] < 0.05)
        outcome_b = (auroc_m4nf >= auroc_m5nf - OUTCOME_B_MARGIN) and (auroc_m4nf > bl_len)
        outcome_c = max(auroc_m4nf, auroc_m5nf) < bl_norm
        return {
            "baseline_source": label,
            "bl_len": bl_len,
            "bl_norm": bl_norm,
            "outcome_A_neural_ode_superior": bool(outcome_a),
            "outcome_B_gp_equivalent_or_better": bool(outcome_b),
            "outcome_C_retract_hypothesis": bool(outcome_c),
        }

    decision_external = decide(EXTERNAL_BL_LEN, EXTERNAL_BL_NORM, "external_spec_TRJ001")
    decision_self = decide(EXTERNAL_BL_LEN, self_bl_norm, "self_measured_bl_norm_pca_space")

    # comparison against the P_opt (supervised) Norm-Free run, if available
    prior_path = REPO_ROOT / "results" / "trj002b_temporal_reanalysis_metrics.json"
    prior_bl_norm = None
    if prior_path.exists():
        prior_bl_norm = json.loads(prior_path.read_text())["results"]["bl_norm"]["test_auroc"]

    final_verdict = {
        "m4_NF_passes_shuffle_validity_gate_p_lt_0.01": bool(m4_valid),
        "m5_NF_passes_shuffle_validity_gate_p_lt_0.01": bool(m5_valid),
        "bl_norm_pca_space": self_bl_norm,
        "bl_norm_p_opt_space_prior_run": prior_bl_norm,
        "bl_norm_dropped_when_unsupervised": (
            bool(prior_bl_norm is not None and self_bl_norm < prior_bl_norm - 0.02)
        ),
        "interpretation": (
            "If bl_norm_dropped_when_unsupervised is True, this confirms "
            "BL-NORM's near-perfect score in the P_opt (supervised) space "
            "was a supervision artifact, not a generic property of "
            "P_opt(h(t)) trajectories -- supporting the recommendation that "
            "BL-NORM comparisons against a supervised projection's own "
            "latent space are not a fair 'generic' baseline. A model must "
            "ALSO beat its own token-shuffled control at p<0.01 (spec "
            "S2.1) to be read as capturing genuine temporal dynamics -- "
            "check the two validity-gate flags above independently of the "
            "raw Outcome A/B/C arithmetic."
        ),
    }

    clean_results = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()}

    out = {
        "note": (
            "Same Norm-Free pipeline as run_trj002b_temporal_reanalysis.py, "
            "but projected through the UNSUPERVISED PCA 64D candidate "
            "(configs/candidates/pca_64d.npz) instead of P_opt (Supervised "
            "Autoencoder), to test whether BL-NORM's near-perfect score was "
            "specific to P_opt's classification-shaped latent norm."
        ),
        "auroc_m4_NF": auroc_m4nf,
        "auroc_m5_NF": auroc_m5nf,
        "auroc_kappa_cos_NF": results["kappa_cos_NF_genuine"]["test_auroc"],
        "significance_tests": sig,
        "decision_external_baselines": decision_external,
        "decision_self_measured_bl_norm": decision_self,
        "final_verdict": final_verdict,
        "results": clean_results,
        "n_test_prompts_mixed": n_test_prompts,
    }

    out_path = REPO_ROOT / "results" / "trj002b_unsupervised_reanalysis_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}", flush=True)

    print("\n=== SUMMARY (Unsupervised PCA Norm-Free re-analysis) ===")
    for name, r in clean_results.items():
        print(f"{name:24s} AUROC={r['test_auroc']:.4f} [{r['test_ci_lo']:.4f}, {r['test_ci_hi']:.4f}]")
    print(f"\nkappa_cos_NF genuine vs shuffled: diff={sig['kappa_cos_NF_genuine_vs_shuffled']['diff']:.4f} p={sig['kappa_cos_NF_genuine_vs_shuffled']['p_value']}")
    print(f"M4-NF genuine vs shuffled: diff={sig['m4_NF_genuine_vs_shuffled']['diff']:.4f} p={sig['m4_NF_genuine_vs_shuffled']['p_value']}  [validity gate p<0.01: {m4_valid}]")
    print(f"M5-NF genuine vs shuffled: diff={sig['m5_NF_genuine_vs_shuffled']['diff']:.4f} p={sig['m5_NF_genuine_vs_shuffled']['p_value']}  [validity gate p<0.01: {m5_valid}]")
    print(f"M5-NF vs M4-NF: diff={sig['m5_NF_vs_m4_NF']['diff']:.4f} p={sig['m5_NF_vs_m4_NF']['p_value']}")
    print(f"\nBL-NORM: PCA space={self_bl_norm:.4f} vs P_opt space (prior run)={prior_bl_norm}")
    print(f"Decision (external baselines): {decision_external}")
    print(f"Decision (self-measured BL-NORM): {decision_self}")
    print(f"\nFinal verdict: {final_verdict}")


if __name__ == "__main__":
    main()
