"""Step 2 (TRJ002a): train the 64D projection candidates on Train/Val only,
select the winner by Validation AUROC (kappa-feature + logistic-regression
pipeline, plan v1.3 S6.1), and freeze it as configs/P_opt_64d.{pt,npz}.

Candidates:
  - PCA 64D                  : sklearn PCA, unsupervised.
  - Supervised Linear 64D    : single linear layer 3584->64 (no bias),
                                trained by backprop through a logistic
                                classification head on the mean-pooled
                                64D trajectory (cross-entropy / metric
                                learning per the TRJ002a spec).
  - Supervised Autoencoder 64D: nonlinear encoder 3584->256->64 (ReLU) /
                                decoder 64->256->3584, trained with a joint
                                reconstruction (MSE) + classification (BCE)
                                loss on the mean-pooled 64D encoding.

Validation Freeze Rule: only Train and Val trajectories are touched here.
Test (configs/data_splits.json "test") is never read by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from trj002a_common import (
    REPO_ROOT,
    build_feature_matrix,
    filter_mixed_prompts,
    kappa_theta_from_train,
    load_split_records,
    select_C_and_fit,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LinearProjector(nn.Module):
    def __init__(self, d_in=3584, d_out=64):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out, bias=False)
        self.classifier = nn.Linear(d_out, 1)

    def forward(self, x):  # x: (T, d_in)
        z = self.proj(x)
        pooled = z.mean(dim=0)
        logit = self.classifier(pooled)
        return z, logit


class AutoencoderProjector(nn.Module):
    def __init__(self, d_in=3584, d_hidden=256, d_out=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out)
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_out, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_in)
        )
        self.classifier = nn.Linear(d_out, 1)

    def forward(self, x):  # x: (T, d_in)
        z = self.encoder(x)
        pooled = z.mean(dim=0)
        logit = self.classifier(pooled)
        recon = self.decoder(z)
        return z, logit, recon


def train_linear(records, epochs=8, lr=1e-3, seed=20260914) -> LinearProjector:
    torch.manual_seed(seed)
    model = LinearProjector().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    idx_order = np.arange(len(records))
    for epoch in range(epochs):
        rng.shuffle(idx_order)
        total_loss = 0.0
        for i in idx_order:
            r = records[i]
            x = torch.tensor(r["hidden_states"], dtype=torch.float32, device=DEVICE)
            y = torch.tensor([float(r["label"])], device=DEVICE)
            opt.zero_grad()
            _, logit = model(x)
            loss = bce(logit, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  [SupLinear] epoch {epoch+1}/{epochs} mean_loss={total_loss/len(records):.4f}", flush=True)
    return model


def train_autoencoder(records, epochs=8, lr=1e-3, recon_weight=1.0, class_weight=1.0, seed=20260914) -> AutoencoderProjector:
    torch.manual_seed(seed)
    model = AutoencoderProjector().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    rng = np.random.default_rng(seed)
    idx_order = np.arange(len(records))
    for epoch in range(epochs):
        rng.shuffle(idx_order)
        total_loss = 0.0
        for i in idx_order:
            r = records[i]
            x = torch.tensor(r["hidden_states"], dtype=torch.float32, device=DEVICE)
            y = torch.tensor([float(r["label"])], device=DEVICE)
            opt.zero_grad()
            _, logit, recon = model(x)
            loss = class_weight * bce(logit, y) + recon_weight * mse(recon, x)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  [SupAE] epoch {epoch+1}/{epochs} mean_loss={total_loss/len(records):.4f}", flush=True)
    return model


def make_pca_project_fn(pca: PCA):
    def fn(z: np.ndarray) -> np.ndarray:
        return pca.transform(z)
    return fn


def make_linear_project_fn(model: LinearProjector):
    model.eval()

    def fn(z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor(z, dtype=torch.float32, device=DEVICE)
            out = model.proj(x)
        return out.cpu().numpy()
    return fn


def make_ae_project_fn(model: AutoencoderProjector):
    model.eval()

    def fn(z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor(z, dtype=torch.float32, device=DEVICE)
            out = model.encoder(x)
        return out.cpu().numpy()
    return fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "trajectories"))
    ap.add_argument("--splits", default=str(REPO_ROOT / "configs" / "data_splits.json"))
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    train_records = load_split_records(data_dir, splits["splits"]["train"])
    val_records = load_split_records(data_dir, splits["splits"]["val"])
    print(f"loaded train={len(train_records)} val={len(val_records)} trajectory samples", flush=True)

    # --- PCA 64D ---
    print("fitting PCA 64D ...", flush=True)
    pooled_train = np.concatenate([r["hidden_states"] for r in train_records], axis=0)
    pca = PCA(n_components=64, random_state=20260914)
    pca.fit(pooled_train)
    pca_fn = make_pca_project_fn(pca)

    # --- Supervised Linear 64D ---
    print("training Supervised Linear 64D ...", flush=True)
    lin_model = train_linear(train_records, epochs=args.epochs)
    lin_fn = make_linear_project_fn(lin_model)

    # --- Supervised Autoencoder 64D ---
    print("training Supervised Autoencoder 64D ...", flush=True)
    ae_model = train_autoencoder(train_records, epochs=args.epochs)
    ae_fn = make_ae_project_fn(ae_model)

    candidates = {
        "pca_64d": pca_fn,
        "supervised_linear_64d": lin_fn,
        "supervised_autoencoder_64d": ae_fn,
    }

    # Encoders above are trained on ALL train records (more data for
    # representation learning); Val-AUROC selection below uses the
    # mixed-prompt-filtered subset (plan v1.3 S5.5), matching the
    # convention run_trj002a_benchmark.py uses for the final Test numbers.
    eval_train_records = filter_mixed_prompts(train_records)
    eval_val_records = filter_mixed_prompts(val_records)
    print(
        f"mixed-prompt filtered for AUROC selection: "
        f"train={len(eval_train_records)} val={len(eval_val_records)}",
        flush=True,
    )

    results = {}
    for name, fn in candidates.items():
        theta = kappa_theta_from_train(eval_train_records, fn)
        X_tr, y_tr, _ = build_feature_matrix(eval_train_records, fn, theta)
        X_val, y_val, _ = build_feature_matrix(eval_val_records, fn, theta)
        _, _, best_c, val_auroc = select_C_and_fit(X_tr, y_tr, X_val, y_val)
        results[name] = {"val_auroc": val_auroc, "best_C": best_c, "theta": theta}
        print(f"{name}: val_auroc={val_auroc:.4f} best_C={best_c} theta={theta:.4g}", flush=True)

    winner = max(results, key=lambda k: results[k]["val_auroc"])
    print(f"WINNER (P_opt): {winner} val_auroc={results[winner]['val_auroc']:.4f}", flush=True)

    # Save every candidate (not just the winner) -- the Step 4 benchmark
    # evaluates all 6 conditions independently; P_opt_64d.{pt,npz} is a
    # separate, additional pointer to whichever candidate wins Val AUROC.
    cand_dir = REPO_ROOT / "configs" / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cand_dir / "pca_64d.npz", type="pca", components=pca.components_, mean=pca.mean_)
    np.savez(cand_dir / "supervised_linear_64d.npz", type="linear", W=lin_model.proj.weight.detach().cpu().numpy())
    ae_sd = ae_model.encoder.state_dict()
    np.savez(
        cand_dir / "supervised_autoencoder_64d.npz",
        type="autoencoder",
        W0=ae_sd["0.weight"].cpu().numpy(),
        b0=ae_sd["0.bias"].cpu().numpy(),
        W2=ae_sd["2.weight"].cpu().numpy(),
        b2=ae_sd["2.bias"].cpu().numpy(),
    )

    out_pt = REPO_ROOT / "configs" / "P_opt_64d.pt"
    out_npz = REPO_ROOT / "configs" / "P_opt_64d.npz"

    if winner == "pca_64d":
        torch.save({"type": "pca", "components": pca.components_, "mean": pca.mean_}, out_pt)
        np.savez(out_npz, type="pca", components=pca.components_, mean=pca.mean_)
    elif winner == "supervised_linear_64d":
        torch.save({"type": "linear", "state_dict": lin_model.proj.state_dict()}, out_pt)
        W = lin_model.proj.weight.detach().cpu().numpy()  # (64, 3584)
        np.savez(out_npz, type="linear", W=W)
    else:
        torch.save({"type": "autoencoder", "state_dict": ae_model.encoder.state_dict()}, out_pt)
        sd = ae_model.encoder.state_dict()
        np.savez(
            out_npz,
            type="autoencoder",
            W0=sd["0.weight"].cpu().numpy(),
            b0=sd["0.bias"].cpu().numpy(),
            W2=sd["2.weight"].cpu().numpy(),
            b2=sd["2.bias"].cpu().numpy(),
        )

    manifest = {
        "winner": winner,
        "val_results": results,
        "notice": (
            "Selected by highest Validation-split AUROC (kappa-feature + "
            "logistic-regression pipeline). Test split was not touched by "
            "this script (Validation Freeze Rule)."
        ),
    }
    (REPO_ROOT / "configs" / "P_opt_64d_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"saved {out_pt}, {out_npz}, and manifest.", flush=True)


if __name__ == "__main__":
    main()
