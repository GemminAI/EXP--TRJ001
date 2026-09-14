"""Phase 4.2 / TRJ002b: train the shared Neural ODE vector field f_theta on
Train, monitor Val loss for early stopping, freeze as configs/M5_ode_state.pt.

Validation Freeze Rule: Test split is never read here. Architecture (64->128
tanh MLP) and solver tolerances (rtol=1e-3, atol=1e-4, dopri5) are fixed
before touching Test, per the spec's Gate 4.

f_theta is a single set of weights shared across all trajectories (a global
model of "how P_opt(h(t)) evolves"), fit by integrating from each
trajectory's own observed z(0) and minimizing reconstruction MSE against
that trajectory's observed z(t) at its own token positions -- the standard
neural-ODE-as-trajectory-model training objective implied by spec S2.1's
zhat(0) = z(0) initial condition.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torchdiffeq import odeint

from trj002a_common import load_split_records
from trj002b_common import DEVICE, ODE_STATE_PATH, SOLVER_KW, ODEFunc, load_p_opt, normalized_time

REPO_ROOT = Path(__file__).resolve().parents[1]


def trajectory_loss(func, project, r) -> torch.Tensor:
    z = project(r["hidden_states"].astype(np.float32))
    T = z.shape[0]
    z_obs = torch.tensor(z, dtype=torch.float32, device=DEVICE)
    t_eval = torch.tensor(normalized_time(T), dtype=torch.float32, device=DEVICE)
    zhat = odeint(func, z_obs[0], t_eval, **SOLVER_KW)
    return ((zhat - z_obs) ** 2).mean()


def evaluate(func, project, records) -> float:
    func.eval()
    losses = []
    with torch.no_grad():
        for r in records:
            losses.append(trajectory_loss(func, project, r).item())
    func.train()
    return float(np.mean(losses))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "trajectories"))
    ap.add_argument("--splits", default=str(REPO_ROOT / "configs" / "data_splits.json"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit-train", type=int, default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    train_records = load_split_records(data_dir, splits["splits"]["train"])
    val_records = load_split_records(data_dir, splits["splits"]["val"])
    if args.limit_train:
        train_records = train_records[: args.limit_train]
        val_records = val_records[: max(1, args.limit_train // 4)]
    print(f"train={len(train_records)} val={len(val_records)}", flush=True)

    project = load_p_opt()

    torch.manual_seed(20260914)
    func = ODEFunc().to(DEVICE)
    opt = torch.optim.Adam(func.parameters(), lr=args.lr)

    rng = np.random.default_rng(20260914)
    idx_order = np.arange(len(train_records))
    best_val = float("inf")
    t0 = time.time()

    for epoch in range(args.epochs):
        rng.shuffle(idx_order)
        total_loss = 0.0
        for count, i in enumerate(idx_order, 1):
            r = train_records[i]
            opt.zero_grad()
            loss = trajectory_loss(func, project, r)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(func.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()
            if count % 200 == 0:
                elapsed = time.time() - t0
                print(
                    f"  epoch {epoch+1}/{args.epochs} [{count}/{len(train_records)}] "
                    f"running_mean_loss={total_loss/count:.5f} elapsed={elapsed/60:.1f}min",
                    flush=True,
                )

        val_loss = evaluate(func, project, val_records)
        print(
            f"epoch {epoch+1}/{args.epochs} DONE train_loss={total_loss/len(train_records):.5f} "
            f"val_loss={val_loss:.5f} elapsed={(time.time()-t0)/60:.1f}min",
            flush=True,
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(func.state_dict(), ODE_STATE_PATH)
            print(f"  -> saved new best (val_loss={val_loss:.5f}) to {ODE_STATE_PATH}", flush=True)

    print(f"DONE. best_val_loss={best_val:.5f}", flush=True)


if __name__ == "__main__":
    main()
