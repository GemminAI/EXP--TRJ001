"""Shared utilities for EXP-TRJ002b (Phase 4.2): P_opt projection, GP (M4)
feature extraction, Neural ODE (M5) model, token-shuffled control, and the
BL-NORM baseline the spec's decision matrix (S3) references. Reuses the
kappa-feature-style logistic-regression + cluster-bootstrap machinery from
trj002a_common.py rather than reimplementing it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trj002a_common import (  # noqa: E402  (re-exported for benchmark script)
    C_GRID,
    cluster_bootstrap_auroc,
    cluster_bootstrap_pvalue_diff,
    filter_mixed_prompts,
    load_split_records,
    select_C_and_fit,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MASK_SEED = 20260912  # configs/mask_protocol.json "mask_seed", reused for the
# token-shuffled control's own permutation draw (a different, documented use
# of the same frozen seed value -- not a reinterpretation of the masking
# protocol itself, which TRJ002b does not use).


# ---------------------------------------------------------------------------
# P_opt (Supervised Autoencoder 64D) projection
# ---------------------------------------------------------------------------


def load_p_opt():
    d = np.load(REPO_ROOT / "configs" / "P_opt_64d.npz", allow_pickle=True)
    kind = str(d["type"])
    manifest = json.loads((REPO_ROOT / "configs" / "P_opt_64d_manifest.json").read_text())
    assert manifest["winner"] == "supervised_autoencoder_64d", (
        "P_opt is expected to be the TRJ002a-selected winner "
        f"(supervised_autoencoder_64d); found {manifest['winner']!r} instead."
    )
    assert kind == "autoencoder"
    W0, b0, W2, b2 = d["W0"], d["b0"], d["W2"], d["b2"]

    def project(z_raw: np.ndarray) -> np.ndarray:
        h = np.maximum(z_raw @ W0.T + b0, 0.0)
        return h @ W2.T + b2

    return project


def load_pca():
    """Unsupervised alternative to P_opt, for ANALYSIS-TRJ002B-STRATEGY-001
    Step 1: the PCA 64D candidate trained (unsupervised, no classification
    loss) alongside P_opt in TRJ002a's train_trj002a_projections.py, saved
    at configs/candidates/pca_64d.npz. Used to check whether BL-NORM's near-
    perfect AUROC on P_opt's space was specific to the Supervised
    Autoencoder's classification-shaped latent norm."""
    d = np.load(REPO_ROOT / "configs" / "candidates" / "pca_64d.npz", allow_pickle=True)
    assert str(d["type"]) == "pca"
    components, mean = d["components"], d["mean"]

    def project(z_raw: np.ndarray) -> np.ndarray:
        return (z_raw - mean) @ components.T

    return project


# ---------------------------------------------------------------------------
# Time axis, arc length, BL-NORM
# ---------------------------------------------------------------------------


def normalized_time(T: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, T)


def arc_length(z: np.ndarray) -> np.ndarray:
    """s(t): cumulative Euclidean arc length in the 64D projected space."""
    diffs = np.linalg.norm(np.diff(z, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(diffs)])


def bl_norm_features(z: np.ndarray) -> np.ndarray:
    """Spec S3 baseline BL-NORM: statistics (mean/max/std) of ||z(t)||."""
    norms = np.linalg.norm(z, axis=1)
    return np.array([norms.mean(), norms.max(), norms.std()])


def token_shuffle(z: np.ndarray, prompt_id: str, sample_index: int) -> np.ndarray:
    """Spec S2.1 Token-Shuffled Control: randomly permute generated-token
    hidden states in time before feature extraction. Deterministic per
    sample via MASK_SEED + sample identity (so the control is reproducible)."""
    seed = MASK_SEED ^ (hash(f"{prompt_id}_{sample_index}") & 0xFFFFFFFF)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(z.shape[0])
    return z[perm]


# ---------------------------------------------------------------------------
# M4: NVS-Kernel Gaussian Process, fit per-trajectory on arc-length s(t)
# ---------------------------------------------------------------------------


def gp_m4_features(z: np.ndarray) -> np.ndarray:
    """Fits GP( Matern5/2 + RBF ) to this single trajectory's s(t) series
    (spec S2.1 formula for M4), and returns
    [mll_per_token, sigma2_mean, sigma2_max, log_length_scale_matern,
     log_length_scale_rbf].

    mll is divided by T (length-normalized, matching TRJ002a S4.2's
    length-normalization convention) so trajectories of different length are
    comparable on the same footing rather than confounded by T.
    """
    T = z.shape[0]
    t = normalized_time(T).reshape(-1, 1)
    s = arc_length(z)
    s_std = s.std() if s.std() > 1e-8 else 1.0
    s_norm = (s - s.mean()) / s_std  # GP works on a unit-scale target

    kernel = Matern(length_scale=0.2, nu=2.5, length_scale_bounds=(1e-2, 10.0)) + RBF(
        length_scale=0.2, length_scale_bounds=(1e-2, 10.0)
    ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1.0))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=0)
    gp.fit(t, s_norm)

    mll_per_token = gp.log_marginal_likelihood_value_ / T
    _, std_pred = gp.predict(t, return_std=True)
    sigma2 = std_pred**2

    k = gp.kernel_
    ls_matern = k.k1.k1.length_scale
    ls_rbf = k.k1.k2.length_scale

    return np.array(
        [
            mll_per_token,
            float(np.mean(sigma2)),
            float(np.max(sigma2)),
            float(np.log(ls_matern)),
            float(np.log(ls_rbf)),
        ]
    )


def gp_m4_nf_features(arclen_series: np.ndarray) -> np.ndarray:
    """M4-NF (norm-free reanalysis): identical GP(Matern5/2 + RBF) fit to
    gp_m4_features, but on an already-computed 1D arc-length series that the
    caller supplies directly -- used with
    norm_free_trajectory.direction_arc_length(direction) so the GP never
    sees anything but the purely angular/directional path."""
    T = arclen_series.shape[0]
    t = np.linspace(0.0, 1.0, T).reshape(-1, 1)
    s = arclen_series
    s_std = s.std() if s.std() > 1e-8 else 1.0
    s_norm = (s - s.mean()) / s_std

    kernel = Matern(length_scale=0.2, nu=2.5, length_scale_bounds=(1e-2, 10.0)) + RBF(
        length_scale=0.2, length_scale_bounds=(1e-2, 10.0)
    ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1.0))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=0)
    gp.fit(t, s_norm)

    mll_per_token = gp.log_marginal_likelihood_value_ / T
    _, std_pred = gp.predict(t, return_std=True)
    sigma2 = std_pred**2
    k = gp.kernel_
    ls_matern = k.k1.k1.length_scale
    ls_rbf = k.k1.k2.length_scale

    return np.array(
        [
            mll_per_token,
            float(np.mean(sigma2)),
            float(np.max(sigma2)),
            float(np.log(ls_matern)),
            float(np.log(ls_rbf)),
        ]
    )


# ---------------------------------------------------------------------------
# M5: Neural ODE vector field
# ---------------------------------------------------------------------------


class ODEFunc(nn.Module):
    """f_theta(z, t): R^64 x [0,1] -> R^64 (spec S2.1 M5)."""

    def __init__(self, d=64, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, d),
        )

    def forward(self, t, z):
        # z: (..., 64); t: scalar tensor. Broadcast t to match z's leading dims.
        if z.dim() == 1:
            t_in = t.reshape(1).expand(1)
            x = torch.cat([z, t_in], dim=-1)
        else:
            t_in = t.reshape(1, 1).expand(z.shape[0], 1)
            x = torch.cat([z, t_in], dim=-1)
        return self.net(x)


ODE_STATE_PATH = REPO_ROOT / "configs" / "M5_ode_state.pt"
ODE_NF_STATE_PATH = REPO_ROOT / "configs" / "M5_NF_ode_state.pt"  # norm-free reanalysis
SOLVER_KW = dict(method="dopri5", rtol=1e-3, atol=1e-4)  # Gate 4: frozen solver tolerances


def load_m5(state_path: Path = ODE_STATE_PATH):
    from torchdiffeq import odeint  # local import: only needed where M5 runs

    func = ODEFunc().to(DEVICE)
    func.load_state_dict(torch.load(state_path, map_location=DEVICE))
    func.eval()
    return func, odeint


def m5_integrate(func, odeint, z: np.ndarray) -> torch.Tensor:
    """Integrates f_theta from z(0) over this trajectory's own normalized
    time axis. Returns predicted trajectory zhat(t), shape (T, 64)."""
    T = z.shape[0]
    t_eval = torch.tensor(normalized_time(T), dtype=torch.float32, device=DEVICE)
    z0 = torch.tensor(z[0], dtype=torch.float32, device=DEVICE)
    zhat = odeint(func, z0, t_eval, **SOLVER_KW)
    return zhat


def m5_features(func, odeint, z: np.ndarray) -> np.ndarray:
    """Spec S2.1 M5 features: reconstruction error, kinetic energy, local
    divergence rate (max eigenvalue of the vector field Jacobian), all
    integrated/aggregated along the trajectory and length-normalized."""
    from torch.func import jacrev, vmap

    T = z.shape[0]
    z_obs = torch.tensor(z, dtype=torch.float32, device=DEVICE)
    t_eval = torch.tensor(normalized_time(T), dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        zhat = m5_integrate(func, odeint, z)  # (T, 64)
        recon_err = ((zhat - z_obs) ** 2).sum(dim=-1)  # (T,)
        # trapezoidal integral over normalized time [0,1] -> length-normalized by construction
        L_recon = torch.trapz(recon_err, t_eval).item()

        f_vals = torch.stack(
            [func(t_eval[i], zhat[i]) for i in range(T)], dim=0
        )  # (T, 64)
        kinetic = (f_vals**2).sum(dim=-1)
        E_kinetic = torch.trapz(kinetic, t_eval).item()

        # Local divergence rate: max eigenvalue (real part) of df/dz, evaluated
        # at a subsampled set of points along zhat(t) (every point is
        # infeasible at scale; a fixed, documented, evenly-spaced subsample of
        # <=20 points per trajectory is used instead, following the spec's
        # "along the trajectory" wording without requiring a T-point Jacobian
        # sweep on every one of ~1,500 test-side trajectories).
        idx = np.linspace(0, T - 1, min(20, T)).astype(int)
        sub_z = zhat[idx]  # (K, 64)
        sub_t = t_eval[idx]  # (K,)

        def f_of_z(z_, t_):
            return func(t_, z_)

        jac_fn = vmap(jacrev(f_of_z), in_dims=(0, 0))
        jacs = jac_fn(sub_z, sub_t)  # (K, 64, 64)
        eigvals = torch.linalg.eigvals(jacs).real  # (K, 64)
        max_eig_per_point = eigvals.max(dim=-1).values  # (K,)
        local_divergence_rate = max_eig_per_point.max().item()

    return np.array([L_recon, E_kinetic, local_divergence_rate])
