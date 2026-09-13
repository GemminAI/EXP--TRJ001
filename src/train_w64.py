"""PCA calibration for EXP-2026-NVS-001 (plan v1.3 §2.3): learns and freezes
the projection (mu, W_128) from the independent calibration corpus.

This is a standalone, data-driven PCA fit on this experiment's own Layer 16
hidden states -- it does NOT depend on or call into `nvs_kernel`. An earlier
draft of this module was framed as "loading NVS-Kernel's standard projection
module," but that framing did not match any code that actually exists:
`nvs_kernel.projection.hashing.HashingProjector` (also defaulting to a 64D
output) is a signed feature-hashing projector over text n-grams, unrelated to
LLM hidden states; `nvs_kernel.projection.trajectory.TrajectoryReducer`
projects a 256D-per-step `hekb-vnext` trajectory format via a fixed *random*
QR-orthogonal matrix (no fitted data, no mu) and collapses all T steps to one
point, which cannot produce the per-token 64D trajectory z(1..T) this
experiment's kappa(t) requires. Neither module does PCA on 4096D LLM hidden
states. §2.3's PCA is this experiment's own artifact (confirmed by the user,
2026-09-12: 64D was chosen for Neural-ODE/curvature compute feasibility on
local hardware, not inherited from any NVS-Kernel convention).

Separation of concerns: extracting Layer 16 hidden states from the
calibration corpus requires a real model (vLLM, per plan §2.1) and cannot run
in this environment. This module therefore only does the PCA math, taking a
[N_tokens, 4096] hidden-state matrix as input -- however it was produced.
`main()` accepts a path to a pre-extracted `.npy` matrix; without one, it
runs a self-test against synthetic data so the fitting logic itself stays
testable without a GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_W_PATH = REPO_ROOT / "configs" / "W_pca128.npy"
DEFAULT_MU_PATH = REPO_ROOT / "configs" / "mu_pca.npy"
DEFAULT_DIAGNOSTICS_PATH = REPO_ROOT / "configs" / "pca_diagnostics.json"

SOURCE_DIMENSION = 3584  # Qwen2.5-7B-Instruct hidden width, §2.1 (v1.4 U8: model
# changed from Meta-Llama-3.1-8B-Instruct, 4096, after Rule 3 failed twice --
# see docs/EXP-2026-NVS-001_v1.4.md changelog)
N_COMPONENTS = 128  # §2.3 (G1): saved dimension
PCA_RANDOM_STATE = 20260912  # matches this project's other frozen seeds


@dataclass
class PCADiagnostics:
    pc1_64_explained_variance: float
    pc1_128_explained_variance: float
    pc1_3_explained_variance: float
    total_samples: int

    def to_dict(self) -> dict:
        return {
            "pc1_64_explained_variance": self.pc1_64_explained_variance,
            "pc1_128_explained_variance": self.pc1_128_explained_variance,
            "pc1_3_explained_variance": self.pc1_3_explained_variance,
            "total_samples": self.total_samples,
        }


def fit_pca_projection(
    hidden_states_matrix: np.ndarray, n_components: int = N_COMPONENTS
) -> tuple[np.ndarray, np.ndarray, PCADiagnostics]:
    """§2.3: fits PCA on a [N_tokens, 4096] calibration matrix.

    Returns (mu, W, diagnostics):
      mu: (4096,) float32 -- the centering vector (`pca.mean_`)
      W:  (4096, n_components) float32, such that a centered hidden state h
          projects as `z = (h - mu) @ W`
      diagnostics: cumulative explained-variance ratios needed for
          `freeze/FREEZE_MANIFEST.md`'s "PCA診断値" table and §6.3's ABL-2
          rationale (PC1-3 occupancy).
    """
    hidden_states_matrix = np.asarray(hidden_states_matrix)
    if hidden_states_matrix.ndim != 2:
        raise ValueError(f"expected a 2D [N, {SOURCE_DIMENSION}] matrix, got shape {hidden_states_matrix.shape}")
    if hidden_states_matrix.shape[1] != SOURCE_DIMENSION:
        raise ValueError(f"expected {SOURCE_DIMENSION}D hidden states, got {hidden_states_matrix.shape[1]}")
    if hidden_states_matrix.shape[0] <= n_components:
        raise ValueError(
            f"need more samples ({hidden_states_matrix.shape[0]}) than components "
            f"({n_components}) to fit PCA meaningfully"
        )

    pca = PCA(n_components=n_components, random_state=PCA_RANDOM_STATE)
    pca.fit(hidden_states_matrix)

    mu = pca.mean_.astype(np.float32)
    # sklearn's components_ has shape (n_components, n_features); transpose
    # to (n_features, n_components) so that `(h - mu) @ W` yields the
    # n_components-dim projection directly.
    W = pca.components_.T.astype(np.float32)

    ratio = pca.explained_variance_ratio_
    diagnostics = PCADiagnostics(
        pc1_64_explained_variance=float(np.sum(ratio[:64])),
        pc1_128_explained_variance=float(np.sum(ratio[:128])),
        pc1_3_explained_variance=float(np.sum(ratio[:3])),
        total_samples=hidden_states_matrix.shape[0],
    )
    return mu, W, diagnostics


def run_calibration(
    hidden_states_matrix: np.ndarray,
    *,
    w_path: Path = DEFAULT_W_PATH,
    mu_path: Path = DEFAULT_MU_PATH,
    diagnostics_path: Path = DEFAULT_DIAGNOSTICS_PATH,
) -> tuple[Path, Path, Path]:
    """Fits PCA and writes (W, mu, diagnostics) to disk for freezing
    (`freeze/FREEZE_MANIFEST.md` rows 2, 3, and the "PCA診断値" table)."""
    mu, W, diagnostics = fit_pca_projection(hidden_states_matrix, n_components=N_COMPONENTS)

    w_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(w_path, W)
    np.save(mu_path, mu)
    diagnostics_path.write_text(json.dumps(diagnostics.to_dict(), indent=2) + "\n", encoding="utf-8")

    print(f"[+] saved mu: {mu_path} (shape {mu.shape})")
    print(f"[+] saved W:  {w_path} (shape {W.shape})")
    print(f"[+] PCA diagnostics: {diagnostics.to_dict()}")
    return w_path, mu_path, diagnostics_path


def project(hidden_states: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Applies a frozen (mu, W): `z = (h - mu) @ W`. Works for a single
    [T, 4096] trajectory or a [N, 4096] batch; output is [..., n_components]."""
    return (np.asarray(hidden_states) - mu) @ W


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hidden-states",
        type=Path,
        default=None,
        help=(
            "Path to a pre-extracted [N_tokens, 4096] float32 .npy matrix of "
            "Layer 16 hidden states over the calibration corpus "
            "(configs/calibration_prompts_v1.json). Extraction itself needs a "
            "real model (vLLM, §2.1) and is not performed by this script."
        ),
    )
    args = parser.parse_args()

    if args.hidden_states is not None:
        hidden_states = np.load(args.hidden_states)
        run_calibration(hidden_states)
    else:
        # Self-test only: never write to the real frozen-artifact paths, so a
        # synthetic run can't be mistaken for (or accidentally overwrite) the
        # real calibration output.
        selftest_dir = REPO_ROOT / "data" / "_train_w64_selftest"
        print(
            "[!] --hidden-states not given: running a self-test against "
            "synthetic 4096D vectors (NOT real calibration data). Writing "
            f"to {selftest_dir}, NOT configs/W_pca128.npy / configs/mu_pca.npy."
        )
        rng = np.random.default_rng(PCA_RANDOM_STATE)
        hidden_states = rng.standard_normal((500, SOURCE_DIMENSION)).astype(np.float32)
        run_calibration(
            hidden_states,
            w_path=selftest_dir / "W_pca128.npy",
            mu_path=selftest_dir / "mu_pca.npy",
            diagnostics_path=selftest_dir / "pca_diagnostics.json",
        )


if __name__ == "__main__":
    main()
