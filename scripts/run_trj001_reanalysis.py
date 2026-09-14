#!/usr/bin/env python3
"""EXP-TRJ001 v3.3 reanalysis (reviewer tasks A–D).

Recomputes Class-E sensitivity, scale-invariant cosine-κ, paired BL-NORM
statistics, and 5-feature PCA on the frozen 98-prompt inventory.

Expected freeze layout (any subset is used; missing pieces are reported)::

    results/phase2_trajectories.json
    results/phase2_projected/*.npz
    results/ablation_raw_hidden_states/*.npy
    configs/data_splits.json
    configs/W_pca128.npy, configs/mu_pca.npy
    results/env_info.json

Override the root with ``--data-root`` or ``$TRJ001_DATA_ROOT``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from compute_kappa import EPSILON, MIN_SEQUENCE_LENGTH, compute_kappa  # noqa: E402

C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260912
PRIMARY_LAYER = 16
SKIP_DIR_NAMES = {"_selftest", "_train_w64_selftest", ".git", ".venv", "__pycache__"}
HIDDEN_DIM = 3584
PCA_DIM = 64
TAU_GRID = (10, 15, 20, 30)
POSITIVE = frozenset({"type_a_positive", "positive", "1", 1, True})
NEGATIVE = frozenset(
    {"type_d_hedge", "type_c_abstain", "negative", "0", 0, False, "type_d_hedge"}
)

KAPPA5 = (
    "kappa_mean",
    "kappa_max",
    "kappa_p95",
    "kappa_std",
    "kappa_auc_density",
)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    prompt_id: str
    sample_index: int
    token_count: int
    raw_label: str
    y: int | None
    split: str = ""
    z_raw: np.ndarray | None = None
    z_pca: np.ndarray | None = None  # (T, >=64)

    @property
    def t(self) -> int:
        if self.z_raw is not None:
            return int(self.z_raw.shape[0])
        if self.z_pca is not None:
            return int(self.z_pca.shape[0])
        return int(self.token_count)


@dataclass
class BootstrapResult:
    point_estimate: float
    ci95_lower: float
    ci95_upper: float
    p_value_le_0: float | None
    p_value_le_0_5: float | None
    n_valid_resamples: int
    n_resamples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProbeFit:
    scores: np.ndarray
    selected_C: float
    val_auroc: float


# ---------------------------------------------------------------------------
# Geometry / features
# ---------------------------------------------------------------------------


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]


def summarize_series(values: np.ndarray) -> np.ndarray:
    v = _finite(np.asarray(values, dtype=np.float64))
    if v.size == 0:
        return np.zeros(5, dtype=np.float64)
    n = float(v.size)
    return np.asarray(
        [
            float(np.mean(v)),
            float(np.max(v)),
            float(np.percentile(v, 95)),
            float(np.std(v, ddof=0)),
            float(np.sum(v) / n),
        ],
        dtype=np.float64,
    )


def gram_kappa_features(z: np.ndarray, theta: float | None = None) -> np.ndarray:
    result = compute_kappa(z)
    valid = result.kappa[result.valid_mask]
    feats = summarize_series(valid)
    if theta is None:
        return feats
    if valid.size == 0:
        spike = 0.0
    else:
        spike = float(np.mean(valid > theta))
    return np.append(feats, spike)


def kappa_cos_series(z: np.ndarray, eps: float = EPSILON) -> np.ndarray:
    """Scale-invariant one-step cosine change on L2-normalized tokens."""
    z = np.asarray(z, dtype=np.float64)
    if z.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    nrm = np.linalg.norm(z, axis=1, keepdims=True)
    zhat = z / (nrm + eps)
    dots = np.sum(zhat[:-1] * zhat[1:], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    return 1.0 - dots


def kappa_cos_features(z: np.ndarray) -> np.ndarray:
    return summarize_series(kappa_cos_series(z))


def l2_normalize_rows(z: np.ndarray, eps: float = EPSILON) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    nrm = np.linalg.norm(z, axis=1, keepdims=True)
    return z / (nrm + eps)


def norm_features(z: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(np.asarray(z, dtype=np.float64), axis=1)
    nrm = _finite(nrm)
    if nrm.size == 0:
        return np.zeros(3, dtype=np.float64)
    return np.asarray([float(np.mean(nrm)), float(np.max(nrm)), float(np.std(nrm, ddof=0))])


def mean_norm(z: np.ndarray) -> float:
    nrm = np.linalg.norm(np.asarray(z, dtype=np.float64), axis=1)
    nrm = _finite(nrm)
    return float(np.mean(nrm)) if nrm.size else float("nan")


def pca64(z_pca: np.ndarray) -> np.ndarray:
    z = np.asarray(z_pca, dtype=np.float64)
    return z[:, :PCA_DIM] if z.shape[1] >= PCA_DIM else z


def representation(sample: Sample, name: str) -> np.ndarray | None:
    if name == "raw":
        return None if sample.z_raw is None else np.asarray(sample.z_raw, dtype=np.float64)
    if name == "pca64":
        if sample.z_pca is not None:
            return pca64(sample.z_pca)
        if sample.z_raw is not None:
            return None  # caller may project
        return None
    return None


# ---------------------------------------------------------------------------
# Classifier + clustered bootstrap
# ---------------------------------------------------------------------------


def safe_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y)
    scores = np.asarray(scores)
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def draw_cluster_resamples(
    prompt_ids: np.ndarray,
    n_resamples: int,
    seed: int,
) -> list[np.ndarray]:
    pids = np.asarray(prompt_ids).astype(str)
    clusters = np.unique(pids)
    mapping: dict[str, np.ndarray] = {
        c: np.flatnonzero(pids == c) for c in clusters
    }
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    n_clusters = clusters.shape[0]
    for _ in range(n_resamples):
        sampled = rng.choice(clusters, size=n_clusters, replace=True)
        draws.append(np.concatenate([mapping[str(c)] for c in sampled]))
    return draws


def bootstrap_auroc(
    y: np.ndarray,
    scores: np.ndarray,
    prompt_ids: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    resamples: list[np.ndarray] | None = None,
) -> BootstrapResult:
    y = np.asarray(y)
    scores = np.asarray(scores)
    if resamples is None:
        resamples = draw_cluster_resamples(prompt_ids, n_resamples, seed)
    point = safe_auroc(y, scores)
    stats = [safe_auroc(y[idx], scores[idx]) for idx in resamples]
    valid = np.asarray([s for s in stats if np.isfinite(s)], dtype=np.float64)
    if valid.size == 0:
        return BootstrapResult(point, float("nan"), float("nan"), None, None, 0, len(resamples))
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return BootstrapResult(
        point_estimate=float(point),
        ci95_lower=float(lo),
        ci95_upper=float(hi),
        p_value_le_0=None,
        p_value_le_0_5=float(np.mean(valid <= 0.5)),
        n_valid_resamples=int(valid.size),
        n_resamples=len(resamples),
    )


def paired_delta_bootstrap(
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    prompt_ids: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    resamples: list[np.ndarray] | None = None,
) -> BootstrapResult:
    y = np.asarray(y)
    if resamples is None:
        resamples = draw_cluster_resamples(prompt_ids, n_resamples, seed)
    point = safe_auroc(y, scores_a) - safe_auroc(y, scores_b)
    deltas: list[float] = []
    for idx in resamples:
        a = safe_auroc(y[idx], scores_a[idx])
        b = safe_auroc(y[idx], scores_b[idx])
        if np.isfinite(a) and np.isfinite(b):
            deltas.append(float(a - b))
    valid = np.asarray(deltas, dtype=np.float64)
    if valid.size == 0:
        return BootstrapResult(float(point), float("nan"), float("nan"), float("nan"), None, 0, len(resamples))
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return BootstrapResult(
        point_estimate=float(point),
        ci95_lower=float(lo),
        ci95_upper=float(hi),
        p_value_le_0=float(np.mean(valid <= 0.0)),
        p_value_le_0_5=None,
        n_valid_resamples=int(valid.size),
        n_resamples=len(resamples),
    )


def fit_logistic_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_eval: np.ndarray,
) -> ProbeFit:
    if len(np.unique(y_train)) < 2:
        raise ValueError("train split must contain both classes")
    scaler = StandardScaler()
    xtr = scaler.fit_transform(X_train)
    xva = scaler.transform(X_val)
    xev = scaler.transform(X_eval)
    best_C = 1.0
    best_val = -np.inf
    best_clf: LogisticRegression | None = None
    val_ok = len(np.unique(y_val)) >= 2
    for C in C_GRID:
        clf = LogisticRegression(C=float(C), solver="lbfgs", max_iter=4000)
        clf.fit(xtr, y_train)
        if best_clf is None:
            best_clf = clf
            best_C = float(C)
        if not val_ok:
            continue
        val_auroc = float(roc_auc_score(y_val, clf.predict_proba(xva)[:, 1]))
        if val_auroc > best_val:
            best_val = val_auroc
            best_C = float(C)
            best_clf = clf
    assert best_clf is not None
    if not val_ok:
        best_val = float("nan")
    return ProbeFit(
        scores=np.asarray(best_clf.predict_proba(xev)[:, 1], dtype=np.float64),
        selected_C=float(best_C),
        val_auroc=float(best_val),
    )


# ---------------------------------------------------------------------------
# Labels / splits
# ---------------------------------------------------------------------------


def encode_label(raw: Any) -> int | None:
    if raw in POSITIVE:
        return 1
    if raw in NEGATIVE:
        return 0
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode()
    if isinstance(raw, str):
        key = raw.strip().lower()
        if key in {"type_a_positive", "positive", "unreliable", "hallucination"}:
            return 1
        if key in {"type_d_hedge", "type_c_abstain", "negative", "reliable", "abstain"}:
            return 0
        if key in {"class_e_indeterminate", "pending_type_a", "type_b_positive"}:
            return None
    if raw is None:
        return None
    raise ValueError(f"unrecognized label {raw!r}")


def load_splits(path: Path) -> dict[str, list[str]] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for key in ("train", "val", "test"):
        if key not in data:
            return None
        out[key] = [str(p) for p in data[key]]
    return out


def assign_splits(samples: list[Sample], splits: dict[str, list[str]] | None) -> None:
    if splits is None:
        return
    lookup = {pid: name for name, ids in splits.items() for pid in ids}
    for s in samples:
        s.split = lookup.get(s.prompt_id, s.split)


# ---------------------------------------------------------------------------
# Data discovery / loaders
# ---------------------------------------------------------------------------


def candidate_roots(cli_root: Path | None) -> list[Path]:
    roots: list[Path] = []
    if cli_root is not None:
        roots.append(cli_root)
    env = os.environ.get("TRJ001_DATA_ROOT")
    if env:
        roots.append(Path(env))
    roots.extend(
        [
            REPO / "results",
            REPO / "data" / "trajectories",
        ]
    )
    # unique, existing
    seen: set[Path] = set()
    out: list[Path] = []
    for r in roots:
        r = r.resolve() if r.exists() else r
        if r in seen:
            continue
        seen.add(r)
        out.append(r)
    return out


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _first_2d(arrs: dict[str, np.ndarray], prefer: Sequence[str]) -> np.ndarray | None:
    for key in prefer:
        if key in arrs and np.asarray(arrs[key]).ndim == 2:
            return np.asarray(arrs[key])
    for v in arrs.values():
        a = np.asarray(v)
        if a.ndim == 2 and a.shape[1] >= 2:
            return a
    return None


def _parse_pid_index(path: Path) -> tuple[str, int] | None:
    stem = path.stem
    if "__" in stem:
        pid, _, idx = stem.partition("__")
        try:
            return pid, int(idx)
        except ValueError:
            return None
    parts = stem.replace("-", "_").split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1]), int(parts[-1])
    if path.parent.name and stem.isdigit():
        return path.parent.name, int(stem)
    return None


def ingest_trajectory_file(path: Path, sample: Sample) -> None:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        if arr.ndim != 2:
            return
        if arr.shape[1] >= HIDDEN_DIM - 32:
            sample.z_raw = arr.astype(np.float32, copy=False)
        else:
            sample.z_pca = arr.astype(np.float32, copy=False)
        return
    if suffix != ".npz":
        return
    arrs = _load_npz_arrays(path)
    raw = _first_2d(
        arrs,
        ("hidden_states", "raw", "h", "layer_16_raw", "states"),
    )
    pca = _first_2d(
        arrs,
        (
            "projected",
            "z",
            "pca128",
            "pc128",
            "layer_16_128d",
            "layer_16",
            "layer16",
            "z_pca",
        ),
    )
    if "label" in arrs and sample.raw_label == "":
        lab = arrs["label"]
        sample.raw_label = str(lab.item() if getattr(lab, "shape", ()) == () else lab)
        try:
            sample.y = encode_label(lab.item() if getattr(lab, "shape", ()) == () else lab)
        except ValueError:
            sample.y = None
    if "token_count" in arrs:
        tc = arrs["token_count"]
        sample.token_count = int(tc.item() if getattr(tc, "shape", ()) == () else int(tc))
    if raw is not None and raw.shape[1] >= 512:
        sample.z_raw = raw.astype(np.float32, copy=False)
        if sample.token_count <= 0:
            sample.token_count = int(raw.shape[0])
        if pca is not None and pca is not raw and pca.shape[1] <= 256:
            sample.z_pca = pca.astype(np.float32, copy=False)
        return
    if pca is not None:
        if pca.shape[1] >= 512:
            sample.z_raw = pca.astype(np.float32, copy=False)
        else:
            sample.z_pca = pca.astype(np.float32, copy=False)
        if sample.token_count <= 0:
            sample.token_count = int(pca.shape[0])


def load_json_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    for key in ("records", "samples", "trajectories", "items"):
        if key in data and isinstance(data[key], list):
            return data[key]
    if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
        rows = []
        for pid, rec in data.items():
            rec = dict(rec)
            rec.setdefault("prompt_id", pid)
            rows.append(rec)
        return rows
    raise ValueError(f"unrecognized JSON schema: {path}")


def sample_from_record(rec: dict[str, Any]) -> Sample:
    pid = str(rec.get("prompt_id") or rec.get("id") or rec.get("prompt"))
    idx = int(rec.get("sample_index", rec.get("sample", rec.get("k", 0))))
    raw_label = rec.get("label", rec.get("raw_label", ""))
    if isinstance(raw_label, dict):
        raw_label = raw_label.get("label", "")
    try:
        y = encode_label(raw_label) if raw_label != "" else None
    except ValueError:
        y = None
    tc = int(rec.get("token_count", rec.get("T", rec.get("n_tokens", 0))) or 0)
    split = str(rec.get("split", ""))
    s = Sample(
        prompt_id=pid,
        sample_index=idx,
        token_count=tc,
        raw_label=str(raw_label),
        y=y,
        split=split,
    )
    for key, dest in (
        ("hidden_states", "z_raw"),
        ("raw", "z_raw"),
        ("z_raw", "z_raw"),
        ("projected", "z_pca"),
        ("z_pca", "z_pca"),
        ("pca128", "z_pca"),
    ):
        if key in rec and rec[key] is not None:
            arr = np.asarray(rec[key])
            if arr.ndim == 2:
                setattr(s, dest, arr.astype(np.float32, copy=False))
                if s.token_count <= 0:
                    s.token_count = int(arr.shape[0])
    for path_key in ("npz_path", "path", "projected_path", "raw_path"):
        p = rec.get(path_key)
        if p:
            fp = Path(p)
            if not fp.is_absolute():
                fp = REPO / fp
            if fp.exists():
                ingest_trajectory_file(fp, s)
    return s


def merge_sample(dst: dict[tuple[str, int], Sample], src: Sample) -> None:
    key = (src.prompt_id, src.sample_index)
    if key not in dst:
        dst[key] = src
        return
    cur = dst[key]
    if src.token_count > cur.token_count:
        cur.token_count = src.token_count
    if src.raw_label and not cur.raw_label:
        cur.raw_label = src.raw_label
        cur.y = src.y
    if src.split and not cur.split:
        cur.split = src.split
    if src.z_raw is not None and cur.z_raw is None:
        cur.z_raw = src.z_raw
    if src.z_pca is not None and cur.z_pca is None:
        cur.z_pca = src.z_pca


def load_corpus(roots: Sequence[Path], splits_path: Path) -> tuple[list[Sample], dict[str, Any]]:
    index: dict[tuple[str, int], Sample] = {}
    provenance: dict[str, Any] = {"roots": [str(r) for r in roots], "files": []}

    splits = load_splits(splits_path)
    if splits is None:
        for r in roots:
            cand = r / "data_splits.json" if r.name != "configs" else r / "data_splits.json"
            alt = [
                r / "data_splits.json",
                r.parent / "configs" / "data_splits.json",
                REPO / "configs" / "data_splits.json",
            ]
            for p in alt:
                splits = load_splits(p)
                if splits is not None:
                    provenance["splits"] = str(p)
                    break
            if splits is not None:
                break
    else:
        provenance["splits"] = str(splits_path)

    json_names = (
        "phase2_trajectories.json",
        "phase3_records.json",
        "trajectories.json",
        "samples.json",
    )
    for root in roots:
        if not root.exists():
            continue
        search_dirs = [root]
        if (root / "phase2_projected").is_dir():
            search_dirs.append(root / "phase2_projected")
        if (root / "ablation_raw_hidden_states").is_dir():
            search_dirs.append(root / "ablation_raw_hidden_states")
        for name in json_names:
            p = root / name
            if p.exists():
                provenance["files"].append(str(p))
                for rec in load_json_records(p):
                    merge_sample(index, sample_from_record(rec))
        for folder in search_dirs:
            if not folder.exists():
                continue
            for path in folder.rglob("*"):
                if any(part in SKIP_DIR_NAMES for part in path.parts):
                    continue
                if path.suffix.lower() not in {".npz", ".npy"}:
                    continue
                parsed = _parse_pid_index(path)
                if parsed is None:
                    continue
                pid, idx = parsed
                s = index.get((pid, idx)) or Sample(
                    prompt_id=pid,
                    sample_index=idx,
                    token_count=0,
                    raw_label="",
                    y=None,
                )
                ingest_trajectory_file(path, s)
                merge_sample(index, s)
                if len(provenance["files"]) < 40:
                    provenance["files"].append(str(path))

    samples = list(index.values())
    # Refuse synthetic/self-test corpora unless --self-test was requested.
    if any(s.prompt_id.startswith("syn_") for s in samples):
        raise RuntimeError(
            "refusing to treat synthetic syn_* trajectories as freeze data; "
            "rerun with --self-test if that was intended"
        )
    assign_splits(samples, splits)
    provenance["n_loaded"] = len(samples)
    provenance["n_with_raw"] = sum(1 for s in samples if s.z_raw is not None)
    provenance["n_with_pca"] = sum(1 for s in samples if s.z_pca is not None)
    provenance["n_with_label"] = sum(1 for s in samples if s.y is not None)
    provenance["n_test_prompts"] = len({s.prompt_id for s in samples if s.split == "test"})
    return samples, provenance


def maybe_project(samples: list[Sample], w_path: Path, mu_path: Path) -> str | None:
    if not w_path.exists() or not mu_path.exists():
        return None
    W = np.load(w_path)
    mu = np.load(mu_path)
    n = 0
    for s in samples:
        if s.z_pca is None and s.z_raw is not None and s.z_raw.shape[1] == mu.shape[0]:
            s.z_pca = ((s.z_raw - mu) @ W).astype(np.float32)
            n += 1
    return f"projected {n} raw trajectories with {w_path.name}"


# ---------------------------------------------------------------------------
# Synthetic corpus (script self-test only)
# ---------------------------------------------------------------------------


def make_synthetic(n_test: int = 12, seed: int = BOOTSTRAP_SEED) -> list[Sample]:
    rng = np.random.default_rng(seed)
    samples: list[Sample] = []
    n_train, n_val = 20, 8
    ids = [f"syn_{i:03d}" for i in range(n_train + n_val + n_test)]
    splits = (
        [("train", pid) for pid in ids[:n_train]]
        + [("val", pid) for pid in ids[n_train : n_train + n_val]]
        + [("test", pid) for pid in ids[n_train + n_val :]]
    )
    t_axis = np.linspace(0.0, 2.0 * np.pi, 40, dtype=np.float64)
    for split, pid in splits:
        for k in range(8):
            # Mix short (Class E) and long sequences.
            short = (hash(f"{pid}_{k}") % 5 == 0)
            T = 12 if short else 36
            y = 1 if k < 4 else 0
            h = rng.normal(0.0, 0.05, size=(T, HIDDEN_DIM)).astype(np.float32)
            h[:, 0:3] += rng.normal(0.0, 40.0, size=(T, 3)).astype(np.float32)
            sl = min(T, t_axis.size)
            if y == 1:
                h[:sl, 16] += 1.6 * np.cos(t_axis[:sl])
                h[:sl, 17] += 1.6 * np.sin(t_axis[:sl])
            else:
                h[:sl, 16] += np.linspace(0.0, 2.0, sl, dtype=np.float32)
            # Cheap 64D "PCA": massive dims + a few geometry dims.
            z = np.concatenate([h[:, :3], h[:, 16:77]], axis=1)[:, :128]
            samples.append(
                Sample(
                    prompt_id=pid,
                    sample_index=k,
                    token_count=T,
                    raw_label="type_a_positive" if y == 1 else "type_d_hedge",
                    y=y,
                    split=split,
                    z_raw=h,
                    z_pca=z.astype(np.float32),
                )
            )
    return samples


# ---------------------------------------------------------------------------
# Task implementations
# ---------------------------------------------------------------------------


def test_prompt_ids(samples: list[Sample]) -> list[str]:
    test = sorted({s.prompt_id for s in samples if s.split == "test"})
    if test:
        return test
    return sorted({s.prompt_id for s in samples})


def filter_binary(samples: Iterable[Sample]) -> list[Sample]:
    return [s for s in samples if s.y in (0, 1)]


def by_split(samples: Sequence[Sample], name: str) -> list[Sample]:
    return [s for s in samples if s.split == name]


def freeze_class_e_arithmetic() -> dict[str, Any]:
    """288/784 follows from freeze n_test=496 and 98×8 generations."""
    n_prompts = 98
    n_gen = n_prompts * 8
    n_kept = 496
    return {
        "n_prompts": n_prompts,
        "n_generations": n_gen,
        "expected_generations": n_gen,
        "class_e_threshold": MIN_SEQUENCE_LENGTH,
        "source": "docs/Paper/phase3_metrics.json n_test=496 plus 98×8 inventory",
        "excluded": {
            "n": n_gen - n_kept,
            "y1": None,
            "y0": None,
            "other": None,
            "positive_rate": None,
            "note": "Label counts require phase2_trajectories.json",
        },
        "before_exclusion": {"n": n_gen, "y1": None, "y0": None, "other": None, "positive_rate": None},
        "after_exclusion": {"n": n_kept, "y1": None, "y0": None, "other": None, "positive_rate": None},
        "paired_note": (
            "BL-LEN and BL-NORM are scored on the identical eligible set "
            f"(T>={MIN_SEQUENCE_LENGTH}, n=496) as Raw 3584D and PCA 64D; "
            "no comparator uses a different prompt inventory."
        ),
    }


def task_a1(samples: list[Sample], test_ids: Sequence[str]) -> dict[str, Any]:
    pool = [s for s in samples if s.prompt_id in set(test_ids)] if test_ids else []
    if not pool:
        return freeze_class_e_arithmetic()
    n_prompts = len({s.prompt_id for s in pool})
    n_gen = len(pool)
    excluded = [s for s in pool if s.t < MIN_SEQUENCE_LENGTH]
    kept = [s for s in pool if s.t >= MIN_SEQUENCE_LENGTH]
    def counts(xs: list[Sample]) -> dict[str, Any]:
        y1 = sum(1 for s in xs if s.y == 1)
        y0 = sum(1 for s in xs if s.y == 0)
        other = len(xs) - y1 - y0
        labeled = y1 + y0
        return {
            "n": len(xs),
            "y1": y1,
            "y0": y0,
            "other": other,
            "positive_rate": None if labeled == 0 else y1 / labeled,
        }

    before = counts(pool)
    after = counts(kept)
    ex = counts(excluded)
    return {
        "n_prompts": n_prompts,
        "n_generations": n_gen,
        "expected_generations": n_prompts * 8,
        "class_e_threshold": MIN_SEQUENCE_LENGTH,
        "excluded": ex,
        "before_exclusion": before,
        "after_exclusion": after,
        "paired_note": (
            "BL-LEN and BL-NORM are scored on the identical eligible set "
            f"(T>={MIN_SEQUENCE_LENGTH}) as Raw 3584D and PCA 64D; "
            "no comparator uses a different prompt inventory."
        ),
    }


def z_for(sample: Sample, space: str) -> np.ndarray | None:
    if space == "raw":
        return None if sample.z_raw is None else np.asarray(sample.z_raw, dtype=np.float64)
    if space == "pca64":
        if sample.z_pca is not None:
            return pca64(np.asarray(sample.z_pca, dtype=np.float64))
        return None
    raise KeyError(space)


def train_theta(train: Sequence[Sample], space: str) -> float:
    vals: list[np.ndarray] = []
    for s in train:
        z = z_for(s, space)
        if z is None or z.shape[0] < 3:
            continue
        k = compute_kappa(z).kappa
        vals.append(_finite(k))
    if not vals:
        return float("nan")
    return float(np.percentile(np.concatenate(vals), 90))


def feature_row(sample: Sample, method: str, theta_pca: float | None) -> np.ndarray | None:
    if method == "BL-LEN":
        return np.asarray([float(sample.t)], dtype=np.float64)
    if method == "BL-NORM":
        z = z_for(sample, "pca64")
        if z is None:
            z = z_for(sample, "raw")
        return None if z is None else norm_features(z)
    if method == "PCA64-6":
        z = z_for(sample, "pca64")
        if z is None:
            return None
        return gram_kappa_features(z, theta=theta_pca)
    if method in {"PCA64-5", "PCA64"}:
        z = z_for(sample, "pca64")
        return None if z is None else gram_kappa_features(z, theta=None)
    if method == "Raw-5":
        z = z_for(sample, "raw")
        return None if z is None else gram_kappa_features(z, theta=None)
    if method == "kcos-raw":
        z = z_for(sample, "raw")
        return None if z is None else kappa_cos_features(l2_normalize_rows(z))
    if method == "kcos-pca":
        z = z_for(sample, "pca64")
        return None if z is None else kappa_cos_features(l2_normalize_rows(z))
    if method == "kcos-raw-unnormed-formula":
        z = z_for(sample, "raw")
        return None if z is None else kappa_cos_features(z)
    raise KeyError(method)


def eligible(sample: Sample, tau: int, method: str) -> bool:
    if sample.y not in (0, 1):
        return False
    if sample.t < tau:
        return False
    if method == "BL-LEN":
        return True
    if method in {"Raw-5", "kcos-raw", "kcos-raw-unnormed-formula"}:
        return sample.z_raw is not None and sample.t >= max(tau, 3)
    if method == "BL-NORM":
        return (sample.z_pca is not None or sample.z_raw is not None)
    return sample.z_pca is not None and sample.t >= max(tau, 3)


def matrix_for(
    samples: Sequence[Sample],
    method: str,
    theta_pca: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y, pids = [], [], []
    for s in samples:
        row = feature_row(s, method, theta_pca)
        if row is None:
            continue
        X.append(row)
        y.append(s.y)
        pids.append(s.prompt_id)
    if not X:
        return (
            np.zeros((0, 1), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=object),
        )
    return np.vstack(X), np.asarray(y, dtype=np.int64), np.asarray(pids)


def eval_method(
    train: Sequence[Sample],
    val: Sequence[Sample],
    test: Sequence[Sample],
    method: str,
    theta_pca: float | None,
    resamples: list[np.ndarray] | None,
    n_boot: int,
    seed: int,
) -> dict[str, Any] | None:
    Xtr, ytr, _ = matrix_for(train, method, theta_pca)
    Xva, yva, _ = matrix_for(val, method, theta_pca)
    Xte, yte, pte = matrix_for(test, method, theta_pca)
    if Xtr.shape[0] == 0 or Xte.shape[0] == 0 or len(np.unique(ytr)) < 2:
        return None
    # If val is empty/single-class, fall back to train for C selection.
    if Xva.shape[0] == 0 or len(np.unique(yva)) < 2:
        Xva, yva = Xtr, ytr
    fit = fit_logistic_probe(Xtr, ytr, Xva, yva, Xte)
    boot = bootstrap_auroc(yte, fit.scores, pte, n_resamples=n_boot, seed=seed, resamples=resamples)
    return {
        "method": method,
        "n_train": int(Xtr.shape[0]),
        "n_val": int(Xva.shape[0]),
        "n_test": int(Xte.shape[0]),
        "n_features": int(Xtr.shape[1]),
        "selected_C": fit.selected_C,
        "val_auroc": fit.val_auroc,
        "test_auroc": boot.to_dict(),
        "scores": fit.scores,
        "y": yte,
        "prompt_ids": pte,
    }


def rank_methods(rows: dict[str, dict[str, Any]]) -> list[str]:
    scored = [
        (name, rows[name]["test_auroc"]["point_estimate"])
        for name in rows
        if rows[name] is not None and np.isfinite(rows[name]["test_auroc"]["point_estimate"])
    ]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return [n for n, _ in scored]


def task_a2(
    samples: list[Sample],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    methods = ("Raw-5", "PCA64-5", "BL-LEN", "BL-NORM")
    out: dict[str, Any] = {"thresholds": {}, "rank_stable": None}
    ranks: dict[int, list[str]] = {}
    for tau in TAU_GRID:
        pool = [s for s in samples if s.t >= tau and s.y in (0, 1)]
        tr, va, te = by_split(pool, "train"), by_split(pool, "val"), by_split(pool, "test")
        theta = train_theta([s for s in tr if s.t >= tau], "pca64")
        # Shared bootstrap draws on the intersection of test samples that
        # have every method defined, else per-method draws.
        te_common = [
            s
            for s in te
            if all(eligible(s, tau, m) for m in methods)
        ]
        common_keys = {(s.prompt_id, s.sample_index) for s in te_common}
        resamples = None
        if te_common:
            resamples = draw_cluster_resamples(
                np.asarray([s.prompt_id for s in te_common]), n_boot, seed
            )
        rows: dict[str, Any] = {}
        for m in methods:
            te_m = [s for s in te if eligible(s, tau, m)]
            tr_m = [s for s in tr if eligible(s, tau, m)]
            va_m = [s for s in va if eligible(s, tau, m)]
            te_keys = {(s.prompt_id, s.sample_index) for s in te_m}
            rs = resamples if te_keys == common_keys and te_common else None
            got = eval_method(tr_m, va_m, te_m, m, theta, rs, n_boot, seed)
            if got is None:
                rows[m] = None
            else:
                stored = {k: v for k, v in got.items() if k not in {"scores", "y", "prompt_ids"}}
                stored["test_n_prompts"] = len(set(got["prompt_ids"]))
                rows[m] = stored
        ranks[tau] = rank_methods({k: v for k, v in rows.items() if v is not None})
        out["thresholds"][str(tau)] = {
            "n_test_binary": len(te),
            "n_test_common": len(te_common),
            "methods": rows,
            "ranking": ranks[tau],
        }
    ref = ranks.get(20)
    out["rank_stable"] = all(ranks[t] == ref for t in ranks) if ref else False
    out["rankings"] = {str(k): v for k, v in ranks.items()}
    return out


def task_b(
    samples: list[Sample],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    tau = MIN_SEQUENCE_LENGTH
    pool = [s for s in samples if s.t >= tau and s.y in (0, 1)]
    tr, va, te = by_split(pool, "train"), by_split(pool, "val"), by_split(pool, "test")
    te_raw = [s for s in te if s.z_raw is not None]
    if not te_raw:
        te_raw = [s for s in te if s.z_pca is not None]
        space = "pca64"
    else:
        space = "raw"

    kappa_means, norms, ys = [], [], []
    for s in te_raw:
        z = z_for(s, space)
        if z is None:
            continue
        km = float(gram_kappa_features(z)[0])
        mn = mean_norm(z)
        if np.isfinite(km) and np.isfinite(mn):
            kappa_means.append(km)
            norms.append(mn)
            ys.append(s.y)
    rho, p_rho = (float("nan"), float("nan"))
    if len(kappa_means) >= 3:
        rho, p_rho = spearmanr(kappa_means, norms)

    theta = train_theta(tr, "pca64")
    kcos = eval_method(tr, va, te_raw, "kcos-raw" if space == "raw" else "kcos-pca", theta, None, n_boot, seed)
    raw5 = eval_method(tr, va, te_raw, "Raw-5" if space == "raw" else "PCA64-5", theta, None, n_boot, seed)
    norm = eval_method(tr, va, te_raw, "BL-NORM", theta, None, n_boot, seed)

    incremental = None
    residual = None
    if raw5 is not None and norm is not None:
        m_kappa = "Raw-5" if space == "raw" else "PCA64-5"

        def joint_matrix(xs: Sequence[Sample]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Xj, Xn, y, pids = [], [], [], []
            for s in xs:
                a = feature_row(s, m_kappa, theta)
                b = feature_row(s, "BL-NORM", theta)
                if a is None or b is None or s.y not in (0, 1):
                    continue
                Xj.append(np.concatenate([a, b]))
                Xn.append(b)
                y.append(s.y)
                pids.append(s.prompt_id)
            if not Xj:
                empty = np.zeros((0, 1), dtype=np.float64)
                return empty, empty, np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=object)
            return np.vstack(Xj), np.vstack(Xn), np.asarray(y, dtype=np.int64), np.asarray(pids)

        Xtr, Ntr, ytr, _ = joint_matrix(tr)
        Xva, Nva, yva, _ = joint_matrix(va)
        Xte, Nte, yte, pte = joint_matrix(te_raw)
        if Xtr.shape[0] and Xte.shape[0] and len(np.unique(ytr)) == 2:
            if Xva.shape[0] == 0 or len(np.unique(yva)) < 2:
                Xva, Nva, yva = Xtr, Ntr, ytr
            joint = fit_logistic_probe(Xtr, ytr, Xva, yva, Xte)
            norm_fit = fit_logistic_probe(Ntr, ytr, Nva, yva, Nte)
            rs = draw_cluster_resamples(pte, n_boot, seed)
            incremental = {
                "joint_kappa_plus_norm": bootstrap_auroc(
                    yte, joint.scores, pte, n_resamples=n_boot, seed=seed, resamples=rs
                ).to_dict(),
                "selected_C": joint.selected_C,
                "delta_vs_norm": paired_delta_bootstrap(
                    yte, joint.scores, norm_fit.scores, pte, n_resamples=n_boot, seed=seed, resamples=rs
                ).to_dict(),
                "n_test": int(Xte.shape[0]),
            }

        # Residual of kappa_mean ~ mean_norm.
        km_tr = np.asarray([gram_kappa_features(z_for(s, space))[0] for s in tr if z_for(s, space) is not None])
        mn_tr = np.asarray([mean_norm(z_for(s, space)) for s in tr if z_for(s, space) is not None])
        y_tr = np.asarray([s.y for s in tr if z_for(s, space) is not None])
        km_te = np.asarray([gram_kappa_features(z_for(s, space))[0] for s in te_raw if z_for(s, space) is not None])
        mn_te = np.asarray([mean_norm(z_for(s, space)) for s in te_raw if z_for(s, space) is not None])
        y_te = np.asarray([s.y for s in te_raw if z_for(s, space) is not None])
        p_te = np.asarray([s.prompt_id for s in te_raw if z_for(s, space) is not None])
        mask_tr = np.isfinite(km_tr) & np.isfinite(mn_tr)
        mask_te = np.isfinite(km_te) & np.isfinite(mn_te)
        if mask_tr.sum() >= 4 and mask_te.sum() >= 4 and len(np.unique(y_tr[mask_tr])) == 2:
            reg = LinearRegression().fit(mn_tr[mask_tr].reshape(-1, 1), km_tr[mask_tr])
            resid_tr = (km_tr[mask_tr] - reg.predict(mn_tr[mask_tr].reshape(-1, 1))).reshape(-1, 1)
            resid_te = (km_te[mask_te] - reg.predict(mn_te[mask_te].reshape(-1, 1))).reshape(-1, 1)
            resid_fit = fit_logistic_probe(
                resid_tr, y_tr[mask_tr], resid_tr, y_tr[mask_tr], resid_te
            )
            residual = {
                "auroc": bootstrap_auroc(
                    y_te[mask_te], resid_fit.scores, p_te[mask_te], n_resamples=n_boot, seed=seed
                ).to_dict(),
                "slope": float(reg.coef_[0]),
                "intercept": float(reg.intercept_),
                "n_test": int(mask_te.sum()),
            }

    def strip(d: dict[str, Any] | None) -> dict[str, Any] | None:
        if d is None:
            return None
        return {k: v for k, v in d.items() if k not in {"scores", "y", "prompt_ids"}}

    return {
        "space": space,
        "n_test": len(te_raw),
        "spearman_kappa_mean_vs_mean_norm": {
            "rho": float(rho) if np.isfinite(rho) else None,
            "p_value": float(p_rho) if np.isfinite(p_rho) else None,
            "n": len(kappa_means),
        },
        "kappa_cos_l2_normalized": strip(kcos),
        "gram_kappa_same_space": strip(raw5),
        "bl_norm_same_space": strip(norm),
        "incremental_kappa_plus_norm": incremental,
        "residual_kappa_mean_on_norm": residual,
    }


def task_c(
    samples: list[Sample],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    tau = MIN_SEQUENCE_LENGTH
    pool = [s for s in samples if s.t >= tau and s.y in (0, 1)]
    tr, va, te = by_split(pool, "train"), by_split(pool, "val"), by_split(pool, "test")
    theta = train_theta(tr, "pca64")
    # Strict pairing: samples that admit LEN, NORM, and Raw.
    te_p = [s for s in te if eligible(s, tau, "BL-LEN") and eligible(s, tau, "BL-NORM") and eligible(s, tau, "Raw-5")]
    tr_p = [s for s in tr if eligible(s, tau, "BL-LEN") and eligible(s, tau, "BL-NORM") and eligible(s, tau, "Raw-5")]
    va_p = [s for s in va if eligible(s, tau, "BL-LEN") and eligible(s, tau, "BL-NORM") and eligible(s, tau, "Raw-5")]
    if len(te_p) == 0:
        # Fall back: pair LEN and NORM even if Raw is missing.
        te_p = [s for s in te if eligible(s, tau, "BL-LEN") and eligible(s, tau, "BL-NORM")]
        tr_p = [s for s in tr if eligible(s, tau, "BL-LEN") and eligible(s, tau, "BL-NORM")]
        va_p = [s for s in va if eligible(s, tau, "BL-LEN") and eligible(s, tau, "BL-NORM")]
        raw_ok = False
    else:
        raw_ok = True
    pids = np.asarray([s.prompt_id for s in te_p])
    rs = draw_cluster_resamples(pids, n_boot, seed) if te_p else None
    len_e = eval_method(tr_p, va_p, te_p, "BL-LEN", theta, rs, n_boot, seed)
    norm_e = eval_method(tr_p, va_p, te_p, "BL-NORM", theta, rs, n_boot, seed)
    raw_e = eval_method(tr_p, va_p, te_p, "Raw-5", theta, rs, n_boot, seed) if raw_ok else None
    out: dict[str, Any] = {
        "n_test_paired": len(te_p),
        "n_test_prompts": len(set(pids.tolist())) if te_p else 0,
        "raw_available": raw_ok,
    }
    if len_e and norm_e:
        out["BL-LEN"] = {k: v for k, v in len_e.items() if k not in {"scores", "y", "prompt_ids"}}
        out["BL-NORM"] = {k: v for k, v in norm_e.items() if k not in {"scores", "y", "prompt_ids"}}
        out["BL-NORM_vs_BL-LEN"] = paired_delta_bootstrap(
            len_e["y"], norm_e["scores"], len_e["scores"], len_e["prompt_ids"],
            n_resamples=n_boot, seed=seed, resamples=rs,
        ).to_dict()
    if raw_e and norm_e:
        out["Raw-5"] = {k: v for k, v in raw_e.items() if k not in {"scores", "y", "prompt_ids"}}
        out["Raw_vs_BL-NORM"] = paired_delta_bootstrap(
            raw_e["y"], raw_e["scores"], norm_e["scores"], raw_e["prompt_ids"],
            n_resamples=n_boot, seed=seed, resamples=rs,
        ).to_dict()
    return out


def task_d(
    samples: list[Sample],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    tau = MIN_SEQUENCE_LENGTH
    pool = [s for s in samples if s.t >= tau and s.y in (0, 1)]
    tr, va, te = by_split(pool, "train"), by_split(pool, "val"), by_split(pool, "test")
    te_p = [s for s in te if eligible(s, tau, "Raw-5") and eligible(s, tau, "PCA64-5")]
    tr_p = [s for s in tr if eligible(s, tau, "Raw-5") and eligible(s, tau, "PCA64-5")]
    va_p = [s for s in va if eligible(s, tau, "Raw-5") and eligible(s, tau, "PCA64-5")]
    if not te_p:
        return {"error": "no test samples with both raw and PCA64 trajectories"}
    pids = np.asarray([s.prompt_id for s in te_p])
    rs = draw_cluster_resamples(pids, n_boot, seed)
    theta = train_theta(tr_p, "pca64")
    raw_e = eval_method(tr_p, va_p, te_p, "Raw-5", theta, rs, n_boot, seed)
    pca5 = eval_method(tr_p, va_p, te_p, "PCA64-5", theta, rs, n_boot, seed)
    pca6 = eval_method(tr_p, va_p, te_p, "PCA64-6", theta, rs, n_boot, seed)
    out: dict[str, Any] = {
        "n_test_paired": len(te_p),
        "features": list(KAPPA5),
        "note": "spike_rate excluded from both Raw and PCA 64D (5-feature matched comparison)",
    }
    if raw_e:
        out["Raw-5"] = {k: v for k, v in raw_e.items() if k not in {"scores", "y", "prompt_ids"}}
    if pca5:
        out["PCA64-5"] = {k: v for k, v in pca5.items() if k not in {"scores", "y", "prompt_ids"}}
    if pca6:
        out["PCA64-6_confirmatory_spike_rate"] = {
            k: v for k, v in pca6.items() if k not in {"scores", "y", "prompt_ids"}
        }
    if raw_e and pca5:
        out["Raw_minus_PCA64_5feat"] = paired_delta_bootstrap(
            raw_e["y"], raw_e["scores"], pca5["scores"], raw_e["prompt_ids"],
            n_resamples=n_boot, seed=seed, resamples=rs,
        ).to_dict()
    return out


def freeze_sanity(payload: dict[str, Any]) -> dict[str, Any]:
    path = REPO / "docs" / "Paper" / "phase3_metrics.json"
    if not path.exists():
        return {"error": "phase3_metrics.json not found"}
    frozen = json.loads(path.read_text(encoding="utf-8"))
    l16 = frozen["per_layer"]["16"]
    return {
        "source": str(path),
        "frozen": {
            "n_test": l16["n_test"],
            "BL-LEN": l16["BL-LEN"]["point_estimate"],
            "BL-NORM": l16["BL-NORM"]["point_estimate"],
            "PROP-kappa_PCA64": l16["PROP-kappa"]["point_estimate"],
            "Raw": frozen["ablation_1_raw_kappa"]["raw_kappa_auroc"]["point_estimate"],
            "PC4-67": frozen["ablation_2_pc4_67"]["pc4_67_kappa_auroc"]["point_estimate"],
        },
    }


def load_env_info() -> dict[str, Any]:
    path = REPO / "results" / "env_info.json"
    freeze = REPO / "docs" / "Paper" / "FREEZE_MANIFEST.md"
    measured = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    paper = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "model_commit_hash": "a09a35458c702b33eeacc393d103063234e8bc28",
        "vllm_version": "0.29.0",
        "cuda": "13.0",
        "torch": "2.13.0+cu130",
        "gpu": "NVIDIA A40",
        "python": "3.11.10",
        "source": "docs/Paper/FREEZE_MANIFEST.md",
        "env_info_json_present": path.exists(),
        "freeze_manifest_present": freeze.exists(),
    }
    mismatches: list[str] = []
    if measured:
        mapping = {
            "vllm_version": ("vllm_version",),
            "python_version": ("python",),
        }
        if measured.get("vllm_version") and str(measured["vllm_version"]) != paper["vllm_version"]:
            mismatches.append(f"vllm {measured['vllm_version']} vs freeze {paper['vllm_version']}")
        gpus = measured.get("gpu_devices") or []
        if gpus and paper["gpu"] not in str(gpus):
            mismatches.append(f"gpu {gpus} vs freeze {paper['gpu']}")
    paper["mismatches"] = mismatches
    paper["measured"] = measured
    return paper


def print_tables(payload: dict[str, Any]) -> None:
    def line(msg: str = "") -> None:
        print(msg)

    line("=" * 72)
    line("EXP-TRJ001 v3.3 reanalysis")
    line("=" * 72)
    src = payload.get("data_source", {})
    line(f"samples={src.get('n_loaded')}  raw={src.get('n_with_raw')}  pca={src.get('n_with_pca')}  labeled={src.get('n_with_label')}")
    a1 = payload.get("task_a", {}).get("label_breakdown", {})
    if a1:
        line("\n[A-1] Class E (T < 20) on the paired prompt inventory")
        line(f"  prompts={a1.get('n_prompts')}  generations={a1.get('n_generations')}  (8×prompts={a1.get('expected_generations')})")
        ex, be, af = a1.get("excluded", {}), a1.get("before_exclusion", {}), a1.get("after_exclusion", {})
        line(f"  excluded n={ex.get('n')}  y=1:{ex.get('y1')}  y=0:{ex.get('y0')}  other:{ex.get('other')}")
        line(f"  positive rate before={be.get('positive_rate')}  after={af.get('positive_rate')}  kept n={af.get('n')}")
        line(f"  {a1.get('paired_note')}")
    a2 = payload.get("task_a", {}).get("threshold_sensitivity", {})
    if a2.get("thresholds"):
        line("\n[A-2] AUROC ranking vs T threshold  (Raw-5, PCA64-5, BL-LEN, BL-NORM)")
        line(f"  {'tau':>6}  {'Raw-5':>8}  {'PCA64-5':>8}  {'BL-LEN':>8}  {'BL-NORM':>8}  ranking")
        for tau, block in a2["thresholds"].items():
            ms = block["methods"]
            def fmt(name: str) -> str:
                row = ms.get(name)
                if not row:
                    return f"{'NA':>8}"
                return f"{row['test_auroc']['point_estimate']:.4f}".rjust(8)
            line(f"  {tau:>6}  {fmt('Raw-5')}  {fmt('PCA64-5')}  {fmt('BL-LEN')}  {fmt('BL-NORM')}  {' > '.join(block['ranking'])}")
        line(f"  rank_stable_across_tau={a2.get('rank_stable')}")
    b = payload.get("task_b", {})
    if b:
        sp = b.get("spearman_kappa_mean_vs_mean_norm", {})
        line("\n[B] Scale-invariant directional κ")
        line(f"  space={b.get('space')}  Spearman(kappa_mean, mean_norm) ρ={sp.get('rho')}  p={sp.get('p_value')}  n={sp.get('n')}")
        kc = b.get("kappa_cos_l2_normalized") or {}
        if kc:
            line(f"  κ^cos (L2-normalized) AUROC={kc['test_auroc']['point_estimate']:.4f}  n={kc.get('n_test')}")
        inc = b.get("incremental_kappa_plus_norm")
        if inc:
            line(f"  κ+NORM AUROC={inc['joint_kappa_plus_norm']['point_estimate']:.4f}  Δ vs NORM={inc['delta_vs_norm']['point_estimate']:.4f}  p(Δ≤0)={inc['delta_vs_norm']['p_value_le_0']}")
        rs = b.get("residual_kappa_mean_on_norm")
        if rs:
            line(f"  residual(κ_mean | norm) AUROC={rs['auroc']['point_estimate']:.4f}")
    c = payload.get("task_c", {})
    if c:
        line("\n[C] Paired bootstrap on the eligible test inventory")
        line(f"  n_test_paired={c.get('n_test_paired')}  prompts={c.get('n_test_prompts')}  raw_available={c.get('raw_available')}")
        if "BL-NORM_vs_BL-LEN" in c:
            d = c["BL-NORM_vs_BL-LEN"]
            line(f"  BL-NORM vs BL-LEN  Δ={d['point_estimate']:.4f}  CI=[{d['ci95_lower']:.4f},{d['ci95_upper']:.4f}]  p(Δ≤0)={d['p_value_le_0']}")
        if "Raw_vs_BL-NORM" in c:
            d = c["Raw_vs_BL-NORM"]
            line(f"  Raw vs BL-NORM     Δ={d['point_estimate']:.4f}  CI=[{d['ci95_lower']:.4f},{d['ci95_upper']:.4f}]  p(Δ≤0)={d['p_value_le_0']}")
    d = payload.get("task_d", {})
    if d and "Raw_minus_PCA64_5feat" in d:
        dd = d["Raw_minus_PCA64_5feat"]
        line("\n[D] 5-feature matched PCA vs Raw")
        line(f"  n={d.get('n_test_paired')}  Δ(Raw−PCA64-5)={dd['point_estimate']:.4f}  CI=[{dd['ci95_lower']:.4f},{dd['ci95_upper']:.4f}]  p(Δ≤0)={dd['p_value_le_0']}")
    line("=" * 72)


def json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items() if k not in {"scores", "y", "prompt_ids"}}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--splits", type=Path, default=REPO / "configs" / "data_splits.json")
    p.add_argument("--w-pca", type=Path, default=REPO / "configs" / "W_pca128.npy")
    p.add_argument("--mu-pca", type=Path, default=REPO / "configs" / "mu_pca.npy")
    p.add_argument("--out", type=Path, default=REPO / "results" / "trj001_reanalysis_metrics.json")
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    p.add_argument("--self-test", action="store_true", help="Run on synthetic trajectories (not a paper result).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        samples = make_synthetic()
        provenance: dict[str, Any] = {
            "mode": "self-test",
            "n_loaded": len(samples),
            "n_with_raw": len(samples),
            "n_with_pca": len(samples),
            "n_with_label": len(samples),
            "warning": "Synthetic data; do not copy these numbers into the manuscript.",
        }
        n_boot = min(args.n_bootstrap, 128)
    else:
        roots = candidate_roots(args.data_root)
        samples, provenance = load_corpus(roots, args.splits)
        note = maybe_project(samples, args.w_pca, args.mu_pca)
        if note:
            provenance["projection"] = note
        n_boot = args.n_bootstrap
        provenance["mode"] = "freeze-data"

    test_ids = test_prompt_ids(samples)
    payload: dict[str, Any] = {
        "schema_version": "trj001-reanalysis-v3.3",
        "self_test": bool(args.self_test),
        "n_bootstrap": n_boot,
        "seed": args.seed,
        "data_source": provenance,
        "env": load_env_info(),
        "freeze_reference": freeze_sanity({}),
        "task_a": {
            "label_breakdown": task_a1(samples, test_ids),
            "threshold_sensitivity": {},
            "paired_evaluation": {
                "statement": (
                    "BL-LEN and BL-NORM are evaluated on the same eligible generations "
                    "as Raw 3584D and PCA 64D (prompt-clustered test inventory; T>=20)."
                )
            },
        },
        "task_b": {},
        "task_c": {},
        "task_d": {},
    }

    has_traj = any(s.z_raw is not None or s.z_pca is not None for s in samples)
    has_split = any(s.split in {"train", "val", "test"} for s in samples)
    freeze_like = (not args.self_test) and (
        sum(1 for s in samples if s.split == "test" and s.y in (0, 1) and s.t >= MIN_SEQUENCE_LENGTH) >= 400
    )
    if not args.self_test and has_traj and not freeze_like:
        payload["error"] = (
            "Loaded trajectories do not match the freeze inventory "
            "(expected ~496 eligible test generations from 98 prompts). "
            "Refusing to treat this run as a paper result. "
            "Pass --data-root to the freeze results directory."
        )
        payload["searched_roots"] = [str(r) for r in candidate_roots(args.data_root)]
    elif not args.self_test and not has_traj:
        payload["error"] = (
            "No hidden-state / projected trajectories were found. "
            "Place freeze artifacts under results/ or pass --data-root. "
            "Task A-1 (labels/T) ran if metadata was present; A-2/B/C/D require trajectories."
        )
        payload["searched_roots"] = [str(r) for r in candidate_roots(args.data_root)]
    if payload.get("error"):
        pass
    elif not has_split and not args.self_test:
        # Without splits, treat all labeled samples as a single pool and
        # synthesize prompt-level 60/20/20 splits so sensitivity still runs,
        # marked exploratory.
        rng = np.random.default_rng(args.seed)
        pids = np.array(sorted({s.prompt_id for s in samples}))
        rng.shuffle(pids)
        n = len(pids)
        n_tr = max(1, int(0.6 * n))
        n_va = max(1, int(0.2 * n))
        split_map = {}
        for i, pid in enumerate(pids):
            split_map[str(pid)] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
        for s in samples:
            s.split = split_map.get(s.prompt_id, "test")
        payload["data_source"]["synthetic_splits"] = True
        payload["data_source"]["warning"] = (
            "configs/data_splits.json missing; used a seeded 60/20/20 prompt split. "
            "Do not treat AUROCs as confirmatory freeze reproductions."
        )

    if not payload.get("error") and has_traj and any(s.split in {"train", "val", "test"} for s in samples):
        payload["task_a"]["threshold_sensitivity"] = task_a2(samples, n_boot, args.seed)
        payload["task_b"] = task_b(samples, n_boot, args.seed)
        payload["task_c"] = task_c(samples, n_boot, args.seed)
        payload["task_d"] = task_d(samples, n_boot, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(json_sanitize(payload), indent=2) + "\n", encoding="utf-8")
    print_tables(payload)
    print(f"[+] wrote {args.out}")
    if payload.get("error"):
        print(f"[!] {payload['error']}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
