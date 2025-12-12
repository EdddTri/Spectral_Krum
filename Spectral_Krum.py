import os
import platform

CORES = os.cpu_count() or 8
PHYS  = max(1, CORES // 2)
DEFAULT_PROCS = max(1, min(PHYS - 4, 92))
PROC_TARGET   = int(os.environ.get("FL_PROCS", str(DEFAULT_PROCS)))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import csv, time, math, random, json
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, datasets

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from tqdm import tqdm
import multiprocessing

EXPERIMENT_NAME = "overnight_run_cifar_noniid_v9"
RESULTS_CSV     = f"{EXPERIMENT_NAME}.csv"

GLOBAL_ROUNDS = 100
N_CLIENTS     = 11
LOCAL_EPOCHS  = 1
BATCH_SIZE    = 64
LR            = 0.02

fl_dev = os.environ.get("FL_DEVICE", "").lower()
if fl_dev == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
DEVICE = "cuda" if (fl_dev == "cuda" or (fl_dev == "" and torch.cuda.is_available())) else "cpu"

# --- Scope ---
SEEDS         = [0]
DATASETS      = [("cifar10","tinycnn")]
ALPHAS        = [0.1]
RUN_MAIN_COMPARISON = True
RUN_ABLATIONS       = True
# -------------

# MAIN comparison knobs
F_GRID_MAIN          = [1, 2]
ATTACKER_COUNTS_MAIN = [0, 2]
TAU_Q_MAIN           = 0.98
R_MAIN               = 50
WARMUP_MAIN          = 3
USE_GUARD_MAIN       = True

# Ablations (SK-only)
ABLATION_DATASET      = ("cifar10","tinycnn")
ABLATION_ALPHA        = 0.1
F_GRID_ABL            = [1, 2, 3]
ATTACKER_COUNTS_ABL   = [0, 2]
TAU_QS_ABL            = [0.95, 0.98, 0.99]
R_LIST_ABL            = [10, 25, 50, 100]
WARMUP_LIST_ABL       = [0, 3, 5]
USE_GUARD_LIST_ABL    = [False, True]

# Attack configs
BASE_ATTACKERS = {
    "none":           {"type": "none"},
    "sign_flip":      {"type": "sign",   "scale": -3.0},
    "label_flip":     {"type": "label",  "map": {k: (k+5)%10 for k in range(10)}},
    "min_max":        {"type": "min_max", "scale": 3.0},
    "adaptive_steer": {"type": "steer",  "gamma": 3.0},
    "buffer_drift":   {"type": "buffer", "drift_rounds": 20, "drift_scale": 0.15, "hit_period": 5, "hit_scale": 1.5},
    "semantic_backdoor": {
        "type": "semantic_backdoor",
        "poison_rate": 0.1,
        "target": 0,
        "trigger_size": 3,
        "trigger_value": 1.0,
        "position": "br"
    },
}

AGGREGATOR_NAMES = [
    "CoordMedian", "TrimmedMean",
    "FullKrum", "MultiKrum", "GeometricMedian",
    "Bulyan",
    "DnC-PMF", "DnC-Cluster", "SpectralKrum"
]

# Core utils
def set_all_seeds(seed:int=0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    fast = os.environ.get("FL_FAST","0") == "1"
    if fast or torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        intra = int(os.environ.get("FL_THREADS", os.environ.get("OMP_NUM_THREADS", "1")))
        torch.set_num_threads(intra)
        torch.set_num_interop_threads(max(1, intra // 2))
    except Exception:
        pass

def model_parameters_to_vector(model: torch.nn.Module) -> np.ndarray:
    with torch.no_grad():
        return torch.cat([p.detach().view(-1) for p in model.parameters()]).cpu().numpy().astype(np.float32)

def vector_to_model_parameters(model: torch.nn.Module, vec: np.ndarray, device: str) -> None:
    vec_t = torch.from_numpy(vec.astype(np.float32)).to(device)
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            numel = p.numel()
            p.copy_(vec_t[idx:idx+numel].view_as(p))
            idx += numel

def coord_median_agg(deltas: np.ndarray, **_) -> Tuple[np.ndarray, Dict[str, Any]]:
    return np.median(deltas, axis=0).astype(np.float32), {}

def trimmed_mean_agg(deltas: np.ndarray, f: int = 0, frac: float = 0.1, **_) -> Tuple[np.ndarray, Dict[str, Any]]:
    m = deltas.shape[0]
    t = f if f > 0 else int(frac * m)
    if 2*t >= m:
        return deltas.mean(axis=0).astype(np.float32), {"fallback":"tm"}

    sort_idx = np.argsort(deltas, axis=0, kind="mergesort")
    kept_rows = sort_idx[t:m-t, np.arange(deltas.shape[1])]

    trimmed_vals = np.empty((m - 2*t, deltas.shape[1]), dtype=np.float32)
    for j in range(deltas.shape[1]):
        trimmed_vals[:, j] = deltas[kept_rows[:, j], j]

    return trimmed_vals.mean(axis=0).astype(np.float32), {"trim_t": t}

class CSVLogger:
    DEFAULT_FIELDS = [
        "seed","dataset","model","alpha","attack","attacker_count","algo",
        "f_param","use_guard","warmup_rounds","r","guard_min_kept","tau_quantile",
        "round","acc","asr",
        "U_rank","pre_guard_selected","num_guard_kept","tau_buffer",
        "guard_target","guard_dropped","guard_fallback_used",
        "time_ms_total","time_ms_proj","time_ms_pairwise","time_ms_guard","time_ms_refresh","time_ms_other",
        "atk_state_used","atk_adv_idx",
        "fallback","bulyan_krum_fallback",
        "final_acc","best_acc",
        "guard_tp","guard_fp","guard_kept_attackers","resid_mean","resid_p95"
    ]
    def __init__(self, path: str, extra_fields: List[str] = None):
        self.path = path
        self.fields = self.DEFAULT_FIELDS + (extra_fields or [])
        self._ensure_header()

    def _ensure_header(self):
        new_or_empty = (not os.path.exists(self.path)) or os.stat(self.path).st_size == 0
        if new_or_empty:
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()

    def write_rows(self, rows: List[Dict[str, Any]]):
        out_rows = [{k: row.get(k, "") for k in self.fields} for row in rows]
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerows(out_rows)

# Aggregators
def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    G = X @ X.T
    nrm = np.sum(X * X, axis=1, keepdims=True)
    D = (nrm + nrm.T - 2*G).astype(np.float32)
    np.maximum(D, 0.0, out=D)
    return D

def _krum_select_and_score(D: np.ndarray, n: int, f: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 2*f + 3:
        raise ValueError(f"Krum requires n >= 2f + 3; got n={n}, f={f}")
    k = max(1, n - f - 2)
    np.fill_diagonal(D, np.inf)
    neigh_idx = np.argpartition(D, kth=k-1, axis=1)[:, :k]
    row_idx = np.arange(n)[:, None]
    neigh_dists = D[row_idx, neigh_idx]
    scores = neigh_dists.sum(axis=1).astype(np.float32)
    order = np.argsort(scores)
    return order[:k], scores

def full_krum_agg(deltas: np.ndarray, f: int = 1, **_) -> Tuple[np.ndarray, Dict[str, Any]]:
    t0 = time.time()
    n = deltas.shape[0]
    info: Dict[str, Any] = {}
    if n < 2*f + 3:
        agg = np.median(deltas, axis=0).astype(np.float32)
        info.update({"fallback": "median_small_n", "pre_guard_selected": n})
    else:
        t_pair0 = time.time()
        D = _pairwise_sq_dists(deltas)
        _, scores = _krum_select_and_score(D, n, f)
        t_pair1 = time.time()
        best_idx = scores.argmin()
        agg = deltas[best_idx].copy().astype(np.float32)
        info.update({"pre_guard_selected": 1, "time_ms_pairwise": (t_pair1 - t_pair0) * 1e3})
    t1 = time.time()
    info["time_ms_total"] = (t1-t0)*1e3
    info.setdefault("time_ms_pairwise", 0.0)
    return agg, info

def multi_krum_agg(deltas: np.ndarray, f: int = 1, **_) -> Tuple[np.ndarray, Dict[str, Any]]:
    t0 = time.time(); n = deltas.shape[0]; info: Dict[str, Any] = {}
    if n < 2*f + 3:
        agg = np.median(deltas, axis=0).astype(np.float32)
        t1 = time.time()
        info.update({
            "fallback": "median_small_n", "pre_guard_selected": n,
            "time_ms_pairwise": 0.0, "time_ms_proj": 0.0, "time_ms_guard": 0.0,
            "time_ms_refresh": 0.0, "time_ms_other": (t1 - t0) * 1e3, "time_ms_total": (t1 - t0) * 1e3,
        })
        return agg, info

    t_pair0 = time.time()
    D = _pairwise_sq_dists(deltas)
    sel, _ = _krum_select_and_score(D, n, f)
    t_pair1 = time.time()
    agg = deltas[sel].mean(axis=0).astype(np.float32)
    t1 = time.time()

    t_pair_ms = (t_pair1 - t_pair0) * 1e3
    t_total_ms = (t1 - t0) * 1e3
    info.update({
        "pre_guard_selected": int(len(sel)),
        "time_ms_pairwise": t_pair_ms,
        "time_ms_proj": 0.0, "time_ms_guard": 0.0, "time_ms_refresh": 0.0,
        "time_ms_other": max(0.0, t_total_ms - t_pair_ms),
        "time_ms_total": t_total_ms,
    })
    return agg, info

def geometric_median_agg(deltas: np.ndarray, f: int = 0, tol: float = 1e-5, max_iter: int = 100, **_) -> Tuple[np.ndarray, Dict[str, Any]]:
    t0 = time.time()
    gm = np.median(deltas, axis=0).astype(np.float32)
    n, d = deltas.shape
    info = {}

    for i in range(max_iter):
        prev_gm = gm.copy()
        dists = np.linalg.norm(deltas - gm, axis=1)
        zero_dist_mask = (dists < tol)
        if np.any(zero_dist_mask):
            if zero_dist_mask.sum() > n / 2.0:
                info.update({"gm_iters": i, "gm_status": "majority_match"})
                break
            dists[zero_dist_mask] = tol

        inv_dists = 1.0 / dists
        weights = inv_dists / np.sum(inv_dists)
        gm = (weights[:, None] * deltas).sum(axis=0)

        if np.linalg.norm(gm - prev_gm) < tol:
            info.update({"gm_iters": i, "gm_status": "converged"})
            break

    if "gm_status" not in info:
        info.update({"gm_iters": max_iter, "gm_status": "max_iter"})

    info["time_ms_total"] = (time.time()-t0)*1e3
    return gm.astype(np.float32), info

def bulyan_agg(deltas: np.ndarray, f: int = 1, **_) -> Tuple[np.ndarray, Dict[str, Any]]:
    t0 = time.time()
    n, d = deltas.shape
    info: Dict[str, Any] = {}

    if n < 4*f + 3:
        agg = np.median(deltas, axis=0).astype(np.float32)
        info["fallback"] = "median_small_n"
        info["time_ms_total"] = (time.time()-t0)*1e3
        return agg, info

    k_sel = n - 2*f
    selected_deltas = []
    X = deltas.copy()

    for i in range(k_sel):
        m = X.shape[0]
        if m <= 0: break

        if m < 2*f + 3:
            info["bulyan_krum_fallback"] = m
            break

        D = _pairwise_sq_dists(X)
        _, scores = _krum_select_and_score(D, m, f)
        best_idx = scores.argmin()
        selected_deltas.append(X[best_idx].copy())
        X = np.delete(X, best_idx, axis=0)

    if not selected_deltas:
        agg = np.median(deltas, axis=0).astype(np.float32)
        info["fallback"] = "median_no_selection"
        info["time_ms_total"] = (time.time()-t0)*1e3
        return agg, info

    X_sel = np.stack(selected_deltas, axis=0)
    m_sel = X_sel.shape[0]

    t_bulyan = min(f, (m_sel - 1)//2)

    if m_sel == 0 or 2*t_bulyan >= m_sel:
        agg = np.mean(X_sel, axis=0).astype(np.float32) if m_sel > 0 else np.median(deltas, axis=0).astype(np.float32)
        info["fallback"] = "bulyan_tm_fallback"
    else:
        sort_idx = np.argsort(X_sel, axis=0, kind="mergesort")
        kept_rows = sort_idx[t_bulyan : m_sel - t_bulyan, np.arange(d)]
        trimmed_vals = np.empty((m_sel - 2*t_bulyan, d), dtype=np.float32)
        for j in range(d):
            trimmed_vals[:, j] = X_sel[kept_rows[:, j], j]
        agg = trimmed_vals.mean(axis=0).astype(np.float32)

    info["bulyan_selected"] = m_sel
    info["bulyan_trimmed"] = t_bulyan
    info["time_ms_total"] = (time.time()-t0)*1e3
    return agg, info

@dataclass
class PMFConfig:
    pca_dim: int = 20
    keep_fraction: float = 0.7
    random_state: int = 0

def dnc_pmf_agg(deltas: np.ndarray, f: int = 1, pmf_cfg: PMFConfig = None, **_) -> Tuple[np.ndarray, Dict[str, Any]]:
    if pmf_cfg is None:
        pmf_cfg = PMFConfig()
    m, d = deltas.shape

    if m < 2:
        return deltas.mean(axis=0).astype(np.float32), {"pmf_kept": m}

    r = min(max(1, pmf_cfg.pca_dim), d, m-1)
    if r <= 0:
        return deltas.mean(axis=0).astype(np.float32), {"pmf_kept": m}

    if not np.isfinite(deltas).all():
        deltas = np.nan_to_num(deltas)

    pca = PCA(n_components=r, svd_solver="randomized", random_state=pmf_cfg.random_state)
    try:
        Z = pca.fit_transform(deltas)
    except Exception:
        return deltas.mean(axis=0).astype(np.float32), {"pmf_kept": m}

    mags = np.linalg.norm(Z, axis=1)
    m_keep = min(m - max(0, f), max(1, int(np.floor(pmf_cfg.keep_fraction * m))))
    keep_idx = np.argpartition(mags, kth=m_keep - 1)[:m_keep]
    agg = deltas[keep_idx].mean(axis=0).astype(np.float32)
    info = {"pmf_kept": int(m_keep), "pmf_r": int(r)}
    return agg, info

@dataclass
class SpectralKrumConfig:
    r: int = 50
    buffer_size: int = 50
    center_mode: str = "mean"
    trim_mode: str = "two_sided"
    trim_frac: float = 0.1
    warmup_rounds: int = 3
    refresh_every: int = 1
    orthE_quantile_from_buffer: float = 0.98
    guard_min_kept: int = 1
    f_byzantine: int = 1
    clip_norm: float = 0.0
    seed: int = 0

class SpectralKrum:
    def __init__(self, cfg: SpectralKrumConfig):
        self.cfg = cfg
        self.buffer: List[np.ndarray] = []
        self.U: Optional[np.ndarray] = None
        self._tau_buffer: float = 0.0
        self.round_idx: int = 0
        self.rng = np.random.RandomState(cfg.seed)

    def _build_subspace_from_buffer(self) -> None:
        if len(self.buffer) == 0:
            self.U = None
            return

        X = np.stack(self.buffer, axis=0)
        if not np.isfinite(X).all():
            X = np.nan_to_num(X)

        mu = np.median(X, axis=0) if self.cfg.center_mode == "median" else X.mean(axis=0)
        Xc = X - mu
        norms = np.linalg.norm(Xc, axis=1)
        b = len(norms)
        t = int(self.cfg.trim_frac * b)

        if self.cfg.trim_mode == "two_sided" and 2 * t < b:
            keep = np.argsort(norms)[t:b - t]
        else:
            keep = np.argsort(norms)[: max(1, b - t)]
        Xk = Xc[keep]

        if Xk.shape[0] < 2:
            self.U = None
            return

        r_use = min(self.cfg.r, Xk.shape[0], Xk.shape[1])
        if r_use <= 0:
            self.U = None
            return

        pca = PCA(n_components=r_use, svd_solver="randomized", random_state=self.cfg.seed)
        try:
            pca.fit(Xk)
            self.U = pca.components_.T.astype(np.float32)
        except Exception:
            self.U = None

    def _refresh_tau(self) -> None:
        if self.U is None or len(self.buffer) == 0:
            self._tau_buffer = 0.0
            return
        X = np.stack(self.buffer, axis=0)
        proj = (X @ self.U) @ self.U.T
        resid = np.linalg.norm(X - proj, axis=1)
        self._tau_buffer = float(np.quantile(resid, self.cfg.orthE_quantile_from_buffer))
    def step(self, deltas: np.ndarray, global_weights: np.ndarray, use_guard: bool = True) -> Dict[str, Any]:
        t_total0 = time.time()
        self.round_idx += 1
        info: Dict[str, Any] = {}

        n = deltas.shape[0]
        f = max(0, int(self.cfg.f_byzantine))
        krum_ok = (n >= 2 * f + 3)

        # warmup or fallback path
        if self.round_idx <= self.cfg.warmup_rounds or self.U is None or (not krum_ok):
            agg = np.median(deltas, axis=0).astype(np.float32)
            self.buffer.append(agg.copy())
            if len(self.buffer) > self.cfg.buffer_size:
                self.buffer.pop(0)

            t_refresh = 0.0
            if (self.round_idx % self.cfg.refresh_every) == 0:
                t_ref0 = time.time()
                self._build_subspace_from_buffer()
                self._refresh_tau()
                t_ref1 = time.time()
                t_refresh = (t_ref1 - t_ref0) * 1e3

            new_global = (global_weights + agg).astype(np.float32)
            t1 = time.time()
            t_total_ms = (t1 - t_total0) * 1e3
            info.update({
                "mode": "warmup",
                "U_rank": 0 if self.U is None else int(self.U.shape[1]),
                "pre_guard_selected": n,
                "num_guard_kept": n,
                "tau_buffer": float(self._tau_buffer),
                "new_global": new_global,
                "time_ms_proj": 0.0,
                "time_ms_pairwise": 0.0,
                "time_ms_guard": 0.0,
                "time_ms_refresh": t_refresh,
                "time_ms_other": max(0.0, t_total_ms - t_refresh),
                "time_ms_total": t_total_ms,
            })
            return info

        # main path
        t_refresh = 0.0
        if (self.round_idx % self.cfg.refresh_every) == 0:
            t_ref0 = time.time()
            self._build_subspace_from_buffer()
            self._refresh_tau()
            t_ref1 = time.time()
            t_refresh = (t_ref1 - t_ref0) * 1e3

        if self.U is None:
            agg = np.median(deltas, axis=0).astype(np.float32)
            self.buffer.append(agg.copy())
            if len(self.buffer) > self.cfg.buffer_size:
                self.buffer.pop(0)
            new_global = (global_weights + agg).astype(np.float32)
            t1 = time.time()
            t_total_ms = (t1 - t_total0) * 1e3
            info.update({
                "mode": "no_subspace",
                "U_rank": 0,
                "pre_guard_selected": n,
                "num_guard_kept": n,
                "tau_buffer": float(self._tau_buffer),
                "new_global": new_global,
                "time_ms_proj": 0.0,
                "time_ms_pairwise": 0.0,
                "time_ms_guard": 0.0,
                "time_ms_refresh": t_refresh,
                "time_ms_other": max(0.0, t_total_ms - t_refresh),
                "time_ms_total": t_total_ms,
            })
            return info


        t_proj0 = time.time()
        Z = deltas @ self.U
        t_proj1 = time.time()

        t_pair0 = time.time()
        Dz = _pairwise_sq_dists(Z)
        pre_sel, _ = _krum_select_and_score(Dz, n, f)
        t_pair1 = time.time()
        t_pair_ms = (t_pair1 - t_pair0) * 1e3

        t_guard0 = time.time()
        kept = pre_sel
        guard_fallback_used = 0

        k_krum = int(pre_sel.size)
        target = max(1, k_krum - f)

        if use_guard and self.U is not None:
            resid = np.linalg.norm(deltas - (Z @ self.U.T), axis=1)
            tau = self._tau_buffer
            mask = resid[pre_sel] <= tau
            kept = pre_sel[mask]

            if kept.size < self.cfg.guard_min_kept:
                guard_fallback_used = 1
                order = np.argsort(resid[pre_sel])[: self.cfg.guard_min_kept]
                kept = pre_sel[order]

        t_guard1 = time.time()
        guard_dropped = int(pre_sel.size - kept.size)

        agg = deltas[kept].mean(axis=0).astype(np.float32)
        self.buffer.append(agg.copy())
        if len(self.buffer) > self.cfg.buffer_size:
            self.buffer.pop(0)

        new_global = (global_weights + agg).astype(np.float32)
        t1 = time.time()

        t_proj_ms = (t_proj1 - t_proj0) * 1e3
        t_guard_ms = (t_guard1 - t_guard0) * 1e3
        t_total_ms = (t1 - t_total0) * 1e3
        t_other_ms = max(0.0, t_total_ms - (t_proj_ms + t_pair_ms + t_guard_ms + t_refresh))

        info.update({
            "mode": "main",
            "U_rank": int(self.U.shape[1] if self.U is not None else 0),
            "pre_guard_selected": int(pre_sel.size),
            "num_guard_kept": int(kept.size),
            "tau_buffer": float(self._tau_buffer),
            "new_global": new_global,
            "time_ms_proj": t_proj_ms,
            "time_ms_pairwise": t_pair_ms,
            "time_ms_guard": t_guard_ms,
            "time_ms_refresh": t_refresh,
            "time_ms_total": t_total_ms,
            "time_ms_other": t_other_ms,
            "guard_target": int(target),
            "guard_dropped": guard_dropped,
            "guard_fallback_used": guard_fallback_used,
        })
        return info

# DnC clustering defense
@dataclass
class DnCConfig:
    pca_dim: int = 10
    max_depth: int = 5
    min_cluster_size: int = 3
    compactness_reg: float = 1e-6
    random_state: int = 0
    verbose: bool = False

class DnCDefense:
    def __init__(self, config: DnCConfig):
        self.cfg = config

    @staticmethod
    def _cluster_radius(X: np.ndarray, idx: np.ndarray) -> float:
        if idx.size == 0: return 0.0
        sub = X[idx]
        center = sub.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(sub - center, axis=1)
        return float(dists.max(initial=0.0))

    def _recursive_partition(self, X: np.ndarray, indices: np.ndarray, depth: int, leaves: List[np.ndarray]) -> None:
        n = indices.size
        if depth >= self.cfg.max_depth or n < 2 * self.cfg.min_cluster_size:
            leaves.append(indices.copy())
            return
        Xnode = X[indices]
        try:
            kmeans = KMeans(n_clusters=2, random_state=self.cfg.random_state + depth, n_init=10)
            labels = kmeans.fit_predict(Xnode)
        except Exception:
            leaves.append(indices.copy()); return
        left = indices[labels == 0]; right = indices[labels == 1]
        if left.size < self.cfg.min_cluster_size or right.size < self.cfg.min_cluster_size:
            leaves.append(indices.copy()); return
        self._recursive_partition(X, left, depth+1, leaves)
        self._recursive_partition(X, right, depth+1, leaves)

    def _select_best_cluster(self, X: np.ndarray, leaves: List[np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        best_score, best_idx = -np.inf, None
        for cl in leaves:
            r = self._cluster_radius(X, cl)
            s = cl.size
            score = s / (r + self.cfg.compactness_reg)
            if score > best_score:
                best_score, best_idx = score, cl
        info = {
            "dnc_num_leaf_clusters": len(leaves),
            "dnc_best_score": float(best_score),
            "dnc_best_cluster_size": int(best_idx.size if best_idx is not None else 0),
        }
        return best_idx, info

    def step(self, deltas: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        n, d = deltas.shape
        if n == 1:
            return deltas[0].copy(), {"dnc_mode":"single_client", "dnc_num_leaf_clusters":1, "dnc_best_cluster_size":1}
        use_pca = 0 < self.cfg.pca_dim < d
        if use_pca and n > self.cfg.pca_dim:
            if not np.isfinite(deltas).all():
                deltas = np.nan_to_num(deltas)
            pca = PCA(n_components=self.cfg.pca_dim, random_state=self.cfg.random_state, svd_solver="randomized")
            try:
                Z = pca.fit_transform(deltas)
                mode = "pca"
            except Exception:
                Z = deltas; mode = "full_fallback"
        else:
            Z = deltas; mode = "full"

        all_idx = np.arange(n, dtype=int)
        leaves: List[np.ndarray] = []
        self._recursive_partition(Z, all_idx, 0, leaves)
        keep_idx, info = self._select_best_cluster(Z, leaves)
        if keep_idx is None or keep_idx.size == 0:
            agg = deltas.mean(axis=0).astype(np.float32)
            info["dnc_mode"] = "fallback_all"
            return agg, info
        agg = deltas[keep_idx].mean(axis=0).astype(np.float32)
        info["dnc_mode"] = mode
        return agg, info

def dnc_agg(deltas: np.ndarray, f: int = 1, dnc_cfg: DnCConfig = None, **_) -> Tuple[np.ndarray, Dict[str, Any]]:
    defense = DnCDefense(dnc_cfg or DnCConfig())
    return defense.step(deltas)

# DataLoader seeding
def make_torch_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g

def make_worker_init_fn(base_seed: int):
    def _init(worker_id: int):
        worker_seed = (base_seed + worker_id) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return _init

# Data and model
def get_client_datasets(dataset_name: str, num_clients: int, alpha: float, seed: int) -> Tuple[List[Subset], Dataset]:
    if dataset_name.lower() != "cifar10":
        raise ValueError("dataset_name must be 'cifar10'")

    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_tf  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    trainset = datasets.CIFAR10(root="./data", train=True, download=False, transform=train_tf)
    testset  = datasets.CIFAR10(root="./data", train=False, download=False, transform=test_tf)
    num_classes = 10

    rng = np.random.default_rng(seed)
    y = np.array(trainset.targets, dtype=np.int64)
    class_indices = [np.where(y==c)[0] for c in range(num_classes)]
    for c in range(num_classes):
        rng.shuffle(class_indices[c])

    P = rng.dirichlet([alpha]*num_clients, size=num_classes)
    client_bins = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx_c = class_indices[c]; n_c = len(idx_c)
        sizes = (P[c]/P[c].sum()) * n_c
        cuts  = np.clip(np.cumsum(sizes).astype(int), 0, n_c)
        parts = np.split(idx_c, cuts[:-1])
        for j in range(num_clients):
            client_bins[j].extend(parts[j].tolist())

    client_subsets = []
    for j in range(num_clients):
        arr = np.array(client_bins[j], dtype=np.int64)
        rng.shuffle(arr)
        client_subsets.append(Subset(trainset, arr.tolist()))
    return client_subsets, testset

class TinyCNN(nn.Module):
    def __init__(self, in_ch: int = 3, n_classes: int = 10):
        super().__init__()
        c1 = 16
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        flat = 32*8*8 if in_ch == 3 else 32*7*7
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 64), nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def make_model(model_name: str, dataset: str) -> nn.Module:
    if model_name.lower() != "tinycnn":
        raise ValueError("model_name must be 'tinycnn'")
    if dataset.lower() != "cifar10":
        raise ValueError("dataset must be 'cifar10'")
    return TinyCNN(in_ch=3, n_classes=10)

@torch.no_grad()
def test_model(model: nn.Module, testset: Dataset, device: str, seed=123) -> float:
    g = make_torch_generator(seed)
    loader = DataLoader(
        testset, batch_size=256, shuffle=False,
        num_workers=(int(os.environ.get("FL_DLOADER_WORKERS","0")) if str(device).startswith('cuda') else 0),
        pin_memory=str(device).startswith('cuda'), generator=g, worker_init_fn=make_worker_init_fn(seed),
        persistent_workers=(str(device).startswith('cuda') and int(os.environ.get("FL_DLOADER_WORKERS","0"))>0),
    )
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred = model(x).argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return 100.0 * correct / max(1, total)

# Backdoor utils
def _stamp_square_trigger(x: torch.Tensor, size: int, value: float, position: str = "br") -> torch.Tensor:
    C, H, W = x.shape
    s = max(1, int(size))
    h0, w0 = {
        "br": (H - s, W - s),
        "bl": (H - s, 0),
        "tr": (0, W - s),
        "tl": (0, 0),
    }.get(position, (H - s, W - s))
    x[:, h0:h0+s, w0:w0+s] = value
    return torch.clamp(x, min=-10.0, max=10.0)

class LabelMapSubset(Dataset):
    def __init__(self, base_subset: Subset, label_map: dict):
        self.base = base_subset
        self.label_map = dict(label_map)
        base_targets = np.array(self.base.dataset.targets)
        self.targets = base_targets[self.base.indices].copy()
        for a, b in self.label_map.items():
            self.targets[self.targets == a] = b
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, _ = self.base[i]
        return x, int(self.targets[i])

class BackdoorSubset(Dataset):
    """Deterministic backdoor with fixed poison mask."""
    def __init__(self, base_subset: Subset, poison_rate: float, target: int,
                 trigger_size: int = 3, trigger_value: float = 1.0, position: str = "br",
                 seed: Optional[int] = None):
        self.base = base_subset
        self.poison_rate = float(poison_rate)
        self.target = int(target)
        self.trigger_size = int(trigger_size)
        self.trigger_value = float(trigger_value)
        self.position = str(position)

        rng = np.random.default_rng(seed)
        N = len(self.base)
        self.poison_mask = (rng.random(N) < self.poison_rate)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        if self.poison_mask[i]:
            x = _stamp_square_trigger(x.clone(), self.trigger_size, self.trigger_value, self.position)
            y = self.target
        return x, y

@torch.no_grad()
def test_backdoor_asr(model: nn.Module, clean_testset: Dataset, device: str,
                      target: int, trigger_size: int, trigger_value: float, position: str = "br",
                      seed: int = 123) -> float:
    class TriggeredTest(Dataset):
        def __init__(self, base): self.base = base
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            x, y = self.base[i]
            x = _stamp_square_trigger(x.clone(), trigger_size, trigger_value, position)
            return x, y
    trig = TriggeredTest(clean_testset)
    g = make_torch_generator(seed)
    loader = DataLoader(
        trig, batch_size=256, shuffle=False,
        num_workers=(int(os.environ.get("FL_DLOADER_WORKERS","0")) if str(device).startswith('cuda') else 0),
        pin_memory=str(device).startswith('cuda'), generator=g, worker_init_fn=make_worker_init_fn(seed),
        persistent_workers=(str(device).startswith('cuda') and int(os.environ.get("FL_DLOADER_WORKERS","0"))>0)
    )
    model.eval()
    hit = total = 0
    with torch.inference_mode():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            pred = model(x).argmax(1)
            hit += (pred.cpu().numpy() == target).sum()
            total += pred.size(0)
    return 100.0 * hit / max(1, total)

# Attack implementations
def subspace_aware_attack(deltas: np.ndarray, U: np.ndarray, tau: float, adv_idx, gamma: float = 3.0, tau_margin: float = 0.95) -> np.ndarray:
    X = deltas.copy().astype(np.float32, copy=False)
    if U is None or tau is None or getattr(U, "ndim", 0) != 2 or U.size == 0 or len(adv_idx) == 0:
        for i in adv_idx:
            if 0 <= i < X.shape[0]: X[i] = -3.0 * X[i]
        return X
    Z = X @ U
    m = Z.shape[0]
    benign = [j for j in range(m) if j not in set(adv_idx)]
    if len(benign) == 0:
        for i in adv_idx:
            if 0 <= i < X.shape[0]: X[i] = -3.0 * X[i]
        return X
    mu_z = Z[benign].mean(axis=0, keepdims=True)
    nrm = float(np.linalg.norm(mu_z)) + 1e-12
    steer_z = (-mu_z / nrm).astype(np.float32)
    steer = (float(gamma) * steer_z) @ U.T
    max_orth = float(tau) * float(tau_margin)
    for i in adv_idx:
        if not (0 <= i < X.shape[0]): continue
        xi = X[i]
        xi_par  = (xi @ U) @ U.T
        xi_orth = xi - xi_par
        orth = float(np.linalg.norm(xi_orth))
        if orth > max_orth and orth > 0:
            xi_orth = xi_orth * (max_orth / orth)
        X[i] = (xi_par + xi_orth + steer[0]).astype(np.float32, copy=False)
    return X

class BufferPoisoner:
    def __init__(self, r_dim: int = None, drift_rounds: int = 20, drift_scale: float = 0.2, hit_period: int = 5, hit_scale: float = 1.5, tau_margin: float = 0.95):
        self.round = 0
        self.z_dir = None
        self.drift_rounds = int(drift_rounds)
        self.drift_scale  = float(drift_scale)
        self.hit_period   = int(hit_period)
        self.hit_scale    = float(hit_scale)
        self.tau_margin   = float(tau_margin)
    def _step_scale(self) -> float:
        self.round += 1
        if self.round <= self.drift_rounds:
            return self.drift_scale
        return self.hit_scale if (self.hit_period > 0 and (self.round % self.hit_period) == 0) else self.drift_scale
    def apply(self, deltas: np.ndarray, U: np.ndarray, tau: float, adv_idx) -> np.ndarray:
        X = deltas.copy().astype(np.float32, copy=False)
        if U is None or tau is None or getattr(U, "ndim", 0) != 2 or U.size == 0 or len(adv_idx) == 0:
            for i in adv_idx:
                if 0 <= i < X.shape[0]: X[i] = -3.0 * X[i]
            return X
        Z = X @ U
        benign = [j for j in range(Z.shape[0]) if j not in set(adv_idx)]
        if len(benign) == 0:
            for i in adv_idx:
                if 0 <= i < X.shape[0]: X[i] = -3.0 * X[i]
            return X
        mu_z = Z[benign].mean(axis=0, keepdims=True)
        nrm  = float(np.linalg.norm(mu_z)) + 1e-12
        z_dir = (-mu_z / nrm).astype(np.float32)
        if self.z_dir is None or self.z_dir.shape != z_dir.shape:
            self.z_dir = z_dir
        else:
            self.z_dir = (0.9 * self.z_dir + 0.1 * z_dir).astype(np.float32)
        scale = float(self._step_scale())
        steer = (scale * self.z_dir) @ U.T
        max_orth = float(tau) * self.tau_margin
        for i in adv_idx:
            if not (0 <= i < X.shape[0]): continue
            xi = X[i]
            xi_par  = (xi @ U) @ U.T
            xi_orth = xi - xi_par
            orth = float(np.linalg.norm(xi_orth))
            if orth > max_orth and orth > 0:
                xi_orth = xi_orth * (max_orth / orth)
            X[i] = (xi_par + xi_orth + steer[0]).astype(np.float32, copy=False)
        return X

# Training and aggregation
def local_train_and_delta(model_init_vec: np.ndarray, dataset_subset: Dataset, model_name: str, dataset_name: str, device: str,
                          epochs: int=1, batch: int=64, lr: float=0.02, seed:int=0) -> np.ndarray:
    set_all_seeds(seed)
    model = make_model(model_name, dataset_name).to(device)
    if str(device).startswith('cuda'):
        try:
            torch.cuda.set_device(int(str(device).split(':')[-1]))
        except Exception:
            torch.cuda.set_device(torch.device(str(device)))
    vector_to_model_parameters(model, model_init_vec, device)
    model.train()
    loader = DataLoader(
        dataset_subset,
        batch_size=batch,
        shuffle=True,
        num_workers=(int(os.environ.get("FL_DLOADER_WORKERS","0")) if str(device).startswith('cuda') else 0),
        pin_memory=str(device).startswith('cuda'),
        generator=make_torch_generator(seed),
        worker_init_fn=make_worker_init_fn(seed),
        persistent_workers=(str(device).startswith('cuda') and int(os.environ.get("FL_DLOADER_WORKERS","0"))>0)
    )
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x,y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad()
                loss = crit(model(x), y)
                loss.backward()
                opt.step()
    new_vec = model_parameters_to_vector(model)
    return (new_vec - model_init_vec).astype(np.float32)

def run_aggregator_with_timing(algo_name: str, agg_fn: Optional[callable], deltas: np.ndarray, ctx: Dict[str,Any]) -> Tuple[np.ndarray, Dict[str,Any]]:
    t0 = time.time()
    info = {}
    if algo_name == "SpectralKrum":
        server: SpectralKrum = ctx["sk_server"]
        out = server.step(deltas=deltas, global_weights=ctx["global_vec"], use_guard=ctx["use_guard"])
        agg_delta = out["new_global"] - ctx["global_vec"]
        info.update(out)
    else:
        kwargs = {"f": ctx.get("f_param", 1)}
        if algo_name == "DnC-Cluster" and "dnc_cfg" in ctx:
            kwargs["dnc_cfg"] = ctx["dnc_cfg"]
        agg_delta, base_info = agg_fn(deltas=deltas, **kwargs)
        info.update(base_info)
        t1 = time.time()
        t_total = (t1 - t0)*1e3
        if "time_ms_total" not in info:
            info.update({
                "time_ms_proj": 0.0, "time_ms_pairwise": 0.0, "time_ms_guard": 0.0,
                "time_ms_refresh": 0.0, "time_ms_other": max(0.0, t_total),
                "time_ms_total": t_total,
            })
    return agg_delta.astype(np.float32), info

# Attack harness
def _build_proxy_state(deltas: np.ndarray, f: int, r_dim: int, tau_q: float, seed: int) -> Tuple[Optional[np.ndarray], float]:
    m, d = deltas.shape
    if m <= max(f, 1): return None, 0.0
    center = np.median(deltas, axis=0, keepdims=True)
    dists  = np.linalg.norm(deltas - center, axis=1)
    k      = max(1, m - 2*f)
    keep   = np.argpartition(dists, kth=k-1)[:k]
    X      = deltas[keep]

    if X.shape[0] < 2:
        return None, 0.0

    r_use  = max(1, min(r_dim, d-1, X.shape[0]-1))
    if r_use <= 0: return None, 0.0

    if not np.isfinite(X).all():
        X = np.nan_to_num(X)

    pca = PCA(n_components=r_use, svd_solver="randomized", random_state=seed)
    try:
        Z = pca.fit_transform(X)
    except Exception:
        return None, 0.0
    U     = pca.components_.T.astype(np.float32)
    resid = np.linalg.norm(X - (Z @ U.T), axis=1)
    tau   = float(np.quantile(resid, tau_q))
    return U, tau

class _BufferPoisonerRuntime:
    def __init__(self, **kwargs):
        self.impl = BufferPoisoner(**kwargs)
    def apply(self, deltas: np.ndarray, U: np.ndarray, tau: float, adv_idx: List[int]) -> np.ndarray:
        return self.impl.apply(deltas, U, tau, adv_idx)

def apply_attack_fair(deltas: np.ndarray, attack_cfg: Dict[str, Any], attacker_idx: List[int], algo_name: str,
                      sk_server: Optional[SpectralKrum], f_param: int, tau_q: float, r_dim: int, seed: int,
                      attack_state: _BufferPoisonerRuntime) -> Tuple[np.ndarray, str]:
    X = deltas.copy()
    t = attack_cfg.get("type", "none")
    if t in ("none", "label", "semantic_backdoor") or len(attacker_idx) == 0:
        return X, "none"
    if t == "sign":
        scale = attack_cfg.get("scale", -3.0)
        for i in attacker_idx:
            if 0 <= i < X.shape[0]: X[i] = scale * X[i]
        return X, "none"
    if t == "min_max":
        m = X.shape[0]
        benign_idx = [j for j in range(m) if j not in set(attacker_idx)]
        if len(benign_idx) == 0:
            for i in attacker_idx: X[i] = -3.0 * X[i]
            return X, "fallback_no_benign"
        benign_updates = X[benign_idx]
        mu_benign = np.mean(benign_updates, axis=0)
        scale = float(attack_cfg.get("scale", 3.0))
        std_benign = np.std(benign_updates, axis=0)
        std_benign[std_benign < 1e-6] = 1e-6
        mal_update = (mu_benign - scale * std_benign).astype(np.float32, copy=False)
        for i in attacker_idx:
            X[i] = mal_update
        return X, "proxy_stats"

    used = "proxy"
    U_atk, tau_atk = None, 0.0
    if (algo_name == "SpectralKrum") and (sk_server is not None) and (getattr(sk_server, "U", None) is not None):
        U_atk   = sk_server.U
        tau_atk = float(getattr(sk_server, "_tau_buffer", 0.0))
        used = "true"
    if U_atk is None or tau_atk in (None, 0.0):
        U_atk, tau_atk = _build_proxy_state(X, f=f_param, r_dim=r_dim, tau_q=tau_q, seed=seed)
        used = "proxy"
    if U_atk is None or U_atk.ndim != 2 or U_atk.shape[1] == 0 or tau_atk in (None, 0.0):
        for i in attacker_idx: X[i] = -3.0 * X[i]
        return X, used
    if t == "steer":
        gamma = float(attack_cfg.get("gamma", 3.0))
        X = subspace_aware_attack(X, U_atk, float(tau_atk), attacker_idx, gamma=gamma)
        return X, used
    if t == "buffer":
        X = attack_state.apply(X, U_atk, float(tau_atk), attacker_idx)
        return X, used
    return X, used

def _make_attack_indices(m: int, count: int, rng: np.random.Generator) -> List[int]:
    if count <= 0: return []
    return rng.choice(m, size=min(count, m), replace=False).tolist()

def _prepare_label_flip_subsets(client_subsets, attack_cfg, attacker_idx):
    if attack_cfg.get("type") != "label" or len(attacker_idx) == 0:
        return client_subsets
    mapped = []
    for j, ds in enumerate(client_subsets):
        mapped.append(LabelMapSubset(ds, attack_cfg["map"]) if j in attacker_idx else ds)
    return mapped

def _prepare_semantic_backdoor_subsets(client_subsets, attack_cfg, attacker_idx, seed: int):
    if attack_cfg.get("type") != "semantic_backdoor" or len(attacker_idx) == 0:
        return client_subsets

    pr = float(attack_cfg.get("poison_rate", 0.1))
    tgt = int(attack_cfg.get("target", 0))
    tsz = int(attack_cfg.get("trigger_size", 3))
    tval= float(attack_cfg.get("trigger_value", 1.0))
    pos = str(attack_cfg.get("position", "br"))

    wrapped = []
    for j, ds in enumerate(client_subsets):
        if j in attacker_idx:
            client_seed = (int(seed) * 10_000 + j) % (2**32)
            wrapped.append(BackdoorSubset(ds, pr, tgt, tsz, tval, pos, seed=client_seed))
        else:
            wrapped.append(ds)
    return wrapped

# Simulation runner
def run_simulation_for_algo(
    seed, dataset_name, model_name, alpha,
    attack_name, base_cfg, attacker_count,
    f_param, use_guard, warmup_rounds, r_dim,
    tau_q, guard_min_kept, device
):
    set_all_seeds(seed)
    run_rows = []

    try:
        client_subsets, testset = get_client_datasets(dataset_name, num_clients=N_CLIENTS, alpha=alpha, seed=seed)
    except Exception as e:
        print(f"ERROR: [Worker {os.getpid()}] loading dataset {dataset_name}: {e}")
        return []

    global_model = make_model(model_name, dataset_name).to(device)
    global_vec   = model_parameters_to_vector(global_model)

    algo_name = base_cfg["algo_name"]
    agg_fn = None; sk_server = None
    if algo_name == "SpectralKrum":
        sk_cfg = SpectralKrumConfig(
            r=r_dim, buffer_size=50, center_mode="mean", trim_mode="two_sided", trim_frac=0.1,
            warmup_rounds=warmup_rounds, refresh_every=1,
            orthE_quantile_from_buffer=tau_q,
            guard_min_kept=guard_min_kept,
            f_byzantine=f_param, clip_norm=0.0, seed=seed
        )
        sk_server = SpectralKrum(sk_cfg)
    elif algo_name == "DnC-Cluster":
        agg_fn = dnc_agg
    elif algo_name == "DnC-PMF":
        agg_fn = dnc_pmf_agg
    elif algo_name == "Bulyan":
        agg_fn = bulyan_agg
    elif algo_name == "FullKrum":
        agg_fn = full_krum_agg
    elif algo_name == "MultiKrum":
        agg_fn = multi_krum_agg
    elif algo_name == "GeometricMedian":
        agg_fn = geometric_median_agg
    elif algo_name == "TrimmedMean":
        agg_fn = trimmed_mean_agg
    elif algo_name == "CoordMedian":
        agg_fn = coord_median_agg
    else:
        print(f"ERROR: [Worker {os.getpid()}] unknown aggregator {algo_name}")
        return []

    rng_global = np.random.default_rng(seed)
    fixed_attackers = _make_attack_indices(N_CLIENTS, attacker_count, rng_global)

    ds_for_run = client_subsets
    if attack_name == "label_flip":
        ds_for_run = _prepare_label_flip_subsets(client_subsets, base_cfg, fixed_attackers)
    elif attack_name == "semantic_backdoor":
        ds_for_run = _prepare_semantic_backdoor_subsets(client_subsets, base_cfg, fixed_attackers, seed)

    buffer_rt_params = {
        "drift_rounds": base_cfg.get("drift_rounds", 20),
        "drift_scale":  base_cfg.get("drift_scale", 0.15),
        "hit_period":   base_cfg.get("hit_period", 5),
        "hit_scale":    base_cfg.get("hit_scale", 1.5),
    }
    buffer_rt = _BufferPoisonerRuntime(**buffer_rt_params)

    best_acc = 0.0

    for round_idx in range(1, GLOBAL_ROUNDS + 1):
        deltas = []
        for j in range(N_CLIENTS):
            d = local_train_and_delta(
                model_init_vec=global_vec,
                dataset_subset=ds_for_run[j],
                model_name=model_name,
                dataset_name=dataset_name,
                device=device,
                epochs=LOCAL_EPOCHS,
                batch=BATCH_SIZE,
                lr=LR,
                seed=seed * 1000 + round_idx * 10 + j
            )
            deltas.append(d.astype(np.float32))
        deltas_np = np.stack(deltas, axis=0)

        adv_idx = fixed_attackers
        attacked_np, atk_state_used = apply_attack_fair(
            deltas=deltas_np,
            attack_cfg=base_cfg,
            attacker_idx=adv_idx,
            algo_name=algo_name,
            sk_server=sk_server,
            f_param=f_param,
            tau_q=tau_q,
            r_dim=r_dim,
            seed=seed + round_idx,
            attack_state=buffer_rt
        )

        ctx = {"global_vec": global_vec, "f_param": f_param, "sk_server": sk_server, "use_guard": use_guard, "adv_idx": adv_idx}
        if algo_name == "DnC-Cluster":
            base_min = max(3, N_CLIENTS // 4)
            cap_min  = max(3, N_CLIENTS - 2*f_param)
            min_size = min(base_min, cap_min)
            ctx["dnc_cfg"] = DnCConfig(pca_dim=r_dim, random_state=seed, min_cluster_size=min_size)
        delta, info = run_aggregator_with_timing(algo_name, agg_fn, attacked_np, ctx)

        global_vec = (global_vec + delta).astype(np.float32)
        vector_to_model_parameters(global_model, global_vec, device)
        acc = test_model(global_model, testset, device, seed=seed)
        if acc > best_acc:
            best_acc = acc

        asr_val = ""
        if attack_name == "semantic_backdoor":
            trig_size = int(base_cfg.get("trigger_size", 3))
            trig_val  = float(base_cfg.get("trigger_value", 1.0))
            target    = int(base_cfg.get("target", 0))
            asr = test_backdoor_asr(global_model, testset, device,
                                    target=target, trigger_size=trig_size,
                                    trigger_value=trig_val, position=str(base_cfg.get("position","br")),
                                    seed=seed)
            asr_val = float(asr)

        row = {
            "seed": seed, "dataset": dataset_name, "model": model_name,
            "alpha": alpha, "attack": attack_name, "attacker_count": attacker_count,
            "algo": algo_name, "f_param": f_param,
            "use_guard": use_guard, "warmup_rounds": warmup_rounds,
            "r": r_dim, "guard_min_kept": guard_min_kept, "tau_quantile": tau_q,
            "round": round_idx, "acc": acc, "asr": asr_val,
            "U_rank": info.get("U_rank", ""),
            "pre_guard_selected": info.get("pre_guard_selected", ""),
            "num_guard_kept": info.get("num_guard_kept", ""),
            "tau_buffer": info.get("tau_buffer", ""),
            "time_ms_total": info.get("time_ms_total", ""),
            "time_ms_proj": info.get("time_ms_proj", ""),
            "time_ms_pairwise": info.get("time_ms_pairwise", ""),
            "time_ms_guard": info.get("time_ms_guard", ""),
            "time_ms_refresh": info.get("time_ms_refresh", ""),
            "time_ms_other": info.get("time_ms_other", ""),
            "atk_state_used": atk_state_used,
            "atk_adv_idx": ";".join(map(str, adv_idx)),
            "fallback": info.get("fallback", ""),
            "bulyan_krum_fallback": info.get("bulyan_krum_fallback", ""),
            "guard_target": info.get("guard_target",""),
            "guard_dropped": info.get("guard_dropped",""),
            "guard_fallback_used": info.get("guard_fallback_used",""),
            "guard_tp": info.get("guard_tp",""),
            "guard_fp": info.get("guard_fp",""),
            "guard_kept_attackers": info.get("guard_kept_attackers",""),
            "resid_mean": info.get("resid_mean",""),
            "resid_p95": info.get("resid_p95",""),
        }
        if round_idx == GLOBAL_ROUNDS:
            row["final_acc"] = acc
            row["best_acc"] = best_acc
        run_rows.append(row)

    return run_rows

def pre_download_datasets():
    print("Pre-downloading CIFAR-10...")
    try:
        datasets.CIFAR10(root="./data", train=True, download=True)
        datasets.CIFAR10(root="./data", train=False, download=True)
        print("CIFAR-10 download complete.")
    except Exception as e:
        print(f"Could not pre-download CIFAR-10: {e}. Workers might fail.")

def _run_job(args):
    t0 = time.time()
    try:
        rows = run_simulation_for_algo(*args)
        dur = time.time() - t0
        return rows, dur
    except Exception as e:
        dur = time.time() - t0
        try:
            print(f"ERROR in job: {e}")
        except Exception:
            pass
        return [], dur


def main():
    multiprocessing.freeze_support()

    try:
        n_cores = int(max(1, min(PROC_TARGET, CORES)))
    except Exception:
        n_cores = 4

    print(f"===== STARTING OVERNIGHT RUN (Parallelized on {n_cores} cores) =====")
    print(f"Using device: {DEVICE}")
    print(f"Results  {RESULTS_CSV}")

    all_job_args: List[Tuple[Any, ...]] = []

    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        devs = [f"cuda:{i}" for i in range(max(1, torch.cuda.device_count()))]
    else:
        devs = [DEVICE]

    if RUN_MAIN_COMPARISON:
        for seed in SEEDS:
            for (dataset_name, model_name) in DATASETS:
                for alpha in ALPHAS:
                    for f_param in F_GRID_MAIN:
                        guard_min_kept = max(1, N_CLIENTS - 2*f_param - 2)
                        for attacker_count in ATTACKER_COUNTS_MAIN:
                            if attacker_count > 0 and f_param < attacker_count:
                                continue
                            if f_param > 0 and N_CLIENTS < 4*f_param + 3:
                                print(f"Skipping f={f_param} for Bulyan (n={N_CLIENTS})")
                                continue
                            for attack_name, base in BASE_ATTACKERS.items():
                                for algo_name in AGGREGATOR_NAMES:
                                    if algo_name == "Bulyan" and N_CLIENTS < 4*f_param + 3:
                                        continue
                                    job_cfg = base.copy()
                                    job_cfg["algo_name"] = algo_name
                                    selected_device = devs[len(all_job_args) % len(devs)]
                                    all_job_args.append((
                                        seed, dataset_name, model_name, alpha,
                                        attack_name, job_cfg, attacker_count,
                                        f_param, USE_GUARD_MAIN, WARMUP_MAIN, R_MAIN,
                                        TAU_Q_MAIN, guard_min_kept, selected_device
                                    ))

    if RUN_ABLATIONS:
        seed = SEEDS[0]
        dataset_name, model_name = ABLATION_DATASET
        alpha = ABLATION_ALPHA
        for f_param in F_GRID_ABL:
            guard_min_kept = max(1, N_CLIENTS - 2*f_param - 2)
            for attacker_count in ATTACKER_COUNTS_ABL:
                if attacker_count > 0 and f_param < attacker_count:
                    continue
                attack_name = "min_max"
                base_cfg = BASE_ATTACKERS[attack_name].copy()
                base_cfg["algo_name"] = "SpectralKrum"
                for use_guard in USE_GUARD_LIST_ABL:
                    for warmup_rounds in WARMUP_LIST_ABL:
                        for r_dim in R_LIST_ABL:
                            for tau_q in TAU_QS_ABL:
                                selected_device = devs[len(all_job_args) % len(devs)]
                                all_job_args.append((
                                    seed, dataset_name, model_name, alpha,
                                    attack_name, base_cfg, attacker_count,
                                    f_param, use_guard, warmup_rounds, r_dim,
                                    tau_q, guard_min_kept, selected_device
                                ))

    print(f"Total jobs: {len(all_job_args)}")
    if len(all_job_args) == 0:
        print("No jobs to run; exiting.")
        return

    limit_jobs = int(os.environ.get("FL_LIMIT_JOBS", "0") or "0")
    if limit_jobs > 0:
        all_job_args = all_job_args[:max(0, limit_jobs)]
        print(f"Limiting to {len(all_job_args)} jobs via FL_LIMIT_JOBS")

    if os.environ.get("FL_DRY_RUN", "0") == "1":
        print("Dry run only (FL_DRY_RUN=1); exiting before download/run.")
        return

    print("Ensuring datasets are downloaded...")
    pre_download_datasets()

    logger = CSVLogger(RESULTS_CSV)
    written = 0

    chunksize = max(1, int(len(all_job_args) / max(1, (n_cores * 4))))
    jobs_done = 0
    cum_dur = 0.0

    with multiprocessing.Pool(processes=n_cores) as pool:
        bar = tqdm(pool.imap_unordered(_run_job, all_job_args, chunksize=chunksize), total=len(all_job_args))
        for result_rows, dur in bar:
            jobs_done += 1
            cum_dur += max(0.0, float(dur))
            if result_rows:
                try:
                    logger.write_rows(result_rows)
                    written += len(result_rows)
                except Exception as e:
                    print(f"ERROR writing rows: {e}")
            avg = (cum_dur / jobs_done) if jobs_done > 0 else 0.0
            rem = max(0, len(all_job_args) - jobs_done)
            eta_sec = (rem * avg) / max(1, n_cores)
            h = int(eta_sec // 3600)
            m = int((eta_sec % 3600) // 60)
            bar.set_postfix_str(f"eta  {h}h {m}m")

    print("\n===== ALL JOBS COMPLETE =====")
    if written > 0:
        print(f"Wrote {written} rows  {RESULTS_CSV}")
    else:
        print("No rows produced; skipping analysis.")


if __name__ == "__main__":
    import sys

    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print("Usage: python final.py [--help] [--download-only] [--dry-run]")
        print("  --download-only  Download CIFAR-10 to ./data then exit")
        print("  --dry-run        Print job count then exit (no download/run)")
        raise SystemExit(0)

    if "--download-only" in sys.argv[1:]:
        pre_download_datasets()
        raise SystemExit(0)

    if "--dry-run" in sys.argv[1:]:
        os.environ["FL_DRY_RUN"] = "1"

    main()