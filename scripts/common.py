"""
Shared paths and utilities for the fusion_lr1e-4_wd_ls CTC-vs-artifact window classifier.

Data: data/train-val-test/Joint-Standardization-Augmented/{split}_X.npz (key X, compressed),
      {split}_{y,mask}.npy, {split}_meta.csv
  X    (N, 2, 2001) float32, jointly standardized (ch1 = green fluorescence,
       ch2 = >700 nm autofluorescence/motion reference), candidate peak at
       index 1000, fs = 2000 Hz.
  y    (N,) 1 = CTC, 0 = artifact.
  mask (N, 2001) bool, True = zero-padding / pulsed off-gap.

Everything here is deterministic given the inputs.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

import parameters

MODEL = "fusion_lr1e-4_wd_ls"      # saved-model prefix and probability column
CUT = parameters.cut_off           # probability cut-off for a detected peak (set in parameters.py)
NEAR_MISS = 0.5                    # rejected candidates with p >= NEAR_MISS are near misses
COHORT = parameters.clearance_cohort   # {cell line: [clearance-cohort-raw scan folders]} of steps 08-12
COHORT_CELL_LINE = {s: cl for cl, scans in COHORT.items() for s in scans}
assert len(COHORT_CELL_LINE) == sum(len(v) for v in COHORT.values()), "a scan is listed twice in parameters.clearance_cohort"

# ---- paths ---------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")
RESULTS = os.path.join(ROOT, "results")

# data/: inputs (and caches derived from them)
RAW_DIR = os.path.join(DATA, "raw-data")                        # {Injected,Control}/*.mat
SELECTED_DIR = os.path.join(DATA, "selected-data")              # {Injected,Control}/{Pulsed,Regular}/*.mat
PREPROCESSED_DIR = os.path.join(DATA, "preprocessed-data")      # {Injected,Control}/{Pulsed,Regular}/<scan>/
PREVIOUS_LABELS_DIR = os.path.join(DATA, "previous-labels")     # Injected/<scan>.csv (earlier labeling)
LABELS_DIR = os.path.join(DATA, "labeled-data")                 # {Injected,Control}/{Pulsed,Regular}/<scan>.csv
SPLITS_DIR = os.path.join(DATA, "train-val-test", "Joint-Standardization-Augmented")
SPLITS_ALS_DIR = os.path.join(DATA, "train-val-test-als-cache")   # ALS + Ch2-scaled split windows (fusion_model.als_scaled)
# clearance cohort (steps 08-12): only the scans of parameters.clearance_cohort
COHORT_RAW_DIR = os.path.join(DATA, "clearance-cohort-raw")                # <scan>/*.mat (+ new zips)
COHORT_PREP_DIR = os.path.join(DATA, "clearance-cohort-preprocessed")      # <scan>/ cache of full_scan_preprocessing
COHORT_ALS_DIR = os.path.join(DATA, "clearance-cohort-als-cache")          # <scan>__<hash>_als.npy
# exploratory scans, one scan per run, never grouped with the cohort or each other
PULSED_RAW_DIR = os.path.join(DATA, "pulsed-exploratory-raw")              # <scan>/*.mat (+ zips), 14_pulsed_exploratory_scan.py
PULSED_PREP_DIR = os.path.join(DATA, "pulsed-exploratory-preprocessed")
PULSED_ALS_DIR = os.path.join(DATA, "pulsed-exploratory-als-cache")
CONTINUOUS_RAW_DIR = os.path.join(DATA, "continuous-exploratory-raw")      # <scan>/*.mat (+ zips), 13_continuous_exploratory_scan.py
CONTINUOUS_PREP_DIR = os.path.join(DATA, "continuous-exploratory-preprocessed")
CONTINUOUS_ALS_DIR = os.path.join(DATA, "continuous-exploratory-als-cache")

# models/: trained seeds, validation/test predictions and metrics, training history, model figures
MODEL_FIG_DIR = os.path.join(MODELS, "figures")

# results/clearance-cohort/: outputs of steps 08-12 (the scans in parameters.clearance_cohort)
COHORT_DIR = os.path.join(RESULTS, "clearance-cohort")
COHORT_SUMMARY = os.path.join(COHORT_DIR, "scan_summary.csv")   # one row per scan (09)
PER_SCAN_DIR = os.path.join(COHORT_DIR, "per-scan")              # <scan>/ split_overlap.csv (08), candidates.csv (09), figures (10)
BARCODES_DIR = os.path.join(COHORT_DIR, "barcodes")
CLEARANCE_DIR = os.path.join(COHORT_DIR, "clearance")
# results/: exploratory scans, <scan>/ each; read by no other step
PULSED_DIR = os.path.join(RESULTS, "pulsed-exploratory")         # 14_pulsed_exploratory_scan.py
CONTINUOUS_DIR = os.path.join(RESULTS, "continuous-exploratory")  # 13_continuous_exploratory_scan.py


def out_dir(*parts):
    """os.path.join(*parts), created."""
    d = os.path.join(*parts)
    os.makedirs(d, exist_ok=True)
    return d


def per_scan(name, fname):
    """results/clearance-cohort/per-scan/<name>/<fname>."""
    return os.path.join(PER_SCAN_DIR, name, fname)


def seed_path(k):
    return os.path.join(MODELS, f"{MODEL}_seed{k}.keras")

FS = 2000
CENTER = 1000
SPLITS = ["train", "val", "test"]

# Prevalence of CTCs among ALL peak candidates in each split BEFORE the
# eval sets were balanced by random subsampling (from split_report.txt:
# CTC / (CTC + artifact) original windows). Used for prevalence-adjusted PPV.
POOL_PREVALENCE = {"val": 234 / (234 + 1994), "test": 433 / (433 + 2300)}

COLORS = {"ctc": "#2a78d6", "artifact": "#eb6834", "third": "#1baf7a",
          "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df"}


def load_split(split):
    X = np.load(os.path.join(SPLITS_DIR, f"{split}_X.npz"))["X"]
    y = np.load(os.path.join(SPLITS_DIR, f"{split}_y.npy"))
    mask = np.load(os.path.join(SPLITS_DIR, f"{split}_mask.npy"))
    meta = pd.read_csv(os.path.join(SPLITS_DIR, f"{split}_meta.csv"))
    assert (meta["label"].values == y).all()
    scans = pd.read_csv(os.path.join(SPLITS_DIR, "scan_assignment.csv"))
    cl = scans.drop_duplicates(["scan", "category"]).set_index(["scan", "category"])["cell_line"]
    meta["cell_line"] = [cl.get((s, c), "unknown") for s, c in zip(meta.scan, meta.category)]
    return X, y, mask, meta


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def metrics_at(y, p, thr=0.5, prevalence=None):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    out = dict(
        n=len(y), tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        auc=roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        ap=average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        acc=(tp + tn) / len(y), sens=sens, spec=spec, ppv=prec,
        f1=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
        bal_acc=np.nanmean([sens, spec]),
    )
    if prevalence is not None:
        q = prevalence
        den = sens * q + (1 - spec) * (1 - q)
        out["ppv_at_pool_prev"] = sens * q / den if den > 0 else np.nan
    return out


def cluster_bootstrap(y, p, groups, stat_fn, n_boot=2000, seed=0):
    """Resample SCANS with replacement (windows from one scan are not
    independent). Returns (point, lo95, hi95). Resamples that lose a class
    are skipped for stats that need both classes."""
    rng = np.random.default_rng(seed)
    g = np.asarray(groups)
    uniq = np.unique(g)
    by = {u: np.where(g == u)[0] for u in uniq}
    point = stat_fn(y, p)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([by[u] for u in pick])
        try:
            v = stat_fn(y[ii], p[ii])
        except ValueError:
            continue
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return point, lo, hi


def stat(name, thr=0.5):
    def f(y, p):
        if name == "auc":
            if len(np.unique(y)) < 2:
                raise ValueError
            return roc_auc_score(y, p)
        yhat = p >= thr
        if name == "sens":
            return yhat[y == 1].mean() if (y == 1).any() else np.nan
        if name == "spec":
            return (~yhat[y == 0]).mean() if (y == 0).any() else np.nan
        if name == "bal_acc":
            if len(np.unique(y)) < 2:
                raise ValueError
            return 0.5 * (yhat[y == 1].mean() + (~yhat[y == 0]).mean())
        raise KeyError(name)
    return f
