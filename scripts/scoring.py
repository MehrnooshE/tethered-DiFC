"""
Scoring of full-scan candidates with `fusion_lr1e-4_wd_ls` (library; used by
08_clearance_cohort_split_overlap, 09_clearance_cohort_score and independent_scan (13, 14)).

Split overlap: which candidates are windows already in the train/val/test splits.
Some clearance-cohort scan folders contain recordings that were also labeled (e.g. the
pulsed B6LLC controls); every file with at least one such window is dropped from the
call counts. A candidate matches a split window (original windows only, no
augmentations) when their centres (+/-HW samples, both channels) agree up to a scale
and an offset: the joint standardization uses the whole 1 s window, so a peak near a
file boundary is standardized differently in the concatenated full scan than in the
zero-padded labeled window, but its centre is the same signal.

Scoring: 5-seed mean probability (Methods S3.7), spectral input standardized with the
TRAIN statistics.
"""
import os
import hashlib
import numpy as np
import pandas as pd

import common as mc
import fusion_model as s8

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

MODEL = mc.MODEL
HW = 100                 # centre half-width (samples) compared
MIN_SIM = 1 - 1e-6       # cosine similarity of the centred segments; non-matches are < 0.999


def centre(X):
    """(N, 2, 2001) -> (N, 2 * (2 HW + 1)) zero-mean, unit-norm centre segments."""
    v = X[:, :, mc.CENTER - HW:mc.CENTER + HW + 1].reshape(len(X), 2 * (2 * HW + 1)).astype(np.float64)
    v -= v.mean(1, keepdims=True)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def split_reference():
    R, split, scan = [], [], []
    for s in mc.SPLITS:
        X = np.load(os.path.join(mc.SPLITS_DIR, f"{s}_X.npz"))["X"]
        meta = pd.read_csv(os.path.join(mc.SPLITS_DIR, f"{s}_meta.csv"))
        keep = np.where(meta.augmentation.values == "none")[0]
        R.append(centre(np.asarray(X[keep])))
        split += [s] * len(keep)
        scan += list(meta.scan.values[keep])
    return np.concatenate(R), np.array(split, object), np.array(scan, object)


def overlap_table(sc, ref):
    """Best split match of every candidate of a prepared scan; ref = split_reference()."""
    R, r_split, r_scan = ref
    peaks, V = sc["peaks"], centre(sc["X"])
    sim = np.empty(len(V))
    j = np.empty(len(V), int)
    for a in range(0, len(V), 2048):
        S = V[a:a + 2048] @ R.T
        j[a:a + 2048] = S.argmax(1)
        sim[a:a + 2048] = S[np.arange(len(S)), j[a:a + 2048]]
    hit = sim >= MIN_SIM
    src = np.empty(len(peaks), dtype=object)
    for fname, s, e in sc["bounds"]:
        src[(peaks >= s) & (peaks < e)] = fname
    return pd.DataFrame({"peak_index": peaks, "source_file": src,
                         "matched_split": np.where(hit, r_split[j], ""), "matched_scan": np.where(hit, r_scan[j], ""),
                         "similarity": sim})


# ---------------------------------------------------------------- scoring ---
def load_scorer():
    """The 5 seeds and the spectral-input standardizer (fitted on TRAIN)."""
    Xtr, _, Mtr, _ = mc.load_split("train")
    zc = s8.Standardizer(s8.centre_spectrum(s8.als_scaled(Xtr, "train"), Mtr), skip=(2,))
    models = [tf.keras.models.load_model(mc.seed_path(k), compile=False) for k in range(5)]
    return models, zc


def score(models, zc, n, X, M, als_cache):
    """5-seed mean probability of each window of scan n (ALS windows cached in als_cache/)."""
    als = s8.als_scaled(X, f"{n}__{hashlib.md5(np.ascontiguousarray(X).tobytes()).hexdigest()[:12]}", als_cache)
    xin = [np.transpose(X, (0, 2, 1)).astype("float32"), zc(s8.centre_spectrum(als, M))]
    return np.mean([m.predict(xin, batch_size=512, verbose=0).ravel() for m in models], 0)


def files_in_splits(sc, o):
    """True for candidates from a file with at least one window in the splits; o = overlap_table
    (or None / a path to its CSV: no exclusion if missing)."""
    seen = np.zeros(len(sc["peaks"]), bool)
    if isinstance(o, str):
        o = pd.read_csv(o) if os.path.exists(o) else None
    if o is not None:
        if np.array_equal(o.peak_index.values, sc["peaks"]):
            bad = set(o[o.matched_split.fillna("") != ""].source_file)
            seen = o.source_file.isin(bad).values
    return seen


def candidate_table(sc, p, seen):
    src = np.empty(len(sc["peaks"]), dtype=object)
    for fname, a, b in sc["bounds"]:
        src[(sc["peaks"] >= a) & (sc["peaks"] < b)] = fname
    return pd.DataFrame({"peak_index": sc["peaks"], "time_h": sc["peaks"] / mc.FS / 3600, "source_file": src,
                         "at_laser_edge": sc["edge"], "file_in_splits": seen, f"p_{MODEL}": p})


def summary_row(sc, p, seen):
    """Per-scan counts at the cut-off; candidates from files in the splits (seen) are left out of
    every count and of the laser-ON hours. Laser-edge candidates are included."""
    det = (p >= mc.CUT) & ~seen
    on_h = sc["info"]["on_hours"] * (1 - seen.mean())
    return {"hours_recorded": round(sc["total"] / mc.FS / 3600, 3), "laser_on_hours": round(on_h, 3),
            "candidates": int((~seen).sum()), "candidates_in_splits": int(seen.sum()),
            "detected_peaks": int(det.sum()), "detected_not_at_laser_edge": int((det & ~sc["edge"]).sum()),
            "near_miss": int(((p >= mc.NEAR_MISS) & (p < mc.CUT) & ~seen).sum()),
            "peaks_per_laser_on_hour": round(det.sum() / on_h, 2) if on_h else np.nan}
