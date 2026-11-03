"""
Full-scan preprocessing for data/<...>-raw/<scan>/*.mat (library;
used by 08_clearance_cohort_split_overlap, 09_clearance_cohort_score and independent_scan (13, 14);
not run on its own).

Per scan folder:
  1. Order .mat files chronologically by the start-hour suffix in the name
     (e.g. _17, _19, ... or _0, _2). The 'time' vector inside each file
     restarts at 0, so it can't be used for ordering.
  2. Preprocess each file on its own exactly as 02_preprocess_labeled_scans.py did for
     the training data: pulsed -> Otsu ON/OFF detection, per-ON-region
     rolling-median background subtraction + adaptive-threshold ch1 peak
     detection, OFF regions zero-filled; continuous -> one ON region. Files
     are placed end to end (sample-contiguous; any wall-clock gap between
     files is not represented).
  3. Windows: same extract_window / compute_invalid_mask / standardize_joint
     as 04_build_splits.py (signal_functions.py), copied below.
  4. Cache everything in data/<...>-preprocessed/<scan>/ (default: the clearance cohort).

unpack_new_zips() first extracts new recordings (zips in a *-raw/ folder).
"""
import os
import re
import glob
import json
import time
import shutil
import zipfile
import numpy as np
import scipy.io

import common as mc
import peak_detection as pp
import parameters as var

PRED_ROOT = mc.COHORT_RAW_DIR
PREP_ROOT = mc.COHORT_PREP_DIR
PREP_METHOD = "per_file_v3"   # each .mat preprocessed on its own (own Otsu), as in 02_preprocess_labeled_scans.py;
                              # v3: ON regions shorter than the 2 s background window treated as OFF
WIN_LEN = 2001
ON_OVERSHOOT, OFF_OVERSHOOT, OTSU_BINS = 5, 5, 20   # as in 02_preprocess_labeled_scans.py
EDGE_MS = 25          # candidates this close to a laser ON/OFF edge are flagged
                      # (onset transients; 02_preprocess_labeled_scans skips only 5 samples)


# ---------------------------------------------------------------- loading ---
def ordered_mat_files(folder):
    files = glob.glob(os.path.join(folder, "*.mat"))

    def key(f):
        m = re.search(r"_(\d+)(?:_F1)?\.mat$", os.path.basename(f))
        return (int(m.group(1)) if m else -1, os.path.basename(f))
    files = sorted(files, key=key)
    dates = {re.match(r"\s*(\d{6})", os.path.basename(f)).group(1) if re.match(r"\s*(\d{6})", os.path.basename(f))
             else None for f in files}
    if len(dates) > 1:
        raise RuntimeError(f"{folder} mixes recording dates {sorted(map(str, dates))}; "
                           f"put each session in its own folder")
    if len(files) > 1:
        keys = [key(f)[0] for f in files]
        if -1 in keys or len(set(keys)) != len(keys):
            raise RuntimeError(f"Cannot infer chronological order for {folder}: {files}")
    return files


# ---------------------------------------------------------- preprocessing ---
def preprocess_pulsed(data):
    """Same steps as 02_preprocess_labeled_scans.preprocess_pulsed_file."""
    fs = var.fs
    otsu = pp.OtsuThreshold(data["Ch1"], bins_num=OTSU_BINS)
    starts, stops = pp.pulse_start_end_index(data, otsu, fs, var.background_sliding_window,
                                             ON_OVERSHOOT, OFF_OVERSHOOT)
    sig1 = np.zeros(len(data))
    sig2 = np.zeros(len(data))
    peaks = []
    prev_stop = 0
    lens = []
    n_short = 0
    for s_idx, e_idx in zip(starts, stops):
        assert s_idx >= prev_stop, "overlapping ON regions"
        if e_idx - s_idx < var.background_sliding_window:
            # e.g. a pulse cut off by the end of a file: pulse_start_end_index
            # only drops short pulses with both edges, and the rolling-median
            # background is undefined (all NaN) on a region shorter than its
            # window -> treat as laser-OFF (stays zero, i.e. masked)
            n_short += 1
            continue
        bs1, bs2, pk, _ = pp.PreProcessing(data[s_idx:e_idx], fs, var.background_sliding_window,
                                           var.rel_thresh, var.min_peak, var.max_peak,
                                           var.prominence_factor)
        sig1[s_idx:e_idx] = np.asarray(bs1)
        sig2[s_idx:e_idx] = np.asarray(bs2)
        peaks.extend(np.asarray(pk, dtype=int) + s_idx)
        prev_stop = e_idx
        lens.append(e_idx - s_idx)
    if not (np.isfinite(sig1).all() and np.isfinite(sig2).all()):
        raise RuntimeError("non-finite values after preprocessing")
    info = {"otsu_threshold": float(otsu), "n_on_regions": len(lens), "n_short_on_dropped": n_short,
            "on_len_s_median": float(np.median(lens) / fs) if lens else np.nan,
            "on_len_s_min": float(np.min(lens) / fs) if lens else np.nan,
            "on_hours": float(np.sum(lens) / fs / 3600)}
    return sig1, sig2, np.array(sorted(peaks), dtype=int), info


def preprocess_regular(data):
    bs1, bs2, pk, _ = pp.PreProcessing(data, var.fs, var.background_sliding_window, var.rel_thresh,
                                       var.min_peak, var.max_peak, var.prominence_factor)
    return np.asarray(bs1), np.asarray(bs2), np.array(sorted(pk), dtype=int), {
        "on_hours": len(data) / var.fs / 3600}


def is_pulsed(ch1):
    """Laser pulsing gives a bimodal ch1 with ~20% ON time; require many
    ON transitions at the Otsu split to call it pulsed. A continuous scan
    whose ch1 decays (photobleaching) is bimodal too, and chatters across the
    split where the decay crosses it, so also require ON and OFF samples to
    be mixed within most 1-min blocks (1.00 in pulsed files, 0.05 in the
    continuous MC38 scan)."""
    thr = pp.OtsuThreshold(ch1, bins_num=OTSU_BINS)
    on = ch1 > thr
    n_on = int(np.sum(np.diff(on.astype(np.int8)) == 1))
    hours = len(ch1) / var.fs / 3600
    blk = 60 * var.fs
    fr = on[:len(on) // blk * blk].reshape(-1, blk).mean(1)
    mixed = np.mean((fr > 0.02) & (fr < 0.98)) if len(fr) else 1.0
    return (0.05 < on.mean() < 0.6) and n_on > 20 * hours and mixed >= 0.5, on.mean(), n_on


def preprocess_per_file(files, pulsed=None):
    """Preprocess each .mat file on its own -- own pulsed/regular decision and
    own Otsu ON/OFF threshold -- exactly as 02_preprocess_labeled_scans.py did for the
    training data, then place the results end to end in chronological order.
    A single threshold over a whole night fails when the signal level drifts
    (e.g. ch1 ON level 1500 mV early vs 250 mV later in the L1210A scans).
    pulsed: None = decided per file by is_pulsed (cohort, pulsed exploratory scans), False = every
    file is a continuous recording (13_continuous_exploratory_scan), True = every file is pulsed."""
    lengths = [dict((n, s) for n, s, _ in scipy.io.whosmat(f))["data"][0] for f in files]
    total = int(sum(lengths))
    sig1 = np.zeros(total)
    sig2 = np.zeros(total)
    peaks, bounds, per_file = [], [], []
    start = 0
    for f, n in zip(files, lengths):
        data = pp.load_scan_data(f)
        if data is None or len(data) != n:
            raise RuntimeError(f"failed to load {f}")   # fail fast: no partial scans
        detected, on_frac, n_on = is_pulsed(data["Ch1"].to_numpy())
        file_pulsed = detected if pulsed is None else pulsed
        s1, s2, pk, inf = (preprocess_pulsed if file_pulsed else preprocess_regular)(data)
        sig1[start:start + n] = s1
        sig2[start:start + n] = s2
        peaks.extend(np.asarray(pk, dtype=int) + start)
        bounds.append((os.path.basename(f), start, start + n))
        per_file.append({"file": os.path.basename(f), "pulsed": bool(file_pulsed), "on_fraction": float(on_frac),
                         "on_transitions": n_on, **inf})
        start += n
        del data
    pulsed_files = [p for p in per_file if p["pulsed"]]
    info = {"pulsed": bool(pulsed_files), "n_pulsed_files": len(pulsed_files), "n_files": len(files),
            "n_on_regions": int(sum(p.get("n_on_regions", 0) for p in per_file)),
            "on_len_s_min": float(min(p["on_len_s_min"] for p in pulsed_files)) if pulsed_files else np.nan,
            "on_hours": float(sum(p["on_hours"] for p in per_file)),
            "hours_recorded": total / var.fs / 3600, "per_file": per_file}
    return sig1, sig2, np.array(sorted(peaks), dtype=int), info, bounds, total


# -------------------- windowing (verbatim logic from functions.py) ----------
def compute_invalid_mask(win, zero_tol=1e-9):
    return (np.abs(win[0]) < zero_tol) & (np.abs(win[1]) < zero_tol)


def standardize_joint(win, invalid_mask=None):
    if invalid_mask is not None and invalid_mask.sum() < win.shape[-1]:
        valid = ~invalid_mask
        flat = np.concatenate([win[0, valid], win[1, valid]])
    else:
        flat = win.reshape(-1)
    mean = flat.mean()
    std = flat.std()
    if std < 1e-12:
        std = 1e-12
    return (win - mean) / std


def extract_window(ch1, ch2, peak_idx, window_len):
    half_left = window_len // 2
    half_right = window_len - half_left
    T = len(ch1)
    start, end = peak_idx - half_left, peak_idx + half_right
    pad_left = max(0, -start)
    a, b = max(0, start), min(T, end)
    win = np.zeros((2, window_len), dtype=np.float64)
    if a < b:
        win[0, pad_left:pad_left + b - a] = ch1[a:b]
        win[1, pad_left:pad_left + b - a] = ch2[a:b]
    return win


def edge_distance_ms(M):
    """Distance from the candidate (window centre) to the nearest invalid
    (OFF-gap / padding) sample on either side, in ms; inf if none in window."""
    c = WIN_LEN // 2
    out = np.full(len(M), np.inf)
    for i, m in enumerate(M):
        left = np.where(m[:c])[0]
        right = np.where(m[c + 1:])[0]
        d = min(c - left[-1] if len(left) else np.inf, right[0] + 1 if len(right) else np.inf)
        out[i] = d / var.fs * 1e3
    return out


def build_windows(sig1, sig2, peaks):
    raw = np.empty((len(peaks), 2, WIN_LEN), dtype=np.float32)
    X = np.empty_like(raw)
    M = np.empty((len(peaks), WIN_LEN), dtype=bool)
    for i, p in enumerate(peaks):
        w = extract_window(sig1, sig2, int(p), WIN_LEN)
        m = compute_invalid_mask(w)
        raw[i], M[i], X[i] = w, m, standardize_joint(w, invalid_mask=m)
    return raw, X, M


# ------------------------------------------------ new data + caching ---
def _session_date(fname):
    m = re.match(r"\s*(\d{6})", os.path.basename(fname))
    return m.group(1) if m else "nodate"


def unpack_new_zips(root=PRED_ROOT, min_free_gb=3.0):
    """Extract .mat files from every zip in `root` into scan folders.
    One recording date -> <zip stem>/; several dates -> <zip stem>_<date>/,
    so different sessions are never concatenated. Only .mat basenames are
    written (no paths from the archive). Files already present with the
    same size are skipped. Returns the scan folder names touched."""
    touched = []
    for z in sorted(glob.glob(os.path.join(root, "*.zip"))):
        with zipfile.ZipFile(z) as zf:
            members = [m for m in zf.infolist() if not m.is_dir()
                       and m.filename.lower().endswith(".mat") and ".ipynb_checkpoints" not in m.filename]
            dates = sorted({_session_date(m.filename) for m in members})
            stem = os.path.basename(z)[:-4]
            target = {d: stem if len(dates) == 1 else f"{stem}_{d}" for d in dates}
            todo = []
            for m in members:
                dest = os.path.join(root, target[_session_date(m.filename)], os.path.basename(m.filename))
                if not (os.path.exists(dest) and os.path.getsize(dest) == m.file_size):
                    todo.append((m, dest))
            if not todo:
                continue
            need = sum(m.file_size for m, _ in todo) / 1e9
            free = shutil.disk_usage(root).free / 1e9
            if free - need < min_free_gb:
                raise RuntimeError(f"{z}: needs {need:.1f} GB, only {free:.1f} GB free")
            for m, dest in todo:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with zf.open(m) as src, open(dest + ".part", "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 22)
                os.replace(dest + ".part", dest)
            touched += sorted(set(target.values()))
            print(f"unzipped {os.path.basename(z)}: {len(todo)} file(s) -> {sorted(set(target.values()))}")
    return touched


def scan_folders(root=PRED_ROOT):
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)) and glob.glob(os.path.join(root, d, "*.mat")))


MC38_TAG = "MC38"     # MC38 scans (.mat names contain "MC38-" / "MC38_") are never in the cohort


def is_mc38(name, root=PRED_ROOT):
    return any(re.search(rf"{MC38_TAG}[-_]", os.path.basename(f)) for f in glob.glob(os.path.join(root, name, "*.mat")))


def cohort_scans(names=None, root=PRED_ROOT):
    """The given scan names, or every cohort scan (parameters.clearance_cohort), sorted by name.
    Every name must be listed in parameters.clearance_cohort, be a clearance-cohort-raw scan
    folder and not be an MC38 scan."""
    names = list(names) if names else sorted(mc.COHORT_CELL_LINE)
    missing = [n for n in names if n not in mc.COHORT_CELL_LINE]
    if missing:
        raise SystemExit(f"{missing} not in the cohort; add them to `clearance_cohort` in parameters.py")
    absent = [n for n in names if not glob.glob(os.path.join(root, n, "*.mat"))]
    if absent:
        raise SystemExit(f"{absent}: no .mat files in {root}/<scan>/ (parameters.clearance_cohort)")
    mc38 = [n for n in names if is_mc38(n, root)]
    if mc38:
        raise SystemExit(f"{mc38}: MC38 scans are not part of the cohort; analyse each with 13_continuous_exploratory_scan.py")
    return names


def prepare_scan(name, pred_root=PRED_ROOT, prep_root=PREP_ROOT, pulsed=None):
    """Preprocess the scan pred_root/<name>/ once and cache it in prep_root/<name>/:
      <name>_preprocessed.npz  background-subtracted ch1, ch2 (float32,
                               compressed; OFF regions are zero) + peaks
      <name>_windows.npz       raw (mV), X (jointly standardized), M, peaks,
                               edge flag -- what every model consumes
      <name>_meta.json         file order/sizes, boundaries, preprocessing info
    pulsed: see preprocess_per_file (None = detected per file).
    The cache is reused only if the folder's .mat files and the pulsed mode are unchanged."""
    mode = {None: "auto", False: "continuous", True: "pulsed"}[pulsed]
    files = ordered_mat_files(os.path.join(pred_root, name))
    if not files:
        raise SystemExit(f"no .mat files in {os.path.join(pred_root, name)} (if it is on the NAS: is it still connected?)")
    sig = [(os.path.basename(f), os.path.getsize(f)) for f in files]
    d = os.path.join(prep_root, name)
    paths = {k: os.path.join(d, f"{name}_{k}") for k in ("preprocessed.npz", "windows.npz", "meta.json")}
    if all(os.path.exists(p) for p in paths.values()):
        with open(paths["meta.json"]) as f:
            meta = json.load(f)
        if ([tuple(x) for x in meta["files"]] == sig and meta.get("method") == PREP_METHOD
                and meta.get("mode", "auto") == mode):
            w = np.load(paths["windows.npz"])
            return {**{k: w[k] for k in w.files}, "bounds": [tuple(b) for b in meta["bounds"]],
                    "total": meta["total_samples"], "info": meta["info"], "files": files, "cached": True}

    t0 = time.time()
    print(f"\n=== {name}: preprocessing {len(files)} file(s) individually: {[s[0] for s in sig]}")
    sig1, sig2, peaks, info, bounds, total = preprocess_per_file(files, pulsed=pulsed)
    for p in info["per_file"]:
        flag = "" if p["pulsed"] and 0.15 <= p["on_fraction"] <= 0.3 else "   <-- check"
        print(f"  {p['file']}: pulsed={p['pulsed']} ON fraction {p['on_fraction']:.2f}, "
              f"{p['on_transitions']} ON transitions, {p.get('n_on_regions', '-')} ON regions kept{flag}")
    raw, X, M = build_windows(sig1, sig2, peaks)
    edge = edge_distance_ms(M) <= EDGE_MS
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(paths["preprocessed.npz"], ch1=sig1.astype(np.float32), ch2=sig2.astype(np.float32),
                        peaks=peaks, fs=var.fs)
    np.savez(paths["windows.npz"], raw=raw, X=X, M=M, peaks=peaks, edge=edge)
    with open(paths["meta.json"], "w") as f:
        json.dump({"method": PREP_METHOD, "mode": mode, "files": sig, "bounds": bounds, "total_samples": total,
                   "info": info}, f, indent=1)
    print(f"  {len(peaks)} peak candidates; cached to {d} ({time.time() - t0:.0f}s)")
    return {"raw": raw, "X": X, "M": M, "peaks": peaks, "edge": edge, "bounds": bounds,
            "total": total, "info": info, "files": files, "cached": False}
