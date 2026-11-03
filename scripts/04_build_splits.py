"""
Step 4. Build train/val/test splits for the peak-candidate window classification task
(Methods S1.3, S2), written to data/train-val-test/Joint-Standardization-Augmented/.

Pipeline (per the agreed spec):
  1. Discover scans under data/preprocessed-data/{Injected,Control}/{Pulsed,Regular}/<scan>/
     with matching label CSVs under data/labeled-data/....
     Injected scans are only kept if their category/setup/scan path contains
     one of INJECTED_INCLUDE_SUBSTRINGS ("MM", "LLC", "L1210A"; case-sensitive).
     Control scans are not filtered.
  2. ALL FOUR categories (Injected/Pulsed, Injected/Regular, Control/Pulsed,
     Control/Regular) now use the SAME assignment procedure, each processed
     INDEPENDENTLY:
       - sort scans descending by their count
            Injected -> number of label==1 peaks (CTCs)
            Control  -> number of label==0 peaks (artifacts / peak candidates)
       - round-robin assign scans to train/test/val (I, I+1, I+2 -> train, test, val, repeat)
       - track cumulative counts per split against absolute-tolerance targets
         (train 70%+-10% -> [60,80]% of that category's total count,
          test  20%+-10% -> [10,30]%,
          val   10%+-10% -> [0,20]%)
       - if adding a scan's count would push its intended split's cumulative
         count above the split's upper bound, the scan is instead assigned to
         TRAIN. ALL of that scan's windows are kept. Only the running
         BOOKKEEPING counter is incremented by 10% of that scan's count.
         Every downstream step uses the ACTUAL contents of each split.
       - the realized (actual) per-split counts and fractions are printed so
         you can see how close each category landed to 70/20/10.
  3. Windows are extracted as 2001-sample slices (peak at index 1000),
     zero-padded at signal edges.
  4. Per split: gather all label-1 windows and all label-0 windows (from every
     scan assigned to that split, regardless of category/setup), standardize
     each window jointly across ch1+ch2 (mask-aware).
  5. TRAIN split only -- augmentation + balancing:
       - CTCs (label-1, N1 originals) are augmented by up to 25%
         (flip:smooth in a 13:12 ratio, disjoint source windows).
       - Artifacts (label-0, N0 originals) are matched to the CTC count:
           * if N0 >= CTC count -> randomly sample that many originals
           * if N0 <  CTC count -> use all originals + augment artifacts,
             capped at 25% of N0.
       - The final per-class count is min(N1 + aug_cap(N1), N0 + aug_cap(N0)),
         so neither class is ever augmented beyond 25% of its own originals.
         If artifacts can't reach the full augmented CTC count even with their
         25% augmentation, CTC augmentation is reduced first; CTC originals
         are only subsampled if N0 * 1.25 < N1. Both cases print a [WARN].
  6. TEST and VAL splits -- NO augmentation at all. Balanced by randomly
     subsampling the larger class down to the size of the smaller class.
     If this drops CTCs, a [WARN] is printed.
  7. Concatenate, shuffle (seeded), and save {split}_X.npz (M,2,2001; key X, compressed so
     that no file exceeds GitHub's 100 MB limit),
     {split}_y.npy (M,), {split}_mask.npy (M,2001), {split}_meta.csv.
  8. Write a summary report (printed + {OUT_DIR}/split_report.txt) and a
     per-scan table ({OUT_DIR}/scan_assignment.csv) covering, per split:
     scans by type (Injected = CTC-bearing, Control), original CTC/artifact
     counts, final saved counts (incl. augmentation), injected scans per
     cell line (MM / LLC / L1210A), and recording hours (Injected vs Control).
     Hours need SAMPLE_RATE_HZ to be set; until then only sample counts are
     reported. Two durations are reported: "recorded" = len(ch1) samples,
     "on-time" = samples where ch1 and ch2 are not both exactly zero (i.e.
     excluding the Pulsed off-gaps, using the same definition as the mask).

ASSUMPTIONS CALLED OUT EXPLICITLY (flagged, not silently baked in):
  - label 's' and '-1' rows are dropped entirely.
  - "Time(Hz)" column is a SAMPLE INDEX into the ch1/ch2 arrays.
  - Control scans are sorted by label-0 count. If a control CSV contains
    label-1 rows (unexpected), they are kept and noted, but they don't
    affect the control sort order or tolerance bookkeeping.
  - The round-robin + upper-bound rule does NOT enforce the lower bound of
    any split (e.g. train >= 60%). This was already true for the Injected
    procedure and now also applies to Control. Check the printed realized
    fractions.
  - Balancing artifacts vs CTCs is done on the POOLED label-0 windows of a
    split (Control artifacts + Injected-scan label-0 windows), as before.
  - INJECTED_INCLUDE_SUBSTRINGS matching is case-sensitive, so "MM" will
    not match "mm" (avoids accidental hits like "20mm").
  - Ties in the descending sort are broken by scan name for reproducibility.

MASK-AWARE STANDARDIZATION:
  - extract_window zero-pads out-of-bounds regions; Pulsed scans also have
    genuine interior zero-gaps. signal_functions.compute_invalid_mask catches both,
    computed on the raw window BEFORE standardization.
  - standardize_joint computes mean/std from valid (non-masked) samples only.
  - The mask is saved per split ({split}_mask.npy, bool).
  - Augmentation propagates the mask: flip reverses it; smoothing keeps it
    as-is (SMOOTH_WIN=3 blurs ~1 sample at on/off transitions; accepted
    approximation).

Run:
    python 04_build_splits.py
"""

import os
import csv
import numpy as np
import pandas as pd

import common as mc
from signal_functions import compute_invalid_mask, standardize_joint, extract_window

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
BASE_PREP = mc.PREPROCESSED_DIR
BASE_LABEL = mc.LABELS_DIR
OUT_DIR = mc.SPLITS_DIR

HALF_WIN = 1000
WIN_LEN = 2 * HALF_WIN + 1  # 2001

SMOOTH_WIN = 3
RANDOM_SEED = 42

# Max augmentation per class = FRAC_AUGMENT_TOTAL of that class's originals.
# Augmented copies are split flip:smooth in the FRAC_FLIP:FRAC_SMOOTH ratio.
FRAC_AUGMENT_TOTAL = 0.25
FRAC_FLIP = 0.13
FRAC_SMOOTH = 0.12
# FRAC_SUPPRESS_CH2 = 0.1
CH2_SUPPRESS_SCALE_LOW = 0.05
CH2_SUPPRESS_SCALE_HIGH = 0.35

assert abs((FRAC_FLIP + FRAC_SMOOTH) - FRAC_AUGMENT_TOTAL) < 1e-9

FRAC_OVERSHOOT_KEEP = 0.10

SPLIT_TARGETS = {
    # split: (fraction, tolerance)  -> absolute bounds computed as
    # [fraction - tol, fraction + tol] of the category total
    "train": (0.70, 0.10),
    "test": (0.20, 0.10),
    "val": (0.10, 0.10),
}
SPLIT_CYCLE = ["train", "test", "val"]

CATEGORIES_INJECTED = [("Injected", "Pulsed"), ("Injected", "Regular")]
CATEGORIES_CONTROL = [("Control", "Pulsed"), ("Control", "Regular")]

# Injected scans are kept ONLY if "<category>/<setup>/<scan>" contains at
# least one of these substrings (case-sensitive). Control scans are not
# filtered.
INJECTED_INCLUDE_SUBSTRINGS = ["MM", "LLC", "L1210A"]
CELL_LINES = INJECTED_INCLUDE_SUBSTRINGS  # used for the per-cell-line report

# Acquisition sample rate (samples per second) per setup, needed to convert
# sample counts into hours. NOT inferred from the data -- set these to your
# real values. Left as None, the report shows sample counts instead of hours.
SAMPLE_RATE_HZ = {"Pulsed": 2000, "Regular": 2000}

SPLITS = ["train", "test", "val"]

rng = np.random.default_rng(RANDOM_SEED)


# --------------------------------------------------------------------------
# Discovery / loading
# --------------------------------------------------------------------------
def find_scans(category, setup, include_substrings=None):
    """Return sorted list of scan folder names for a given category/setup.
    If include_substrings is given, keep only scans whose
    '<category>/<setup>/<scan>' path contains at least one of them
    (case-sensitive)."""
    dir_path = os.path.join(BASE_PREP, category, setup)
    if not os.path.isdir(dir_path):
        print(f"[WARN] Preprocessed dir not found: {dir_path}")
        return []
    all_scans = sorted(
        d for d in os.listdir(dir_path)
        if os.path.isdir(os.path.join(dir_path, d))
    )

    if not include_substrings:
        return all_scans

    kept, dropped = [], []
    for d in all_scans:
        path_str = os.path.join(category, setup, d)
        if any(sub in path_str for sub in include_substrings):
            kept.append(d)
        else:
            dropped.append(d)

    print(f"[INFO] {category}/{setup}: keeping {len(kept)}/{len(all_scans)} scan(s) "
          f"matching {include_substrings}")
    if dropped:
        print(f"[INFO] {category}/{setup}: dropped (no match): {dropped}")

    return kept


def cell_line_of(category, setup, scan):
    """Cell-line tag for a scan from CELL_LINES substrings (case-sensitive).
    Control scans -> 'Control'. No match -> 'none'. More than one match ->
    'MULTI(a+b)' with a warning, rather than silently picking one."""
    if category != "Injected":
        return "Control"
    path_str = os.path.join(category, setup, scan)
    hits = [c for c in CELL_LINES if c in path_str]
    if not hits:
        return "none"
    if len(hits) > 1:
        print(f"[WARN] {path_str} matches multiple cell lines {hits}")
        return "MULTI(" + "+".join(hits) + ")"
    return hits[0]


def load_scan_signals(category, setup, scan):
    """Load ch1, ch2 numpy arrays for a scan."""
    d = os.path.join(BASE_PREP, category, setup, scan)
    ch1 = np.load(os.path.join(d, f"{scan}_bs_ch1.npy"))
    ch2 = np.load(os.path.join(d, f"{scan}_bs_ch2.npy"))
    return ch1, ch2


def load_scan_labels(category, setup, scan):
    """
    Load label CSV for a scan and split peak indices into label0 / label1
    lists. 's' rows are dropped. Unknown label values are dropped with a
    warning (not silently coerced).

    Returns (label0_indices, label1_indices) as lists of int sample indices.
    """
    path = os.path.join(BASE_LABEL, category, setup, f"{scan}.csv")
    if not os.path.isfile(path):
        print(f"[WARN] Label file not found: {path}")
        return [], []

    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if "Time(Hz)" not in df.columns or "Label" not in df.columns:
        raise ValueError(
            f"Expected columns 'Time(Hz)' and 'Label' in {path}, "
            f"found {list(df.columns)}"
        )

    label0, label1 = [], []
    for _, row in df.iterrows():
        raw_label = str(row["Label"]).strip()
        raw_idx = row["Time(Hz)"]

        if raw_label.lower() == "s":
            continue  # skip label, per instruction

        try:
            idx = int(round(float(raw_idx)))
        except (TypeError, ValueError):
            print(f"[WARN] Unparseable Time(Hz) value '{raw_idx}' in {path}, skipping row")
            continue

        try:
            lab_val = int(round(float(raw_label)))
        except (TypeError, ValueError):
            print(f"[WARN] Unknown label value '{raw_label}' in {path}, skipping row")
            continue

        if lab_val == 1:
            label1.append(idx)
        elif lab_val == 0:
            label0.append(idx)
        elif lab_val == -1:
            continue  # ignore, same treatment as 's'
        else:
            print(f"[WARN] Unexpected label value '{raw_label}' in {path}, skipping row")

    return label0, label1


# --------------------------------------------------------------------------
# Window extraction / standardization / augmentation
# --------------------------------------------------------------------------
# extract_window imported from functions.py (see MASK-AWARE STANDARDIZATION
# note in module docstring). Superseded local version kept only for history:
#
# def extract_window(ch1, ch2, peak_idx):
#     T = len(ch1)
#     assert len(ch2) == T, "ch1/ch2 length mismatch"
#     start = peak_idx - HALF_WIN
#     end = peak_idx + HALF_WIN + 1  # exclusive
#     pad_left = max(0, -start)
#     pad_right = max(0, end - T)
#     actual_start = max(0, start)
#     actual_end = min(T, end)
#     win = np.zeros((2, WIN_LEN), dtype=np.float64)
#     if actual_start < actual_end:
#         seg_len = actual_end - actual_start
#         win[0, pad_left:pad_left + seg_len] = ch1[actual_start:actual_end]
#         win[1, pad_left:pad_left + seg_len] = ch2[actual_start:actual_end]
#     else:
#         print(f"[WARN] peak_idx {peak_idx} yields empty window (T={T}); returning all-zero window")
#     return win


def convolve_smooth(data, win):
    window = np.ones(int(win)) / float(win)
    return np.convolve(data, window, "same")


def augment_flip(win, mask):
    """Flip in time; both channels together, shape preserved. Mask flips
    identically so it stays aligned with the (now time-reversed) window."""
    return win[:, ::-1].copy(), mask[::-1].copy()


def augment_smooth(win, mask):
    """Smooth independently per channel with SMOOTH_WIN. Mask is kept
    as-is (see MASK-AWARE STANDARDIZATION note in module docstring re:
    minor boundary blur this doesn't account for)."""
    out = np.empty_like(win)
    out[0] = convolve_smooth(win[0], SMOOTH_WIN)
    out[1] = convolve_smooth(win[1], SMOOTH_WIN)
    return out, mask.copy()


# def augment_suppress_ch2(win, mask):
#     """
#     Suppress the WHOLE Ch2 trace by a single random multiplicative factor.
#     (kept commented for history; see previous version for full rationale)
#     """
#     aug_win = win.copy()
#     scale = rng.uniform(CH2_SUPPRESS_SCALE_LOW, CH2_SUPPRESS_SCALE_HIGH)
#     aug_win[1, :] = aug_win[1, :] * scale
#     aug_mask = mask.copy()
#     return aug_win, aug_mask


def max_aug_count(n_orig):
    """Hard cap on augmented copies for a class: floor(25% of its originals).
    floor (not round) so the cap is never exceeded."""
    return int(np.floor(FRAC_AUGMENT_TOTAL * n_orig + 1e-9))


def augment_items(items, n_aug):
    """
    Return a list of n_aug augmented copies (NOT including originals),
    each made from a DISTINCT source window (sampled without replacement,
    seeded). Copies are split flip:smooth in the FRAC_FLIP:FRAC_SMOOTH
    ratio. n_aug is clipped to max_aug_count(len(items)).
    """
    n = len(items)
    n_aug = min(int(n_aug), max_aug_count(n))
    if n == 0 or n_aug <= 0:
        return []

    n_flip = int(round(n_aug * FRAC_FLIP / FRAC_AUGMENT_TOTAL))
    chosen = rng.choice(n, size=n_aug, replace=False)

    out = []
    for j, i in enumerate(chosen):
        src = items[i]
        if j < n_flip:
            aug_win, aug_mask = augment_flip(src["win"], src["mask"])
            tag = "flip"
        else:
            aug_win, aug_mask = augment_smooth(src["win"], src["mask"])
            tag = "smooth"
        out.append({**src, "win": aug_win, "mask": aug_mask, "aug": tag})
    return out


def sample_items(items, k):
    """Random subset of k items without replacement (seeded)."""
    if k >= len(items):
        return list(items)
    idx = rng.choice(len(items), size=k, replace=False)
    return [items[i] for i in idx]


# --------------------------------------------------------------------------
# Step 1: assign scans to splits (same procedure for Injected and Control)
# --------------------------------------------------------------------------
def assign_category_round_robin(category, setup, count_label, include_substrings=None):
    """
    Sorted round-robin scan-level split assignment for one category/setup,
    honoring the 70/20/10 +-10% absolute upper bounds on the per-scan count,
    with the "overshoot -> reassign to train" rule.

    count_label:
        1 -> scans are sorted/tracked by number of label-1 peaks (Injected)
        0 -> scans are sorted/tracked by number of label-0 peaks (Control)

    Returns dict: scan_name -> {
        'split': 'train'|'test'|'val',
        'label0_idx': [...],       # ALL label-0 peak indices for this scan
        'label1_idx_used': [...],  # ALL label-1 peak indices for this scan
        'overshoot': bool,
    }
    """
    scans = find_scans(category, setup, include_substrings)
    if not scans:
        return {}

    count_name = "label-1 (CTC)" if count_label == 1 else "label-0 (artifact)"

    scan_info = {}
    for s in scans:
        l0, l1 = load_scan_labels(category, setup, s)
        n = len(l1) if count_label == 1 else len(l0)
        scan_info[s] = {"label0_idx": l0, "label1_idx": l1, "n": n}
        if count_label == 0 and l1:
            print(f"  [NOTE] {category} scan {s} has {len(l1)} label-1 rows in its CSV "
                  f"(unexpected for a control scan) - keeping them as-is.")

    order = sorted(scans, key=lambda s: (-scan_info[s]["n"], s))
    N_total = sum(scan_info[s]["n"] for s in scans)

    bounds = {}
    for split, (frac, tol) in SPLIT_TARGETS.items():
        lo = max(0.0, (frac - tol)) * N_total
        hi = min(1.0, (frac + tol)) * N_total
        bounds[split] = (lo, hi)

    cum = {"train": 0, "test": 0, "val": 0}      # bookkeeping (tolerance checks)
    actual = {"train": 0, "test": 0, "val": 0}   # real counts per split
    n_scans = {"train": 0, "test": 0, "val": 0}
    assignment = {}

    print(f"\n=== {category}/{setup}: {len(scans)} scans, total {count_name} = {N_total} ===")
    print(f"    bounds ({count_name} counts): {bounds}")

    for i, s in enumerate(order):
        intended = SPLIT_CYCLE[i % 3]
        n = scan_info[s]["n"]
        would_be = cum[intended] + n
        upper = bounds[intended][1]

        if n > 0 and would_be > upper:
            # overshoot -> whole scan goes to train; only the bookkeeping
            # counter is incremented by 10% of n.
            k = int(round(FRAC_OVERSHOOT_KEEP * n))
            split = "train"
            cum["train"] += k
            overshoot = True
            print(f"  [OVERSHOOT] scan={s} intended={intended} n={n} "
                  f"-> forced to train, ALL windows kept "
                  f"(bookkeeping count added to running tally: {k})")
        else:
            split = intended
            cum[intended] += n
            overshoot = False

        actual[split] += n
        n_scans[split] += 1
        assignment[s] = {
            "split": split,
            "label0_idx": scan_info[s]["label0_idx"],
            "label1_idx_used": list(scan_info[s]["label1_idx"]),
            "overshoot": overshoot,
        }

    print(f"    bookkeeping cumulative counts: {cum}")
    if N_total > 0:
        fracs = {k: f"{100.0 * v / N_total:.1f}%" for k, v in actual.items()}
    else:
        fracs = {k: "n/a" for k in actual}
    print(f"    ACTUAL {count_name} counts: {actual} -> {fracs} "
          f"(targets: train~70%, test~20%, val~10%)")
    print(f"    scans per split: {n_scans}")

    return assignment


# --------------------------------------------------------------------------
# Step 2: build per-split window pools
# --------------------------------------------------------------------------
def build_all_assignments():
    """
    Run assignment for all 4 categories and merge into one flat structure:
    list of records, one per scan:
      {scan, category, setup, split, label0_idx, label1_idx_used}
    """
    records = []

    jobs = (
        [(c, s, 1, INJECTED_INCLUDE_SUBSTRINGS) for c, s in CATEGORIES_INJECTED]
        + [(c, s, 0, None) for c, s in CATEGORIES_CONTROL]
    )

    for category, setup, count_label, include in jobs:
        assign = assign_category_round_robin(category, setup, count_label, include)
        for scan, info in assign.items():
            records.append({
                "scan": scan, "category": category, "setup": setup,
                "cell_line": cell_line_of(category, setup, scan),
                "split": info["split"],
                "label0_idx": info["label0_idx"],
                "label1_idx_used": info["label1_idx_used"],
            })

    return records


def extract_label_pools(records):
    """
    For every scan record, extract & standardize its label-1 windows and
    ALL its label-0 windows. Returns:
      label1_pool[split] -> list of dicts {win, mask, scan, category, setup, orig_idx, aug}
      label0_pool[split] -> list of dicts {win, mask, scan, category, setup, orig_idx, aug}

    mask is computed on the RAW window (before standardize_joint), then
    used to make standardize_joint's mean/std robust to off-gaps/padding.
    """
    label1_pool = {"train": [], "test": [], "val": []}
    label0_pool = {"train": [], "test": [], "val": []}

    for rec in records:
        ch1, ch2 = load_scan_signals(rec["category"], rec["setup"], rec["scan"])
        split = rec["split"]

        # duration bookkeeping for the report (raw signal, before windowing)
        rec["n_samples"] = int(len(ch1))
        rec["n_on_samples"] = int(np.count_nonzero(~((ch1 == 0) & (ch2 == 0))))

        for pool, idx_list in ((label1_pool, rec["label1_idx_used"]),
                               (label0_pool, rec["label0_idx"])):
            for idx in idx_list:
                win_raw = extract_window(ch1, ch2, idx, WIN_LEN)
                mask = compute_invalid_mask(win_raw)
                win = standardize_joint(win_raw, invalid_mask=mask)
                pool[split].append({
                    "win": win, "mask": mask, "scan": rec["scan"],
                    "category": rec["category"], "setup": rec["setup"],
                    "orig_idx": idx, "aug": "none",
                })

    return label1_pool, label0_pool


# --------------------------------------------------------------------------
# Step 3: balancing (+ augmentation for train only)
# --------------------------------------------------------------------------
def balance_train(l1_items, l0_items):
    """
    TRAIN: augment CTCs by up to 25%, then match artifacts to the CTC count,
    augmenting artifacts (also capped at 25%) only if there aren't enough
    original artifacts. Final count per class = min(max1, max0).
    """
    N1, N0 = len(l1_items), len(l0_items)
    max1 = N1 + max_aug_count(N1)   # CTC originals + up to 25% augmented
    max0 = N0 + max_aug_count(N0)   # artifact originals + up to 25% augmented
    target = min(max1, max0)

    print(f"    originals: CTC={N1}, artifact={N0}")
    print(f"    max reachable: CTC={max1} (25% aug cap={max_aug_count(N1)}), "
          f"artifact={max0} (25% aug cap={max_aug_count(N0)})")
    print(f"    per-class target = {target}")

    # ---- label-1 (CTC) ----
    if target >= N1:
        l1_aug = augment_items(l1_items, target - N1)
        l1_final = list(l1_items) + l1_aug
        if target < max1:
            print(f"[WARN] train: artifacts can only reach {max0} even with 25% "
                  f"augmentation -> CTC augmentation reduced from "
                  f"{max_aug_count(N1)} to {len(l1_aug)} copies.")
    else:
        l1_final = sample_items(l1_items, target)
        print(f"[WARN] train: only {N0} original artifacts (max {max0} with 25% aug) "
              f"< {N1} original CTCs -> CTC originals SUBSAMPLED to {target}, "
              f"no CTC augmentation. Consider whether this is acceptable.")

    # ---- label-0 (artifact) ----
    if target <= N0:
        l0_final = sample_items(l0_items, target)
    else:
        l0_aug = augment_items(l0_items, target - N0)
        l0_final = list(l0_items) + l0_aug
        print(f"    artifacts: all {N0} originals + {len(l0_aug)} augmented "
              f"({100.0 * len(l0_aug) / max(N0, 1):.1f}% of originals)")

    return l1_final, l0_final


def balance_eval(split_name, l1_items, l0_items):
    """
    TEST / VAL: no augmentation. Subsample the larger class down to the
    size of the smaller class.
    """
    N1, N0 = len(l1_items), len(l0_items)
    target = min(N1, N0)
    print(f"    originals: CTC={N1}, artifact={N0} -> per-class target = {target} "
          f"(no augmentation)")

    if N1 > target:
        print(f"[WARN] {split_name}: fewer artifacts ({N0}) than CTCs ({N1}) -> "
              f"{N1 - target} CTC windows dropped to keep the split balanced.")
    return sample_items(l1_items, target), sample_items(l0_items, target)


# --------------------------------------------------------------------------
# Step 4: assemble + save
# --------------------------------------------------------------------------
def assemble_split(split_name, label1_pool, label0_pool):
    l1_items = label1_pool[split_name]
    l0_items = label0_pool[split_name]

    print(f"\n--- Split '{split_name}' ---")

    if split_name == "train":
        l1_final, l0_final = balance_train(l1_items, l0_items)
    else:
        l1_final, l0_final = balance_eval(split_name, l1_items, l0_items)

    all_items = list(l1_final) + list(l0_final)
    all_labels = [1] * len(l1_final) + [0] * len(l0_final)

    if len(all_items) == 0:
        print(f"[WARN] Split '{split_name}' is completely empty -- returning empty "
              f"arrays. Check scan counts / label counts for this split.")
        return (np.empty((0, 2, WIN_LEN), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0, WIN_LEN), dtype=bool),
                [])

    # shuffle (seeded) so label-1/label-0 rows aren't blocked together
    order = np.arange(len(all_items))
    rng.shuffle(order)
    all_items = [all_items[i] for i in order]
    all_labels = [all_labels[i] for i in order]

    X = np.stack([it["win"] for it in all_items], axis=0).astype(np.float32)
    y = np.array(all_labels, dtype=np.int64)
    mask = np.stack([it["mask"] for it in all_items], axis=0).astype(bool)

    meta_rows = []
    for it, lab in zip(all_items, all_labels):
        meta_rows.append({
            "scan": it["scan"],
            "category": it["category"],
            "setup": it["setup"],
            "orig_idx": it["orig_idx"],
            "label": lab,
            "augmentation": it["aug"],
        })

    aug_counts = {}
    for it, lab in zip(all_items, all_labels):
        key = (lab, it["aug"])
        aug_counts[key] = aug_counts.get(key, 0) + 1
    print(f"    final split size: {len(y)} (label-1={int(np.sum(y == 1))}, "
          f"label-0={int(np.sum(y == 0))})")
    print(f"    breakdown (label, augmentation): {dict(sorted(aug_counts.items()))}")
    return X, y, mask, meta_rows


def save_split(split_name, X, y, mask, meta_rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, f"{split_name}_X.npz"), X=X)
    np.save(os.path.join(OUT_DIR, f"{split_name}_y.npy"), y)
    np.save(os.path.join(OUT_DIR, f"{split_name}_mask.npy"), mask)

    meta_path = os.path.join(OUT_DIR, f"{split_name}_meta.csv")
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scan", "category", "setup", "orig_idx", "label", "augmentation"])
        writer.writeheader()
        for row in meta_rows:
            writer.writerow(row)

    print(f"Saved {split_name}: X{X.shape}, y{y.shape}, mask{mask.shape} -> {OUT_DIR}")


# --------------------------------------------------------------------------
# Step 5: reporting
# --------------------------------------------------------------------------
def build_scan_table(records):
    """One row per scan with split, type, cell line, original window counts
    and durations. Hours are NaN when that setup's SAMPLE_RATE_HZ is None."""
    rows = []
    for r in records:
        fs = SAMPLE_RATE_HZ.get(r["setup"])
        rows.append({
            "scan": r["scan"],
            "category": r["category"],
            "setup": r["setup"],
            "cell_line": r["cell_line"],
            "split": r["split"],
            "n_ctc": len(r["label1_idx_used"]),
            "n_artifact": len(r["label0_idx"]),
            "n_samples": r.get("n_samples", 0),
            "n_on_samples": r.get("n_on_samples", 0),
            "hours_recorded": r.get("n_samples", 0) / fs / 3600.0 if fs else np.nan,
            "hours_on": r.get("n_on_samples", 0) / fs / 3600.0 if fs else np.nan,
        })
    return pd.DataFrame(rows)


def _split_table(df, index, values, aggfunc):
    """Pivot with splits as columns (train/test/val + TOTAL), zero-filled so
    a split with no scans still shows up."""
    if df.empty:
        return "(no rows)"
    t = pd.pivot_table(df, index=index, columns="split", values=values,
                       aggfunc=aggfunc, fill_value=0, margins=True,
                       margins_name="TOTAL")
    t = t.reindex(columns=SPLITS + ["TOTAL"], fill_value=0)
    # round only for display, after summing (avoids per-scan rounding error)
    return t.to_string(float_format=lambda v: f"{v:.2f}")


def write_report(records, meta_by_split):
    df = build_scan_table(records)
    lines = []

    def section(title, body):
        lines.append(f"\n### {title}")
        lines.append(body)

    df["scan_type"] = np.where(df["category"] == "Injected",
                               "Injected (CTC-bearing)", "Control")

    # 1. scans per split
    section("Scans per split, by type",
            _split_table(df, "scan_type", "scan", "count"))
    section("Scans per split, by type and setup",
            _split_table(df, ["scan_type", "setup"], "scan", "count"))

    n_inj_zero = int(((df["category"] == "Injected") & (df["n_ctc"] == 0)).sum())
    n_ctl_ctc = int(((df["category"] == "Control") & (df["n_ctc"] > 0)).sum())
    if n_inj_zero:
        lines.append(f"[NOTE] {n_inj_zero} Injected scan(s) have 0 labeled CTCs "
                     f"(counted as Injected above).")
    if n_ctl_ctc:
        lines.append(f"[NOTE] {n_ctl_ctc} Control scan(s) have label-1 rows.")

    # 2. original windows per split (before balancing / augmentation)
    section("ORIGINAL CTC windows per split (before balancing/augmentation), by scan type",
            _split_table(df, "scan_type", "n_ctc", "sum"))
    section("ORIGINAL artifact windows per split (before balancing/augmentation), by scan type",
            _split_table(df, "scan_type", "n_artifact", "sum"))

    # 3. final saved windows per split
    meta_frames = []
    for split, rows in meta_by_split.items():
        if rows:
            m = pd.DataFrame(rows)
            m["split"] = split
            meta_frames.append(m)
    if meta_frames:
        meta = pd.concat(meta_frames, ignore_index=True)
        meta["class"] = np.where(meta["label"] == 1, "CTC", "artifact")
        meta["n"] = 1
        section("FINAL saved windows per split, by class and augmentation",
                _split_table(meta, ["class", "augmentation"], "n", "sum"))
        section("FINAL saved windows per split, by class and source category",
                _split_table(meta, ["class", "category"], "n", "sum"))
    else:
        section("FINAL saved windows per split", "(all splits empty)")

    # 4. cell lines (Injected only)
    inj = df[df["category"] == "Injected"]
    section("Injected scans per split, by cell line",
            _split_table(inj, "cell_line", "scan", "count"))
    section("ORIGINAL CTC windows per split, by cell line",
            _split_table(inj, "cell_line", "n_ctc", "sum"))

    # 5. durations
    missing_fs = sorted({s for s in df["setup"].unique() if not SAMPLE_RATE_HZ.get(s)})
    if missing_fs:
        lines.append(f"\n[WARN] SAMPLE_RATE_HZ not set for setup(s) {missing_fs} -> "
                     f"hours cannot be computed; showing sample counts instead. "
                     f"Set SAMPLE_RATE_HZ at the top of the script.")
        section("Recorded samples per split, by type and setup",
                _split_table(df, ["scan_type", "setup"], "n_samples", "sum"))
        section("On-time (non-zero) samples per split, by type and setup",
                _split_table(df, ["scan_type", "setup"], "n_on_samples", "sum"))
    else:
        section("Recorded hours per split, by type",
                _split_table(df, "scan_type", "hours_recorded", "sum"))
        section("Recorded hours per split, by type and setup",
                _split_table(df, ["scan_type", "setup"], "hours_recorded", "sum"))
        section("On-time hours per split (excludes Pulsed off-gaps), by type and setup",
                _split_table(df, ["scan_type", "setup"], "hours_on", "sum"))

    report = "\n".join(lines)
    print("\n" + "=" * 70 + "\nSPLIT REPORT" + "\n" + "=" * 70 + report)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "split_report.txt"), "w") as f:
        f.write(report.lstrip("\n") + "\n")
    df.drop(columns=["scan_type"]).sort_values(["split", "category", "setup", "scan"]) \
      .to_csv(os.path.join(OUT_DIR, "scan_assignment.csv"), index=False)
    print(f"\nReport -> {os.path.join(OUT_DIR, 'split_report.txt')}")
    print(f"Per-scan table -> {os.path.join(OUT_DIR, 'scan_assignment.csv')}")


def main():
    records = build_all_assignments()

    print("\n=== Scan -> split assignment summary ===")
    for r in records:
        print(f"  {r['category']}/{r['setup']}/{r['scan']}: split={r['split']}, "
              f"label1_used={len(r['label1_idx_used'])}, label0={len(r['label0_idx'])}")

    label1_pool, label0_pool = extract_label_pools(records)

    meta_by_split = {}
    for split_name in SPLITS:
        X, y, mask, meta_rows = assemble_split(split_name, label1_pool, label0_pool)
        save_split(split_name, X, y, mask, meta_rows)
        meta_by_split[split_name] = meta_rows

    write_report(records, meta_by_split)


if __name__ == "__main__":
    main()