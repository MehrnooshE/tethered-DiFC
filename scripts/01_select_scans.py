"""
Step 1. Sort the raw .mat files of data/raw-data/{Injected,Control} into
data/selected-data/<dataset>/{Regular,Pulsed} based on the behavior of channel 1 (Ch1).

Logic
-----
1.  Only consider files whose name contains "MM", "LLC" or "L1210A"
    (case-insensitive). All other scans are discarded.
2.  Load Ch1 / Ch2 from the .mat file.
3.  Decide if the trace is "pulsed":
        pulsed  <=>  min(Ch1) < OFF_THRESH  AND  max(Ch1) > ON_THRESH
    - Not pulsed  -> copy file to data/selected-data/<dataset>/Regular
    - Pulsed      -> find the ON/OFF split point with Otsu's method
                     (histogram bin size = 20), measure the ON and OFF
                     durations (for reporting only), and copy the file to
                     data/selected-data/<dataset>/Pulsed
                     All pulsed scans are kept, regardless of duty cycle.

This same pipeline is applied to both the "Injected" and "Control"
datasets.

Sampling frequency: 2000 Hz (0.5 ms / sample).
Usage:  python 01_select_scans.py
"""
import os
import shutil
import glob
import numpy as np
from scipy.io import loadmat

import common as mc

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATASETS = [
    {
        "name": name,
        "raw_dir": os.path.join(mc.RAW_DIR, name),
        "regular_out_dir": os.path.join(mc.SELECTED_DIR, name, "Regular"),
        "pulsed_out_dir": os.path.join(mc.SELECTED_DIR, name, "Pulsed"),
    }
    for name in ("Injected", "Control")
]

FS = 2000.0          # Hz
OFF_THRESH = 150.0   # used to decide pulsed vs. not
ON_THRESH = 300.0    # used to decide pulsed vs. not
OTSU_BIN_SIZE = 20    # bin width for the Otsu histogram

REQUIRED_NAME_TAGS = ["MM", "LLC", "L1210A"]  # keep a file if its name contains any of these


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def should_skip_file(filename):
    """Return True if this file should be excluded up front."""
    base = os.path.basename(filename).upper()
    return not any(tag in base for tag in REQUIRED_NAME_TAGS)


def get_channels(mat_dict):
    """
    Pull Ch1 / Ch2 arrays out of a loaded .mat dict.

    Expected layout: mat_dict['data'] is an (N, 2) array where
    column 0 = Ch1 and column 1 = Ch2.
    """
    if "data" not in mat_dict:
        raise KeyError("Could not find 'data' variable in this .mat file")

    data = np.asarray(mat_dict["data"])

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Unexpected shape for 'data': {data.shape}")

    ch1 = data[:, 0]
    ch2 = data[:, 1]

    return ch1, ch2


def otsu_threshold(data, bin_size=20):
    """
    Otsu's method computed on a histogram with fixed bin width.
    Returns the bin-center value that maximizes between-class variance.
    """
    data = np.asarray(data, dtype=float)
    lo, hi = data.min(), data.max()

    if hi <= lo:
        return lo

    edges = np.arange(lo, hi + bin_size, bin_size)
    hist, edges = np.histogram(data, bins=edges)
    hist = hist.astype(float)
    centers = (edges[:-1] + edges[1:]) / 2.0

    total = hist.sum()
    sum_total = np.dot(hist, centers)

    w_bg = 0.0
    sum_bg = 0.0
    best_var = -1.0
    best_thresh = centers[0]

    for i in range(len(hist)):
        w_bg += hist[i]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break

        sum_bg += centers[i] * hist[i]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_total - sum_bg) / w_fg

        between_var = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if between_var > best_var:
            best_var = between_var
            best_thresh = centers[i]

    return best_thresh


def get_runs(mask):
    """Return list of (value, run_length) for a boolean array."""
    runs = []
    if len(mask) == 0:
        return runs

    current_val = mask[0]
    count = 1
    for v in mask[1:]:
        if v == current_val:
            count += 1
        else:
            runs.append((current_val, count))
            current_val = v
            count = 1
    runs.append((current_val, count))
    return runs


def measure_on_off_durations(ch1, threshold, fs):
    """
    Binarize ch1 at `threshold`, find ON/OFF runs, drop the first and
    last run (likely truncated at the edges of the recording), and
    return the median ON duration and median OFF duration, in seconds.
    """
    mask = ch1 >= threshold
    runs = get_runs(mask)

    # drop first/last run - may be partial
    if len(runs) > 2:
        runs = runs[1:-1]

    on_durations_s = [count / fs for val, count in runs if val]
    off_durations_s = [count / fs for val, count in runs if not val]

    if not on_durations_s or not off_durations_s:
        return None, None

    return np.median(on_durations_s), np.median(off_durations_s)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def process_dataset(name, raw_dir, regular_out_dir, pulsed_out_dir):
    os.makedirs(regular_out_dir, exist_ok=True)
    os.makedirs(pulsed_out_dir, exist_ok=True)

    mat_files = sorted(glob.glob(os.path.join(raw_dir, "*.mat")))
    print(f"\n=== {name}: found {len(mat_files)} .mat files in {raw_dir} ===")

    n_skipped_name = 0
    n_regular = 0
    n_pulsed = 0
    n_errors = 0

    for filepath in mat_files:
        fname = os.path.basename(filepath)

        if should_skip_file(fname):
            n_skipped_name += 1
            continue

        try:
            mat_dict = loadmat(filepath)
            ch1, ch2 = get_channels(mat_dict)
        except Exception as e:
            print(f"  [ERROR] Could not read {fname}: {e}")
            n_errors += 1
            continue

        is_pulsed = (np.min(ch1) < OFF_THRESH) and (np.max(ch1) > ON_THRESH)

        if not is_pulsed:
            shutil.copy2(filepath, os.path.join(regular_out_dir, fname))
            n_regular += 1
            print(f"  [REGULAR] {fname}")
            continue

        # Pulsed: keep every one; durations are reported for reference only
        threshold = otsu_threshold(ch1, bin_size=OTSU_BIN_SIZE)
        on_dur_s, off_dur_s = measure_on_off_durations(ch1, threshold, FS)

        shutil.copy2(filepath, os.path.join(pulsed_out_dir, fname))
        n_pulsed += 1
        on_str = f"{on_dur_s:.2f}" if on_dur_s is not None else "NA"
        off_str = f"{off_dur_s:.2f}" if off_dur_s is not None else "NA"
        print(f"  [PULSED] {fname} (on={on_str}s, off={off_str}s, thresh={threshold:.1f})")

    print(f"\n{name} summary")
    print("-" * (len(name) + 8))
    print(f"Skipped by name filter : {n_skipped_name}")
    print(f"Regular (not pulsed)   : {n_regular}")
    print(f"Pulsed                 : {n_pulsed}")
    print(f"Errors                 : {n_errors}")


def main():
    for dataset in DATASETS:
        process_dataset(
            name=dataset["name"],
            raw_dir=dataset["raw_dir"],
            regular_out_dir=dataset["regular_out_dir"],
            pulsed_out_dir=dataset["pulsed_out_dir"],
        )


if __name__ == "__main__":
    main()
