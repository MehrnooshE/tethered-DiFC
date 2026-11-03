"""
Step 2. Preprocessing (Methods S1.1-S1.2) of the four selected-data folders:

    data/selected-data/Injected/Pulsed
    data/selected-data/Injected/Regular
    data/selected-data/Control/Pulsed
    data/selected-data/Control/Regular

Pulsed files  : each ON region is preprocessed individually, then stitched
                back together with zero-filled OFF regions. The stitched
                signal is padded to match the original recording length.
Regular files : the whole signal is preprocessed in one pass.

Outputs saved to data/preprocessed-data/<group>/<kind>/<info>/
Control label CSVs (every candidate = artifact, label 0) saved to
data/labeled-data/Control/<kind>/<info>.csv; Injected labels come from
03_transfer_labels.py.

No normalization is applied (Ch2 rescale and min-max removed).
Usage:  python 02_preprocess_labeled_scans.py
"""

import os
import glob
import numpy as np
import pandas as pd

import common as mc
import peak_detection as pp
import parameters as var


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
fs                      = var.fs
background_sliding_window = var.background_sliding_window
rel_thresh              = var.rel_thresh
min_peak                = var.min_peak
max_peak                = var.max_peak
prominence_factor       = var.prominence_factor

on_overshoot  = 5
off_overshoot = 5
OTSU_BINS     = 20

# ----------------------------------------------------------------------
# Directories
# ----------------------------------------------------------------------
SELECTED_ROOT    = mc.SELECTED_DIR
PREPROCESSED_ROOT = mc.PREPROCESSED_DIR
LABELS_ROOT      = mc.LABELS_DIR

DATASETS = [
    ("Injected", "Pulsed"),
    ("Injected", "Regular"),
    ("Control",  "Pulsed"),
    ("Control",  "Regular"),
]

CH1_SUFFIX   = "_bs_ch1.npy"
CH2_SUFFIX   = "_bs_ch2.npy"
PEAKS_SUFFIX = "_filtered_peaks_1.npy"




# ----------------------------------------------------------------------
# Per-file logic
# ----------------------------------------------------------------------
def preprocess_pulsed_file(data):
    """Preprocess each ON region separately and stitch with zero-filled
    OFF regions. Returns (signal_ch1, signal_ch2, global_peaks)."""
    otsu_threshold = pp.OtsuThreshold(data["Ch1"], bins_num=OTSU_BINS)
    pulse_start_idx, pulse_stop_idx = pp.pulse_start_end_index(
        data, otsu_threshold, fs, background_sliding_window,
        on_overshoot, off_overshoot
    )

    accumulated_signal1 = np.zeros(0)
    accumulated_signal2 = np.zeros(0)
    global_peaks = []

    for num, (s_idx, e_idx) in enumerate(zip(pulse_start_idx, pulse_stop_idx)):
        data_bs1, data_bs2, filtered_peaks_1, _ = pp.PreProcessing(
            data[s_idx:e_idx], fs, background_sliding_window,
            rel_thresh, min_peak, max_peak, prominence_factor
        )

        # shift local peak indices to global coordinates
        filtered_peaks_1 = filtered_peaks_1 + s_idx
        global_peaks.extend(filtered_peaks_1)

        # zeros for the OFF region preceding this pulse
        gap = s_idx if num == 0 else s_idx - pulse_stop_idx[num - 1]
        zero_filling = np.zeros(gap)

        accumulated_signal1 = np.concatenate((accumulated_signal1, zero_filling, np.asarray(data_bs1)))
        accumulated_signal2 = np.concatenate((accumulated_signal2, zero_filling, np.asarray(data_bs2)))

    # pad trailing OFF region so the stitched signal matches original length
    if len(accumulated_signal1) < len(data):
        tail = np.zeros(len(data) - len(accumulated_signal1))
        accumulated_signal1 = np.concatenate((accumulated_signal1, tail))
        accumulated_signal2 = np.concatenate((accumulated_signal2, tail))

    return accumulated_signal1, accumulated_signal2, global_peaks


def preprocess_regular_file(data):
    """Preprocess the whole signal at once. Returns (signal_ch1, signal_ch2, peaks)."""
    data_bs1, data_bs2, filtered_peaks_1, _ = pp.PreProcessing(
        data, fs, background_sliding_window,
        rel_thresh, min_peak, max_peak, prominence_factor
    )
    return np.asarray(data_bs1), np.asarray(data_bs2), list(filtered_peaks_1)


def save_outputs(info, group, kind, sig1, sig2, global_peaks):
    """Save preprocessed signals + peaks; write a zero-label CSV for Control."""
    # control label CSV (written regardless of peak count)
    if group == "Control":
        label_dir = os.path.join(LABELS_ROOT, "Control", kind)
        os.makedirs(label_dir, exist_ok=True)
        labels = pd.DataFrame({"Time(Hz)": global_peaks, "Label": 0})
        labels.to_csv(os.path.join(label_dir, info + ".csv"), index=False)

    # signals + peaks (only when at least one peak was detected)
    if len(global_peaks) > 0:
        out_dir = os.path.join(PREPROCESSED_ROOT, group, kind, info)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, info + CH1_SUFFIX), sig1)
        np.save(os.path.join(out_dir, info + CH2_SUFFIX), sig2)
        np.save(os.path.join(out_dir, info + PEAKS_SUFFIX), np.array(global_peaks))
        return True
    return False


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def main():
    for group, kind in DATASETS:
        in_dir = os.path.join(SELECTED_ROOT, group, kind)
        mat_files = sorted(glob.glob(os.path.join(in_dir, "*.mat")))
        print(f"\n=== {group}/{kind}: {len(mat_files)} files in {in_dir} ===")

        for fp in mat_files:
            info = os.path.basename(fp).replace(".mat", "")

            data = pp.load_scan_data(fp)
            if data is None:
                print(f"  [SKIP] could not load {info}")
                continue

            try:
                if kind == "Pulsed":
                    sig1, sig2, peaks = preprocess_pulsed_file(data)
                else:
                    sig1, sig2, peaks = preprocess_regular_file(data)
            except Exception as e:
                print(f"  [ERROR] {info}: {e}")
                continue

            saved = save_outputs(info, group, kind, sig1, sig2, peaks)
            status = "saved" if saved else "no peaks – signal not saved"
            print(f"  [{status}] {info}  (peaks={len(peaks)})")


if __name__ == "__main__":
    main()