"""
Pipeline parameters: acquisition and peak detection (Methods S1.1-S1.2), used by
peak_detection.py, 02_preprocess_labeled_scans.py and full_scan_preprocessing.py,
the classifier cut-off, used by every step that calls peaks (06, 09-14), and the
cohort scans of steps 08-12.
"""
fs = 2000                                   # sampling rate (Hz)
background_sliding_window = 2 * fs + 1      # 2 s rolling-median background window (samples)
rel_thresh = [4.5, 4.5]                     # adaptive peak threshold (x background std), Ch1 / Ch2
min_peak = [0, 0]
max_peak = [100000, 100000]
prominence_factor = 1
narrow_win_len = 200                        # Ch1 candidates closer than this (100 ms) are merged

# ---- classifier ----
cut_off = 0.6                               # 5-seed mean probability >= cut_off -> detected peak

# ---- clearance cohort (steps 08-12) ----
# The scan folders (data/clearance-cohort-raw/<scan>/) that form the clearance cohort, per
# cell line. Only these scans are scored and plotted by steps 08-12, and the cell line of a
# scan is taken from here. Exploratory scans (continuous, step 13; pulsed, step 14) are
# never part of any cohort result. Edit the lists and rerun 08-12.
clearance_cohort = {
    "LLC": ["BCLLC-11", "BCLLC-12", "BCLLC-14_250717", "BCLLC-17", "BCLLC-18",
            # "BCLLC-14_250724",            # second BCLLC14 session
            ],
    "L1210A": ["BCL12-3-1", "BCL12-3-2", "BCL12-3-4_250810", "BCL12-3-5", "BCL12-3-6",
               # "BCL12-3-4_250825",        # second BCL12-4 session, 1.06 h only
               ],
    "MM": ["BCMM1", "BCMM2", "BCMM3", "BCMM4", "BCMM6"],
    "Control": ["B6_Control_01", "B6_Control_02", "B6_Control_03", "BCControl02",
                "Control-B6LLC5-D15", "Control-B6LLC5-D21", "Control-B6LLC6-D0", "Control-B6LLC6-D25"],
}
