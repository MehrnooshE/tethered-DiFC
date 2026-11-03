"""
Step 10 (clearance cohort). Publication-style per-scan figures of every cohort scan.

Cohort = the scans listed in `clearance_cohort` in parameters.py.
Inputs (no re-scoring): per-candidate probabilities saved by 09_clearance_cohort_score
(results/clearance-cohort/per-scan/<scan>/candidates.csv) and the cached windows of
full_scan_preprocessing.prepare_scan (data/clearance-cohort-preprocessed/<scan>/).
All candidates are plotted, including those within 25 ms of a laser ON/OFF edge.

Per scan, in results/clearance-cohort/per-scan/<scan>/ (cut-off: parameters.cut_off;
scan_figures.scan_outputs):
  timeline.{png,pdf}           barcode of detected peaks (count annotated) above
                               the probability of every candidate vs time
  windows_detected[_pN].png    every detected peak, raw (mV) Ch1/Ch2, +/-250 ms
  windows_rejected_random.png  random 24 rejected candidates (p < 0.5)
  windows_rejected_nearmiss[_pN].png  every near-miss (0.5 <= p < cut-off)
Style as the other figures: no titles, no grid, labels 13 pt, ticks 12 pt.
Usage:  python 10_clearance_cohort_scan_figures.py [scan_folder_name ...]
        (with scan names, only those scans are redrawn)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common as mc
import full_scan_preprocessing as p5
import scan_figures as sf

MODEL = mc.MODEL
PREP = mc.COHORT_PREP_DIR
plt.rcParams.update(sf.STYLE)


def main():
    col = f"p_{MODEL}"
    for name in p5.cohort_scans(sys.argv[1:]):
        path = mc.per_scan(name, "candidates.csv")
        if not os.path.exists(path):
            raise SystemExit(f"{name} not scored yet; run 09_clearance_cohort_score.py first")
        if not os.path.exists(os.path.join(PREP, name, f"{name}_windows.npz")):
            raise SystemExit(f"{name}: no preprocessing cache in {os.path.join(PREP, name)}; put the scan in "
                             f"data/clearance-cohort-raw/ and run 08_clearance_cohort_split_overlap.py first")
        d = pd.read_csv(path)
        w = np.load(os.path.join(PREP, name, f"{name}_windows.npz"))
        meta = json.load(open(os.path.join(PREP, name, f"{name}_meta.json")))
        assert np.array_equal(w["peaks"], d.peak_index.values), name
        d = d.assign(p=lambda t: t[col])
        hours = meta["total_samples"] / mc.FS / 3600
        det, near = sf.scan_outputs(w["raw"], d, hours, os.path.join(mc.PER_SCAN_DIR, name), write_candidates=False)
        print(f"{name}: {len(d)} candidates plotted, {len(det)} detected, {len(near)} near misses", flush=True)


if __name__ == "__main__":
    main()
