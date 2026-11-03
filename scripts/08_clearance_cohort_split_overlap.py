"""
Step 8 (clearance cohort). Which candidates of the cohort scans are windows already in the
train/val/test splits (scoring.overlap_table: centre +/-100 samples equal up to a
scale and an offset). 09_clearance_cohort_score drops every file with at least one such
window from the counts.

Cohort = the data/clearance-cohort-raw/ scans listed in `clearance_cohort` in parameters.py.
New zips in data/clearance-cohort-raw/ are unpacked and each scan is preprocessed once
(full_scan_preprocessing, cached in data/clearance-cohort-preprocessed/). Run before 09.
Writes results/clearance-cohort/per-scan/<scan>/split_overlap.csv (peak_index, source_file,
matched_split, matched_scan, similarity).

Usage:  python 08_clearance_cohort_split_overlap.py [scan_folder_name ...]
"""
import os
import sys

import common as mc
import full_scan_preprocessing as p5
import scoring


def main():
    p5.unpack_new_zips()
    names = p5.cohort_scans(sys.argv[1:])
    ref = scoring.split_reference()
    for n in names:
        sc = p5.prepare_scan(n)
        o = scoring.overlap_table(sc, ref)
        o.to_csv(os.path.join(mc.out_dir(mc.PER_SCAN_DIR, n), "split_overlap.csv"), index=False)
        hit = o.matched_split.values != ""
        files = sorted(set(o.source_file[hit]))
        print(f"{n}: {len(o)} candidates, {hit.sum()} in the splits"
              + (f", from {len(files)} file(s): {files}" if files else ""))


if __name__ == "__main__":
    main()
