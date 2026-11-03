"""
Step 9 (clearance cohort). Score every candidate of the cohort scans with `fusion_lr1e-4_wd_ls`
(5-seed mean, Methods S3.7) and count the detected peaks at the cut-off (parameters.cut_off).

Cohort = the data/clearance-cohort-raw/ scans listed in `clearance_cohort` in parameters.py
(cell line from there too). Scans are preprocessed with full_scan_preprocessing (cached in
data/clearance-cohort-preprocessed/; ALS windows in data/clearance-cohort-als-cache/). Files
whose windows overlap the train/val/test splits are left out of the counts
(per-scan/<scan>/split_overlap.csv from 08_clearance_cohort_split_overlap; run 08 first, a
scan without that file is counted without exclusion).
Writes, in results/clearance-cohort/:
  per-scan/<scan>/candidates.csv  every candidate with its probability and `detected` (p >= cut-off,
                                  laser-edge candidates included; input of 10-12)
  scan_summary.csv                one row per scan: cell line, hours recorded, laser-ON hours,
                                  candidates, candidates in the splits, detected peaks (all / not at
                                  a laser edge), near misses (scoring.summary_row; files in the
                                  splits left out)

Usage:  python 09_clearance_cohort_score.py [scan_folder_name ...]
        (with names, only those scans are scored and their rows replaced in scan_summary.csv)
"""
import os
import sys
import pandas as pd

import common as mc
import full_scan_preprocessing as p5
import scoring


def main():
    pd.set_option("display.width", 220)
    models, zc = scoring.load_scorer()
    p5.unpack_new_zips()                                  # new recordings, one folder per date
    rows = []
    for n in p5.cohort_scans(sys.argv[1:]):
        sc = p5.prepare_scan(n)
        p = scoring.score(models, zc, n, sc["X"], sc["M"], mc.COHORT_ALS_DIR)
        seen = scoring.files_in_splits(sc, mc.per_scan(n, "split_overlap.csv"))
        scoring.candidate_table(sc, p, seen).assign(detected=p >= mc.CUT).to_csv(
            os.path.join(mc.out_dir(mc.PER_SCAN_DIR, n), "candidates.csv"), index=False)
        rows.append({"scan": n, "cell_line": mc.COHORT_CELL_LINE[n], **scoring.summary_row(sc, p, seen)})
    S = pd.DataFrame(rows)
    if os.path.exists(mc.COHORT_SUMMARY) and sys.argv[1:]:
        old = pd.read_csv(mc.COHORT_SUMMARY)
        S = pd.concat([old[~old.scan.isin(S.scan)], S], ignore_index=True)
    S.to_csv(mc.COHORT_SUMMARY, index=False)
    print(f"\nDetected peaks at cut-off {mc.CUT} (laser-edge included; files overlapping the splits excluded):\n",
          S[["scan", "cell_line", "candidates", "detected_peaks", "near_miss"]].to_string(index=False))
    counts = ["detected_peaks", "detected_not_at_laser_edge", "near_miss"]
    for cl, g in S.groupby("cell_line", sort=False):
        print(f"\n{cl} ({len(g)} scans, {g.laser_on_hours.sum():.1f} laser-ON h):",
              {c: int(g[c].sum()) for c in counts})


if __name__ == "__main__":
    main()
