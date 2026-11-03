"""
Step 13 (exploratory). Analyse one continuous (not pulsed) scan on its own: it is never
grouped with the clearance-cohort scans, with other continuous scans or with the pulsed
exploratory scans of step 14, and it needs no other step. Any cell line (e.g. MC38).

Same pipeline as 14_pulsed_exploratory_scan (independent_scan.analyse); the only
difference is that continuous recordings are not pulsed, so every file is preprocessed as
one laser-ON segment instead of being tested for laser pulsing.

The scan's .mat files (or a zip of them) go in data/continuous-exploratory-raw/<scan>/;
caches go in data/continuous-exploratory-preprocessed/<scan>/ and
data/continuous-exploratory-als-cache/. Everything is written to
results/continuous-exploratory/<scan>/ (summary, candidates, barcode, timeline, window
galleries, clearance fit; see independent_scan.py). Cell line and mouse label come from the
.mat file names (scan_figures.cell_line). Steps 08-12 never read these folders.

Usage:  python 13_continuous_exploratory_scan.py <scan_folder_name>   (cut-off: parameters.cut_off)
"""
import re
import sys

import common as mc
import independent_scan
import scan_figures as sf


def mouse_label(scan, files):
    """102726_C57-M5data_MC38-CFSE_CB_F1.mat -> C57-M5; scan_figures.label otherwise."""
    lines = "|".join(sf.CELL_LINES)
    m = re.match(rf"\s*\d{{6}}[-_](.+?)(?:data)?[-_](?:{lines})[-_]", files[0])
    return (m.group(1),) if m else sf.label(scan, files)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    independent_scan.analyse(sys.argv[1], mc.CONTINUOUS_RAW_DIR, mc.CONTINUOUS_PREP_DIR, mc.CONTINUOUS_ALS_DIR,
                             mc.CONTINUOUS_DIR, pulsed=False, label=mouse_label)


if __name__ == "__main__":
    main()
