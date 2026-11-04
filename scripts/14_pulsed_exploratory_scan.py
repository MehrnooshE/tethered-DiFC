"""
Step 14 (exploratory). Analyse one pulsed scan on its own: it is never grouped with the
clearance-cohort scans, with the continuous exploratory scans of step 13 or with other
pulsed exploratory scans, and it needs no other step.

Same pipeline as 13_continuous_exploratory_scan (independent_scan.analyse), without the
clearance fit (fit_clearance=False); pulsed or
continuous recording is detected per file, as for the cohort, and a warning is printed
if a file is continuous (use step 13 for a continuous scan).

Input: the folder with the scan's .mat files, either
  - its path: local, /Volumes/<share>/... or smb://<server>/<share>/... of a connected NAS
    (quote it if it has spaces). It is linked into data/pulsed-exploratory-raw/<folder name>/
    and only read; the scan name is the folder name.
  - or a scan name: a folder (or zip) already in data/pulsed-exploratory-raw/.
Caches go in data/pulsed-exploratory-preprocessed/<scan>/ and
data/pulsed-exploratory-als-cache/. Everything is written to
results/pulsed-exploratory/<scan>/ (summary, candidates, barcode, timeline, window
galleries; see independent_scan.py). Steps 08-12 never read these folders.

Usage:  python 14_pulsed_exploratory_scan.py "<path to the scan folder>"   (cut-off: parameters.cut_off)
        python 14_pulsed_exploratory_scan.py <scan_folder_name>
e.g.    python 14_pulsed_exploratory_scan.py "smb://<server>/<share>/<folder>/<scan>"
"""
import os
import sys

import common as mc
import independent_scan


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    arg = sys.argv[1]
    is_path = arg.startswith(("smb://", "/", "~", ".")) or os.sep in arg
    name = independent_scan.link_scan(arg, mc.PULSED_RAW_DIR) if is_path else arg
    if os.path.isdir(os.path.join(mc.COHORT_RAW_DIR, name)):
        print(f"note: {name} is also a clearance-cohort-raw folder (no cohort result is changed)")
    independent_scan.analyse(name, mc.PULSED_RAW_DIR, mc.PULSED_PREP_DIR, mc.PULSED_ALS_DIR, mc.PULSED_DIR,
                             fit_clearance=False)


if __name__ == "__main__":
    main()
