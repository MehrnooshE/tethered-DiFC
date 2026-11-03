"""
Step 11 (clearance cohort). Barcode plots of detected peaks per cell line (one figure per cell line, one
row per mouse/session), styled after the lab's L1210A / LLC / MM count figures:
cell-line colours L1210A #E69F00, LLC #CC79A7, MM #0072B2; 0-20 h axis; mouse
label on the left; detected-peak count on the right. No title, no grid; fonts as
in the other figures.

Scans and cell lines: `clearance_cohort` in parameters.py (one figure per cell line, Control included).
Detections: fusion_lr1e-4_wd_ls at the cut-off (results/clearance-cohort/per-scan/<scan>/candidates.csv
from 09_clearance_cohort_score), laser-edge candidates included.
Writes results/clearance-cohort/barcodes/barcodes_{L1210A,LLC,MM,Control}.{png,pdf} and
barcode_counts.csv (input of 12_clearance_cohort_clearance_fit).
Usage:  python 11_clearance_cohort_barcodes.py   (cut-off: parameters.cut_off)
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as mc
from scan_figures import CELL_COLORS as COLORS, X_MAX, SHORT_H, label

MODEL = mc.MODEL
CUT = mc.CUT
PREP = mc.COHORT_PREP_DIR
# row labels set from the scan folder where the .mat file names are wrong
# (BCL12-3-6 files are named BCL1210A1 but the recording is mouse BCL12-6)
LABEL_OVERRIDE = {"BCL12-3-1": "BCL12-1", "BCL12-3-2": "BCL12-2", "BCL12-3-4_250810": "BCL12-4",
                  "BCL12-3-5": "BCL12-5", "BCL12-3-6": "BCL12-6",
                  # control mice (no injected cells); file names follow several conventions
                  "B6_Control_01": "BL6-01", "B6_Control_02": "BL6-02", "B6_Control_03": "BL6-03",
                  "BCControl02": "BC2", "Control-B6LLC5-D15": "B6LLC5", "Control-B6LLC5-D21": "B6LLC5",
                  "Control-B6LLC6-D0": "B6LLC6", "Control-B6LLC6-D25": "B6LLC6"}
plt.rcParams.update({"axes.grid": False, "font.size": 12, "axes.labelsize": 13,
                     "xtick.labelsize": 12, "ytick.labelsize": 12})


def main():
    col = f"p_{MODEL}"
    rows = []
    for scan in sorted(mc.COHORT_CELL_LINE):
        path = mc.per_scan(scan, "candidates.csv")
        if not os.path.exists(path):
            raise SystemExit(f"{scan} not scored yet; run 09_clearance_cohort_score.py first")
        if not os.path.exists(os.path.join(PREP, scan, f"{scan}_meta.json")):
            raise SystemExit(f"{scan}: no preprocessing cache in {os.path.join(PREP, scan)}; put the scan in "
                             f"data/clearance-cohort-raw/ and run 08_clearance_cohort_split_overlap.py first")
        meta = json.load(open(os.path.join(PREP, scan, f"{scan}_meta.json")))
        files = [x[0] for x in meta["files"]]
        cl = mc.COHORT_CELL_LINE[scan]
        d = pd.read_csv(path)
        det = d[d[col] >= CUT]
        mouse, date = label(scan, files)
        mouse = LABEL_OVERRIDE.get(scan, mouse)
        rows.append({"scan": scan, "cell_line": cl, "mouse": mouse, "date": date,
                     "hours": meta["total_samples"] / mc.FS / 3600, "n_peaks": len(det),
                     "times_h": det.time_h.values})
    R = pd.DataFrame(rows)
    # mouse name only; the recording date is added only for control mice recorded more than once
    dup = (R.cell_line == "Control") & R.duplicated("mouse", keep=False)
    R["row_label"] = np.where(dup, R.mouse + " (" + R.date.str[2:4] + "/" + R.date.str[4:6] + ")", R.mouse)
    R.drop(columns="times_h").to_csv(os.path.join(mc.out_dir(mc.BARCODES_DIR), "barcode_counts.csv"), index=False)
    print(R.drop(columns="times_h").to_string(index=False))

    for cl, g in R.groupby("cell_line"):
        # recordings of <= SHORT_H hours get a 0-6 h axis (same panel width), the rest 0-20 h;
        # short and long recordings form separate blocks, each with its own time axis
        g = g.assign(xmax=np.where(g.hours <= SHORT_H, 6, X_MAX))
        g = g.sort_values(["xmax", "mouse", "date"], ascending=[False, True, True]).reset_index(drop=True)
        blocks = [b for _, b in g.groupby("xmax", sort=False)]
        n_rows = len(g) + (len(blocks) - 1)                  # one spacer row between blocks
        ratios = []
        for k, b in enumerate(blocks):
            ratios += [1] * len(b) + ([0.55] if k < len(blocks) - 1 else [])
        fig = plt.figure(figsize=(8.0, 0.95 * len(g) + 0.8 * len(blocks)))
        gs = fig.add_gridspec(n_rows, 1, height_ratios=ratios, hspace=0.25)
        row = 0
        for k, b in enumerate(blocks):
            axes = []
            for r in b.itertuples():
                ax = fig.add_subplot(gs[row, 0])
                row += 1
                ax.vlines(r.times_h, 0, 1, color=COLORS[cl], lw=1.0)
                ax.set_ylim(0, 1)
                ax.set_xlim(0, r.xmax)
                ax.set_yticks([])
                ax.set_ylabel(r.row_label, rotation=0, ha="right", va="center", fontsize=12)
                ax.text(1.01, 0.5, f"{r.n_peaks}", transform=ax.transAxes, ha="left", va="center", fontsize=12)
                step = 2 if r.xmax > 6 else 1
                ax.set_xticks(range(0, int(r.xmax) + 1, step))
                ax.tick_params(labelbottom=False)
                for sp in ax.spines.values():
                    sp.set_linewidth(1.0)
                axes.append(ax)
            axes[-1].tick_params(labelbottom=True)
            axes[-1].set_xlabel("Time (h)")
            row += 1 if k < len(blocks) - 1 else 0
        fig.subplots_adjust(left=0.2, right=0.92, top=0.98, bottom=0.07 if len(blocks) > 1 else 0.1)
        for ext in ("png", "pdf"):
            out = os.path.join(mc.BARCODES_DIR, f"barcodes_{cl}.{ext}")
            fig.savefig(out, dpi=300, bbox_inches="tight")
        print(os.path.join(mc.BARCODES_DIR, f"barcodes_{cl}.png"))
        plt.close(fig)


if __name__ == "__main__":
    main()
