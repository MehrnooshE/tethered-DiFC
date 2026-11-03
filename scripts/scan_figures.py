"""
Per-scan figures and labels shared by the full-scan steps (library; used by
10_clearance_cohort_scan_figures, 11_clearance_cohort_barcodes and independent_scan (13, 14)).

  timeline   barcode of detected peaks (count annotated) above the probability of every
             candidate vs time
  gallery    raw (mV) Ch1/Ch2 windows, +/-250 ms, time and probability in each panel
  scan_outputs  timeline + galleries of every detected peak, every near miss
             (0.5 <= p < cut-off) and N_REJ random rejected candidates + candidates.csv
  cell_line, label, CELL_LINES, CELL_COLORS  cell line and mouse label from the .mat file names
Matplotlib style: plt.rcParams.update(STYLE) (not applied on import).
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as mc

PEAK, ART, INK, INK2 = mc.COLORS["ctc"], mc.COLORS["artifact"], mc.COLORS["ink"], mc.COLORS["ink2"]
N_REJ, NCOL, MAX_ROWS = 24, 6, 6
NEAR_MISS = mc.NEAR_MISS           # rejected candidates with p >= NEAR_MISS are shown as near misses

STYLE = {"axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
         "font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
         "legend.fontsize": 11}
CELL_LINES = ("L1210A", "LLC", "MM", "MC38")      # recognised in the .mat file names (cell_line)
CELL_COLORS = {"L1210A": "#E69F00", "LLC": "#CC79A7", "MM": "#0072B2", "MC38": "#009E73", "Control": "#595959"}
X_MAX = 20
SHORT_H = 6.5          # recordings up to this length are drawn on a 0-6 h axis


def save(fig, path, pdf=True):
    fig.savefig(path + ".png", dpi=300)
    if pdf:
        fig.savefig(path + ".pdf", dpi=300)
    plt.close(fig)


def timeline(d, hours, path):
    det = d.p >= mc.CUT
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 3.6), sharex=True, gridspec_kw={"height_ratios": [1, 2.6]})
    a1.vlines(d.time_h[det], 0, 1, color=PEAK, lw=1.0)
    a1.set_ylim(0, 1)
    a1.set_yticks([])
    a1.spines["left"].set_visible(False)
    a1.text(0.0, 1.06, f"{int(det.sum())} detected peaks", transform=a1.transAxes, fontsize=12, ha="left",
            va="bottom")
    a2.scatter(d.time_h[~det], d.p[~det], s=7, color=ART, alpha=0.6, lw=0, label="Rejected candidate",
               rasterized=True)
    a2.scatter(d.time_h[det], d.p[det], s=12, color=PEAK, lw=0, label="Detected peak", rasterized=True)
    a2.axhline(mc.CUT, color=INK, lw=1.1, ls="--", label=f"Cut-off {mc.CUT:g}")
    a2.set_ylim(-0.03, 1.03)
    a2.set_yticks([0, 0.5, 1])
    a2.set_yticklabels(["0", "0.5", "1"])
    a2.set_ylabel("Probability")
    a2.set_xlabel("Time (h)")
    a2.set_xlim(0, hours)
    a2.set_xticks(np.arange(0, hours + 0.01, 2 if hours > 6 else 1))
    a2.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), handlelength=1.4, borderaxespad=0.2)
    fig.tight_layout(h_pad=0.4)
    save(fig, path)


def gallery(raw, d, idx, path):
    """Raw (mV) windows, +/-250 ms, NCOL columns; time and probability annotated in each panel."""
    t = (np.arange(raw.shape[-1]) - raw.shape[-1] // 2) / mc.FS * 1e3
    sl = np.abs(t) <= 250
    pages = [idx[k:k + NCOL * MAX_ROWS] for k in range(0, len(idx), NCOL * MAX_ROWS)]
    for pg, chunk in enumerate(pages):
        nrows = int(np.ceil(len(chunk) / NCOL))
        fig, axes = plt.subplots(nrows, NCOL, figsize=(2.3 * NCOL, 1.85 * nrows + 0.5), squeeze=False)
        for k, ax in enumerate(axes.ravel()):
            if k >= len(chunk):
                ax.axis("off")
                continue
            i = chunk[k]
            ax.plot(t[sl], raw[i, 1, sl], color=ART, lw=0.7, label="Ch2")
            ax.plot(t[sl], raw[i, 0, sl], color=PEAK, lw=0.8, label="Ch1")
            ax.text(0.0, 1.02, f"{d.time_h.iat[i]:.2f} h, p = {d.p.iat[i]:.2f}", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=9)
            ax.tick_params(labelsize=9)
            ax.set_xticks([-200, 0, 200])
            if k // NCOL == nrows - 1:
                ax.set_xlabel("Time (ms)", fontsize=10)
            if k % NCOL == 0:
                ax.set_ylabel("Signal (mV)", fontsize=10)
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h[::-1], l[::-1], frameon=False, loc="upper right", ncol=2, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / (1.85 * nrows + 0.5)))
        save(fig, path if len(pages) == 1 else f"{path}_p{pg + 1}", pdf=False)


def scan_outputs(raw, d, hours, out, exclude=None, write_candidates=True):
    """Timeline, galleries and (write_candidates) candidates.csv of one scan in out/ (old figures removed).
    d: candidate table with probability column `p`; exclude: candidates left out (files in
    the splits) or None. Returns the indices of the detected peaks and of the near misses."""
    keep = np.ones(len(d), bool) if exclude is None else ~np.asarray(exclude)
    os.makedirs(out, exist_ok=True)
    for old in os.listdir(out):
        if old.endswith((".png", ".pdf")):
            os.remove(os.path.join(out, old))
    timeline(d if exclude is None else d[keep], hours, os.path.join(out, "timeline"))
    det = np.where((d.p >= mc.CUT) & keep)[0]
    rej = np.where((d.p < mc.CUT) & keep)[0]
    near = rej[d.p.values[rej] >= NEAR_MISS]
    rest = np.setdiff1d(rej, near)
    rng = np.random.default_rng(0)
    if len(det):
        gallery(raw, d, list(det), os.path.join(out, "windows_detected"))
    if len(rest):
        gallery(raw, d, sorted(rng.choice(rest, min(N_REJ, len(rest)), replace=False)),
                os.path.join(out, "windows_rejected_random"))
    if len(near):
        gallery(raw, d, list(near), os.path.join(out, "windows_rejected_nearmiss"))
    if write_candidates:
        d.assign(detected=(d.p >= mc.CUT) & keep).drop(columns="p").to_csv(os.path.join(out, "candidates.csv"),
                                                                         index=False)
    return det, near


def cell_line(files):
    s = " ".join(files)
    hits = [c for c in CELL_LINES if re.search(rf"{c}[-_]", s)]
    return hits[0] if len(hits) == 1 else None


def label(scan, files):
    """Mouse name from the recording file names, plus the date when a mouse has several sessions."""
    f = files[0]
    m = re.match(r"\s*(\d{6})-(BC[A-Za-z0-9]+?)[-_]", f)
    mouse = m.group(2) if m else scan
    mouse = mouse.replace("BCL1210A", "BCL12-")           # BCL1210A1 -> BCL12-1 (as in the lab figures)
    dm = re.match(r"\s*(?:test)?(\d{6})", f)              # date = leading YYMMDD of the file name
    return mouse, (dm.group(1) if dm else "")
