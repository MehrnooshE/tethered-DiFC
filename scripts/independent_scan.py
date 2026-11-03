"""
Analysis of one exploratory scan, never grouped with any other scan (library; used by
13_continuous_exploratory_scan and 14_pulsed_exploratory_scan, which differ only in where the
scan is read from / written to and in continuous scans being preprocessed as continuous).

Same processing as the cohort: full_scan_preprocessing, split-overlap check (files with
a window already in train/val/test are left out), 5-seed scoring (scoring.py), the plots
of scan_figures.py, and the clearance model of clearance_fit.py on this scan alone.

Writes <out_root>/<scan>/:
  summary.csv                  files, pulsed / continuous, hours, candidates, detected peaks
                               (all / not at a laser edge), near misses, peaks per laser-ON hour
  candidates.csv               every candidate with its probability and `detected`
  barcode.{png,pdf}            detected peaks vs time, one row, detected-peak count on the right
  timeline.{png,pdf}           barcode above the probability of every candidate
  windows_detected[_pN].png, windows_rejected_nearmiss[_pN].png, windows_rejected_random.png
  clearance_bins.csv           detections per 5-min bin, normalized to the maximum bin
  clearance_fit.csv            single exponential (tau, half-life) and double exponential fits
  clearance_decay.{png,pdf}    normalized counts with both fits
  split_overlap.csv            candidates that are windows already in the splits
"""
import os
import sys
import urllib.parse
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as mc
import full_scan_preprocessing as p5
import scoring
import scan_figures as sf
import clearance_fit as cf

plt.rcParams.update(sf.STYLE)


def barcode(times_h, n, row_label, hours, color, path):
    """One-row barcode; time axis = recording length rounded up to an even number of hours."""
    fig, axes = plt.subplots(1, 1, figsize=(8.0, 0.95 + 0.8), squeeze=False)
    ax = axes[0, 0]
    xmax = 2 * int(np.ceil(hours / 2))
    ax.vlines(times_h, 0, 1, color=color, lw=1.0)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, xmax)
    ax.set_yticks([])
    ax.set_xticks(range(0, xmax + 1, 2 if xmax > 6 else 1))
    ax.set_ylabel(row_label, rotation=0, ha="right", va="center", fontsize=12)
    ax.text(1.01, 0.5, f"{n}", transform=ax.transAxes, ha="left", va="center", fontsize=12)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.0)
    ax.set_xlabel("Time (h)")
    fig.subplots_adjust(left=0.2, right=0.92, top=0.98, bottom=0.1, hspace=0.25)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def clearance(times_h, hours, tag, color, out):
    """Clearance of this scan (method of 12_clearance_cohort_clearance_fit): counts per 5-min bin normalized to
    the maximum bin, single exponential exp(-t / tau) and double exponential; time zero = start
    of the recording. Returns the fit row, or None with fewer than 3 bins or no detection."""
    t, n = cf.binned(times_h, hours)
    if len(t) < 3 or n.max() == 0:
        print(f"clearance not fitted: {len(t)} complete 5-min bins, {int(n.sum())} detections")
        return None
    y = n / n.max()
    pd.DataFrame({"bin_start_h": t, "count": n, "normalized": y}).to_csv(os.path.join(out, "clearance_bins.csv"),
                                                                          index=False)
    (tau,), cov = curve_fit(cf.single, t, y, p0=[0.5], bounds=(1e-3, 50))
    row = {"n_peaks": int(n.sum()), "initial_count": int(n[0]), "max_bin_count": int(n.max()),
           "max_bin_time_h": float(t[np.argmax(n)]), "tau_h": tau, "tau_se_h": float(np.sqrt(cov[0, 0])),
           "half_life_h": tau * np.log(2), "red_chi2_single": cf.red_chi2(y, cf.single(t, tau), 1)}
    try:
        p2, _ = curve_fit(cf.double, t, y, p0=[0.8, 0.2, 0.2, 2.0], bounds=([0, 1e-3, 0, 1e-3], [5, 50, 5, 50]),
                          maxfev=20000)
        if p2[1] > p2[3]:                                    # report the fast phase as tau1
            p2 = np.array([p2[2], p2[3], p2[0], p2[1]])
        row.update({"N0": p2[0], "tau1_h": p2[1], "N1": p2[2], "tau2_h": p2[3],
                    "red_chi2_double": cf.red_chi2(y, cf.double(t, *p2), 4)})
    except RuntimeError:
        p2 = None
        print("double exponential did not converge")
    pd.DataFrame([row]).to_csv(os.path.join(out, "clearance_fit.csv"), index=False)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(t, y, color=color, lw=0.8, alpha=0.8, marker="o", ms=2.5)
    xmax = 2 * int(np.ceil(hours / 2))
    tt = np.linspace(0, xmax, 800)
    ax.plot(tt, cf.single(tt, tau), color=color, lw=2.2, label=f"Single exp., t½ = {tau * np.log(2) * 60:.0f} min")
    if p2 is not None:
        ax.plot(tt, cf.double(tt, *p2), color=mc.COLORS["ink"], lw=1.4, ls="--", label="Double exp.")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(range(0, xmax + 1, 2 if xmax > 6 else 1))
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Normalized counts")
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.02, 0.97, tag, transform=ax.transAxes, ha="left", va="top", fontsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"clearance_decay.{ext}"), dpi=300)
    plt.close(fig)
    return row


def link_scan(path, raw_root, name=None):
    """Link a scan folder given by its path into raw_root/<name> (name: the folder name) and return
    the name. path: a local folder, /Volumes/<share>/... or smb://<server>/<share>/... of a connected
    NAS (macOS mounts it at /Volumes/<share>). The folder is only read, never written."""
    path = urllib.parse.unquote(path.strip())            # Finder copies spaces as %20
    if path.startswith("smb://"):
        parts = path[len("smb://"):].split("/", 1)
        if len(parts) < 2 or not parts[1]:
            raise SystemExit(f"{path}: expected smb://<server>/<share>/<folder>")
        path = os.path.join("/Volumes", parts[1])
    path = os.path.abspath(os.path.expanduser(path.rstrip("/")))
    if not os.path.isdir(path):
        raise SystemExit(f"{path} not found. Is the NAS connected (Finder > Go > Connect to Server)?")
    name = name or os.path.basename(path)
    link = os.path.join(raw_root, name)
    os.makedirs(raw_root, exist_ok=True)
    if os.path.lexists(link):
        if os.path.realpath(link) != os.path.realpath(path):
            raise SystemExit(f"{link} already exists and is {os.path.realpath(link)}, not {path}")
    else:
        os.symlink(path, link)
        print(f"linked {link} -> {path}")
    return name


def analyse(name, in_root, prep_root, als_root, out_root, pulsed=None, label=sf.label, cell_line=None, color=None):
    """Full analysis of in_root/<name>/ (.mat files; zips in in_root are unpacked first); caches in
    prep_root/<name>/ and als_root/, results in out_root/<name>/.
    pulsed: None = detect per file (as the cohort), False = continuous recording.
    label(scan, files) -> (row label, ...); cell_line / color: fixed, or from the file names."""
    os.makedirs(in_root, exist_ok=True)
    p5.unpack_new_zips(root=in_root)
    if not os.path.isdir(os.path.join(in_root, name)):
        sys.exit(f"no folder {os.path.join(in_root, name)}; put the scan's .mat files there")
    out = mc.out_dir(out_root, name)

    sc = p5.prepare_scan(name, pred_root=in_root, prep_root=prep_root, pulsed=pulsed)
    cont = [f["file"] for f in sc["info"]["per_file"] if not f["pulsed"]]
    if pulsed is None and cont:
        print(f"WARNING: {cont} detected as continuous (not pulsed) recordings; for a continuous scan "
              f"use 13_continuous_exploratory_scan.py")
    ov = scoring.overlap_table(sc, scoring.split_reference())
    ov.to_csv(os.path.join(out, "split_overlap.csv"), index=False)
    seen = scoring.files_in_splits(sc, ov)
    models, zc = scoring.load_scorer()
    p = scoring.score(models, zc, name, sc["X"], sc["M"], als_root)

    d = scoring.candidate_table(sc, p, seen).assign(p=p)
    hours = sc["total"] / mc.FS / 3600
    files = [os.path.basename(f) for f in sc["files"]]
    cl = cell_line or ("Control" if "Control" in name else sf.cell_line(files))
    color = color or sf.CELL_COLORS.get(cl, mc.COLORS["ink2"])
    row_label = label(name, files)[0]
    det, near = sf.scan_outputs(sc["raw"], d, hours, out, exclude=seen)
    times_h = d.time_h.values[det]
    barcode(times_h, len(det), row_label, hours, color, os.path.join(out, "barcode"))
    clearance(times_h, hours, cl or row_label, color, out)

    r = scoring.summary_row(sc, p, seen)
    assert (r["detected_peaks"], r["near_miss"]) == (len(det), len(near))
    summary = {"scan": name, "cell_line": cl or "unknown", "label": row_label, "n_files": len(files),
               "pulsed": sc["info"]["pulsed"], "hours_recorded": r["hours_recorded"],
               "laser_on_hours": r["laser_on_hours"], "cut-off": mc.CUT,
               **{k: r[k] for k in ("candidates", "detected_peaks", "detected_not_at_laser_edge", "near_miss",
                                    "peaks_per_laser_on_hour", "candidates_in_splits")}}
    pd.DataFrame([summary]).to_csv(os.path.join(out, "summary.csv"), index=False)
    print(pd.Series(summary).to_string())
    if seen.any():
        print(f"\n{int(seen.sum())} candidates come from files already in the train/val/test splits "
              f"({sorted(set(d.source_file[seen]))}); they are left out of the counts and plots.")
    print(f"\n-> {out}")
    return summary
