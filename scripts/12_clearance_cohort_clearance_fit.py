"""
Step 12 (clearance cohort). Clearance analysis of injected cells (every cohort cell line except Control) from the detected peaks of
the current model, following the draft's Methods ("Cell clearance"):

  * detections = fusion_lr1e-4_wd_ls, cut-off 0.6, laser-edge candidates included;
    the scans of the grouped barcode figures (results/clearance-cohort/barcodes/barcode_counts.csv,
    11_clearance_cohort_barcodes, which must run first), i.e. the scans in `clearance_cohort` in parameters.py
  * counts summed per 5-min bin, normalized to the maximum bin of each scan
  * single exponential f(t) = N0 exp(-t / tau), N0 fixed at 1, plateau fixed at 0
    (fitted with scipy.optimize.curve_fit), per scan and on the pooled scans of
    each cell line; half-life = tau * ln 2
  * double exponential f(t) = N0 exp(-t / tau1) + N1 exp(-t / tau2) on the pooled
    scans, compared with the single exponential by reduced chi-squared
  * initial count = the maximum 5-min count of each scan (the normalization factor; in
    14/15 scans within the first 15 min); the first-5-min count is also reported
  * pairwise comparisons between cell lines: Welch t-test (shown in the figure)
    and Mann-Whitney U, both two-sided (n = scans per cell line in parameters.clearance_cohort)
Time zero = start of the recording (taken as the time of injection).

Writes results/clearance-cohort/clearance/clearance_per_scan.csv, clearance_fits.csv, clearance_tests.csv,
clearance_decay.{png,pdf} and clearance_half_life_initial.{png,pdf}.
Usage:  python 12_clearance_cohort_clearance_fit.py
"""
import os
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as mc
from clearance_fit import single, double, red_chi2
import clearance_fit

MODEL, CUT = mc.MODEL, mc.CUT
OUT = mc.CLEARANCE_DIR
LINES = [c for c in mc.COHORT if c != "Control"]          # order of parameters.clearance_cohort
COLORS = {**{c: "#595959" for c in LINES}, "L1210A": "#E69F00", "LLC": "#CC79A7", "MM": "#0072B2"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
                     "font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 11})


def binned(scan, hours):
    d = pd.read_csv(mc.per_scan(scan, "candidates.csv"))
    return clearance_fit.binned(d.time_h[d[f"p_{MODEL}"] >= CUT].values, hours)


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def main():
    B = pd.read_csv(os.path.join(mc.BARCODES_DIR, "barcode_counts.csv"))
    os.makedirs(OUT, exist_ok=True)
    B = B[B.cell_line.isin(LINES)]
    rows, pooled = [], {c: ([], []) for c in LINES}
    for r in B.itertuples():
        t, n = binned(r.scan, r.hours)
        y = n / n.max()
        (tau,), _ = curve_fit(single, t, y, p0=[0.5], bounds=(1e-3, 50))
        rows.append({"cell_line": r.cell_line, "mouse": r.row_label, "scan": r.scan, "n_peaks": int(n.sum()),
                     "initial_count": int(n[0]), "max_bin_count": int(n.max()),
                     "max_bin_time_h": float(t[np.argmax(n)]), "tau_h": tau, "half_life_h": tau * np.log(2),
                     "red_chi2_single": red_chi2(y, single(t, tau), 1)})
        pooled[r.cell_line][0].append(t)
        pooled[r.cell_line][1].append(y)
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(OUT, "clearance_per_scan.csv"), index=False)

    fits = []
    for c in LINES:
        t, y = np.concatenate(pooled[c][0]), np.concatenate(pooled[c][1])
        (tau,), cov = curve_fit(single, t, y, p0=[0.5], bounds=(1e-3, 50))
        p2, _ = curve_fit(double, t, y, p0=[0.8, 0.2, 0.2, 2.0], bounds=([0, 1e-3, 0, 1e-3], [5, 50, 5, 50]),
                          maxfev=20000)
        if p2[1] > p2[3]:                                    # report the fast phase as tau1
            p2 = np.array([p2[2], p2[3], p2[0], p2[1]])
        fits.append({"cell_line": c, "tau_h": tau, "tau_se_h": float(np.sqrt(cov[0, 0])),
                     "half_life_h": tau * np.log(2), "red_chi2_single": red_chi2(y, single(t, tau), 1),
                     "N0": p2[0], "tau1_h": p2[1], "N1": p2[2], "tau2_h": p2[3],
                     "red_chi2_double": red_chi2(y, double(t, *p2), 4)})
    F = pd.DataFrame(fits)
    F.to_csv(os.path.join(OUT, "clearance_fits.csv"), index=False)

    tests = []
    for metric in ("half_life_h", "max_bin_count", "initial_count"):
        for a, b in itertools.combinations(LINES, 2):
            x, z = S[S.cell_line == a][metric].values, S[S.cell_line == b][metric].values
            tests.append({"metric": metric, "a": a, "b": b,
                          "welch_p": stats.ttest_ind(x, z, equal_var=False).pvalue,
                          "mannwhitney_p": stats.mannwhitneyu(x, z, alternative="two-sided").pvalue})
    T = pd.DataFrame(tests)
    T.to_csv(os.path.join(OUT, "clearance_tests.csv"), index=False)

    pd.set_option("display.width", 220)
    print(S.round(3).to_string(index=False))
    print("\nPer cell line (mean +- sd over scans):")
    print(S.groupby("cell_line")[["half_life_h", "max_bin_count", "initial_count", "n_peaks"]].agg(["mean", "std"]).round(3)
          .reindex(LINES).to_string())
    print("\nPooled fits:\n", F.round(4).to_string(index=False))
    print("\nTests:\n", T.round(4).to_string(index=False))

    # ---- figure 1: normalized counts + fits ---------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    tt = np.linspace(0, 20, 800)
    for ax, c in zip(axes, LINES):
        for t, y in zip(*pooled[c]):
            ax.plot(t, y, color=COLORS[c], lw=0.8, alpha=0.3)
        f = F[F.cell_line == c].iloc[0]
        ax.plot(tt, single(tt, f.tau_h), color=COLORS[c], lw=2.2,
                label=f"t½ = {f.half_life_h * 60:.0f} min")
        ax.set_xlim(0, 20)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xticks(range(0, 21, 4))
        ax.set_xlabel("Time (h)")
        ax.legend(frameon=False, loc="upper right")
        ax.text(0.02, 0.97, c, transform=ax.transAxes, ha="left", va="top", fontsize=13)
    axes[0].set_ylabel("Normalized counts")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"clearance_decay.{ext}"), dpi=300)
    plt.close(fig)

    # ---- figure 2: half-life and initial count bars --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    rng = np.random.default_rng(0)
    for ax, metric, lab in ((axes[0], "half_life_h", "Half-life (h)"), (axes[1], "max_bin_count", "Initial count")):
        vals = [S[S.cell_line == c][metric].values for c in LINES]
        means = [v.mean() for v in vals]
        sds = [v.std(ddof=1) for v in vals]
        ax.bar(range(3), means, color=[COLORS[c] for c in LINES], edgecolor="black", lw=0.8, width=0.7)
        ax.errorbar(range(3), means, yerr=sds, fmt="none", ecolor="#333333", capsize=4, lw=1.5)
        for i, v in enumerate(vals):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=28, facecolor="white", edgecolor="black",
                       lw=0.9, zorder=3)
        top = max(max(v.max() for v in vals), max(m + s for m, s in zip(means, sds)))
        for k, (a, b) in enumerate([(0, 1), (1, 2), (0, 2)]):
            p = T[(T.metric == metric) & (T.a == LINES[a]) & (T.b == LINES[b])].welch_p.iloc[0]
            h = top * (1.08 + 0.12 * k)
            ax.plot([a, a, b, b], [h - top * 0.02, h, h, h - top * 0.02], color="black", lw=1)
            ax.text((a + b) / 2, h, stars(p), ha="center", va="bottom", fontsize=11)
        ax.set_ylim(0, top * 1.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(LINES)
        ax.set_xlabel("Cell type")
        ax.set_ylabel(lab)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"clearance_half_life_initial.{ext}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
