"""
Step 6. Validation / test evaluation of the `fusion_lr1e-4_wd_ls` model (5-seed mean
probability, models/predictions_{val,test}.csv from 05_train.py). In models/figures/:

  loss_curves.{png,pdf}                 training / validation BCE per epoch, 5 seeds
                                        (models/training_history.csv); dot = restored epoch
  val_roc_probs_cut<c>.{png,pdf}        validation: ROC, predicted-probability distribution
  test_roc_probs_cm_cut<c>.{png,pdf}    test: ROC, probability distribution, confusion matrix
Vertical line / operating point / confusion matrix at the cut-off. No titles,
no grid, square panels.
Metrics at the cut-off (with 95% scan-level cluster-bootstrap CIs) are printed
and written to models/metrics_at_<c>.csv (Table S2). The cut-off is
parameters.cut_off (0.6).
Usage:  python 06_evaluate.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

import common as mc

COL = mc.MODEL                                                  # prediction column
THR = mc.CUT                                                    # probability cut-off
TAG = f"_cut{THR:g}"
CTC, ART, INK, INK2 = mc.COLORS["ctc"], mc.COLORS["artifact"], mc.COLORS["ink"], mc.COLORS["ink2"]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
                     "font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 11})


def save(fig, name):
    d = mc.out_dir(mc.MODEL_FIG_DIR)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(d, f"{name}.{ext}"), dpi=300)
    print(os.path.join(d, f"{name}.png"))
    plt.close(fig)


def roc_panel(ax, y, p):
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot([0, 1], [0, 1], color="#c9c8c2", lw=1)
    ax.plot(fpr, tpr, color=CTC, lw=2, label=f"AUC = {roc_auc_score(y, p):.3f}")
    yhat = p >= THR
    sens, fpr_op = yhat[y == 1].mean(), yhat[y == 0].mean()
    ax.plot(fpr_op, sens, "o", ms=7, color=INK, mec="white", mew=1.2, label=f"cut-off {THR}")
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, loc="lower right")
    ax.set_box_aspect(1)


def prob_panel(ax, y, p):
    bins = np.linspace(0, 1, 41)
    ax.hist(p[y == 0], bins=bins, color=ART, alpha=0.75, label="Artifact", edgecolor="white", linewidth=0.3)
    ax.hist(p[y == 1], bins=bins, color=CTC, alpha=0.75, label="Peak", edgecolor="white", linewidth=0.3)
    ax.axvline(THR, color=INK, lw=1.3, ls="--", label=f"Cut-off {THR}")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_xlabel("Predicted probability of peak")
    ax.set_ylabel("Windows")
    # legend in the empty middle, kept left of the cut-off line
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(THR - 0.03, 1.02), handlelength=1.2,
              borderaxespad=0)
    ax.set_box_aspect(1)


def cm_panel(ax, y, p):
    cm = confusion_matrix(y, (p >= THR).astype(int), labels=[0, 1])
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
    for i in range(2):
        for j in range(2):
            pct = 100 * cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)", ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > cm.max() * 0.6 else INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Artifact", "Peak"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Artifact", "Peak"], rotation=90, va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_box_aspect(1)
    return cm


def main():
    P = {s: pd.read_csv(os.path.join(mc.MODELS, f"predictions_{s}.csv")) for s in ("val", "test")}

    # ---- loss curves ------------------------------------------------------
    H = pd.read_csv(os.path.join(mc.MODELS, "training_history.csv"))     # from 05_train.py
    H = H[H.config == COL.replace("fusion_", "")].rename(columns={"val_bce": "val_loss"})
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    for seed, g in H.groupby("seed"):
        g = g[g.epoch <= g[g.best].epoch.iloc[0]]            # only up to the selected (restored) epoch
        ax.plot(g.epoch, g.train_loss, color=CTC, lw=1.4, alpha=0.85, label="Training" if seed == 0 else None)
        ax.plot(g.epoch, g.val_loss, color=ART, lw=1.4, alpha=0.85, label="Validation" if seed == 0 else None)
        b = g[g.best]
        ax.plot(b.epoch, b.val_loss, "o", ms=6, color=INK, mec="white", mew=1, zorder=5,
                label="Selected epoch" if seed == 0 else None)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy")
    last = int(H[H.best].epoch.max())
    step = 4 if last <= 20 else 8
    ax.set_xticks(range(0, last + 2, step))
    ax.legend(frameon=False, loc="upper right", handlelength=1.5)
    ax.set_box_aspect(0.75)                              # shorter than wide
    fig.tight_layout()
    save(fig, "loss_curves")

    # ---- validation ---------------------------------------------------------
    y, p = P["val"].label.values, P["val"][COL].values
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    roc_panel(axes[0], y, p)
    prob_panel(axes[1], y, p)
    fig.tight_layout()
    save(fig, f"val_roc_probs{TAG}")

    # ---- test ---------------------------------------------------------------
    y, p = P["test"].label.values, P["test"][COL].values
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    roc_panel(axes[0], y, p)
    prob_panel(axes[1], y, p)
    cm_panel(axes[2], y, p)
    fig.tight_layout()
    save(fig, f"test_roc_probs_cm{TAG}")

    # ---- metrics at the cut-off ----------------------------------------------
    rows = []
    for s, d in P.items():
        y, p, g = d.label.values, d[COL].values, d.scan.values
        r = mc.metrics_at(y, p, THR, prevalence=mc.POOL_PREVALENCE[s])
        out = {"split": s, "n": len(y), "TP": r["tp"], "FP": r["fp"], "TN": r["tn"], "FN": r["fn"]}
        for k in ("auc", "sens", "spec", "bal_acc"):
            pt, lo, hi = mc.cluster_bootstrap(y, p, g, mc.stat(k, THR), n_boot=2000, seed=1)
            out[k] = f"{pt:.3f} [{lo:.3f}, {hi:.3f}]"
        out["PPV (balanced set)"] = round(r["ppv"], 3)
        out["PPV at pre-balancing prevalence"] = round(r["ppv_at_pool_prev"], 3)
        rows.append(out)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(mc.MODELS, f"metrics_at_{THR:g}.csv"), index=False)
    print(t.set_index("split").T.to_string())


if __name__ == "__main__":
    main()
