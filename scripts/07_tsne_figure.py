"""
Step 7. Publication-style t-SNE figure (Fig. S4) of the `fusion_lr1e-4_wd_ls` model's branch embeddings
(seed 0): temporal (128-d) and spectral (64-d), training set (original windows)
and test set, CTC vs artifact. 2 x 2 panels of equal size, no titles, no grid.

The 2-D embeddings are cached in models/tsne_seed0.npz, so restyling
does not rerun t-SNE (delete the file to recompute).
Writes models/figures/tsne_branches.{png,pdf}.
Usage:  python 07_tsne_figure.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as mc

SEED = 0
CACHE = os.path.join(mc.MODELS, f"tsne_seed{SEED}.npz")


def representation_model(path):
    """Outputs [temporal 128-d, spectral 64-d, concatenated 192-d]. The
    Concatenate layer's inputs are the two branch outputs in the outer graph."""
    import tensorflow as tf
    m = tf.keras.models.load_model(path, compile=False)
    concat = [l for l in m.layers if isinstance(l, tf.keras.layers.Concatenate)][0]
    return tf.keras.Model(m.inputs, list(concat.input) + [concat.output])


def compute():
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    import fusion_model as s8
    data = {s: mc.load_split(s) for s in ("train", "test")}
    zc = s8.Standardizer(s8.centre_spectrum(s8.als_scaled(data["train"][0], "train"), data["train"][2]), skip=(2,))
    rep = representation_model(mc.seed_path(SEED))
    out = {}
    for s, (X, y, M, meta) in data.items():
        sel = (meta.augmentation.values == "none") if s == "train" else np.ones(len(y), bool)
        inp = [np.transpose(X, (0, 2, 1)).astype("float32"), zc(s8.centre_spectrum(s8.als_scaled(X, s), M))]
        zt, zs, _ = rep.predict(inp, batch_size=512, verbose=0)
        for name, Z in (("temporal", zt[sel]), ("spectral", zs[sel])):
            out[f"{s}_{name}"] = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(
                StandardScaler().fit_transform(Z))
        out[f"{s}_y"] = y[sel]
    np.savez(CACHE, **out)
    return out


def main():
    E = dict(np.load(CACHE)) if os.path.exists(CACHE) else compute()

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
                         "font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "legend.fontsize": 11})
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.6))
    for r, split in enumerate(("train", "test")):
        y = E[f"{split}_y"]
        size = 3 if split == "train" else 9
        for c, branch in enumerate(("temporal", "spectral")):
            ax = axes[r, c]
            P = E[f"{split}_{branch}"]
            ax.scatter(P[y == 0, 0], P[y == 0, 1], s=size, color=mc.COLORS["artifact"], alpha=0.6, lw=0,
                       label="Artifact", rasterized=True)
            ax.scatter(P[y == 1, 0], P[y == 1, 1], s=size, color=mc.COLORS["ctc"], alpha=0.6, lw=0,
                       label="Peak", rasterized=True)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.set_box_aspect(1)                      # identical square plot areas
    # shared legend above the panels, clear of the data
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles[::-1], labels[::-1], frameon=False, markerscale=3, ncol=2, loc="upper center",
                     bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, columnspacing=1.5)
    for h in leg.legend_handles:
        h.set_alpha(1)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        out = os.path.join(mc.out_dir(mc.MODEL_FIG_DIR), f"tsne_branches.{ext}")
        fig.savefig(out, dpi=300)
        print(out)
    for s in ("train", "test"):
        print(f"{s}: n={len(E[f'{s}_y'])} (CTC {int(E[f'{s}_y'].sum())}, artifact {int((E[f'{s}_y'] == 0).sum())})")


if __name__ == "__main__":
    main()
