"""
Spectral-branch input preparation (Fig. S1) of the `fusion_lr1e-4_wd_ls` model, step by step,
for one training window on a bumpy background: a CTC (#5596, clear Ch1-only peak) or an
artifact (#6235, Control scan, the same peak in Ch1 and Ch2).

Every step is the code the model was trained with (fusion_model.py / signal_functions.py):
  A  jointly standardized window, 2 x 2001
  B  ALS baseline per channel (lam = 1e7, p = 0.01, dead time cropped)
  C  baseline removed; Ch2 x sigma_bg,Ch1 / sigma_bg,Ch2 (sigma-clipped, +/-0.15 s
     around the candidate and 25 ms edges excluded)
  D  centre 0.2 s (401 samples) x mask-aware cosine taper (50-sample ramp)
  E  |rFFT| / sum(w), 5 Hz bins; only <= 300 Hz kept (SR560 low-pass)
  F  log10 magnitude of the kept bins
  G  cos(phase Ch1 - phase Ch2)
  H  the (61 x 3) array that enters the spectral CNN

ILLUSTRATIVE (default): the window is redrawn for display -- same ALS baseline,
residual smoothed (Gaussian, SMOOTH_SIGMA samples), the recorded peak and the side
bumps near it (PRESETS[idx]["replace"]) swapped for quiet noise from the same channel,
and the peak drawn as the same Gaussian (PEAK_AMP, PEAK_SIGMA) in the channels of
PRESETS[idx]["peak"] (height after Ch2 scaling, relative to PEAK_AMP) -- and then
passed through the real pipeline. Pass --real to plot the recorded window unchanged.

Usage:  python fig_spectral_input.py [train index, default 5596] [--real]
Outputs (in models/figures/): spectral_input<suffix>.{png,pdf} and the
editable slide spectral_input<suffix>.pptx (panel images + native PowerPoint
text/arrows; panel PNGs in spectral_input<suffix>_panels/). Requires python-pptx.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter1d

import common as mc
import signal_functions as labf
import fusion_model as s8

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
IDX = int(ARGS[0]) if ARGS else 5596
ILLUSTRATIVE = "--real" not in sys.argv
SMOOTH_SIGMA = 1.0                            # samples (0.5 ms; keeps most of the 0-300 Hz band)
PEAK_AMP, PEAK_SIGMA = 4.5, 24.0              # standardized units; samples (FWHM ~28 ms)
# replace: channel -> ((start, stop) of the stretch replaced, start of the quiet source), samples from
# the centre; peak: channel -> Gaussian height after Ch2 scaling, as a fraction of PEAK_AMP
PRESETS = {
    5596: dict(suffix="",                     # CTC
               replace={0: ((-60, 260), -600),          # Ch1: recorded peak + bump at +60..+120 ms
                        1: ((-260, -100), 300)},        # Ch2: bump at -125..-55 ms
               peak={0: 1.0}),
    6235: dict(suffix="_artifact",            # artifact, 240328-BCLLC1-Awake-Control_R1_F1
               replace={0: ((-220, 220), 500),          # recorded peak + dip/bumps within +/-0.11 s
                        1: ((-220, 220), 500)},
               peak={0: 1.0, 1: 1.0}),
}
PRESET = PRESETS.get(IDX, dict(suffix=f"_{IDX}"))
ZLIM = 5.0                                    # fixed colour range of H, comparable across windows
BLEND = 20                                    # samples crossfaded at each end of a replaced stretch
MODEL = mc.MODEL
C1, C2, CPH = "#1a9850", "#e0522a", "#6a3a9a"  # Ch1 green, Ch2 orange (as in earlier figures), phase purple
B1, B2 = "#0b5a2e", "#8c2a10"
SHADE = "#d9d9d9"
INK2 = mc.COLORS["ink2"]
EXCL_HW = 300                                 # compute_background_std(peak_exclude_half_width)
EDGE = 50                                     # compute_background_std(edge_margin)
OUT = os.path.join(mc.out_dir(mc.MODEL_FIG_DIR), "spectral_input" + PRESET["suffix"])
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})

TITLES = {  # (title, subtitle) -- matplotlib titles in the PNG, native text on the slide
    "A": ("Standardized window", "2 × 2001, joint Ch1/Ch2 z-score"),
    "B": ("ALS baseline estimation", "λ = 10⁷, p = 0.01, per channel"),
    "C": ("Baseline removed, Ch2 scaled", "Ch2 × σbg,Ch1 / σbg,Ch2"),
    "D": ("Central 0.2 s × cosine taper", "2 × 401, 0.025 s ramps"),
    "E": ("FFT magnitude", "|rFFT| / Σw, Δf = 5 Hz"),
    "F": ("Log magnitude, 0–300 Hz", "61 bins, Δf = 5 Hz"),
    "G": ("Cross-channel phase", "cos(φCh1 − φCh2)"),
    "H": ("Spectral-CNN input", "61 × 3, z-scored per bin"),
}


def ch_steps(x):
    """Same calls as functions.ALS_BGremoved_Ch2scaled, keeping the intermediates."""
    c = x.shape[-1] // 2
    df = labf.remove_baseline(pd.DataFrame(x.T, columns=["Ch1", "Ch2"]), method="als", crop_dead_time=True)
    base = np.stack([df.Ch1_baseline.to_numpy(), df.Ch2_baseline.to_numpy()])
    corr = np.stack([df.Ch1_corrected.to_numpy(), df.Ch2_corrected.to_numpy()])
    sd = [labf.compute_background_std(corr[k], c, labf.build_invalid_mask(corr[k]))[1] for k in (0, 1)]
    scale = sd[0] / sd[1] if np.isfinite(sd).all() and sd[1] >= 1e-9 else 1.0
    return base, np.stack([corr[0], corr[1] * scale]), scale


def illustrative(x, base):
    """Display version of x: same baseline, smoothed residual, recorded peak and side bumps -> quiet
    noise, peak -> Gaussian."""
    c = x.shape[-1] // 2
    e = x - base
    for k, ((a, b), src) in PRESET["replace"].items():
        n = b - a
        w = np.clip(np.minimum(np.arange(n), n - 1 - np.arange(n)) / BLEND, 0.0, 1.0)
        e[k, c + a:c + b] = w * e[k, c + src:c + src + n] + (1 - w) * e[k, c + a:c + b]
    out = base + gaussian_filter1d(e, SMOOTH_SIGMA, axis=-1)
    scale = ch_steps(out)[2]                  # background-only, so unchanged by the peak added below
    g = PEAK_AMP * np.exp(-0.5 * ((np.arange(x.shape[-1]) - c) / PEAK_SIGMA) ** 2)
    for k, frac in PRESET["peak"].items():
        out[k] += frac * g / (scale if k == 1 else 1.0)
    return out


def prepare():
    X, y, M, meta = mc.load_split("train")
    x, m = X[IDX].astype(np.float64), M[IDX]
    A_train = s8.als_scaled(X, "train")
    base, xs, scale = ch_steps(x)
    assert np.allclose(xs, A_train[IDX], atol=1e-4), "recomputed ALS differs from the training cache"
    if ILLUSTRATIVE:
        x = illustrative(x, base)
        base, xs, scale = ch_steps(x)

    c, hw = X.shape[-1] // 2, s8.CENTER_HW
    seg = xs[:, c - hw:c + hw + 1]
    w = s8.taper_weights(~m[c - hw:c + hw + 1])
    freqs = np.fft.rfftfreq(2 * hw + 1, 1 / mc.FS)
    keep = freqs <= s8.F_MAX
    Z = np.fft.rfft(seg * w, axis=-1) / max(w.sum(), 1e-8)
    raw = s8.centre_spectrum(xs[None].astype(np.float32), m[None])[0]          # (61, 3)
    zc = s8.Standardizer(s8.centre_spectrum(A_train, M), skip=(2,))
    z = zc(raw[None])[0]
    assert np.allclose(raw[:, 0], np.log10(np.abs(Z[0, keep]) + s8.LOG_EPS), atol=1e-4)

    from tensorflow import keras
    xin = [np.transpose(x[None], (0, 2, 1)).astype("float32"), z[None]]
    p = np.mean([keras.models.load_model(mc.seed_path(k), compile=False)
                 .predict(xin, verbose=0).ravel()[0] for k in range(5)])
    print(f"train #{IDX} ({'CTC' if y[IDX] else 'artifact'}): {meta.scan[IDX]} ({meta.cell_line[IDX]}, {meta.setup[IDX]}), "
          f"{'illustrative' if ILLUSTRATIVE else 'recorded'}, ch2 scale {scale:.3f}, max |z| {np.abs(z[:, :2]).max():.1f}, P(CTC) {MODEL} = {p:.3f}")
    return dict(x=x, base=base, xs=xs, scale=scale, seg=seg, w=w, freqs=freqs, keep=keep, Z=Z, raw=raw,
                zc=zc, z=z, p=p, t=(np.arange(X.shape[-1]) - c) / mc.FS,
                tc=(np.arange(2 * hw + 1) - hw) / mc.FS, meta=meta.iloc[IDX])


# -------------------------------------------------------------- panels ---
# Each draws into one axes; fs = base font size (larger for the full figure, smaller for slide images).
def panel_A(a, d, fs):
    a.plot(d["t"], d["x"][0], color=C1, lw=0.9, label="Ch1")
    a.plot(d["t"], d["x"][1], color=C2, lw=0.9, label="Ch2", alpha=0.85)
    a.set(xlabel="Time from candidate (s)", ylabel="Standardized amplitude", xlim=(-0.5, 0.5))
    a.legend(loc="upper right", ncol=2, fontsize=fs - 1)


def panel_B(a, d, fs):
    a.plot(d["t"], d["x"][0], color=C1, lw=0.8, alpha=0.6, label="Ch1")
    a.plot(d["t"], d["x"][1], color=C2, lw=0.8, alpha=0.5, label="Ch2")
    a.plot(d["t"], d["base"][0], "--", color=B1, lw=2.0, label="ALS baseline (Ch1)")
    a.plot(d["t"], d["base"][1], "--", color=B2, lw=2.0, label="ALS baseline (Ch2)")
    a.set(xlabel="Time from candidate (s)", ylabel="Standardized amplitude", xlim=(-0.5, 0.5))
    a.legend(loc="upper right", fontsize=fs - 1.5, ncol=1, handlelength=1.6)


def panel_C(a, d, fs):
    hw = s8.CENTER_HW
    for e in ((-EXCL_HW / mc.FS, EXCL_HW / mc.FS), (-0.5, -0.5 + EDGE / mc.FS), (0.5 - EDGE / mc.FS, 0.5)):
        a.axvspan(*e, color=SHADE, alpha=0.6, lw=0)
    a.plot(d["t"], d["xs"][1], color=C2, lw=0.8, alpha=0.9,
           label=r"Ch2 × $\sigma_{bg,Ch1}/\sigma_{bg,Ch2}$" + f"\n(= {d['scale']:.2f})")
    a.plot(d["t"], d["xs"][0], color=C1, lw=0.9, label="Ch1")
    ylo, yhi = a.get_ylim()
    yhi *= 1.45
    a.add_patch(Rectangle((-hw / mc.FS, ylo), 2 * hw / mc.FS, (yhi - ylo) * 0.68, fill=False, ec="k", lw=1.0,
                          ls=":"))
    a.set(xlabel="Time from candidate (s)", ylabel="Amplitude (a.u.)", xlim=(-0.5, 0.5), ylim=(ylo, yhi))
    a.text(0.03, 0.97, "grey: excluded from σ$_{bg}$\n(±0.15 s, edges)\ndotted: central 0.2 s",
           transform=a.transAxes, ha="left", va="top", fontsize=fs - 1.5, color=INK2)
    a.legend(loc="upper right", fontsize=fs - 1.5, handlelength=1.4)


def panel_D(a, d, fs):
    """Signal x taper; the taper itself in a strip above (no second y-axis). Returns the strip."""
    at = a.inset_axes([0, 1.08, 1, 0.3], sharex=a)
    at.set_facecolor("none")
    at.fill_between(d["tc"], d["w"], color="#9e9e9e", alpha=0.5, lw=0)
    at.plot(d["tc"], d["w"], color="#555", lw=1.1)
    at.set_ylim(0, 1.15)
    at.set_yticks([0, 1])
    at.set_ylabel("w", rotation=0, labelpad=6)
    at.tick_params(labelbottom=False)
    at.spines["top"].set_visible(False)
    at.spines["right"].set_visible(False)
    a.plot(d["tc"], d["seg"][1] * d["w"], color=C2, lw=1.0, label="Ch2 · w")
    a.plot(d["tc"], d["seg"][0] * d["w"], color=C1, lw=1.1, label="Ch1 · w")
    a.set(xlabel="Time from candidate (s)", ylabel="Amplitude (a.u.)", xlim=(-0.1, 0.1),
          xticks=[-0.1, -0.05, 0, 0.05, 0.1], xticklabels=["−0.1", "−0.05", "0", "0.05", "0.1"])
    a.legend(loc="upper right", fontsize=fs - 1.5)
    return at


def panel_E(a, d, fs):
    f, Z, keep = d["freqs"], d["Z"], d["keep"]
    a.axvspan(s8.F_MAX, f[-1], color=SHADE, alpha=0.6, lw=0)
    for k, col, lab in ((1, C2, "Ch2"), (0, C1, "Ch1")):
        a.semilogy(f, np.abs(Z[k]) + s8.LOG_EPS, color=col, lw=0.9, label=lab)
        a.plot(f[keep], np.abs(Z[k, keep]) + s8.LOG_EPS, "o", ms=2.2, color=col)
    a.set(xlabel="Frequency (Hz)", ylabel="|FFT| / Σw (a.u.)", xlim=(0, f[-1]))
    a.text((s8.F_MAX + f[-1]) / 2, 0.97, "discarded\n> 300 Hz", ha="center", va="top",
           transform=a.get_xaxis_transform(), fontsize=fs - 1.5, color=INK2)
    a.text(s8.F_MAX / 2, 0.97, "kept:\n61 bins", ha="center", va="top", transform=a.get_xaxis_transform(),
           fontsize=fs - 1.5, color=INK2)
    a.legend(loc="lower left", fontsize=fs - 1.5)


def panel_F(a, d, fs):
    fk, raw = d["freqs"][d["keep"]], d["raw"]
    for k, col, lab in ((1, C2, "log₁₀|Ch2|"), (0, C1, "log₁₀|Ch1|")):
        a.plot(fk, raw[:, k], "o-", color=col, lw=1.1, ms=2.5, label=lab)
    a.set(xlabel="Frequency (Hz)", ylabel="log₁₀ magnitude", xlim=(0, s8.F_MAX))
    a.legend(loc="upper right", fontsize=fs - 1.5)


def panel_G(a, d, fs):
    a.axhline(0, color="#bbb", lw=0.8)
    a.plot(d["freqs"][d["keep"]], d["raw"][:, 2], "o-", color=CPH, lw=1.1, ms=2.5)
    a.set(xlabel="Frequency (Hz)", ylabel="cos Δφ", xlim=(0, s8.F_MAX), ylim=(-1.1, 1.1))
    a.set_yticks([-1, 0, 1])
    a.set_yticklabels(["−1\nopposite", "0", "+1\nin phase"], fontsize=fs - 1.5)


def panel_H(a, d, fs, cax=None):
    z, fk = d["z"], d["freqs"][d["keep"]]
    lim = ZLIM
    im = a.imshow(z.T, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest",
                  extent=(-2.5, fk[-1] + 2.5, 2.5, -0.5))
    a.set_yticks([0, 1, 2])
    a.set_yticklabels(["z log|Ch1|", "z log|Ch2|", "cos Δφ"])
    for lbl, col in zip(a.get_yticklabels(), (C1, C2, CPH)):
        lbl.set_color(col)
    a.axhline(1.5, color="k", lw=1)
    a.set(xlabel="Frequency (Hz)")
    a.spines["right"].set_visible(True)
    a.spines["top"].set_visible(True)
    cb = plt.colorbar(im, ax=None if cax else a, cax=cax, fraction=0.05, pad=0.02)
    cb.set_label("z-score / cos Δφ", fontsize=fs - 1)
    cb.ax.tick_params(labelsize=fs - 1.5)


PANELS = {"A": panel_A, "B": panel_B, "C": panel_C, "D": panel_D,
          "E": panel_E, "F": panel_F, "G": panel_G, "H": panel_H}


# --------------------------------------------------------- full figure ---
def full_figure(d):
    fs = 10
    plt.rcParams.update({"font.size": fs, "axes.titlesize": 11.5})
    fig = plt.figure(figsize=(20, 9.2))
    gs = fig.add_gridspec(2, 4, hspace=0.5, wspace=0.32, left=0.045, right=0.965, top=0.92, bottom=0.08)
    for i, (k, fn) in enumerate(PANELS.items()):
        a = fig.add_subplot(gs[i // 4, i % 4])
        if k == "D":                           # room for the taper strip drawn above it
            pos = a.get_position()
            a.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.72])
        top = fn(a, d, fs) or a
        top.set_title(TITLES[k][0] + "\n", fontsize=11.5)
        top.text(0.5, 1.02, TITLES[k][1], transform=top.transAxes, ha="center", va="bottom",
                 fontsize=9, color=INK2)
    fig.text(0.965, 0.012, "→ spectral CNN (3 × Conv1D–BN–ReLU → Flatten → FC 64)", ha="right",
             fontsize=10, color=INK2)
    fig.savefig(OUT + ".png", dpi=200)
    fig.savefig(OUT + ".pdf")
    plt.close(fig)
    print(OUT + ".png")


# --------------------------------------------------------------- slide ---
PW, PH = 2.5, 2.2                              # panel image size on the slide (inches)


def panel_images(d):
    """One PNG per panel at its exact slide size, without titles (those are native text)."""
    fs = 7
    plt.rcParams.update({"font.size": fs})
    pdir = OUT + "_panels"
    os.makedirs(pdir, exist_ok=True)
    paths = {}
    for k, fn in PANELS.items():
        fig = plt.figure(figsize=(PW, PH))
        top = 0.66 if k == "D" else 0.96
        right = 0.8 if k == "H" else 0.97
        left = 0.25 if k in ("G", "H") else 0.2
        a = fig.add_axes([left, 0.19, right - left, top - 0.19])
        if k == "H":
            fn(a, d, fs, cax=fig.add_axes([0.83, 0.19, 0.035, top - 0.19]))
        else:
            fn(a, d, fs)
        paths[k] = os.path.join(pdir, f"{k}.png")
        fig.savefig(paths[k], dpi=300, transparent=True)
        plt.close(fig)
    return paths


def slide(d, paths):
    from pptx import Presentation
    from pptx.util import Inches
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
    import fig_architecture as a17
    box, label, arrow = a17.box, a17.label, a17.arrow

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(6.7)
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    box(sl, 0.15, 0.12, 13.03, 6.45, fill=a17.S_FILL, line=a17.S_LINE, lw=1.25, radius=0.02,
        name="spectral_branch")

    xs = [0.35, 3.25, 6.15, 9.05]
    rows = [0.35, 3.6]
    hdr = 0.55
    for i, k in enumerate(PANELS):
        x, y = xs[i % 4], rows[i // 4]
        label(sl, x, y, PW, hdr, [[(TITLES[k][0], 11, {"bold": True})],
                                  [(TITLES[k][1], 9, {"color": a17.INK2})]],
              align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, margin=0)
        pic = sl.shapes.add_picture(paths[k], Inches(x), Inches(y + hdr), Inches(PW), Inches(PH))
        pic.name = f"panel_{k}"
        if i % 4 < 3:
            ym = y + hdr + PH / 2
            arrow(sl, x + PW + 0.05, ym, xs[i % 4 + 1] - 0.05, ym, color=a17.S_BLOCK_LINE, lw=1.75)
    # row 1 -> row 2 (D down, back to E)
    xd, yb, ymid = xs[3] + PW / 2, rows[0] + hdr + PH + 0.05, (rows[0] + hdr + PH + rows[1]) / 2
    arrow(sl, xd, yb, xd, ymid, color=a17.S_BLOCK_LINE, lw=1.75, head=False)
    arrow(sl, xd, ymid, xs[0] + PW / 2, ymid, color=a17.S_BLOCK_LINE, lw=1.75, head=False)
    arrow(sl, xs[0] + PW / 2, ymid, xs[0] + PW / 2, rows[1] - 0.03, color=a17.S_BLOCK_LINE, lw=1.75)

    # H -> spectral CNN
    yc = rows[1] + hdr + PH / 2
    arrow(sl, xs[3] + PW + 0.05, yc, 11.85, yc, color=a17.S_BLOCK_LINE, lw=1.75)
    box(sl, 11.85, yc - 1.0, 1.2, 2.0, fill=a17.WHITE, line=a17.S_LINE, lw=1.25, radius=0.08,
        paras=[[("Spectral", 12, {"bold": True})], [("CNN", 12, {"bold": True})],
               {"runs": [("3 × [Conv1D", 9, {"color": a17.INK2})], "space_after": 0},
               [("→ BN → ReLU]", 9, {"color": a17.INK2})],
               [("Flatten → FC 64", 9, {"color": a17.INK2})],
               [("→ z", 11, {"italic": True, "color": a17.S_TITLE}),
                ("s", 11, {"italic": True, "color": a17.S_TITLE, "baseline": -25000})]], name="spectral_cnn")
    prs.save(OUT + ".pptx")
    print(OUT + ".pptx")


def main():
    d = prepare()
    full_figure(d)
    slide(d, panel_images(d))


if __name__ == "__main__":
    main()
