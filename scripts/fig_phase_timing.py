"""
How the cross-channel phase (panel G of fig_spectral_input.py) encodes timing.

Synthetic, noise-free: the same Gaussian peak as the illustrative figures in Ch1 at 0 and
in Ch2 at a lag dt (0, 25, 50 ms), through the spectral branch's crop, taper and FFT
(fusion_model.py). Columns:
  1  the two channels in time
  2  their 10 Hz and 20 Hz components (one FFT bin each, reconstructed as sinusoids):
     a lag dt shifts every component by dt, i.e. by f * dt of its cycle
  3  cos(phase Ch1 - phase Ch2) per 5 Hz bin, with cos(2 pi f dt) overlaid; grey where the
     peak has < 1% of its peak spectral magnitude, so the phase there is not reliable
Output: models/figures/phase_timing_demo.{png,pdf}
Usage:  python fig_phase_timing.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as mc
import fusion_model as s8

LAGS = [0.0, 0.025, 0.050]                    # s
PEAK_SIGMA = 24.0                             # samples, as fig_spectral_input.PEAK_SIGMA
SHOW_F = (10.0, 20.0)                         # Hz, components drawn in column 2
F_SHOW_MAX = 60.0
LOW_ENERGY = 0.01                             # |X(f)| / max |X| below which the phase is unreliable
C1, C2, CPH = "#1a9850", "#e0522a", "#6a3a9a"
INK2 = mc.COLORS["ink2"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})


def main():
    hw = s8.CENTER_HW
    n = np.arange(-hw, hw + 1)
    t = n / mc.FS
    w = s8.taper_weights(np.ones(len(n), bool))
    freqs = np.fft.rfftfreq(len(n), 1 / mc.FS)
    keep = freqs <= F_SHOW_MAX
    ff = np.linspace(0, F_SHOW_MAX, 400)

    fig, ax = plt.subplots(len(LAGS), 3, figsize=(15, 3.1 * len(LAGS)),
                           gridspec_kw=dict(width_ratios=[1, 1.25, 1.1], hspace=0.55, wspace=0.3))
    for r, dt in enumerate(LAGS):
        x = np.stack([np.exp(-0.5 * (n / PEAK_SIGMA) ** 2),
                      np.exp(-0.5 * ((n - dt * mc.FS) / PEAK_SIGMA) ** 2)])
        Z = np.fft.rfft(x * w, axis=-1)
        lag = f"Δt = {dt * 1e3:.0f} ms"

        a = ax[r, 0]
        a.plot(t, x[0], color=C1, lw=1.6, label="Ch1")
        a.plot(t, x[1], color=C2, lw=1.6, ls="--", label="Ch2")
        a.set(xlim=(-0.1, 0.1), ylim=(-0.05, 1.25), ylabel="Amplitude (a.u.)",
              xticks=[-0.1, -0.05, 0, 0.05, 0.1], xticklabels=["−0.1", "−0.05", "0", "0.05", "0.1"])
        a.set_title(f"Peaks in time: {lag}", fontsize=11)
        a.legend(loc="upper left", fontsize=9)

        a = ax[r, 1]
        for j, f0 in enumerate(SHOW_F):
            k = int(np.argmin(np.abs(freqs - f0)))
            off = 1.6 * j
            for c, col, ls in ((0, C1, "-"), (1, C2, "--")):
                comp = np.abs(Z[c, k]) * np.cos(2 * np.pi * freqs[k] * t + np.angle(Z[c, k]))
                a.plot(t, off + comp / np.abs(Z[0, k]) * 0.55, color=col, lw=1.4, ls=ls)
            dphi = np.degrees(np.angle(Z[0, k] * np.conj(Z[1, k])))
            a.text(-0.1, off + 0.62, f"{freqs[k]:.0f} Hz: shift = {dt * freqs[k]:.2g} cycle  →  "
                                     f"Δφ = {abs(dphi):.0f}°, cos Δφ = {np.cos(np.radians(dphi)):+.2f}",
                   va="bottom", fontsize=8.5, color=INK2)
        a.set(xlim=(-0.1, 0.1), ylim=(-0.65, 0.95 + 1.6 * (len(SHOW_F) - 1)), yticks=[],
              xticks=[-0.1, -0.05, 0, 0.05, 0.1], xticklabels=["−0.1", "−0.05", "0", "0.05", "0.1"])
        a.spines["left"].set_visible(False)
        a.set_title("10 Hz (bottom) and 20 Hz (top) components", fontsize=11)

        a = ax[r, 2]
        a.axhline(0, color="#bbb", lw=0.8)
        low = freqs[keep][np.minimum(np.abs(Z[0]), np.abs(Z[1]))[keep] < LOW_ENERGY * np.abs(Z[0]).max()]
        if len(low):
            a.axvspan(low.min() - 2.5, F_SHOW_MAX, color="#d9d9d9", alpha=0.6, lw=0)
            a.text((low.min() + F_SHOW_MAX) / 2, -1.08, "little peak\nenergy", ha="center", va="bottom",
                   fontsize=8, color=INK2)
        a.plot(ff, np.cos(2 * np.pi * ff * dt), color="#999", lw=1.2, label="cos(2π f Δt)")
        a.plot(freqs[keep], np.real(Z[0] * np.conj(Z[1]))[keep] / np.abs(Z[0] * Z[1])[keep],
               "o", color=CPH, ms=5, label="cos Δφ per 5 Hz bin")
        a.set(xlim=(0, F_SHOW_MAX), ylim=(-1.15, 1.15), ylabel="cos Δφ", yticks=[-1, 0, 1])
        a.set_title("Cross-channel phase" + ("" if dt == 0 else f": repeats every 1/Δt = {1 / dt:.0f} Hz"),
                    fontsize=11)
        if r == 0:
            a.legend(loc="lower left", fontsize=9)
    for a in ax[-1, :2]:
        a.set_xlabel("Time from candidate (s)")
    ax[-1, 2].set_xlabel("Frequency (Hz)")

    out = os.path.join(mc.out_dir(mc.MODEL_FIG_DIR), "phase_timing_demo")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print(out + ".png")


if __name__ == "__main__":
    main()
