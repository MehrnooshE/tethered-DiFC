"""
Inputs and architecture of the time + spectral `fusion` model.

Preprocessing reuses the lab code in signal_functions.py: ALS_BGremoved_Ch2scaled (ALS baseline removal, lam=1e7,
p=0.01, then Ch2 scaled to Ch1's background std) and the same cosine taper
as compute_fft_masked.

  time branch      conv trunk on the jointly standardized 2 x 2001 window
                   (build_encoder, 128-d)
  spectral branch  centre 401 samples (+/-100 ms) of the ALS + Ch2-scaled
                   window: log|Ch1|, log|Ch2|, cos(phase Ch1 - phase Ch2),
                   0-300 Hz (5 Hz bins) (spectral_branch, 64-d)
  head             concatenated 192-d -> dropout / dense 32 / sigmoid
Only bins <= 300 Hz are used: the SR560 pre-amps low-pass at 300 Hz.
Library for 05_train, scoring (09, 13, 14), 07_tsne_figure, fig_spectral_input and
fig_phase_timing; not run on its own.
"""
import os
import numpy as np
from scipy.ndimage import distance_transform_edt

import common as mc
import signal_functions as labf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
KERAS_VERSION = getattr(tf.keras, "__version__", None) or tf.__version__   # TF <= 2.15: tf.keras is Keras 2
if not KERAS_VERSION.startswith("2."):                                     # the seeds are Keras 2 (TF 2.13) files
    import sys
    raise SystemExit(f"{sys.executable} has TensorFlow {tf.__version__} with Keras {KERAS_VERSION}, which "
                     f"cannot load the saved models (Keras 2, TensorFlow 2.13). Use the t-difc-env environment "
                     f"built from requirements.txt (in a notebook: select it as the kernel).")
from tensorflow.keras import layers, models

F_MAX = 300.0
CENTER_HW = 200
RAMP = 50                        # taper length, as compute_fft_masked's default
LOG_EPS = 1e-5
CACHE = mc.SPLITS_ALS_DIR          # default: the train/val/test windows


# ------------------------------------------------------------ features ---
def als_scaled(X, name, cache=CACHE):
    """functions.ALS_BGremoved_Ch2scaled, cached in cache/<name>_als.npy (~30 ms/window)."""
    os.makedirs(cache, exist_ok=True)
    p = os.path.join(cache, f"{name}_als.npy")
    if os.path.exists(p):
        out = np.load(p)
        if out.shape == X.shape:
            return out
    out, n_fail = labf.ALS_BGremoved_Ch2scaled(X.astype(np.float64))
    print(f"  ALS {name}: {len(X)} windows, ch2 scaling failed on {n_fail}")
    np.save(p, out.astype(np.float32))
    return out.astype(np.float32)


def taper_weights(valid):
    """Same cosine ramp as functions.compute_fft_masked (edges and off-gaps)."""
    d = distance_transform_edt(np.pad(valid, 1))[1:-1]
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(d / RAMP, 0.0, 1.0))


def centre_spectrum(Xals, M):
    """(N, 2, 2001) ALS windows -> (N, n_bins, 3): log|Ch1|, log|Ch2|, cos dphi."""
    c = Xals.shape[-1] // 2
    L = 2 * CENTER_HW + 1
    freqs = np.fft.rfftfreq(L, 1 / mc.FS)
    keep = freqs <= F_MAX
    out = np.empty((len(Xals), keep.sum(), 3), dtype=np.float32)
    for i in range(len(Xals)):
        seg = Xals[i, :, c - CENTER_HW:c + CENTER_HW + 1].astype(np.float64)
        w = taper_weights(~M[i, c - CENTER_HW:c + CENTER_HW + 1])
        Z = np.fft.rfft(seg * w, axis=-1)[:, keep] / max(w.sum(), 1e-8)
        mag = np.abs(Z)
        cross = Z[0] * np.conj(Z[1])
        out[i, :, 0] = np.log10(mag[0] + LOG_EPS)
        out[i, :, 1] = np.log10(mag[1] + LOG_EPS)
        out[i, :, 2] = np.real(cross) / np.maximum(np.abs(cross), 1e-12)
    return out


class Standardizer:
    """Per-(bin, channel) z-score fitted on TRAIN; cos-phase channel left as is."""

    def __init__(self, A, skip=()):
        self.mu = A.mean(0, keepdims=True)
        self.sd = A.std(0, keepdims=True) + 1e-6
        for k in skip:
            self.mu[..., k], self.sd[..., k] = 0.0, 1.0

    def __call__(self, A):
        return ((A - self.mu) / self.sd).astype(np.float32)


# -------------------------------------------------------------- models ---
def build_encoder():
    """Time branch: conv trunk (the CNN of the earlier model comparison), up to the pooled 128-d feature."""
    inp = layers.Input((2001, 2))
    x = inp
    for filt, k in [(16, 9), (32, 9), (64, 7), (64, 5), (64, 5)]:
        x = layers.Conv1D(filt, k, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool1D(2)(x)
    x = layers.Concatenate()([layers.GlobalAveragePooling1D()(x), layers.GlobalMaxPooling1D()(x)])
    return models.Model(inp, x, name="encoder")


def spectral_branch(n_bins, n_ch):
    """Conv over frequency, then Flatten (keeps which-frequency information)."""
    inp = layers.Input((n_bins, n_ch))
    x = inp
    for filt, k, pool in [(16, 5, 2), (32, 5, 2), (32, 3, 1)]:
        x = layers.Conv1D(filt, k, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        if pool > 1:
            x = layers.MaxPool1D(pool)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    return models.Model(inp, x, name="spectral")


def head(inputs, z):
    x = layers.Dropout(0.3)(z)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = models.Model(inputs, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
              metrics=[tf.keras.metrics.AUC(name="auc")])
    return m


def build(arm, shapes):
    if arm == "fusion":
        t_in = layers.Input((2001, 2))
        f_in = layers.Input(shapes["spec_center"])
        z = layers.Concatenate()([build_encoder()(t_in), spectral_branch(*shapes["spec_center"])(f_in)])
        return head([t_in, f_in], z)
    raise KeyError(arm)
