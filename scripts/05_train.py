"""
Step 5. Train the final time + spectral `fusion_lr1e-4_wd_ls` model (5 seeds; Methods S3.1-S3.4).

Trimmed from 21_fusion_stabilize.py: only the chosen config is trained, with the
same architecture (fusion_model.build("fusion")), inputs, seeds (0-4), batch 64,
<=60 epochs, ReduceLROnPlateau (x0.5, patience 4, min 1e-6) and early stopping
(patience 8, best weights restored), both on the PLAIN validation BCE (`val_bce`):
  Adam 1e-4 + decoupled weight decay 1e-4 + label smoothing 0.05

Writes models/fusion_lr1e-4_wd_ls_seed{0-4}.keras,
       models/training_history.csv (per-epoch losses; read by 06_evaluate.py),
       models/predictions_{val,test}.csv (meta + per-seed and 5-seed mean probability).
Usage:  python 05_train.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import common as mc
import fusion_model as s8

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks

SEEDS = [0, 1, 2, 3, 4]
NAME = "lr1e-4_wd_ls"
CFG = dict(lr=1e-4, wd=1e-4, ls=0.05)
COL = mc.MODEL                                  # "fusion_" + NAME


class DecoupledWeightDecay(callbacks.Callback):
    """AdamW-style decoupled weight decay: after every step, kernels *= (1 - lr * wd).
    Needed because on Apple-silicon TF 2.13 swaps optimizers for their legacy versions:
    keras.optimizers.AdamW becomes a string (fit fails) and Adam(weight_decay=...) silently
    loses the decay."""

    def __init__(self, wd):
        super().__init__()
        self.wd = wd

    def on_train_batch_end(self, batch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        for v in self.model.trainable_weights:
            if "kernel" in v.name:
                v.assign(v * (1.0 - lr * self.wd))


def build(cfg, shapes):
    m = s8.build("fusion", shapes)                       # identical architecture; recompile below
    opt = keras.optimizers.legacy.Adam(learning_rate=cfg["lr"])   # what plain Adam becomes here anyway
    m.compile(optimizer=opt, loss=keras.losses.BinaryCrossentropy(label_smoothing=cfg["ls"]),
              metrics=[keras.metrics.BinaryCrossentropy(name="bce"), keras.metrics.AUC(name="auc")])
    return m


def fit(m, cfg, inp, y):
    cbs = [callbacks.EarlyStopping(monitor="val_bce", patience=8, restore_best_weights=True),
           callbacks.ReduceLROnPlateau(monitor="val_bce", factor=0.5, patience=4, min_lr=1e-6),
           DecoupledWeightDecay(cfg["wd"])]
    return m.fit(inp["train"], y["train"], batch_size=64, shuffle=True, validation_data=(inp["val"], y["val"]),
                 epochs=60, verbose=0, callbacks=cbs)


def main():
    os.makedirs(mc.MODELS, exist_ok=True)
    data = {s: mc.load_split(s) for s in mc.SPLITS}
    y = {s: data[s][1] for s in mc.SPLITS}
    zc = s8.Standardizer(s8.centre_spectrum(s8.als_scaled(data["train"][0], "train"), data["train"][2]), skip=(2,))
    inp = {s: [np.transpose(data[s][0], (0, 2, 1)).astype("float32"),
               zc(s8.centre_spectrum(s8.als_scaled(data[s][0], s), data[s][2]))] for s in mc.SPLITS}
    shapes = {"spec_center": inp["train"][1].shape[1:]}

    hist, P = [], {"val": [], "test": []}
    for seed in SEEDS:
        tf.keras.utils.set_random_seed(seed)
        m = build(CFG, shapes)
        h = fit(m, CFG, inp, y).history
        best = int(np.argmin(h["val_bce"]))
        hist.append(pd.DataFrame({"config": NAME, "seed": seed, "epoch": np.arange(1, len(h["loss"]) + 1),
                                  "train_loss": h["bce"], "val_bce": h["val_bce"],
                                  "best": np.arange(len(h["loss"])) == best}))
        for s in P:
            P[s].append(m.predict(inp[s], batch_size=512, verbose=0).ravel())
        m.save(mc.seed_path(seed))
        print(f"{NAME} seed {seed}: epochs {len(h['loss'])}, best epoch {best + 1}, "
              f"min val BCE {min(h['val_bce']):.4f}", flush=True)
    pd.concat(hist, ignore_index=True).to_csv(os.path.join(mc.MODELS, "training_history.csv"), index=False)

    for s in P:
        d = data[s][3].reset_index(drop=True).copy()
        for k, p in enumerate(P[s]):
            d[f"{COL}_seed{k}"] = p
        d[COL] = np.mean(P[s], 0)
        d.to_csv(os.path.join(mc.MODELS, f"predictions_{s}.csv"), index=False)
        print(f"{s}: ensemble AUC {roc_auc_score(y[s], d[COL]):.4f}")


if __name__ == "__main__":
    main()
