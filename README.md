# t-DiFC peak classifier: `fusion_lr1e-4_wd_ls` pipeline

Code, trained model and results for a two-branch (time + spectral) classifier that separates peaks of circulating cells from artifacts in t-DiFC recordings:
- Adam 1e-4, decoupled weight decay 1e-4, label smoothing 0.05;
- 5 seeds, averaged;
- cut-off 0.6 (`cut_off` in `scripts/parameters.py`, read by every step that calls peaks).

The model is described in `methods_fusion_lr1e-4_wd_ls_v5.md` and `tDiFC_ML_Methods_v5.docx`.

## Layout

```
├── scripts/          all code: numbered steps, library modules, the step-13 and step-14 notebooks
├── data/             inputs and caches (folder structure only, plus the train/val/test splits)
├── models/           the trained model, its validation / test evaluation, model figures
├── results/          full-scan results and figures
└── requirements.txt  Python packages (see Setup)
```

### What the repository contains

- **Included:** the code, the 5 trained seeds and their evaluation (`models/`), the train/val/test splits the model was trained on (`data/train-val-test/Joint-Standardization-Augmented/`), the clearance-cohort results (`results/clearance-cohort/`), the step-14 notebook with the outputs of its last run, the step-13 notebook (without outputs), and every `data/` and `results/` folder (empty folders hold a `.gitkeep`).
- **Not included:** the full-scan recordings (`.mat`) and everything derived from them locally (caches, exploratory-scan results), and the Python environment. `.gitignore` keeps these out.

So step 6 (model evaluation) runs right after Setup. The full-scan steps need recordings: put them in the `data/` folders described below. A step that needs data that is not there stops with a message naming the missing folder or file.

### `data/`

| Folder | Contents |
|---|---|
| `train-val-test/Joint-Standardization-Augmented/` | **Included.** The labeled train/val/test windows: `{split}_X.npz` (windows, compressed, key `X`), `{split}_y.npy`, `{split}_mask.npy`, `{split}_meta.csv`, `scan_assignment.csv`, `split_report.txt` |
| `train-val-test-als-cache/` | ALS + Ch2-scaled split windows (spectral input). Built automatically the first time a step needs it (a few minutes) |
| `clearance-cohort-raw/<scan>/` | Clearance-cohort recordings only (steps 8–12): one folder of `.mat` files per scan. A `.zip` placed here is unpacked automatically |
| `clearance-cohort-preprocessed/`, `clearance-cohort-als-cache/` | Caches of steps 8–9, built automatically |
| `pulsed-exploratory-raw/<scan>/` | One pulsed exploratory scan per folder (step 14) |
| `pulsed-exploratory-preprocessed/`, `pulsed-exploratory-als-cache/` | Caches of step 14 |
| `continuous-exploratory-raw/<scan>/` | One continuous (not pulsed) exploratory scan per folder, any cell line (step 13) |
| `continuous-exploratory-preprocessed/`, `continuous-exploratory-als-cache/` | Caches of step 13 |

A scan folder can be a real folder or a link to one elsewhere (`ln -s /path/to/<scan> data/clearance-cohort-raw/<scan>`), so large recordings do not have to be copied. Caches can be deleted at any time; they are rebuilt when needed.

**Where to put a scan.** `clearance-cohort-raw/` is only for the clearance cohort: put a scan there only if it is listed in `clearance_cohort` in `scripts/parameters.py`. A scan that is only for exploration goes in its own folder, `continuous-exploratory-raw/<scan>/` for a continuous recording (step 13) or `pulsed-exploratory-raw/<scan>/` for a pulsed one (step 14); its results never touch the cohort results. To move an exploratory scan into the cohort later, move its folder to `clearance-cohort-raw/`, add it to `clearance_cohort`, and rerun steps 8–12.

Steps 1–4 (building the labeled splits) read `data/raw-data/` and `data/previous-labels/` and write `data/selected-data/`, `data/preprocessed-data/` and `data/labeled-data/`. These inputs are not in the repository; the splits they produce are.

### `models/`

| File | Contents |
|---|---|
| `fusion_lr1e-4_wd_ls_seed{0-4}.keras` | The 5 trained seeds |
| `training_history.csv` | Training / validation BCE per epoch and seed, with the restored epoch flagged |
| `predictions_{val,test}.csv` | Window metadata, per-seed probabilities and the 5-seed mean (column `fusion_lr1e-4_wd_ls`) |
| `metrics_at_0.6.csv` | Table S2: AUC, sensitivity, specificity and balanced accuracy with scan-level bootstrap CIs; PPV |
| `tsne_seed0.npz` | Cached t-SNE embeddings for `figures/tsne_branches` |
| `figures/` | `loss_curves` (Fig. S2); `val_roc_probs_cut0.6` and `test_roc_probs_cm_cut0.6` (Fig. S3); `tsne_branches` (Fig. S4); `architecture` (Fig. 1); `spectral_input[_artifact]` (Fig. S1); `phase_timing_demo` |

### `results/`

```
results/
├── clearance-cohort/              steps 8–12, the scans in `clearance_cohort`
│   ├── scan_summary.csv           one row per scan
│   ├── per-scan/<scan>/           split_overlap.csv, candidates.csv, timeline, window galleries
│   ├── barcodes/                  barcode figures per cell line, barcode_counts.csv
│   └── clearance/                 clearance fits, tests and figures
├── pulsed-exploratory/<scan>/     step 14, one scan on its own
└── continuous-exploratory/<scan>/ step 13, one scan on its own
```

| Path | Contents |
|---|---|
| `clearance-cohort/scan_summary.csv` | Per cohort scan: cell line, hours recorded, laser-ON hours, candidates, `candidates_in_splits`, detected peaks at the cut-off (all, and not at a laser edge), near misses (0.5 ≤ p < cut-off), peaks per laser-ON hour. Candidates from files with a window in the train/val/test splits are left out of every count. |
| `clearance-cohort/per-scan/<scan>/split_overlap.csv` | Candidates that are windows already in the train/val/test splits; their files are left out of the counts |
| `clearance-cohort/per-scan/<scan>/candidates.csv` | Every peak candidate: time, source file, laser-edge flag, `file_in_splits`, probability `p_fusion_lr1e-4_wd_ls`, `detected` (p ≥ cut-off) |
| `clearance-cohort/per-scan/<scan>/` figures | `timeline`, galleries of detected / near-miss / random rejected windows |
| `clearance-cohort/barcodes/` | Barcode figures per cell line (L1210A, LLC, MM, Control) and `barcode_counts.csv` |
| `clearance-cohort/clearance/` | Clearance decay and half-life figures; `clearance_per_scan/fits/tests.csv` |
| `pulsed-exploratory/<scan>/`, `continuous-exploratory/<scan>/` | Steps 14 and 13: `summary.csv`, `candidates.csv`, `barcode`, `timeline`, window galleries, `split_overlap.csv`; step 13 also `clearance_bins.csv`, `clearance_fit.csv`, `clearance_decay`. No other step reads them. |

## Setup

**Getting the code.** Clone the repository once:

```
git clone https://github.com/MehrnooshE/tethered-DiFC.git
```

**Updating to the latest version.** To bring an existing copy up to date with what is on GitHub, run inside the repository folder:

```
cd <repository folder>
git pull
```

This only changes files that are tracked in the repository; your recordings in `data/`, the caches, the exploratory results and `t-difc-env/` are not touched (`.gitignore` keeps them out). If `git pull` stops because you edited a tracked file (for example, a notebook you ran, or `results/clearance-cohort/` after rerunning steps 8–12), either keep your changes aside with `git stash`, then `git pull` (and `git stash pop` to bring them back), or discard them with `git checkout -- <file>` before pulling.

**Python environment.** The scripts need **Python 3.11** and the packages in `requirements.txt` (NumPy, pandas, SciPy, scikit-learn, Matplotlib, TensorFlow 2.13, tqdm; python-pptx and lxml for the two `.pptx` figure scripts; ipykernel and IPython for the notebook). The versions are pinned to the ones the pipeline was run and checked with. TensorFlow runs on the CPU; no GPU is needed.

The environment is called **`t-difc-env`**. Create it once, inside the repository folder:

```
cd <repository folder>
python3.11 -m venv t-difc-env           # creates the environment in t-difc-env/
source t-difc-env/bin/activate          # macOS / Linux  (Windows: t-difc-env\Scripts\activate)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`python3.11 --version` should print 3.11; if `python3.11` points to a Python inside another virtual environment, use the full path of a Python 3.11 installation instead. The environment takes about 1.5 GB, mostly TensorFlow; `python -m pip check` should report no broken requirements.

**No Python 3.11?** Python 3.12 and 3.13, the default on most new machines, do not work (TensorFlow 2.13 stops at 3.11), so install 3.11 next to them; it does not replace the existing Python. Any one of these:

- **Conda** (simplest; [Miniconda](https://docs.anaconda.com/miniconda/) brings its own Python). Conda downloads Python 3.11 into the environment, so the `venv` step is not needed:
  ```
  conda create -n t-difc-env python=3.11
  conda activate t-difc-env
  python -m pip install -r requirements.txt
  ```
  Then activate it with `conda activate t-difc-env` instead of `source t-difc-env/bin/activate`.
- **Homebrew** (macOS): `brew install python@3.11`, then the `venv` commands above with `$(brew --prefix)/bin/python3.11 -m venv t-difc-env`.
- **pyenv** (macOS / Linux): `pyenv install 3.11`, then `$(pyenv root)/versions/<3.11.x>/bin/python3.11 -m venv t-difc-env`.
- **python.org installer** (macOS / Windows): the 3.11 release page has installers up to 3.11.9 (later 3.11 releases are source only); on Windows tick "Add python.exe to PATH" and use `py -3.11 -m venv t-difc-env`.

Activate the environment in every new terminal before running a script; `deactivate` leaves it. The `t-difc-env/` folder holds only the installed packages and is not part of the repository; delete and recreate it at any time.

**Notebook kernel.** Register the environment once as a Jupyter kernel (venv; with conda, activate the environment and run the same command with `python`):

```
t-difc-env/bin/python -m ipykernel install --user --name t-difc-env --display-name "t-difc-env (Python 3.11)"
```

Then in VS Code: *Select Kernel* (top right of the notebook) > *Jupyter Kernel…* > **t-difc-env (Python 3.11)**; if it is not listed, run *Developer: Reload Window* from the Command Palette (Cmd/Ctrl+Shift+P). In Jupyter Lab / Notebook, choose the same kernel name. The notebook's setup cell prints the Python it runs on, which should end in `t-difc-env/bin/python`. To remove the kernel entry again: `t-difc-env/bin/jupyter kernelspec uninstall t-difc-env`.

**Why these versions.** The saved models are Keras 2 files, made with TensorFlow 2.13. TensorFlow 2.13 supports Python 3.8–3.11 and needs NumPy ≤ 1.24.3 and typing-extensions < 4.6, which is why python-pptx (0.6.23) and IPython (8.18.1) are pinned to older versions. TensorFlow 2.16 and later (the versions for Python 3.12+) come with Keras 3, which cannot load the models; `fusion_model.py` checks the Keras version and stops with a message if a newer TensorFlow is used.

## Running

With the environment active, run a step with `python scripts/<script>.py` (from any folder).

### A. Labeled data (steps 1–4)

These steps need the raw recordings and the earlier labels (`data/raw-data/`, `data/previous-labels/`), which are not in the repository. Their output, the splits in `data/train-val-test/`, is.

| Step | Script | What it does | Output |
|---|---|---|---|
| 1 | `01_select_scans.py` | Keeps the MM / LLC / L1210A scans and sorts them into Regular or Pulsed (Ch1 min < 150 and max > 300 → pulsed) | `data/selected-data/` |
| 2 | `02_preprocess_labeled_scans.py` | Otsu laser-ON segmentation; 2 s rolling-median background removal; adaptive 4.5σ peak detection, with candidates closer than 100 ms merged (S1.1–S1.2). Writes Control label CSVs (all 0). | `data/preprocessed-data/`, `data/labeled-data/Control/` |
| 3 | `03_transfer_labels.py` | Gives each new Injected candidate the label of the earlier labeled peak within ±5 samples (unmatched: −1); prints label counts | `data/labeled-data/Injected/` |
| 4 | `04_build_splits.py` | Scan-level 70/10/20 split; 2,001-sample windows; joint-channel standardization; train-set augmentation and 1:1 class balancing (S1.3, S2) | `data/train-val-test/` |

### B. Model (steps 5–7)

| Step | Script | What it does | Output |
|---|---|---|---|
| 5 | `05_train.py` | Trains the 5 seeds (S3.1–S3.4). Overwrites the models in `models/` | `models/*.keras`, `training_history.csv`, `predictions_{val,test}.csv` |
| 6 | `06_evaluate.py` | Loss curves, ROC, probability distributions, confusion matrix; metrics with scan-level bootstrap CIs | `models/metrics_*.csv`, `models/figures/` |
| 7 | `07_tsne_figure.py` | t-SNE of the temporal and spectral branch features, seed 0 | `models/figures/tsne_branches.*` |

### C. Clearance cohort (steps 8–12)

The clearance cohort is the set of `data/clearance-cohort-raw/` scans listed per cell line in `clearance_cohort` in `scripts/parameters.py` (LLC 5, L1210A 5, MM 5, Control 8 scans). These steps use only those scans, take each scan's cell line from the list, and group them in the summary, barcodes and clearance fits. To change the cohort, edit the lists and rerun 8–12. Two second sessions, `BCLLC-14_250724` and `BCL12-3-4_250825`, are in the lists as comments, i.e. left out. MC38 scans are refused (they are analysed with step 13).

| Step | Script | What it does | Output (in `results/clearance-cohort/`) |
|---|---|---|---|
| 8 | `08_clearance_cohort_split_overlap.py` | Unpacks new zips and preprocesses each cohort scan once (cached). Flags the candidates that are windows already in the splits. | `per-scan/<scan>/split_overlap.csv` |
| 9 | `09_clearance_cohort_score.py` | Scores every candidate with the 5-seed mean (S3.7); counts at the cut-off, per scan and per cell line | `per-scan/<scan>/candidates.csv`, `scan_summary.csv` |
| 10 | `10_clearance_cohort_scan_figures.py` | Per-scan timeline and window galleries | `per-scan/<scan>/` figures |
| 11 | `11_clearance_cohort_barcodes.py` | Barcode plots per cell line | `barcodes/` |
| 12 | `12_clearance_cohort_clearance_fit.py` | Exponential clearance fits, half-lives, cell-line tests (needs 11) | `clearance/` |

Steps 8, 9 and 10 also accept scan names, e.g. `python scripts/09_clearance_cohort_score.py BCMM4`. The names must be in the cohort lists. Only those scans are processed, and their rows are replaced in `scan_summary.csv`.

### D. Exploratory scans (steps 13 and 14)

These scans are analysed one at a time, on their own: a scan is never grouped with the cohort or with any other scan, and the step needs no other step to run first. Both steps run the same pipeline (`independent_scan.py`): preprocessing, split-overlap check, 5-seed scoring, then summary, barcode, timeline and window galleries of that scan; step 13 (continuous) also fits the clearance of that scan, step 14 (pulsed) does not. Cell line and mouse label are read from the `.mat` file names.

| Step | Script | Input | Output |
|---|---|---|---|
| 13 | `13_continuous_exploratory_scan.py <scan>` | `data/continuous-exploratory-raw/<scan>/`, every file preprocessed as a continuous recording; any cell line | `results/continuous-exploratory/<scan>/` |
| 14 | `14_pulsed_exploratory_scan.py "<path>"` or `<scan>` | the scan folder at `<path>`, linked into `data/pulsed-exploratory-raw/<folder name>/`, or a folder already in `data/pulsed-exploratory-raw/`; pulsed or continuous detected per file, as for the cohort; prints a warning if a file is continuous | `results/pulsed-exploratory/<scan>/` |

For example, put a continuous scan's `.mat` files in `data/continuous-exploratory-raw/<scan>/` and run `python scripts/13_continuous_exploratory_scan.py <scan>`.

Step 14 takes the path of a scan folder anywhere, including a network drive: `python scripts/14_pulsed_exploratory_scan.py "/path/to/<scan>"`, or on macOS an `smb://<server>/<share>/<folder>` address of a connected share (read from `/Volumes/<share>/<folder>`, where macOS mounts it). Quote the path if it has spaces (`%20` also works). The folder is linked, not copied, and the folder name becomes the scan name. The `.mat` files are only read; caches and results are written inside the repository. A network drive must stay connected while the scan is rerun. For step 13, link a folder by hand: `ln -s "/path/to/<scan>" data/continuous-exploratory-raw/<scan>`.

**Notebook version of step 14:** `scripts/14_pulsed_exploratory_scan.ipynb`. Set `DATA_PATH` in the first cell to the scan folder (a path or an `smb://` address, as for the script) and run all cells with the `t-difc-env (Python 3.11)` kernel (see Setup). It runs the same analysis as the script (same files in `results/pulsed-exploratory/<scan>/`) and shows the summary, timeline, barcode, the candidate table and the 1-s raw window of every candidate with its probability, in two parts: the detected peaks (p ≥ cut-off) and the artifacts (p < cut-off), each sorted by time or probability (`SORT_BY`, `MAX_WINDOWS`). The saved outputs in the repository are from its last run.

**Notebook version of step 13:** `scripts/13_continuous_exploratory_scan.ipynb`. Set `SCAN` in the first cell to a folder in `data/continuous-exploratory-raw/`, or `DATA_PATH` to a scan folder anywhere (local or NAS, linked in as by step 14), and run all cells with the `t-difc-env (Python 3.11)` kernel. It runs the same analysis as the script (same files in `results/continuous-exploratory/<scan>/`) and shows the summary, timeline, barcode, the clearance fit (decay plot, fit values, counts per 5-min bin), the candidate table with a histogram of the probabilities, and the 1-s raw windows of the detected peaks and the artifacts (`SORT_BY`, `TIME_RANGE_H`, `MAX_WINDOWS`; at most 48 windows per part by default, since a continuous scan can have thousands of peaks). It is saved without outputs.

### Method figures

| Script | Figure |
|---|---|
| `fig_architecture.py` | Fig. 1, model architecture (editable `.pptx`) |
| `fig_spectral_input.py [5596\|6235] [--real]` | Fig. S1, spectral-input preparation for a peak (5596) or an artifact (6235) |
| `fig_phase_timing.py` | How cos Δφ encodes the Ch1–Ch2 timing (synthetic) |

### Library modules (imported, not run)

| File | Contents |
|---|---|
| `parameters.py` | **Settings:** sampling rate, peak-detection parameters, the classifier cut-off (`cut_off = 0.6`) and the clearance-cohort scans per cell line (`clearance_cohort`) |
| `common.py` | Model name; the cut-off and cohort (from `parameters.py`); every path; split loading; metrics; scan-level cluster bootstrap |
| `peak_detection.py` | File loading, Otsu threshold, laser-ON segments, background removal, peak detection |
| `signal_functions.py` | Window extraction, masking, joint standardization; ALS baseline removal with Ch2 scaling; label matching |
| `full_scan_preprocessing.py` | Full-scan loading, preprocessing and caching (same steps as step 2): `prepare_scan(name, pred_root, prep_root, pulsed)`, where `pulsed=False` forces continuous; `cohort_scans` (validated cohort list) |
| `scoring.py` | Split-overlap check, the 5 seeds, 5-seed scoring, candidate table, per-scan summary row (shared by steps 9, 13, 14) |
| `scan_figures.py` | Timeline, window galleries, per-scan outputs, cell-line colours and labels |
| `clearance_fit.py` | 5-min binning, single and double exponential, reduced χ² |
| `independent_scan.py` | The pipeline of steps 13 and 14 for one exploratory scan; `link_scan` (scan folder given by a path) |
| `fusion_model.py` | Spectral input (`als_scaled(X, name, cache)`, `centre_spectrum`, `Standardizer`) and the architecture (`build_encoder`, `spectral_branch`, `head`, `build("fusion")`) |

## Reproducibility

- Step 6 reproduces `models/metrics_at_0.6.csv` exactly (test set: AUC 0.980, sensitivity 0.843, specificity 0.986 at the 0.6 cut-off). Re-scoring the validation and test windows with the saved seeds matches `predictions_{val,test}.csv` to 3e-8.
- Rerunning steps 8–12 on the cohort recordings reproduces `results/clearance-cohort/` exactly. Detected peaks per cell line: Control 7, L1210A 1,357, LLC 627, MM 1,589.
- Python 3.11 and Python 3.8 (with TensorFlow 2.13) give identical tables (probabilities, detections, metrics, clearance fits) and pixel-identical figures.
- The ALS caches rebuilt from scratch in a fresh clone are identical to the ones the results were made with.
- Step 5 (training) was not rerun for these checks: TensorFlow training is not bit-identical across machines, so retrained seeds can give slightly different metrics.
