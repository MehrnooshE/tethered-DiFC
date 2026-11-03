"""
Step 3. Transfer the earlier labels to the new peak candidates of the
Injected scans and write the new label CSVs, with a per-folder summary
report and the label counts.

Assumptions (change constants below if wrong):
- New candidate locations (.npy) are in the same units/scale as the
  'Time(Hz)' column in the old label CSVs (i.e. directly comparable).
- Matching tolerance is +/- 5 units (samples) in that same scale.
- If a new candidate matches multiple old labeled peaks within tolerance,
  the closest one is used.
- Unmatched new candidates get Label = -1.

Directory layout expected:
  Preprocessed dir: data/preprocessed-data/Injected/{Pulsed,Regular}/<name>/<name>_filtered_peaks_1.npy
                    (02_preprocess_labeled_scans.py)
  Old labels:       data/previous-labels/Injected/<name>.csv   (columns: Time(Hz), Label)
  New labels:       data/labeled-data/Injected/{Pulsed,Regular}/<name>.csv
Control labels (all 0) are written by 02_preprocess_labeled_scans.py.
Usage:  python 03_transfer_labels.py
"""

import os
import glob
from collections import Counter
import numpy as np
import pandas as pd

import common as mc
import signal_functions as sf

# ---- Config ----
PREPROCESSED_DIR = os.path.join(mc.PREPROCESSED_DIR, "Injected")
OLD_LABELS_DIR = os.path.join(mc.PREVIOUS_LABELS_DIR, "Injected")
NEW_LABELS_BASE_DIR = os.path.join(mc.LABELS_DIR, "Injected")

CATEGORIES = ["Pulsed", "Regular"]
TOLERANCE = 5  # +/- samples/units for matching
UNLABELED_VALUE = -1


def process_category(category: str, report: list):
    category_dir = os.path.join(PREPROCESSED_DIR, category)
    if not os.path.isdir(category_dir):
        print(f"[SKIP] Preprocessed category dir not found: {category_dir}")
        return

    out_dir = os.path.join(NEW_LABELS_BASE_DIR, category)
    os.makedirs(out_dir, exist_ok=True)

    folder_names = sorted(
        d for d in os.listdir(category_dir)
        if os.path.isdir(os.path.join(category_dir, d))
    )

    for name in folder_names:
        # npy filename is "<name>_filtered_peaks_1.npy", not "<name>.npy"
        npy_matches = sorted(glob.glob(os.path.join(category_dir, name, f"{name}_filtered_peaks_*.npy")))
        old_csv_path = os.path.join(OLD_LABELS_DIR, f"{name}.csv")
        out_csv_path = os.path.join(out_dir, f"{name}.csv")

        if not npy_matches:
            print(f"[WARN] Missing npy for '{name}' in {category}: "
                  f"{os.path.join(category_dir, name, f'{name}_filtered_peaks_*.npy')}")
            continue
        if len(npy_matches) > 1:
            print(f"[WARN] Multiple npy matches for '{name}' in {category}, using {npy_matches[0]}: {npy_matches}")

        npy_path = npy_matches[0]

        new_locations = np.load(npy_path)
        new_locations = np.asarray(new_locations).ravel()

        if os.path.isfile(old_csv_path):
            old_df = pd.read_csv(old_csv_path)
            if "Time(Hz)" not in old_df.columns or "Label" not in old_df.columns:
                print(f"[WARN] Old label CSV missing expected columns for '{name}': "
                      f"{old_csv_path} (found columns: {list(old_df.columns)})")
                old_df = pd.DataFrame(columns=["Time(Hz)", "Label"])
        else:
            old_df = pd.DataFrame(columns=["Time(Hz)", "Label"])

        labels = sf.match_labels(new_locations, old_df, tolerance=TOLERANCE, unlabeled_value=UNLABELED_VALUE)

        out_df = pd.DataFrame({
            "Time(Hz)": new_locations,
            "Label": labels,
        })
        out_df.to_csv(out_csv_path, index=False)

        total = len(new_locations)
        n_labeled = int(sum(1 for l in labels if l != UNLABELED_VALUE))
        n_unlabeled = total - n_labeled

        report.append({
            "Category": category,
            "Folder": name,
            "Total": total,
            "Labeled": n_labeled,
            "Unlabeled": n_unlabeled,
        })

        print(f"[{category}] {name}: total={total}, labeled={n_labeled}, unlabeled={n_unlabeled} "
              f"-> saved to {out_csv_path}")


def label_counts():
    """Label 1 / 0 / -1 counts of the new label CSVs, per category and overall."""
    overall_counts = Counter()

    for category in CATEGORIES:
        cat_dir = os.path.join(NEW_LABELS_BASE_DIR, category)
        if not os.path.isdir(cat_dir):
            print(f"[SKIP] {cat_dir} not found")
            continue

        cat_counts = Counter()
        csv_files = sorted(f for f in os.listdir(cat_dir) if f.endswith(".csv"))

        for fname in csv_files:
            df = pd.read_csv(os.path.join(cat_dir, fname))
            labels = df["Label"].astype(str)
            cat_counts.update(labels)

        print(f"\n=== {category} ({len(csv_files)} files) ===")
        for val in ["1", "0", "-1"]:
            print(f"  Label {val}: {cat_counts.get(val, 0)}")
        other = {k: v for k, v in cat_counts.items() if k not in ["1", "0", "-1"]}
        if other:
            print(f"  Other values: {other}")

        overall_counts.update(cat_counts)

    print(f"\n=== Overall (Pulsed + Regular) ===")
    for val in ["1", "0", "-1"]:
        print(f"  Label {val}: {overall_counts.get(val, 0)}")
    other = {k: v for k, v in overall_counts.items() if k not in ["1", "0", "-1"]}
    if other:
        print(f"  Other values: {other}")


def main():
    report = []
    for category in CATEGORIES:
        process_category(category, report)

    if report:
        report_df = pd.DataFrame(report)
        print("\n=== Summary ===")
        print(report_df.to_string(index=False))

        totals = report_df[["Total", "Labeled", "Unlabeled"]].sum()
        print(f"\nOverall: total={totals['Total']}, labeled={totals['Labeled']}, "
              f"unlabeled={totals['Unlabeled']}")
    else:
        print("No folders processed.")
    label_counts()


if __name__ == "__main__":
    main()
