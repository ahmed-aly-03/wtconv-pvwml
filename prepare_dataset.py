"""Reorganize registered FLAIR volumes into the 4-class train/val/test
ImageFolder layout used by train_multitask.py / train_tdlf_finetune.py,
driven by WML_Prevalance.xlsx's `New_Cohort` column.

Class grouping (as specified):
    Non-vascular: ADMCI, ALS, FTD, LBD, PD, SCI   (SCI -- the sheet has no
                  "SC" value; SCI is almost certainly what was meant. If
                  that's wrong, edit NON_VASCULAR_CODES below.)
    Vascular:     CVD
    Control:      CN, with Age > --control-max-age (default 70) excluded,
                  and rows with missing Age excluded (can't verify the cutoff).
    vMCIAD:       vMCIAD

Each Excel row is one (subject, visit) volume. Splitting is done at the
SUBJECT level (by the `ID` column), not per-row, so a subject's multiple
visits never end up split across train/val/test. All valid visits/rows for
an included subject are used (not baseline-only) -- override this design
choice by filtering the dataframe yourself if you want baseline-only.

The Database -> folder name mapping handles a naming inconsistency: the
Excel's Database column says "ADNI", but the folder on disk was
"ANDI_Registered" as of 2026-08 -- override with --adni-folder-name if
that gets fixed server-side.

Run with --dry-run first: it does everything except the slow slice
extraction, and prints/writes a full coverage report so you can catch a
wrong --adni-folder-name or path assumption before committing to the full
run (which processes ~7,000 volumes).

Example:
    python prepare_dataset.py \
        --excel-path /path/to/WML_Prevalance.xlsx \
        --data-root "/home/sharedFolder/Spatial WML/Data" \
        --output-dir /home/ra/aaly/WTConv/wtconvnext/swml_vols_full_4class \
        --dry-run

    # then, once the coverage report looks right, drop --dry-run.
"""
import argparse
import os
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from nifti_utils import SliceExtractionConfig, extract_and_save  # noqa: E402

NON_VASCULAR_CODES = {"ADMCI", "ALS", "FTD", "LBD", "PD", "SCI"}
VASCULAR_CODES = {"CVD"}
CONTROL_CODES = {"CN"}
VMCIAD_CODES = {"vMCIAD"}

CLASS_FOLDER_NAMES = {
    "control": "Control",
    "non_vascular": "Non-vascular (ADMCI, ALS, FTD, LBD, PD, SC)",
    "vascular": "Vascular (CVD)",
    "vmciad": "vMCIAD",
}


def classify_cohort(new_cohort: str) -> Optional[str]:
    if new_cohort in NON_VASCULAR_CODES:
        return "non_vascular"
    if new_cohort in VASCULAR_CODES:
        return "vascular"
    if new_cohort in CONTROL_CODES:
        return "control"
    if new_cohort in VMCIAD_CODES:
        return "vmciad"
    return None


def build_filepath(row, data_root: str, adni_folder_name: str) -> str:
    database = row["Database"]
    folder_name = adni_folder_name if database == "ADNI" else database
    filename = str(row["FileName"])
    if not filename.endswith(".nii.gz"):
        filename += ".nii.gz"
    return os.path.join(data_root, f"{folder_name}_Registered", folder_name, "vols", filename)


def subject_level_split(subjects_by_class: Dict[str, List[str]], train_frac: float, val_frac: float, test_frac: float, seed: int) -> Dict[str, str]:
    """Returns {subject_id: split_name}, stratified by class at the subject level."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "fractions must sum to 1.0"
    rng = random.Random(seed)
    subject_to_split: Dict[str, str] = {}

    for cls, subjects in subjects_by_class.items():
        subjects = sorted(set(subjects))
        rng.shuffle(subjects)
        n = len(subjects)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        for s in subjects[:n_train]:
            subject_to_split[s] = "train"
        for s in subjects[n_train:n_train + n_val]:
            subject_to_split[s] = "val"
        for s in subjects[n_train + n_val:]:
            subject_to_split[s] = "test"

    return subject_to_split


@dataclass
class Job:
    filepath: str
    out_dir: str
    base_name: str


def _run_job(job: Job, cfg: SliceExtractionConfig):
    try:
        written = extract_and_save(job.filepath, job.out_dir, job.base_name, cfg)
        return job.filepath, len(written), None
    except Exception as e:  # noqa: BLE001 -- one bad volume must not kill a 7000-file batch
        return job.filepath, 0, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel-path", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True, help='e.g. "/home/sharedFolder/Spatial WML/Data"')
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--adni-folder-name", type=str, default="ANDI",
                         help="Folder name on disk for Database=='ADNI' rows (observed typo as of 2026-08).")
    parser.add_argument("--control-max-age", type=float, default=70.0,
                         help="Control (CN) rows with Age > this are excluded. Rows with missing Age are also excluded.")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--slice-fraction", type=float, default=0.5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--min-nonzero-frac", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--dry-run", action="store_true",
                         help="Do filtering/mapping/splitting and write the coverage report and manifest, but skip the slow slice extraction.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_excel(args.excel_path, sheet_name="Sheet1")
    total_rows = len(df)

    # Age can contain non-numeric values in real clinical exports (e.g. ADNI's
    # ">90" top-coding for de-identification, or blank placeholders), which
    # makes the whole column dtype 'object' and breaks numeric comparison.
    # Coerce to numeric; anything unparseable becomes NaN, which the
    # missing-age branch below already excludes Control rows for.
    age_raw_non_numeric = df["Age"].apply(lambda v: pd.notna(v) and not isinstance(v, (int, float)))
    if age_raw_non_numeric.any():
        examples = df.loc[age_raw_non_numeric, "Age"].unique()[:10]
        print(f"Note: {age_raw_non_numeric.sum()} rows have a non-numeric Age value (e.g. {list(examples)}); "
              f"coercing to numeric, unparseable ones become NaN (Control rows with NaN Age are excluded below).")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    # --- filter: valid label ---
    df["class_key"] = df["New_Cohort"].apply(lambda v: classify_cohort(v) if pd.notna(v) else None)
    unmapped = df[df["class_key"].isna()]
    if len(unmapped) > 0:
        unmapped_codes = Counter(unmapped["New_Cohort"].dropna().tolist())
        print(f"Excluding {len(unmapped)} rows with unmapped/missing New_Cohort values: {dict(unmapped_codes)}")
        unmapped.to_csv(os.path.join(args.output_dir, "excluded_unmapped_cohort.csv"), index=False)
    df = df[df["class_key"].notna()].copy()

    # --- filter: control age cutoff ---
    is_control = df["class_key"] == "control"
    age_missing = is_control & df["Age"].isna()
    age_too_old = is_control & df["Age"].notna() & (df["Age"] > args.control_max_age)
    excluded_control = df[age_missing | age_too_old]
    if len(excluded_control) > 0:
        print(f"Excluding {len(excluded_control)} Control rows (age > {args.control_max_age} or missing age): "
              f"{age_too_old.sum()} too old, {age_missing.sum()} missing age.")
        excluded_control.to_csv(os.path.join(args.output_dir, "excluded_control_age.csv"), index=False)
    df = df[~(age_missing | age_too_old)].copy()

    # --- build filepaths, check existence ---
    df["filepath"] = df.apply(lambda r: build_filepath(r, args.data_root, args.adni_folder_name), axis=1)
    df["exists"] = df["filepath"].apply(os.path.exists)

    coverage = df.groupby(["Database", "class_key"])["exists"].agg(["sum", "count"])
    print("\n=== File coverage by Database x class (found / total) ===")
    print(coverage)

    missing = df[~df["exists"]]
    if len(missing) > 0:
        print(f"\n{len(missing)}/{len(df)} rows point to a file that does not exist on disk -- these are EXCLUDED.")
        print("First 10 missing paths:")
        for p in missing["filepath"].head(10):
            print(f"  {p}")
        missing.to_csv(os.path.join(args.output_dir, "missing_files.csv"), index=False)
        print(f"Full list written to {os.path.join(args.output_dir, 'missing_files.csv')}")
    df = df[df["exists"]].copy()

    if len(df) == 0:
        print("\nNo files found on disk -- stopping. Check --data-root and --adni-folder-name.")
        return

    # --- subject-level class assignment (mode, in case of rare inconsistency) ---
    subject_class = df.groupby("ID")["class_key"].agg(lambda s: s.value_counts().idxmax())
    inconsistent = df.groupby("ID")["class_key"].nunique()
    inconsistent_ids = inconsistent[inconsistent > 1].index.tolist()
    if inconsistent_ids:
        print(f"\n{len(inconsistent_ids)} subjects have inconsistent New_Cohort across visits; using their majority class. IDs: {inconsistent_ids[:10]}{' ...' if len(inconsistent_ids) > 10 else ''}")

    subjects_by_class: Dict[str, List[str]] = {}
    for sid, cls in subject_class.items():
        subjects_by_class.setdefault(cls, []).append(sid)

    print("\n=== Subjects per class (before split) ===")
    for cls, subs in subjects_by_class.items():
        print(f"  {CLASS_FOLDER_NAMES[cls]}: {len(subs)} subjects, {len(df[df['ID'].isin(subs)])} volumes")

    subject_to_split = subject_level_split(subjects_by_class, args.train_frac, args.val_frac, args.test_frac, args.seed)
    df["split"] = df["ID"].map(subject_to_split)

    print("\n=== Split summary (subjects / volumes) ===")
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        n_subj = split_df["ID"].nunique()
        print(f"  {split}: {n_subj} subjects, {len(split_df)} volumes")
        for cls in CLASS_FOLDER_NAMES:
            cls_df = split_df[split_df["class_key"] == cls]
            print(f"    {CLASS_FOLDER_NAMES[cls]}: {cls_df['ID'].nunique()} subjects, {len(cls_df)} volumes")

    df["out_dir"] = df.apply(
        lambda r: os.path.join(args.output_dir, r["split"], CLASS_FOLDER_NAMES[r["class_key"]]), axis=1
    )
    df["base_name"] = df.apply(
        lambda r: f"{r['Database']}_" + os.path.basename(r["filepath"]).replace(".nii.gz", ""), axis=1
    )

    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    df[["FileName", "ID", "Database", "New_Cohort", "class_key", "Age", "split", "filepath", "out_dir", "base_name"]].to_csv(manifest_path, index=False)
    print(f"\nManifest written to {manifest_path} ({len(df)} volumes total).")

    if args.dry_run:
        print("\n--dry-run set: skipping slice extraction. Review the coverage report and manifest above, then rerun without --dry-run.")
        return

    # --- extraction ---
    cfg = SliceExtractionConfig(
        slice_fraction=args.slice_fraction,
        img_size=args.img_size,
        min_nonzero_frac=args.min_nonzero_frac,
    )

    jobs = [Job(filepath=r["filepath"], out_dir=r["out_dir"], base_name=r["base_name"]) for _, r in df.iterrows()]
    print(f"\nExtracting slices from {len(jobs)} volumes using {args.num_workers} workers...")

    total_slices = 0
    errors = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(_run_job, job, cfg): job for job in jobs}
        for future in as_completed(futures):
            filepath, n_slices, error = future.result()
            done += 1
            total_slices += n_slices
            if error:
                errors.append((filepath, error))
            if done % 200 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] volumes processed, {total_slices} slices written so far, {len(errors)} errors")

    if errors:
        errors_path = os.path.join(args.output_dir, "extraction_errors.csv")
        pd.DataFrame(errors, columns=["filepath", "error"]).to_csv(errors_path, index=False)
        print(f"\n{len(errors)} volumes failed to extract -- see {errors_path}")

    print(f"\nDone. {total_slices} slice PNGs written across {len(jobs) - len(errors)} volumes into {args.output_dir}")
    print("Final per-split/class slice counts:")
    for split in ["train", "val", "test"]:
        for cls, folder_name in CLASS_FOLDER_NAMES.items():
            d = os.path.join(args.output_dir, split, folder_name)
            n = len(os.listdir(d)) if os.path.isdir(d) else 0
            print(f"  {split}/{folder_name}: {n}")


if __name__ == "__main__":
    main()
