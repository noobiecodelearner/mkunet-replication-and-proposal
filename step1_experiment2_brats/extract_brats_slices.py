"""
Experiment 2 (Domain Generalization): BraTS2020 -> 2D axial slice extraction
for MK-UNet binary tumor segmentation.

Design decisions (documented here for the Step 1 report):
- Modality: FLAIR only (best whole-tumor visibility; edema dominates tumor
  volume and is most visible on FLAIR).
- Grayscale FLAIR slice is replicated across 3 channels to match MK-UNet's
  in_channels=3 expectation, rather than modifying the architecture's first
  conv layer. This keeps the architecture identical to the reference-replicated
  version (Experiment 1), isolating the *data* domain shift as the only
  variable -- modifying the model itself would confound the generalization
  test.
- Multi-class BraTS labels (1=necrotic/non-enhancing, 2=edema, 4=enhancing)
  are collapsed to a single binary whole-tumor mask (label > 0), matching
  MK-UNet's binary segmentation setup used across all 6 of its benchmarks.
- Only slices containing at least `min_tumor_pixels` tumor pixels are kept.
  This avoids the dataset being dominated by empty (all-background) slices,
  which would trivially inflate Dice/accuracy without testing anything
  meaningful. The discarded-slice count is logged explicitly so the class-
  imbalance decision is transparent, not hidden.
- Splits are done at the PATIENT level (not slice level) to prevent data
  leakage between train/val/test -- slices from the same patient are highly
  spatially correlated.
- Resolution: 256x256, matching MK-UNet's resolution for its non-polyp
  benchmarks (BUSI/ISIC18/DSB18/EM) in the original paper, since BraTS's
  native 240x240 in-plane resolution is much closer to 256 than to the
  352x352 used only for polyp datasets.

Usage:
    python extract_brats_slices.py --num_patients 50
"""
import argparse
import os
import random
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm

RAW_ROOT = "raw/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
OUT_ROOT = "brats_2d"


def load_patient_ids(raw_root, num_patients, seed=42):
    all_ids = sorted([
        d for d in os.listdir(raw_root)
        if d.startswith("BraTS20_Training_") and os.path.isdir(os.path.join(raw_root, d))
    ])
    rng = random.Random(seed)
    rng.shuffle(all_ids)
    return all_ids[:num_patients] if num_patients else all_ids


def split_patients(patient_ids, train_frac=0.8, val_frac=0.1, seed=42):
    ids = patient_ids[:]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }


def extract_patient_slices(patient_id, raw_root, out_root, split, img_size, min_tumor_pixels):
    patient_dir = os.path.join(raw_root, patient_id)
    flair_path = os.path.join(patient_dir, f"{patient_id}_flair.nii")
    seg_path = os.path.join(patient_dir, f"{patient_id}_seg.nii")

    if not (os.path.exists(flair_path) and os.path.exists(seg_path)):
        return 0, 0  # kept, discarded

    flair_vol = nib.load(flair_path).get_fdata()  # (H, W, num_slices)
    seg_vol = nib.load(seg_path).get_fdata()

    img_out_dir = os.path.join(out_root, split, "images")
    mask_out_dir = os.path.join(out_root, split, "masks")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    num_slices = flair_vol.shape[2]
    kept, discarded = 0, 0

    for z in range(num_slices):
        seg_slice = seg_vol[:, :, z]
        binary_mask = (seg_slice > 0).astype(np.uint8)

        if binary_mask.sum() < min_tumor_pixels:
            discarded += 1
            continue

        flair_slice = flair_vol[:, :, z]

        # Normalize FLAIR slice to 0-255 (per-slice min-max; simple and
        # standard for this kind of MRI intensity normalization task)
        f_min, f_max = flair_slice.min(), flair_slice.max()
        if f_max - f_min < 1e-6:
            discarded += 1
            continue
        flair_norm = ((flair_slice - f_min) / (f_max - f_min) * 255).astype(np.uint8)

        flair_resized = cv2.resize(flair_norm, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(binary_mask * 255, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

        # Replicate grayscale across 3 channels (see module docstring)
        flair_rgb = np.stack([flair_resized] * 3, axis=-1)

        slice_name = f"{patient_id}_z{z:03d}.png"
        cv2.imwrite(os.path.join(img_out_dir, slice_name), flair_rgb)
        cv2.imwrite(os.path.join(mask_out_dir, slice_name), mask_resized)
        kept += 1

    return kept, discarded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_patients", type=int, default=50,
                        help="Number of patients to use (None/0 = all 369)")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--min_tumor_pixels", type=int, default=50,
                        help="Minimum tumor pixels (in native 240x240 space) for a slice to be kept")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    patient_ids = load_patient_ids(RAW_ROOT, args.num_patients, seed=args.seed)
    splits = split_patients(patient_ids, seed=args.seed)

    print(f"Total patients selected: {len(patient_ids)}")
    for split_name, ids in splits.items():
        print(f"  {split_name}: {len(ids)} patients")

    summary = {}
    for split_name, ids in splits.items():
        total_kept, total_discarded = 0, 0
        for pid in tqdm(ids, desc=f"Extracting {split_name}"):
            kept, discarded = extract_patient_slices(
                pid, RAW_ROOT, OUT_ROOT, split_name, args.img_size, args.min_tumor_pixels
            )
            total_kept += kept
            total_discarded += discarded
        summary[split_name] = (total_kept, total_discarded)

    print("\n=== Extraction summary ===")
    for split_name, (kept, discarded) in summary.items():
        total = kept + discarded
        pct = 100 * kept / total if total else 0
        print(f"{split_name}: {kept} slices kept, {discarded} discarded "
              f"(empty/low-tumor) out of {total} total ({pct:.1f}% kept)")

    # Save the patient-level split assignment for reproducibility / reporting
    with open(os.path.join(OUT_ROOT, "patient_split.txt"), "w") as f:
        for split_name, ids in splits.items():
            f.write(f"# {split_name} ({len(ids)} patients)\n")
            for pid in ids:
                f.write(f"{split_name}\t{pid}\n")


if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)
    main()
