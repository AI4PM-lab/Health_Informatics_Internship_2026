#!/usr/bin/env python
"""
Inference con singolo modello per fold (0..4), estratto dal notebook originale.
"""

import argparse
from pathlib import Path

import inspect
import re
import shutil
import sys
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from scipy.ndimage import binary_erosion, generate_binary_structure
    from scipy.spatial import cKDTree
except ImportError:
    binary_erosion = None
    generate_binary_structure = None
    cKDTree = None

from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader, MetaTensor, decollate_batch
from monai.transforms import (
    AsDiscreted, Compose, CropForegroundd, EnsureChannelFirstd,
    Invertd, KeepLargestConnectedComponent, KeepLargestConnectedComponentd,
    LoadImaged, Orientationd, ScaleIntensityRanged, Spacingd,
)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def autocast_context():
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    if device.type == "mps":
        return torch.amp.autocast("mps")
    return nullcontext()


print(f"Using device: {device}")


PATIENT_PATTERN = re.compile(r"^TAVI_\d+$")

PROJECT_ROOT = None
DATA_ROOT = None
SPLIT_CSV = None
CHECKPOINTS_ROOT = None
COMPARISON_OUTPUT_DIR = None
INFERENCE_FOLDS = None
MODEL_TYPE = None
MODEL_FILENAME = None
MODEL_LABEL = None
CASE_NAMES = None
MAX_PATIENTS = None
CURRENT_FOLD_INDEX = None

SLICE_PERCENTAGES = [13, 16, 19, 22, 25, 27, 29, 31, 34, 37, 40, 43, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80]
TARGET_SPACING = (1.0, 1.0, 1.0)
ROI_SIZE = (96, 96, 96)
SW_BATCH_SIZE = 4
OVERLAP = 0.5
SW_MODE = "gaussian"
USE_EMA_WEIGHTS = True
NUM_WORKERS = 0
DISPLAY_FIRST_N_PATIENTS = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Single-model inference per fold, senza ensemble.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument("--checkpoints-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-type", choices=["baseline", "finetuned"], default="finetuned")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Fold su cui fare inferenza. Default: 0 1 2 3 4")
    parser.add_argument("--case-names", nargs="+", default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--display-first-n-patients", type=int, default=0)
    parser.add_argument("--no-ema", action="store_true")
    return parser.parse_args()


def initialise_config(args):
    global PROJECT_ROOT, DATA_ROOT, SPLIT_CSV, CHECKPOINTS_ROOT, COMPARISON_OUTPUT_DIR
    global INFERENCE_FOLDS, MODEL_TYPE, MODEL_FILENAME, MODEL_LABEL, CASE_NAMES, MAX_PATIENTS
    global USE_EMA_WEIGHTS, DISPLAY_FIRST_N_PATIENTS

    if args.project_root is not None:
        PROJECT_ROOT = args.project_root.resolve()
    else:
        cwd = Path.cwd().resolve()
        PROJECT_ROOT = cwd.parent if cwd.name in ("scripts", "notebooks") else cwd

    DATA_ROOT = (args.data_root or (PROJECT_ROOT / "notebooks" / "nifti_data")).resolve()
    # Default coerente con il notebook originale: prova prima cv_splits.csv.
    # Se non esiste ma esiste cv_splits_qc.csv, usa automaticamente quello.
    if args.split_csv is not None:
        SPLIT_CSV = args.split_csv.resolve()
    else:
        default_split = PROJECT_ROOT / "data" / "cv_splits.csv"
        qc_split = PROJECT_ROOT / "data" / "cv_splits_qc.csv"
        SPLIT_CSV = (default_split if default_split.exists() else qc_split).resolve()

    CHECKPOINTS_ROOT = (args.checkpoints_root or (PROJECT_ROOT / "notebooks" / "output" / "named-outputs" / "output_model")).resolve()

    INFERENCE_FOLDS = list(args.folds)
    invalid_folds = [fold for fold in INFERENCE_FOLDS if fold not in range(5)]
    if invalid_folds:
        raise ValueError(f"Fold non validi: {invalid_folds}. Usa solo fold 0, 1, 2, 3, 4.")
    MODEL_TYPE = args.model_type
    MODEL_FILENAME = {"baseline": "best_metric_model_baseline.pth", "finetuned": "best_metric_model_finetuned.pth"}[MODEL_TYPE]
    MODEL_LABEL = {"baseline": "Baseline", "finetuned": "Fine-tuned"}[MODEL_TYPE]

    CASE_NAMES = args.case_names
    MAX_PATIENTS = args.max_patients
    USE_EMA_WEIGHTS = not args.no_ema
    DISPLAY_FIRST_N_PATIENTS = int(args.display_first_n_patients)

    folds_tag = "_".join(map(str, INFERENCE_FOLDS))
    COMPARISON_OUTPUT_DIR = (args.output_dir or (PROJECT_ROOT / "notebooks" / "output" / f"{MODEL_TYPE}_single_model_folds_{folds_tag}")).resolve()
    COMPARISON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Controlli iniziali: fallisce subito con un messaggio chiaro se manca qualcosa.
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"DATA_ROOT non trovato: {DATA_ROOT}")
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(f"SPLIT_CSV non trovato: {SPLIT_CSV}")
    if not CHECKPOINTS_ROOT.exists():
        raise FileNotFoundError(f"CHECKPOINTS_ROOT non trovato: {CHECKPOINTS_ROOT}")


def get_model_spec_for_fold(fold_index):
    checkpoint_dir = CHECKPOINTS_ROOT / f"fold_{fold_index}"
    return {
        "fold": fold_index,
        "name": f"{MODEL_TYPE}_fold_{fold_index}",
        "label": f"{MODEL_LABEL} fold {fold_index}",
        "path": checkpoint_dir / MODEL_FILENAME,
        "segmentation_filename": f"segmentation_{MODEL_TYPE}_fold_{fold_index}.nii.gz",
    }


def find_valid_cases(input_dir):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Data root does not exist: {input_dir}")
    cases = []
    for folder in sorted(input_dir.iterdir()):
        if not (folder.is_dir() and PATIENT_PATTERN.match(folder.name)):
            continue
        img = folder / "CT_LATE.nii.gz"
        lbl = folder / "registration_mask.nii.gz"
        if img.exists() and lbl.exists():
            cases.append({"image": str(img), "label": str(lbl), "name": folder.name})
    return cases


def select_cases(cases, split_csv, fold_index, case_names=None, max_patients=None):
    case_by_name = {case["name"]: case for case in cases}
    if case_names:
        selected_names = list(case_names)
    else:
        split_csv = Path(split_csv)
        if not split_csv.exists():
            raise FileNotFoundError(f"CV split file does not exist: {split_csv}")
        split_df = pd.read_csv(split_csv)
        fold_rows = split_df[split_df["fold"].astype(int) == int(fold_index)]
        selected_names = fold_rows["patient_id"].astype(str).tolist()
    missing = [name for name in selected_names if name not in case_by_name]
    if missing:
        print(f"Skipping {len(missing)} fold_{fold_index} patients not found under {DATA_ROOT}:")
        print(", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))
    selected = [case_by_name[name] for name in selected_names if name in case_by_name]
    if max_patients is not None:
        selected = selected[: int(max_patients)]
    if not selected:
        raise ValueError(f"No valid patients selected for fold_{fold_index}.")
    return selected


def verify_model_specs(model_specs):
    missing = [spec for spec in model_specs if not Path(spec["path"]).exists()]
    if missing:
        raise FileNotFoundError("Missing inference checkpoints:\n" + "\n".join(str(spec["path"]) for spec in missing))
    for spec in model_specs:
        path = Path(spec["path"])
        print(f"fold_{spec['fold']} | {spec['label']}: {path} ({path.stat().st_size / (1024**2):.1f} MB)")
    return model_specs


infer_transforms = None
val_transforms_with_label = None
post_transforms = None
keep_largest_cc_preprocessed = None


def initialise_transforms():
    global infer_transforms, val_transforms_with_label, post_transforms, keep_largest_cc_preprocessed
    infer_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        CropForegroundd(keys=["image"], source_key="image"),
        Spacingd(keys=["image"], pixdim=TARGET_SPACING, mode=("bilinear",)),
        ScaleIntensityRanged(keys=["image"], a_min=-100.0, a_max=400.0, b_min=0.0, b_max=1.0, clip=True),
    ])
    val_transforms_with_label = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Spacingd(keys=["image", "label"], pixdim=TARGET_SPACING, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-100.0, a_max=400.0, b_min=0.0, b_max=1.0, clip=True),
    ])
    post_transforms = Compose([
        Invertd(keys="pred", transform=infer_transforms, orig_keys="image", meta_keys="pred_meta_dict", orig_meta_keys="image_meta_dict", meta_key_postfix="meta_dict", nearest_interp=False, to_tensor=True),
        AsDiscreted(keys="pred", argmax=True),
        KeepLargestConnectedComponentd(keys="pred", applied_labels=[1], is_onehot=False, independent=True, connectivity=1, num_components=1),
    ])
    keep_largest_cc_preprocessed = KeepLargestConnectedComponent(applied_labels=[1], is_onehot=False, independent=True, connectivity=1, num_components=1)


MODEL_CONFIG = {
    "spatial_dims": 3,
    "in_channels": 1,
    "out_channels": 2,
    "init_filters": 16,
    "blocks_down": [1, 2, 2, 4],
    "blocks_up": [1, 1, 1],
}


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def build_model():
    return SegResNet(**MODEL_CONFIG)


def torch_load_checkpoint(path):
    load_kwargs = {"map_location": "cpu"}
    try:
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = False
    except (TypeError, ValueError):
        pass

    # Some older checkpoints reference numpy._core during unpickling.
    if hasattr(np, "core"):
        sys.modules["numpy._core"] = np.core
        sys.modules["numpy._core.multiarray"] = np.core.multiarray

    return torch.load(path, **load_kwargs)


def normalize_state_dict_keys(state_dict):
    return {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}


def load_checkpoint_to_model(model, checkpoint_path, use_ema_weights=True):
    checkpoint = torch_load_checkpoint(checkpoint_path)
    state_key = "model_state_dict"

    if isinstance(checkpoint, dict):
        if use_ema_weights and checkpoint.get("ema_model_state_dict"):
            state_key = "ema_model_state_dict"
        state_dict = checkpoint.get(state_key, checkpoint)
    else:
        state_dict = checkpoint

    model.load_state_dict(normalize_state_dict_keys(state_dict))
    model.to(device)
    model.eval()

    metadata = {
        "checkpoint": str(checkpoint_path),
        "state_key": state_key,
        "step": checkpoint.get("step") if isinstance(checkpoint, dict) else None,
        "best_dice": checkpoint.get("best_dice") if isinstance(checkpoint, dict) else None,
        "loss_name": checkpoint.get("loss_name") if isinstance(checkpoint, dict) else None,
    }
    return metadata


def load_comparison_models(model_specs):
    models = {}
    metadata = {}
    for spec in model_specs:
        model = build_model()
        meta = load_checkpoint_to_model(model, spec["path"], USE_EMA_WEIGHTS)
        models[spec["name"]] = {**spec, "model": model}
        metadata[spec["name"]] = meta
        dice_text = "n/a" if meta["best_dice"] is None else f"{float(meta['best_dice']):.4f}"
        print(f"Loaded {spec['label']} from {meta['state_key']} | checkpoint best Dice: {dice_text}")
    return models, metadata


def prepare_patient_batch(patient_case):
    ds = Dataset(data=[patient_case], transform=infer_transforms)
    loader = DataLoader(ds, batch_size=1, num_workers=NUM_WORKERS)
    return next(iter(loader))


def load_preprocessed_patient(patient_case):
    sample = val_transforms_with_label(patient_case)
    ct_vol = as_numpy(sample["image"])[0].astype(np.float32)
    gt_mask = as_numpy(sample["label"])[0].astype(np.uint8)
    return ct_vol, gt_mask


def meta_to_plain_dict(meta):
    if meta is None:
        return {}
    if not isinstance(meta, dict):
        meta = dict(meta)
    plain = {}
    for key, value in meta.items():
        plain[key] = as_numpy(value) if isinstance(value, torch.Tensor) else value
    return plain


def affine_from_meta(*metas):
    for meta in metas:
        meta = meta_to_plain_dict(meta)
        affine = meta.get("affine")
        if affine is not None:
            affine = as_numpy(affine)
            if affine.ndim == 3 and affine.shape[0] == 1:
                affine = affine[0]
            return np.asarray(affine, dtype=np.float64)
    return np.eye(4, dtype=np.float64)


def squeeze_mask(mask):
    mask = as_numpy(mask)
    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask[0]
    elif mask.ndim == 4 and mask.shape[0] > 1:
        mask = np.argmax(mask, axis=0)
    return mask.astype(np.uint8)


def save_mask_like_reference(mask, reference_path, output_path):
    reference_img = nib.load(str(reference_path))
    mask_np = squeeze_mask(mask)
    if mask_np.shape != reference_img.shape:
        raise ValueError(
            f"Cannot save {output_path}: mask shape {mask_np.shape} "
            f"does not match reference shape {reference_img.shape}"
        )

    header = reference_img.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1, 0)

    output_img = nib.Nifti1Image(mask_np.astype(np.uint8), reference_img.affine, header=header)
    qform_code = int(reference_img.header["qform_code"]) or 1
    sform_code = int(reference_img.header["sform_code"]) or 1
    output_img.set_qform(reference_img.affine, qform_code)
    output_img.set_sform(reference_img.affine, sform_code)
    nib.save(output_img, str(output_path))
    return mask_np


def save_prediction_in_original_space(probs, batch, output_path, reference_path):
    src_meta = batch["image"].meta if hasattr(batch["image"], "meta") else batch.get("image_meta_dict")
    predictions = MetaTensor(probs.detach().cpu(), meta=src_meta)
    batch_data = {"image": batch["image"], "pred": predictions}

    for item in decollate_batch(batch_data):
        if "image_meta_dict" not in item:
            item["image_meta_dict"] = batch["image"].meta if hasattr(batch["image"], "meta") else src_meta
        if "pred_meta_dict" not in item:
            item["pred_meta_dict"] = dict(meta_to_plain_dict(item["image_meta_dict"]))

        item["pred_meta_dict"] = meta_to_plain_dict(item.get("pred_meta_dict"))
        item["image_meta_dict"] = meta_to_plain_dict(item.get("image_meta_dict"))
        item = post_transforms(item)

        pred_np = squeeze_mask(item["pred"])
        return save_mask_like_reference(pred_np, reference_path, output_path)

    raise RuntimeError("No prediction item produced after decollation.")


def copy_slicer_inputs(patient_case, patient_dir):
    copies = {
        "volume_CT_LATE.nii.gz": patient_case["image"],
        "ground_truth_mask.nii.gz": patient_case["label"],
    }
    for filename, source in copies.items():
        destination = patient_dir / filename
        if not destination.exists():
            shutil.copy2(source, destination)


def release_device_cache():
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def run_models_for_patient(patient_case, models, patient_dir):
    batch = prepare_patient_batch(patient_case)
    inputs = batch["image"].to(device)
    results = {}

    with torch.inference_mode():
        for model_name, model_payload in models.items():
            print(f"    {model_payload['label']} inference ...")
            with autocast_context():
                logits = sliding_window_inference(
                    inputs,
                    ROI_SIZE,
                    SW_BATCH_SIZE,
                    model_payload["model"],
                    overlap=OVERLAP,
                    mode=SW_MODE,
                )
                probs = torch.softmax(logits, dim=1)

            pred_mask_raw = torch.argmax(probs, dim=1).detach().cpu().numpy()[0].astype(np.uint8)
            pred_mask_preprocessed = as_numpy(
                keep_largest_cc_preprocessed(torch.as_tensor(pred_mask_raw[None, ...], dtype=torch.uint8))
            )[0].astype(np.uint8)
            pred_original_path = patient_dir / model_payload["segmentation_filename"]
            pred_mask_original = save_prediction_in_original_space(
                probs,
                batch,
                pred_original_path,
                patient_dir / "volume_CT_LATE.nii.gz",
            )

            results[model_name] = {
                "label": model_payload["label"],
                "pred_mask": pred_mask_preprocessed,
                "pred_mask_raw": pred_mask_raw,
                "pred_original_shape": tuple(pred_mask_original.shape),
                "prediction_path": pred_original_path,
            }
            del logits, probs
            release_device_cache()

    return results


def dice_score_binary(pred_mask, gt_mask):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    denom = pred_mask.sum() + gt_mask.sum()
    if denom == 0:
        return 1.0
    return 2.0 * np.logical_and(pred_mask, gt_mask).sum() / denom


def surface_voxels(mask):
    mask = mask.astype(bool)
    if not mask.any():
        return np.empty((0, 3), dtype=np.int32)

    if binary_erosion is not None:
        structure = generate_binary_structure(3, 1)
        eroded = binary_erosion(mask, structure=structure, border_value=0)
        return np.argwhere(mask & ~eroded)

    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = padded[1:-1, 1:-1, 1:-1]
    eroded = interior.copy()
    eroded &= padded[:-2, 1:-1, 1:-1]
    eroded &= padded[2:, 1:-1, 1:-1]
    eroded &= padded[1:-1, :-2, 1:-1]
    eroded &= padded[1:-1, 2:, 1:-1]
    eroded &= padded[1:-1, 1:-1, :-2]
    eroded &= padded[1:-1, 1:-1, 2:]
    return np.argwhere(interior & ~eroded)


def nearest_surface_distances(source_coords, target_coords, spacing, chunk_size=2048):
    if len(source_coords) == 0 or len(target_coords) == 0:
        return np.array([np.nan], dtype=np.float32)

    source_mm = source_coords.astype(np.float32) * spacing
    target_mm = target_coords.astype(np.float32) * spacing

    if cKDTree is not None:
        distances, _ = cKDTree(target_mm).query(source_mm, k=1)
        return distances.astype(np.float32)

    distances = []
    for start in range(0, len(source_mm), chunk_size):
        chunk = source_mm[start:start + chunk_size]
        diff = chunk[:, None, :] - target_mm[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        distances.append(np.sqrt(np.min(dist_sq, axis=1)))
    return np.concatenate(distances)


def segmentation_metrics(pred_mask, gt_mask, spacing):
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)
    dice = dice_score_binary(pred_bool, gt_bool)

    if not pred_bool.any() and not gt_bool.any():
        hd95 = 0.0
        asd = 0.0
    elif not pred_bool.any() or not gt_bool.any():
        hd95 = 500.0
        asd = 500.0
    else:
        spacing_arr = np.asarray(spacing, dtype=np.float32)
        pred_surface = surface_voxels(pred_bool)
        gt_surface = surface_voxels(gt_bool)
        pred_to_gt = nearest_surface_distances(pred_surface, gt_surface, spacing_arr)
        gt_to_pred = nearest_surface_distances(gt_surface, pred_surface, spacing_arr)
        all_dist = np.concatenate([pred_to_gt, gt_to_pred])
        asd = float(np.clip(np.nan_to_num(np.nanmean(all_dist), nan=500), 0, 500))
        hd95 = float(np.clip(np.nan_to_num(np.nanpercentile(all_dist, 95), nan=500), 0, 500))

    pred_voxels = int(pred_bool.sum())
    gt_voxels = int(gt_bool.sum())
    vox_vol_mm3 = float(np.prod(spacing))

    return {
        "dice": float(dice),
        "hd95_mm": float(hd95),
        "asd_mm": float(asd),
        "gt_voxels": gt_voxels,
        "pred_voxels": pred_voxels,
        "gt_volume_ml": gt_voxels * vox_vol_mm3 / 1000.0,
        "pred_volume_ml": pred_voxels * vox_vol_mm3 / 1000.0,
        "overlap_fraction": float(np.logical_and(pred_bool, gt_bool).sum() / max(gt_voxels, 1)),
    }


def z_index_for_percent(depth, percent):
    return int(np.clip(round((float(percent) / 100.0) * (depth - 1)), 0, depth - 1))


def overlay_gt(ax, ct_slice, gt_slice, title):
    ax.imshow(ct_slice.T, cmap="gray", origin="lower", vmin=0, vmax=1)
    gt_rgba = np.zeros((*gt_slice.shape, 4), dtype=np.float32)
    gt_rgba[gt_slice > 0] = [0.0, 1.0, 0.0, 0.55]
    ax.imshow(gt_rgba.transpose(1, 0, 2), origin="lower")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def overlay_gt_pred(ax, ct_slice, gt_slice, pred_slice, title):
    ax.imshow(ct_slice.T, cmap="gray", origin="lower", vmin=0, vmax=1)

    gt_only = (gt_slice > 0) & ~(pred_slice > 0)
    pred_only = (pred_slice > 0) & ~(gt_slice > 0)
    overlap = (gt_slice > 0) & (pred_slice > 0)

    rgba = np.zeros((*gt_slice.shape, 4), dtype=np.float32)
    rgba[gt_only] = [0.0, 1.0, 0.0, 0.50]
    rgba[pred_only] = [1.0, 0.1, 0.1, 0.50]
    rgba[overlap] = [1.0, 1.0, 0.0, 0.65]

    ax.imshow(rgba.transpose(1, 0, 2), origin="lower")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def add_overlay_legend(fig):
    patches = [
        mpatches.Patch(color="green", alpha=0.6, label="Ground truth only"),
        mpatches.Patch(color="red", alpha=0.6, label="Prediction only"),
        mpatches.Patch(color="yellow", alpha=0.8, label="Overlap"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.005))


def save_slice_percentage_figure(patient_id, ct_vol, gt_mask, model_results, metrics_by_model, patient_dir, show=False):
    rows = len(SLICE_PERCENTAGES)
    model_items = list(model_results.items())
    cols = 1 + len(model_items)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.25 * rows), squeeze=False)
    fig.suptitle(f"{patient_id} | fold_{CURRENT_FOLD_INDEX} single {MODEL_TYPE} model | requested axial z-percent slices", fontsize=13, fontweight="bold")

    for row, percent in enumerate(SLICE_PERCENTAGES):
        z = z_index_for_percent(ct_vol.shape[2], percent)
        overlay_gt(axes[row, 0], ct_vol[:, :, z], gt_mask[:, :, z], f"GT | {percent}% (z={z})")
        for col, (model_name, result) in enumerate(model_items, start=1):
            metric = metrics_by_model[model_name]
            title = (
                f"{result['label']} | {percent}% (z={z})\n"
                f"Dice {metric['dice']:.3f} | HD95 {metric['hd95_mm']:.1f} mm"
            )
            overlay_gt_pred(
                axes[row, col],
                ct_vol[:, :, z],
                gt_mask[:, :, z],
                result["pred_mask"][:, :, z],
                title,
            )

    add_overlay_legend(fig)
    plt.tight_layout(rect=(0, 0.025, 1, 0.985))
    figure_path = patient_dir / f"slices_z_percentages_{MODEL_TYPE}_fold_{CURRENT_FOLD_INDEX}.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return figure_path


def centroid_slice(mask):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return tuple((np.array(mask.shape) // 2).astype(int))
    return tuple(coords.mean(axis=0).astype(int))


def save_three_plane_figure(patient_id, ct_vol, gt_mask, model_results, metrics_by_model, patient_dir, show=False):
    cx, cy, cz = centroid_slice(gt_mask)
    model_items = list(model_results.items())
    fig, axes = plt.subplots(len(model_items), 3, figsize=(13, 4.3 * len(model_items)), squeeze=False)
    fig.suptitle(f"{patient_id} | centroid three-plane comparison", fontsize=13, fontweight="bold")

    for row, (model_name, result) in enumerate(model_items):
        pred_mask = result["pred_mask"]
        metric = metrics_by_model[model_name]
        planes = [
            (f"Axial z={cz}", ct_vol[:, :, cz], gt_mask[:, :, cz], pred_mask[:, :, cz]),
            (f"Coronal y={cy}", ct_vol[:, cy, :], gt_mask[:, cy, :], pred_mask[:, cy, :]),
            (f"Sagittal x={cx}", ct_vol[cx, :, :], gt_mask[cx, :, :], pred_mask[cx, :, :]),
        ]
        for col, (plane_name, ct_slice, gt_slice, pred_slice) in enumerate(planes):
            title = f"{result['label']} | {plane_name}\nDice {metric['dice']:.3f} | ASD {metric['asd_mm']:.1f} mm"
            overlay_gt_pred(axes[row, col], ct_slice, gt_slice, pred_slice, title)

    add_overlay_legend(fig)
    plt.tight_layout(rect=(0, 0.04, 1, 0.95))
    figure_path = patient_dir / f"three_plane_{MODEL_TYPE}_fold_{CURRENT_FOLD_INDEX}.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return figure_path



def run_inference():
    global CURRENT_FOLD_INDEX
    initialise_transforms()

    all_cases = find_valid_cases(DATA_ROOT)
    selected_cases_by_fold = {}
    for fold_index in INFERENCE_FOLDS:
        selected_cases_by_fold[fold_index] = select_cases(all_cases, SPLIT_CSV, fold_index, CASE_NAMES, MAX_PATIENTS)

    print(f"Project root  : {PROJECT_ROOT}")
    print(f"Data root     : {DATA_ROOT}")
    print(f"CV split file : {SPLIT_CSV}")
    print(f"Checkpoints   : {CHECKPOINTS_ROOT}")
    print(f"Output dir    : {COMPARISON_OUTPUT_DIR}")
    print(f"Model type    : {MODEL_TYPE}")
    print(f"Folds         : {INFERENCE_FOLDS}")

    all_metric_rows = []
    patient_output_rows = []

    for fold_index in INFERENCE_FOLDS:
        CURRENT_FOLD_INDEX = fold_index
        print("\n==============================")
        print(f"Running inference for fold_{fold_index}")
        print("==============================")

        fold_model_specs = verify_model_specs([get_model_spec_for_fold(fold_index)])
        comparison_models, model_metadata = load_comparison_models(fold_model_specs)

        fold_cases = selected_cases_by_fold[fold_index]
        fold_output_dir = COMPARISON_OUTPUT_DIR / f"fold_{fold_index}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Patients selected for fold_{fold_index}: {len(fold_cases)}")
        print("First patients:", ", ".join(case["name"] for case in fold_cases[:10]))

        for patient_index, patient_case in enumerate(fold_cases, start=1):
            patient_id = patient_case["name"]
            patient_dir = fold_output_dir / patient_id
            patient_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[fold_{fold_index} | {patient_index}/{len(fold_cases)}] Processing {patient_id}")

            copy_slicer_inputs(patient_case, patient_dir)
            ct_vol, gt_mask = load_preprocessed_patient(patient_case)
            model_results = run_models_for_patient(patient_case, comparison_models, patient_dir)

            metrics_by_model = {}
            per_patient_rows = []

            for spec in fold_model_specs:
                model_name = spec["name"]
                metric_raw = segmentation_metrics(model_results[model_name]["pred_mask_raw"], gt_mask, TARGET_SPACING)
                metric = segmentation_metrics(model_results[model_name]["pred_mask"], gt_mask, TARGET_SPACING)
                metrics_by_model[model_name] = metric

                row = {
                    "fold": fold_index,
                    "patient_id": patient_id,
                    "model": model_name,
                    "model_label": spec["label"],
                    "checkpoint": model_metadata[model_name]["checkpoint"],
                    "checkpoint_state": model_metadata[model_name]["state_key"],
                    "checkpoint_step": model_metadata[model_name]["step"],
                    "checkpoint_best_dice": model_metadata[model_name]["best_dice"],
                    "checkpoint_loss_name": model_metadata[model_name]["loss_name"],
                    "preprocessed_shape": "x".join(map(str, gt_mask.shape)),
                    "original_prediction_shape": "x".join(map(str, model_results[model_name]["pred_original_shape"])),
                    "prediction_path": str(model_results[model_name]["prediction_path"]),
                    "raw_dice": metric_raw["dice"],
                    "raw_hd95_mm": metric_raw["hd95_mm"],
                    "raw_asd_mm": metric_raw["asd_mm"],
                    "raw_pred_volume_ml": metric_raw["pred_volume_ml"],
                    **metric,
                }
                per_patient_rows.append(row)
                all_metric_rows.append(row)

            patient_metrics_df = pd.DataFrame(per_patient_rows)
            metrics_path = patient_dir / "metrics.csv"
            patient_metrics_df.to_csv(metrics_path, index=False)

            show_figures = patient_index <= DISPLAY_FIRST_N_PATIENTS
            slice_figure = save_slice_percentage_figure(patient_id, ct_vol, gt_mask, model_results, metrics_by_model, patient_dir, show=show_figures)
            three_plane_figure = save_three_plane_figure(patient_id, ct_vol, gt_mask, model_results, metrics_by_model, patient_dir, show=show_figures)

            only_spec = fold_model_specs[0]
            only_model_name = only_spec["name"]
            patient_output_rows.append({
                "fold": fold_index,
                "patient_id": patient_id,
                "patient_dir": str(patient_dir),
                "volume": str(patient_dir / "volume_CT_LATE.nii.gz"),
                "ground_truth": str(patient_dir / "ground_truth_mask.nii.gz"),
                "segmentation": str(patient_dir / only_spec["segmentation_filename"]),
                "model": only_model_name,
                "checkpoint": str(only_spec["path"]),
                "slice_figure": str(slice_figure),
                "three_plane_figure": str(three_plane_figure),
                "metrics_csv": str(metrics_path),
            })

            dice_text = f"{only_spec['label']}: Dice {metrics_by_model[only_model_name]['dice']:.3f}"
            print(f"    Saved to {patient_dir}")
            print(f"    {dice_text}")

        del comparison_models
        release_device_cache()

    folds_tag = "_".join(map(str, INFERENCE_FOLDS))
    summary_df = pd.DataFrame(all_metric_rows)
    summary_path = COMPARISON_OUTPUT_DIR / f"{MODEL_TYPE}_single_model_folds_{folds_tag}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    patient_outputs_df = pd.DataFrame(patient_output_rows)
    patient_outputs_path = COMPARISON_OUTPUT_DIR / f"{MODEL_TYPE}_single_model_folds_{folds_tag}_patient_outputs.csv"
    patient_outputs_df.to_csv(patient_outputs_path, index=False)
    print(f"\nSaved summary metrics to: {summary_path}")
    print(f"Saved patient output index to: {patient_outputs_path}")


def main():
    args = parse_args()
    initialise_config(args)
    run_inference()


if __name__ == "__main__":
    main()
