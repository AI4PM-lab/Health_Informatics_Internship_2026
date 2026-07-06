"""Run local fold -1 inference with SwinUNETR on Apple Silicon MPS.

Predicted masks are written under notebooks/nifti_data/TAVI_XXX/SwinUNETR/.
The script reuses the project validation transforms and computes the same
binary mask metrics used by the existing local inference runner.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import math
import shutil
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import cKDTree
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.swin_UNETR import swin_kwargs_for_signature, with_swin_defaults


DEFAULT_CHECKPOINT = "notebooks/output/models/SwinUNETR/best_metric_model.pth"
DEFAULT_INPUT_DATA = "notebooks/nifti_data"
DEFAULT_OUTPUT_ROOT = "notebooks/nifti_data"
DEFAULT_OUTPUT_SUBDIR = "SwinUNETR"
DEFAULT_SPLIT_CSV = "data/cv_splits_qc.csv"
DEFAULT_CONFIG = "config/train_config.yaml"
DEFAULT_REPORT_DIR = "notebooks/output/models/SwinUNETR"


@dataclass(frozen=True)
class PatientCase:
    patient_id: str
    image: Path
    label: Path
    fold: int


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def select_cases_from_split(
    input_data: str | Path,
    split_csv: str | Path,
    test_fold: int = -1,
    case_names: list[str] | None = None,
) -> list[PatientCase]:
    input_root = resolve_project_path(input_data)
    split_path = resolve_project_path(split_csv)
    requested_names = set(case_names or [])
    cases: list[PatientCase] = []
    missing: list[str] = []

    with split_path.open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            fold = int(row["fold"])
            patient_id = str(row["patient_id"])
            if fold != int(test_fold):
                continue
            if requested_names and patient_id not in requested_names:
                continue

            image_path = input_root / patient_id / "CT_LATE.nii.gz"
            label_path = input_root / patient_id / "registration_mask.nii.gz"
            if image_path.exists() and label_path.exists():
                cases.append(
                    PatientCase(
                        patient_id=patient_id,
                        image=image_path,
                        label=label_path,
                        fold=fold,
                    )
                )
            else:
                missing.append(patient_id)

    if requested_names:
        found_names = {case.patient_id for case in cases}
        not_found = sorted(requested_names - found_names)
        if not_found:
            raise ValueError(
                "Requested case(s) were not found with complete image/label files: "
                + ", ".join(not_found)
            )

    if not cases:
        raise ValueError(f"No complete cases found for fold {test_fold} using {split_path}.")

    if missing:
        print(
            "Skipping fold "
            f"{test_fold} case(s) with missing CT_LATE.nii.gz or registration_mask.nii.gz: "
            + ", ".join(missing),
            flush=True,
        )

    return cases


def as_binary_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 4 and 1 in (mask.shape[0], mask.shape[-1]):
        mask = np.squeeze(mask)
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, received shape {mask.shape}.")
    return mask > 0


def surface_voxels(mask: np.ndarray) -> np.ndarray:
    foreground = as_binary_mask(mask)
    if not foreground.any():
        return foreground
    structure = generate_binary_structure(rank=3, connectivity=1)
    eroded = binary_erosion(foreground, structure=structure, border_value=0)
    return foreground & ~eroded


def _directed_distances(
    source: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    source_points = np.argwhere(source) * np.asarray(spacing, dtype=float)
    target_points = np.argwhere(target) * np.asarray(spacing, dtype=float)
    if source_points.size == 0 or target_points.size == 0:
        return np.asarray([], dtype=float)
    return cKDTree(target_points).query(source_points, k=1)[0]


def _symmetric_distance_stats(
    first: np.ndarray,
    second: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[float, float]:
    first = as_binary_mask(first)
    second = as_binary_mask(second)
    if not first.any() and not second.any():
        return 0.0, 0.0
    if not first.any() or not second.any():
        return math.inf, math.inf

    distances = np.concatenate(
        [
            _directed_distances(first, second, spacing),
            _directed_distances(second, first, spacing),
        ]
    )
    return float(np.max(distances)), float(np.percentile(distances, 95))


def binary_mask_metrics(
    pred_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> dict[str, float | int]:
    pred = as_binary_mask(pred_mask)
    gt = as_binary_mask(ground_truth_mask)
    pred_voxels = int(pred.sum())
    gt_voxels = int(gt.sum())
    intersection = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())

    if pred_voxels + gt_voxels == 0:
        dice = 1.0
    else:
        dice = 2.0 * intersection / float(pred_voxels + gt_voxels)
    iou = 1.0 if union == 0 else intersection / float(union)

    surface_hd, surface_hd95 = _symmetric_distance_stats(
        surface_voxels(pred),
        surface_voxels(gt),
        spacing,
    )
    full_hd, full_hd95 = _symmetric_distance_stats(pred, gt, spacing)
    voxel_volume_ml = float(np.prod(spacing)) / 1000.0
    pred_volume_ml = pred_voxels * voxel_volume_ml
    gt_volume_ml = gt_voxels * voxel_volume_ml

    return {
        "dice": float(dice),
        "surface_hausdorff_mm": surface_hd,
        "surface_hausdorff95_mm": surface_hd95,
        "hausdorff_mm": full_hd,
        "hausdorff95_mm": full_hd95,
        "iou": float(iou),
        "pred_volume_ml": float(pred_volume_ml),
        "ground_truth_volume_ml": float(gt_volume_ml),
        "volume_difference_ml": float(pred_volume_ml - gt_volume_ml),
        "volume_ratio_pred_to_gt": math.inf if gt_volume_ml == 0 else float(pred_volume_ml / gt_volume_ml),
        "pred_voxels": pred_voxels,
        "ground_truth_voxels": gt_voxels,
    }


def nifti_spacing(path: str | Path) -> tuple[float, float, float]:
    image = nib.load(str(path))
    return tuple(float(value) for value in image.header.get_zooms()[:3])


def load_binary_nifti(path: str | Path) -> np.ndarray:
    return as_binary_mask(nib.load(str(path)).get_fdata()).astype(np.uint8)


def _nifti_read_error(path: str | Path) -> str | None:
    try:
        image = nib.load(str(path))
        _ = image.shape
        _ = image.header.get_zooms()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def filter_readable_cases(cases: list[PatientCase]) -> tuple[list[PatientCase], list[dict[str, str]]]:
    readable: list[PatientCase] = []
    skipped: list[dict[str, str]] = []
    for case in cases:
        image_error = _nifti_read_error(case.image)
        label_error = _nifti_read_error(case.label)
        if image_error or label_error:
            row = {
                "patient_id": case.patient_id,
                "fold": str(case.fold),
                "image_path": str(case.image),
                "label_path": str(case.label),
            }
            if image_error:
                row["image_error"] = image_error
            if label_error:
                row["label_error"] = label_error
            skipped.append(row)
        else:
            readable.append(case)
    return readable, skipped


def save_mask_like_reference(mask: np.ndarray, reference_path: str | Path, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference = nib.load(str(reference_path))
    data = as_binary_mask(mask).astype(np.uint8)
    image = nib.Nifti1Image(data, affine=reference.affine, header=reference.header.copy())
    image.set_data_dtype(np.uint8)
    qform_code = int(np.asarray(reference.header["qform_code"]).item())
    sform_code = int(np.asarray(reference.header["sform_code"]).item())
    image.set_qform(reference.get_qform(), code=qform_code)
    image.set_sform(reference.get_sform(), code=sform_code)
    nib.save(image, str(output_path))


def write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def existing_case_metric_rows(output_root: str | Path, output_subdir: str, cases: list[PatientCase]) -> list[dict]:
    output_root = Path(output_root)
    rows: list[dict] = []
    for case in cases:
        metrics_path = output_root / case.patient_id / output_subdir / "metrics.csv"
        mask_path = output_root / case.patient_id / output_subdir / "segmentation_model.nii.gz"
        if mask_path.exists() and metrics_path.exists():
            rows.extend(read_csv_rows(metrics_path))
    return rows


def summarize_metric_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        return []

    numeric_keys = [
        "dice",
        "surface_hausdorff_mm",
        "surface_hausdorff95_mm",
        "hausdorff_mm",
        "hausdorff95_mm",
        "iou",
        "pred_volume_ml",
        "ground_truth_volume_ml",
        "volume_difference_ml",
        "volume_ratio_pred_to_gt",
    ]
    summaries = []
    for mask_type in sorted({row["mask_type"] for row in rows}):
        subset = [row for row in rows if row["mask_type"] == mask_type]
        summary = {"mask_type": mask_type, "case_count": len(subset)}
        for key in numeric_keys:
            values = np.asarray([float(row[key]) for row in subset], dtype=float)
            finite = values[np.isfinite(values)]
            summary[f"mean_{key}"] = float(np.mean(finite)) if finite.size else math.inf
            summary[f"median_{key}"] = float(np.median(finite)) if finite.size else math.inf
        summaries.append(summary)
    return summaries


def write_cohort_outputs(report_dir: str | Path, output_root: str | Path, cases: list[PatientCase], metric_rows: list[dict]) -> None:
    report_dir = Path(report_dir)
    output_root = Path(output_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = sorted(metric_rows, key=lambda row: (row["patient_id"], row["mask_type"]))
    write_csv(report_dir / "fold_minus1_metrics.csv", metric_rows)
    write_csv(report_dir / "fold_minus1_metrics_summary.csv", summarize_metric_rows(metric_rows))
    write_csv(
        report_dir / "patient_output_folders.csv",
        [
            {
                "patient_id": case.patient_id,
                "output_dir": str(output_root / case.patient_id / DEFAULT_OUTPUT_SUBDIR),
                "mask_path": str(
                    output_root / case.patient_id / DEFAULT_OUTPUT_SUBDIR / "segmentation_model.nii.gz"
                ),
                "ground_truth_path": str(output_root / case.patient_id / DEFAULT_OUTPUT_SUBDIR / "ground_truth_mask.nii.gz"),
            }
            for case in cases
        ],
    )


def _select_device(torch, requested: str):
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    mps_built = bool(mps_backend and mps_backend.is_built())

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is False.")
    if requested == "mps" and not mps_available:
        raise RuntimeError(
            "MPS was requested with --device mps, but torch.backends.mps.is_available() "
            f"is False. torch.backends.mps.is_built()={mps_built}. Run on Apple Silicon with MPS enabled or use --device cpu."
        )
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "mps":
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available:
        return torch.device("mps")
    return torch.device("cpu")


def _load_checkpoint_state(torch, checkpoint_path: Path, use_ema_weights: bool) -> dict:
    if not hasattr(np, "_core") and hasattr(np, "core"):
        sys.modules.setdefault("numpy._core", np.core)

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if use_ema_weights and checkpoint.get("ema_model_state_dict") is not None:
            state_dict = checkpoint["ema_model_state_dict"]
            print("Loaded ema_model_state_dict from checkpoint.", flush=True)
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print("Loaded model_state_dict from checkpoint.", flush=True)
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print("Loaded state_dict from checkpoint.", flush=True)
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format in {checkpoint_path}.")

    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def _as_3d_numpy_mask(monai_pred) -> np.ndarray:
    if hasattr(monai_pred, "detach"):
        array = monai_pred.detach().cpu().numpy()
    else:
        array = np.asarray(monai_pred)
    return as_binary_mask(array).astype(np.uint8)


def _build_swin_unetr_model(cfg):
    from monai.networks.nets import SwinUNETR

    signature_keys = inspect.signature(SwinUNETR).parameters.keys()
    model_cfg = OmegaConf.to_container(cfg.get("model", {}), resolve=True)
    roi_size = OmegaConf.to_container(cfg.transforms.roi_size, resolve=True)
    swin_cfg = with_swin_defaults(model_cfg, roi_size)
    kwargs = swin_kwargs_for_signature(swin_cfg, signature_keys)
    return SwinUNETR(**kwargs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run local SwinUNETR inference on the reserved fold -1 cases and "
            "write masks to notebooks/nifti_data/TAVI_XXX/SwinUNETR/."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--input_data", type=Path, default=Path(DEFAULT_INPUT_DATA))
    parser.add_argument("--output_root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--split_csv", type=Path, default=Path(DEFAULT_SPLIT_CSV))
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--report_dir", type=Path, default=Path(DEFAULT_REPORT_DIR))
    parser.add_argument("--output_subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--test_fold", type=int, default=-1)
    parser.add_argument(
        "--case_name",
        action="append",
        default=None,
        help="Optional TAVI_* case to run. Can be repeated. Defaults to all fold -1 cases.",
    )
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="mps")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--use_raw_model_weights",
        action="store_true",
        help="Use model_state_dict even when ema_model_state_dict is available.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip cases that already have segmentation_model.nii.gz.",
    )
    parser.add_argument(
        "--fail_on_unreadable",
        action="store_true",
        help="Fail instead of skipping cases whose image/label files cannot be read by nibabel.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List selected cases and check paths without importing MONAI/PyTorch.",
    )
    return parser


def run_inference(args: argparse.Namespace) -> list[dict]:
    import torch
    from monai.data import DataLoader, Dataset, MetaTensor, decollate_batch
    from monai.inferers import sliding_window_inference
    from monai.transforms import AsDiscreted, Compose, Invertd
    from scripts.lcc_postprocessing import keep_largest_cc_numpy
    from lems_ct.src.utils.transforms import get_transforms

    checkpoint_path = resolve_project_path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_root = resolve_project_path(args.output_root)
    report_dir = resolve_project_path(args.report_dir)
    config_path = resolve_project_path(args.config)
    cfg = OmegaConf.load(config_path)

    cases = select_cases_from_split(
        args.input_data,
        args.split_csv,
        test_fold=args.test_fold,
        case_names=args.case_name,
    )
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]

    cases, skipped_cases = filter_readable_cases(cases)
    if skipped_cases:
        output_root.mkdir(parents=True, exist_ok=True)
        write_csv(report_dir / "skipped_cases.csv", skipped_cases)
        message = (
            f"Skipping {len(skipped_cases)} unreadable case(s). "
            f"Details: {report_dir / 'skipped_cases.csv'}"
        )
        if args.fail_on_unreadable:
            raise RuntimeError(message)
        print(message, flush=True)
    else:
        stale_skipped_report = report_dir / "skipped_cases.csv"
        if stale_skipped_report.exists():
            stale_skipped_report.unlink()

    if not cases:
        print("No cases to run after applying selection filters.", flush=True)
        return []

    _, val_transforms = get_transforms(**cfg.transforms)
    post_transforms = Compose(
        [
            Invertd(
                keys="pred",
                transform=val_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", argmax=True),
        ]
    )

    device = _select_device(torch, args.device)
    amp_context = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()
    roi_size = tuple(int(value) for value in cfg.transforms.roi_size)
    sw_batch_size = int(cfg.inference.get("sw_batch_size", 4))
    overlap = float(cfg.inference.get("overlap", 0.5))
    mode = str(cfg.inference.get("mode", "gaussian"))

    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Output root: {output_root}", flush=True)
    print(f"Masks subdir: {args.output_subdir}", flush=True)
    print(f"Running {len(cases)} fold {args.test_fold} case(s) on device={device}.", flush=True)

    model = _build_swin_unetr_model(cfg).to(device)
    state_dict = _load_checkpoint_state(
        torch,
        checkpoint_path,
        use_ema_weights=not args.use_raw_model_weights,
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    metric_rows: list[dict] = []
    if args.skip_existing:
        metric_rows.extend(existing_case_metric_rows(output_root, args.output_subdir, cases))

    pending_cases = cases
    if args.skip_existing:
        pending_cases = [
            case
            for case in cases
            if not (output_root / case.patient_id / args.output_subdir / "segmentation_model.nii.gz").exists()
        ]
        skipped_count = len(cases) - len(pending_cases)
        if skipped_count:
            print(
                f"Skipping {skipped_count} case(s) with existing segmentation_model.nii.gz.",
                flush=True,
            )

    if not pending_cases:
        write_cohort_outputs(report_dir, output_root, cases, metric_rows)
        print(f"Saved masks and metrics under: {output_root}", flush=True)
        return metric_rows

    dataset = Dataset(
        data=[{"image": str(case.image), "label": str(case.label)} for case in pending_cases],
        transform=val_transforms,
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=int(args.num_workers))

    with torch.inference_mode():
        for case, batch in zip(pending_cases, loader):
            case_dir = output_root / case.patient_id / args.output_subdir
            case_dir.mkdir(parents=True, exist_ok=True)
            mask_path = case_dir / "segmentation_model.nii.gz"
            metrics_path = case_dir / "metrics.csv"

            print(f"[{case.patient_id}] inference", flush=True)
            inputs = batch["image"].to(device)
            with amp_context:
                logits = sliding_window_inference(
                    inputs,
                    roi_size,
                    sw_batch_size,
                    model,
                    overlap=overlap,
                    mode=mode,
                )
                probs = torch.softmax(logits, dim=1)

            src_meta = batch["image"].meta if hasattr(batch["image"], "meta") else batch["image_meta_dict"]
            pred_tensor = MetaTensor(probs.detach().cpu(), meta=src_meta)
            item = decollate_batch({"image": batch["image"], "pred": pred_tensor})[0]
            if "image_meta_dict" not in item:
                item["image_meta_dict"] = item["image"].meta if hasattr(item["image"], "meta") else src_meta
            if "pred_meta_dict" not in item:
                item["pred_meta_dict"] = dict(item["image_meta_dict"])
            item = post_transforms(item)

            raw_mask = _as_3d_numpy_mask(item["pred"])
            final_mask = keep_largest_cc_numpy(raw_mask)

            save_mask_like_reference(final_mask, case.label, mask_path)
            shutil.copy2(case.label, case_dir / "ground_truth_mask.nii.gz")
            shutil.copy2(case.image, case_dir / "volume_CT_LATE.nii.gz")

            gt_mask = load_binary_nifti(case.label)
            spacing = nifti_spacing(case.label)
            row = {
                "patient_id": case.patient_id,
                "fold": case.fold,
                "mask_type": "postprocessed",
                "mask_path": str(mask_path),
            }
            row.update(binary_mask_metrics(final_mask, gt_mask, spacing))
            metric_rows.append(row)
            write_csv(metrics_path, [row])

            print(
                f"[{case.patient_id}] Dice={row['dice']:.4f} "
                f"IoU={row['iou']:.4f} "
                f"surfaceHD={row['surface_hausdorff_mm']:.2f}mm "
                f"HD={row['hausdorff_mm']:.2f}mm",
                flush=True,
            )

    write_cohort_outputs(report_dir, output_root, cases, metric_rows)
    print(f"Saved masks and metrics under: {output_root}", flush=True)
    return metric_rows


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    checkpoint_path = resolve_project_path(args.checkpoint)
    cases = select_cases_from_split(
        args.input_data,
        args.split_csv,
        test_fold=args.test_fold,
        case_names=args.case_name,
    )
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]

    if args.dry_run:
        readable_cases, skipped_cases = filter_readable_cases(cases)
        print(f"Checkpoint exists: {checkpoint_path.exists()} | {checkpoint_path}")
        print(f"Selected {len(cases)} case(s) from fold {args.test_fold}:")
        print(f"Readable case(s): {len(readable_cases)}")
        print(f"Unreadable case(s): {len(skipped_cases)}")
        for skipped in skipped_cases:
            print(f"  SKIP {skipped['patient_id']}: {skipped}")
        for case in readable_cases:
            print(f"  {case.patient_id}: {case.image}")
        return 0

    run_inference(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())