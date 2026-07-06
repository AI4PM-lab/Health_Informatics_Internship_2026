"""Evaluate fold -1 TotalSegmentator myocardium masks against registration masks."""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_local_fold_minus1_inference import (  # noqa: E402
    as_binary_mask,
    binary_mask_metrics,
    load_binary_nifti,
    nifti_spacing,
    resolve_project_path,
    summarize_metric_rows,
    surface_voxels,
    write_csv,
)

DEFAULT_INPUT_DATA = Path("notebooks/nifti_data")
DEFAULT_SPLIT_CSV = Path("data/cv_splits_qc.csv")
DEFAULT_OUTPUT_CSV = Path("notebooks/nifti_data/fold_minus1_totalsegmentator_metrics.csv")
TOTALSEG_MASK_RELATIVE_PATH = Path(
    "TotalSegmentator/CT_LATE/heartchambers_highres/heart_myocardium.nii.gz"
)
PATIENT_METRICS_NAME = "metrics.csv"
MASK_TYPE = "heart_myocardium"

OUTPUT_COLUMNS = [
    "patient_id",
    "fold",
    "mask_type",
    "mask_path",
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
    "pred_voxels",
    "ground_truth_voxels",
    "hd95_surface_mm",
    "hd95_full_mask_mm",
    "totalsegmentator_volume_ml",
    "volume_ratio_totalsegmentator_to_gt",
    "totalsegmentator_voxels",
    "totalsegmentator_surface_voxels",
    "ground_truth_surface_voxels",
    "intersection_voxels",
    "union_voxels",
    "voxel_spacing_mm",
    "totalsegmentator_mask",
    "ground_truth_mask",
]


@dataclass(frozen=True)
class TotalSegmentatorCase:
    patient_id: str
    fold: int
    totalsegmentator_mask: Path
    ground_truth_mask: Path
    output_dir: Path


def _format_spacing(spacing: tuple[float, float, float]) -> str:
    return "x".join(f"{value:g}" for value in spacing)


def _ordered_row(row: dict) -> dict:
    return {key: row.get(key, "") for key in OUTPUT_COLUMNS}


def _read_split_rows(split_csv: str | Path) -> list[dict[str, str]]:
    split_path = resolve_project_path(split_csv)
    with split_path.open(newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def select_totalsegmentator_cases(
    input_data: str | Path = DEFAULT_INPUT_DATA,
    split_csv: str | Path = DEFAULT_SPLIT_CSV,
    test_fold: int = -1,
    case_names: list[str] | None = None,
) -> tuple[list[TotalSegmentatorCase], list[dict[str, str]]]:
    input_root = resolve_project_path(input_data)
    requested_names = set(case_names or [])
    cases: list[TotalSegmentatorCase] = []
    skipped: list[dict[str, str]] = []
    seen_requested: set[str] = set()

    for row in _read_split_rows(split_csv):
        patient_id = str(row["patient_id"])
        fold = int(row["fold"])
        if fold != int(test_fold):
            continue
        if requested_names and patient_id not in requested_names:
            continue
        seen_requested.add(patient_id)

        patient_dir = input_root / patient_id
        ground_truth_mask = patient_dir / "registration_mask.nii.gz"
        totalsegmentator_mask = patient_dir / TOTALSEG_MASK_RELATIVE_PATH
        output_dir = patient_dir / "TotalSegmentator"

        if not ground_truth_mask.exists():
            skipped.append(
                {
                    "patient_id": patient_id,
                    "fold": str(fold),
                    "reason": "missing_ground_truth_mask",
                    "path": str(ground_truth_mask),
                }
            )
            continue
        if not totalsegmentator_mask.exists():
            skipped.append(
                {
                    "patient_id": patient_id,
                    "fold": str(fold),
                    "reason": "missing_totalsegmentator_mask",
                    "path": str(totalsegmentator_mask),
                }
            )
            continue

        cases.append(
            TotalSegmentatorCase(
                patient_id=patient_id,
                fold=fold,
                totalsegmentator_mask=totalsegmentator_mask,
                ground_truth_mask=ground_truth_mask,
                output_dir=output_dir,
            )
        )

    missing_requested = sorted(requested_names - seen_requested)
    if missing_requested:
        raise ValueError(
            "Requested case(s) were not found in the selected fold: "
            + ", ".join(missing_requested)
        )
    return cases, skipped


def evaluate_case(
    case: TotalSegmentatorCase,
    write_patient_csv: bool = True,
) -> dict:
    pred = load_binary_nifti(case.totalsegmentator_mask)
    gt = load_binary_nifti(case.ground_truth_mask)
    if pred.shape != gt.shape:
        raise ValueError(
            f"{case.patient_id}: mask shape mismatch "
            f"{case.totalsegmentator_mask} has {pred.shape}, "
            f"{case.ground_truth_mask} has {gt.shape}."
        )

    spacing = nifti_spacing(case.ground_truth_mask)
    metrics = binary_mask_metrics(pred, gt, spacing)
    pred_bool = as_binary_mask(pred)
    gt_bool = as_binary_mask(gt)
    intersection = int(np.logical_and(pred_bool, gt_bool).sum())
    union = int(np.logical_or(pred_bool, gt_bool).sum())
    pred_surface = int(surface_voxels(pred_bool).sum())
    gt_surface = int(surface_voxels(gt_bool).sum())

    row = {
        "patient_id": case.patient_id,
        "fold": case.fold,
        "mask_type": MASK_TYPE,
        "mask_path": str(case.totalsegmentator_mask),
        **metrics,
        "hd95_surface_mm": metrics["surface_hausdorff95_mm"],
        "hd95_full_mask_mm": metrics["hausdorff95_mm"],
        "totalsegmentator_volume_ml": metrics["pred_volume_ml"],
        "volume_ratio_totalsegmentator_to_gt": metrics["volume_ratio_pred_to_gt"],
        "totalsegmentator_voxels": metrics["pred_voxels"],
        "totalsegmentator_surface_voxels": pred_surface,
        "ground_truth_surface_voxels": gt_surface,
        "intersection_voxels": intersection,
        "union_voxels": union,
        "voxel_spacing_mm": _format_spacing(spacing),
        "totalsegmentator_mask": str(case.totalsegmentator_mask),
        "ground_truth_mask": str(case.ground_truth_mask),
    }
    ordered = _ordered_row(row)

    if write_patient_csv:
        write_csv(case.output_dir / PATIENT_METRICS_NAME, [ordered])
    return ordered


def _summary_csv_path(output_csv: Path) -> Path:
    return output_csv.with_name(f"{output_csv.stem}_summary.csv")


def _skipped_csv_path(output_csv: Path) -> Path:
    return output_csv.with_name(f"{output_csv.stem}_skipped.csv")


def run_evaluation(
    input_data: str | Path = DEFAULT_INPUT_DATA,
    split_csv: str | Path = DEFAULT_SPLIT_CSV,
    test_fold: int = -1,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    case_names: list[str] | None = None,
    max_cases: int | None = None,
    write_patient_csv: bool = True,
) -> tuple[list[dict], list[dict[str, str]]]:
    cases, skipped = select_totalsegmentator_cases(
        input_data=input_data,
        split_csv=split_csv,
        test_fold=test_fold,
        case_names=case_names,
    )
    if max_cases is not None:
        cases = cases[: int(max_cases)]

    rows: list[dict] = []
    for case in cases:
        try:
            row = evaluate_case(case, write_patient_csv=write_patient_csv)
        except Exception as exc:
            skipped.append(
                {
                    "patient_id": case.patient_id,
                    "fold": str(case.fold),
                    "reason": "evaluation_error",
                    "path": str(case.totalsegmentator_mask),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        rows.append(row)
        print(
            f"[{case.patient_id}] Dice={float(row['dice']):.4f} "
            f"IoU={float(row['iou']):.4f} "
            f"surfaceHD95={float(row['surface_hausdorff95_mm']):.2f}mm "
            f"HD95={float(row['hausdorff95_mm']):.2f}mm",
            flush=True,
        )

    output_csv = resolve_project_path(output_csv)
    if rows:
        rows = sorted(rows, key=lambda item: item["patient_id"])
        write_csv(output_csv, rows)
        write_csv(_summary_csv_path(output_csv), summarize_metric_rows(rows))
    if skipped:
        write_csv(_skipped_csv_path(output_csv), skipped)
    if not rows:
        raise ValueError(
            f"No TotalSegmentator cases were evaluated for fold {test_fold}. "
            f"Skipped case count: {len(skipped)}."
        )
    return rows, skipped


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate TotalSegmentator heart_myocardium.nii.gz masks against "
            "registration_mask.nii.gz for patients in the requested fold."
        )
    )
    parser.add_argument("--input_data", type=Path, default=DEFAULT_INPUT_DATA)
    parser.add_argument("--split_csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--test_fold", type=int, default=-1)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--case_name",
        action="append",
        default=None,
        help="Optional TAVI_* case to evaluate. Can be repeated. Defaults to all fold -1 cases.",
    )
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument(
        "--no_patient_csv",
        action="store_true",
        help="Only write the cohort CSV files; skip per-patient TotalSegmentator/metrics.csv files.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List selected and skipped cases without loading masks or writing metrics.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cases, skipped = select_totalsegmentator_cases(
        input_data=args.input_data,
        split_csv=args.split_csv,
        test_fold=args.test_fold,
        case_names=args.case_name,
    )
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]
    if args.dry_run:
        print(f"Selected {len(cases)} case(s) from fold {args.test_fold}.")
        for case in cases:
            print(f"  {case.patient_id}: {case.totalsegmentator_mask}")
        print(f"Skipped {len(skipped)} case(s).")
        for skipped_case in skipped:
            print(
                "  SKIP "
                f"{skipped_case['patient_id']}: {skipped_case['reason']} "
                f"{skipped_case.get('path', '')}"
            )
        return 0

    rows, skipped = run_evaluation(
        input_data=args.input_data,
        split_csv=args.split_csv,
        test_fold=args.test_fold,
        output_csv=args.output_csv,
        case_names=args.case_name,
        max_cases=args.max_cases,
        write_patient_csv=not args.no_patient_csv,
    )
    output_csv = resolve_project_path(args.output_csv)
    print(f"Saved {len(rows)} row(s) to: {output_csv}", flush=True)
    print(f"Saved summary to: {_summary_csv_path(output_csv)}", flush=True)
    if skipped:
        print(
            f"Skipped {len(skipped)} case(s); details: {_skipped_csv_path(output_csv)}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
