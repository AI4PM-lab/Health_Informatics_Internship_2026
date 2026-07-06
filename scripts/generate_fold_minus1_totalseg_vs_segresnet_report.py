from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_SEGRESNET = "SegResNet"
MODEL_TOTALSEG = "TotalSegmentator"
PANEL_TITLES = [
    "Ground truth",
    "TotalSegmentator",
    "SegResNet",
    "Model overlap",
    "TotalSegmentator error",
    "SegResNet error",
]
DEFAULT_SEGRESNET_METRICS = Path("notebooks/output/models/dice_focal/masks/fold_minus1_metrics.csv")
DEFAULT_SEGRESNET_MASK_ROOT = Path("notebooks/output/models/dice_focal/masks")
DEFAULT_TOTALSEG_METRICS = Path("notebooks/total_segmentator_heart_myocardium_metrics.csv")
DEFAULT_OUTPUT_DIR = Path("notebooks/output/fold_minus1_totalseg_vs_segresnet")
SLICE_TRIM_COUNT = 3


@dataclass(frozen=True)
class ModelFoldSpec:
    fold: int
    metrics_csv: Path
    mask_root: Path
    output_dir_name: str


def _fold_output_dir_name(fold: int) -> str:
    return "fold_minus1" if fold == -1 else f"fold_{fold}"


def build_model_fold_specs(folds: list[int]) -> list[ModelFoldSpec]:
    specs = []
    for fold in folds:
        if fold == -1:
            specs.append(
                ModelFoldSpec(
                    fold=-1,
                    metrics_csv=DEFAULT_SEGRESNET_METRICS,
                    mask_root=DEFAULT_SEGRESNET_MASK_ROOT,
                    output_dir_name=_fold_output_dir_name(-1),
                )
            )
        else:
            mask_root = Path(f"notebooks/output/models/dice_focal_fold_{fold}/masks")
            specs.append(
                ModelFoldSpec(
                    fold=fold,
                    metrics_csv=mask_root / "fold_minus1_metrics.csv",
                    mask_root=mask_root,
                    output_dir_name=_fold_output_dir_name(fold),
                )
            )
    return specs


def _parse_model_folds(value: str) -> list[int]:
    folds = []
    for item in value.split(","):
        item = item.strip()
        if item:
            folds.append(int(item))
    return folds


def slice_indices_to_render(slice_count: int, trim_count: int = SLICE_TRIM_COUNT) -> list[int]:
    if slice_count <= trim_count * 2:
        return []
    return list(range(trim_count, slice_count - trim_count))


def _winner_for_higher(segresnet_value: float, totalseg_value: float) -> str:
    if pd.isna(segresnet_value) or pd.isna(totalseg_value):
        return ""
    if segresnet_value == totalseg_value:
        return "Tie"
    return MODEL_SEGRESNET if segresnet_value > totalseg_value else MODEL_TOTALSEG


def _winner_for_lower(segresnet_value: float, totalseg_value: float) -> str:
    if pd.isna(segresnet_value) or pd.isna(totalseg_value):
        return ""
    if segresnet_value == totalseg_value:
        return "Tie"
    return MODEL_SEGRESNET if segresnet_value < totalseg_value else MODEL_TOTALSEG


def _resolve_segresnet_mask_path(mask_path: str, mask_root: Path | None) -> str:
    path = Path(str(mask_path))
    if path.exists() or mask_root is None:
        return str(path)
    try:
        patient_id = path.parent.name
        fallback = mask_root / patient_id / path.name
    except IndexError:
        return str(path)
    return str(fallback) if fallback.exists() else str(path)


def load_comparison_rows(
    segresnet_metrics_csv: Path,
    totalseg_metrics_csv: Path,
    segresnet_mask_root: Path | None = None,
    model_fold: int = -1,
) -> pd.DataFrame:
    segresnet = pd.read_csv(segresnet_metrics_csv)
    totalseg = pd.read_csv(totalseg_metrics_csv)

    segresnet = segresnet[
        (segresnet["fold"].astype(int) == model_fold)
        & (segresnet["mask_type"].astype(str) == "postprocessed")
    ].copy()

    segresnet_normalized = segresnet.rename(
        columns={
            "mask_type": "segresnet_mask_type",
            "mask_path": "segresnet_mask_path",
            "dice": "segresnet_dice",
            "iou": "segresnet_iou",
            "surface_hausdorff95_mm": "segresnet_surface_hd95_mm",
            "hausdorff95_mm": "segresnet_hd95_mm",
            "pred_volume_ml": "segresnet_volume_ml",
            "ground_truth_volume_ml": "ground_truth_volume_ml_segresnet",
        }
    )
    totalseg_normalized = totalseg.rename(
        columns={
            "dice": "totalseg_dice",
            "iou": "totalseg_iou",
            "hd95_surface_mm": "totalseg_surface_hd95_mm",
            "hd95_full_mask_mm": "totalseg_hd95_mm",
            "totalsegmentator_volume_ml": "totalseg_volume_ml",
            "ground_truth_volume_ml": "ground_truth_volume_ml_totalseg",
            "totalsegmentator_mask": "totalseg_mask_path",
            "ground_truth_mask": "ground_truth_mask_path",
        }
    )

    segresnet_columns = [
        "patient_id",
        "segresnet_mask_type",
        "segresnet_mask_path",
        "segresnet_dice",
        "segresnet_iou",
        "segresnet_hd95_mm",
        "segresnet_surface_hd95_mm",
        "segresnet_volume_ml",
        "ground_truth_volume_ml_segresnet",
    ]
    totalseg_columns = [
        "patient_id",
        "totalseg_mask_path",
        "ground_truth_mask_path",
        "totalseg_dice",
        "totalseg_iou",
        "totalseg_hd95_mm",
        "totalseg_surface_hd95_mm",
        "totalseg_volume_ml",
        "ground_truth_volume_ml_totalseg",
    ]

    comparison = segresnet_normalized[segresnet_columns].merge(
        totalseg_normalized[totalseg_columns],
        on="patient_id",
        how="inner",
    )
    comparison = comparison.sort_values("patient_id").reset_index(drop=True)
    comparison["segresnet_mask_path"] = comparison["segresnet_mask_path"].map(
        lambda value: _resolve_segresnet_mask_path(value, segresnet_mask_root)
    )
    comparison["ground_truth_volume_ml"] = comparison[
        "ground_truth_volume_ml_segresnet"
    ].fillna(comparison["ground_truth_volume_ml_totalseg"])
    comparison["dice_delta_segresnet_minus_totalseg"] = (
        comparison["segresnet_dice"] - comparison["totalseg_dice"]
    )
    comparison["iou_delta_segresnet_minus_totalseg"] = (
        comparison["segresnet_iou"] - comparison["totalseg_iou"]
    )
    comparison["hd95_delta_segresnet_minus_totalseg_mm"] = (
        comparison["segresnet_hd95_mm"] - comparison["totalseg_hd95_mm"]
    )
    comparison["surface_hd95_delta_segresnet_minus_totalseg_mm"] = (
        comparison["segresnet_surface_hd95_mm"] - comparison["totalseg_surface_hd95_mm"]
    )
    comparison["volume_delta_segresnet_minus_totalseg_ml"] = (
        comparison["segresnet_volume_ml"] - comparison["totalseg_volume_ml"]
    )
    comparison["segresnet_volume_error_ml"] = (
        comparison["segresnet_volume_ml"] - comparison["ground_truth_volume_ml"]
    )
    comparison["totalseg_volume_error_ml"] = (
        comparison["totalseg_volume_ml"] - comparison["ground_truth_volume_ml"]
    )
    comparison["best_dice_model"] = comparison.apply(
        lambda row: _winner_for_higher(row["segresnet_dice"], row["totalseg_dice"]),
        axis=1,
    )
    comparison["best_hd95_model"] = comparison.apply(
        lambda row: _winner_for_lower(row["segresnet_hd95_mm"], row["totalseg_hd95_mm"]),
        axis=1,
    )
    comparison["comparison_image"] = ""
    comparison["comparison_slice_images"] = "[]"

    output_columns = [
        "patient_id",
        "segresnet_mask_type",
        "segresnet_dice",
        "totalseg_dice",
        "dice_delta_segresnet_minus_totalseg",
        "best_dice_model",
        "segresnet_iou",
        "totalseg_iou",
        "iou_delta_segresnet_minus_totalseg",
        "segresnet_hd95_mm",
        "totalseg_hd95_mm",
        "hd95_delta_segresnet_minus_totalseg_mm",
        "segresnet_surface_hd95_mm",
        "totalseg_surface_hd95_mm",
        "surface_hd95_delta_segresnet_minus_totalseg_mm",
        "best_hd95_model",
        "segresnet_volume_ml",
        "totalseg_volume_ml",
        "ground_truth_volume_ml",
        "segresnet_volume_error_ml",
        "totalseg_volume_error_ml",
        "volume_delta_segresnet_minus_totalseg_ml",
        "segresnet_mask_path",
        "totalseg_mask_path",
        "ground_truth_mask_path",
        "comparison_image",
        "comparison_slice_images",
    ]
    return comparison[output_columns]


def write_comparison_csv(comparison: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_csv, index=False)


def _load_nifti_data(path: Path) -> np.ndarray:
    import nibabel as nib

    image = nib.load(str(path))
    return np.asarray(image.dataobj)


def _window_ct_slice(ct_slice: np.ndarray) -> np.ndarray:
    finite = ct_slice[np.isfinite(ct_slice)]
    if finite.size == 0:
        return np.zeros_like(ct_slice, dtype=float)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros_like(ct_slice, dtype=float)
    return np.clip((ct_slice - low) / (high - low), 0, 1)


def _choose_axial_slice(*masks: np.ndarray) -> int:
    combined = np.zeros(masks[0].shape, dtype=bool)
    for mask in masks:
        combined |= mask.astype(bool)
    if not combined.any():
        return combined.shape[2] // 2
    counts = combined.sum(axis=(0, 1))
    return int(np.argmax(counts))


def _overlay_mask(base_rgb: np.ndarray, mask: np.ndarray, color: tuple[float, float, float], alpha: float) -> np.ndarray:
    output = base_rgb.copy()
    mask_bool = mask.astype(bool)
    for channel, color_value in enumerate(color):
        output[..., channel] = np.where(
            mask_bool,
            (1 - alpha) * output[..., channel] + alpha * color_value,
            output[..., channel],
        )
    return output


def _mask_rgb(mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    rgb = np.ones(mask.shape + (3,), dtype=float)
    mask_bool = mask.astype(bool)
    for channel, color_value in enumerate(color):
        rgb[..., channel] = np.where(mask_bool, color_value, 0.95)
    return rgb


def _load_patient_arrays(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    seg_path = Path(str(row.get("segresnet_mask_path", "")))
    tot_path = Path(str(row.get("totalseg_mask_path", "")))
    gt_path = Path(str(row.get("ground_truth_mask_path", "")))
    ct_path = seg_path.parent / "volume_CT_LATE.nii.gz"
    required = [seg_path, tot_path, gt_path, ct_path]
    if any(not path.exists() for path in required):
        return None

    try:
        ct = _load_nifti_data(ct_path)
        seg = _load_nifti_data(seg_path).astype(bool)
        tot = _load_nifti_data(tot_path).astype(bool)
        gt = _load_nifti_data(gt_path).astype(bool)
    except Exception:
        return None

    if not (ct.shape == seg.shape == tot.shape == gt.shape) or len(ct.shape) != 3:
        return None
    return ct, seg, tot, gt


def _slice_panels(
    ct: np.ndarray,
    gt: np.ndarray,
    seg: np.ndarray,
    tot: np.ndarray,
    z_index: int,
) -> list[tuple[str, np.ndarray]]:
    ct_slice = np.rot90(_window_ct_slice(ct[:, :, z_index]))
    gt_slice = np.rot90(gt[:, :, z_index])
    seg_slice = np.rot90(seg[:, :, z_index])
    tot_slice = np.rot90(tot[:, :, z_index])
    base_rgb = np.repeat(ct_slice[..., None], 3, axis=2)
    return [
        (PANEL_TITLES[0], _overlay_mask(base_rgb, gt_slice, (0.12, 0.62, 0.54), 0.55)),
        (PANEL_TITLES[1], _overlay_mask(base_rgb, tot_slice, (0.61, 0.36, 0.90), 0.55)),
        (PANEL_TITLES[2], _overlay_mask(base_rgb, seg_slice, (0.12, 0.48, 0.55), 0.55)),
        (
            PANEL_TITLES[3],
            _overlay_mask(
                _overlay_mask(base_rgb, tot_slice, (0.61, 0.36, 0.90), 0.45),
                seg_slice,
                (0.12, 0.48, 0.55),
                0.45,
            ),
        ),
        (PANEL_TITLES[4], _mask_rgb(tot_slice ^ gt_slice, (0.90, 0.35, 0.28))),
        (PANEL_TITLES[5], _mask_rgb(seg_slice ^ gt_slice, (0.90, 0.35, 0.28))),
    ]


def _write_slice_comparison_png(
    patient_id: str,
    z_index: int,
    panels: list[tuple[str, np.ndarray]],
    output_png: Path,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    panel_size = 360
    title_h = 32
    header_h = 42
    gap = 18
    margin = 22
    canvas_w = margin * 2 + panel_size * 3 + gap * 2
    canvas_h = margin * 2 + header_h + (panel_size + title_h) * 2 + gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), "#f5f7fa")
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("Arial.ttf", 24)
        label_font = ImageFont.truetype("Arial.ttf", 18)
    except OSError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()

    title = f"{patient_id} axial slice z={z_index}"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(
        ((canvas_w - (title_box[2] - title_box[0])) / 2, margin - 4),
        title,
        fill="#111820",
        font=title_font,
    )

    for index, (label, image_array) in enumerate(panels):
        row = index // 3
        col = index % 3
        x = margin + col * (panel_size + gap)
        y = margin + header_h + row * (panel_size + title_h + gap)
        label_box = draw.textbbox((0, 0), label, font=label_font)
        draw.text(
            (x + (panel_size - (label_box[2] - label_box[0])) / 2, y),
            label,
            fill="#17202a",
            font=label_font,
        )
        image = Image.fromarray(np.uint8(np.clip(image_array, 0, 1) * 255), mode="RGB")
        image = image.resize((panel_size, panel_size), Image.Resampling.BILINEAR)
        canvas.paste(image, (x, y + title_h))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png, optimize=True)


def generate_patient_comparison_image(row: pd.Series, output_png: Path) -> bool:
    arrays = _load_patient_arrays(row)
    if arrays is None:
        return False
    ct, seg, tot, gt = arrays
    z_index = _choose_axial_slice(gt, seg, tot)
    _write_slice_comparison_png(
        str(row.get("patient_id", "Patient")),
        z_index,
        _slice_panels(ct, gt, seg, tot, z_index),
        output_png,
    )
    return True


def generate_patient_slice_comparison_images(row: pd.Series, output_patient_dir: Path) -> list[str]:
    arrays = _load_patient_arrays(row)
    if arrays is None:
        return []
    ct, seg, tot, gt = arrays
    patient_id = str(row.get("patient_id", output_patient_dir.name))
    output_patient_dir.mkdir(parents=True, exist_ok=True)
    relative_paths = []
    for z_index in slice_indices_to_render(ct.shape[2]):
        output_png = output_patient_dir / f"slice_{z_index:03d}.png"
        if not output_png.exists():
            _write_slice_comparison_png(
                patient_id,
                z_index,
                _slice_panels(ct, gt, seg, tot, z_index),
                output_png,
            )
        relative_paths.append(f"images/{patient_id}/slice_{z_index:03d}.png")
    return relative_paths


def _fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _summary_stat(comparison: pd.DataFrame, column: str) -> float:
    return float(comparison[column].mean()) if len(comparison) else float("nan")


def _records_for_html(comparison: pd.DataFrame) -> list[dict[str, object]]:
    records = []
    for raw_record in comparison.fillna("").to_dict(orient="records"):
        record = {}
        for key, value in raw_record.items():
            if hasattr(value, "item"):
                value = value.item()
            if key == "comparison_slice_images":
                if isinstance(value, str) and value:
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        value = []
                elif not isinstance(value, list):
                    value = []
            record[key] = value
        records.append(record)
    return records


def render_html(
    comparison: pd.DataFrame,
    output_html: Path,
    image_dir_name: str = "images",
) -> None:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    rows = comparison.copy()
    if "comparison_image" not in rows.columns:
        rows["comparison_image"] = rows["patient_id"].map(
            lambda patient_id: f"{image_dir_name}/{patient_id}_comparison.png"
        )
    if "comparison_slice_images" not in rows.columns:
        rows["comparison_slice_images"] = rows["comparison_image"].map(
            lambda value: json.dumps([value] if value else [])
        )

    records = _records_for_html(rows)
    json_rows = json.dumps(records, ensure_ascii=True)
    segresnet_dice = _summary_stat(rows, "segresnet_dice")
    totalseg_dice = _summary_stat(rows, "totalseg_dice")
    segresnet_hd95 = _summary_stat(rows, "segresnet_hd95_mm")
    totalseg_hd95 = _summary_stat(rows, "totalseg_hd95_mm")
    dice_wins = rows["best_dice_model"].value_counts().to_dict() if len(rows) else {}

    table_rows = []
    for _, row in rows.sort_values("dice_delta_segresnet_minus_totalseg", ascending=False).iterrows():
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(row['patient_id']))}</td>"
            f"<td>{_fmt(row['segresnet_dice'])}</td>"
            f"<td>{_fmt(row['totalseg_dice'])}</td>"
            f"<td>{_fmt(row['dice_delta_segresnet_minus_totalseg'])}</td>"
            f"<td>{_fmt(row['segresnet_hd95_mm'], 2)}</td>"
            f"<td>{_fmt(row['totalseg_hd95_mm'], 2)}</td>"
            f"<td>{escape(str(row['best_dice_model']))}</td>"
            "</tr>"
        )

    options = "\n".join(
        f'<option value="{escape(str(record["patient_id"]))}">{escape(str(record["patient_id"]))}</option>'
        for record in records
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fold -1 Model Comparison</title>
  <style>
    :root {{
      --bg: #f5f7fa;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #5f6f82;
      --line: #dbe2ea;
      --seg: #1f7a8c;
      --tot: #9b5de5;
      --accent: #e76f51;
      --ok: #2a9d8f;
      --shadow: 0 12px 34px rgba(30, 42, 56, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.45;
    }}
    header {{
      padding: 28px 32px 18px;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0 0 6px; font-size: 28px; letter-spacing: 0; }}
    .subtitle {{ color: var(--muted); max-width: 980px; }}
    main {{ padding: 24px 32px 36px; max-width: 1500px; margin: 0 auto; }}
    .grid {{ display: grid; gap: 16px; }}
    .cards {{ grid-template-columns: repeat(4, minmax(180px, 1fr)); margin-bottom: 18px; }}
    .card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .card {{ padding: 16px; }}
    .label {{ color: var(--muted); font-size: 13px; margin-bottom: 6px; }}
    .value {{ font-size: 26px; font-weight: 700; }}
    .pair {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
    .layout {{ grid-template-columns: 360px minmax(0, 1fr); align-items: start; }}
    .panel {{ padding: 18px; }}
    h2 {{ margin: 0 0 14px; font-size: 18px; }}
    select {{
      width: 100%;
      min-height: 40px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      background: #fff;
      color: var(--ink);
      font-size: 14px;
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: var(--seg);
    }}
    .slice-control {{
      display: grid;
      gap: 8px;
      margin: 12px 0 4px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
    }}
    .slice-control-header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .metric-list {{ display: grid; gap: 10px; margin-top: 16px; }}
    .metric-row {{ display: grid; grid-template-columns: 1fr auto; gap: 12px; border-bottom: 1px solid var(--line); padding-bottom: 9px; }}
    .metric-row:last-child {{ border-bottom: 0; padding-bottom: 0; }}
    .metric-control {{ max-width: 260px; display: grid; gap: 6px; color: var(--muted); font-size: 13px; margin-bottom: 14px; }}
    .model-key {{ display: flex; gap: 10px; margin: 14px 0; flex-wrap: wrap; }}
    .pill {{ border-radius: 999px; padding: 5px 10px; color: #fff; font-size: 12px; font-weight: 700; }}
    .seg {{ background: var(--seg); }}
    .tot {{ background: var(--tot); }}
    .gt {{ background: var(--ok); }}
    .viewer img {{
      width: 100%;
      display: block;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #101820;
    }}
    .missing {{
      display: none;
      padding: 22px;
      border: 1px dashed var(--line);
      border-radius: 8px;
      color: var(--muted);
      background: #fbfcfd;
    }}
    .table-wrap {{ overflow-x: auto; margin-top: 18px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: #fff; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 10px; text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child, th:last-child, td:last-child {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 700; background: #f9fbfd; position: sticky; top: 0; }}
    .bar-row {{ display: grid; grid-template-columns: 130px 1fr 55px; gap: 10px; align-items: center; margin: 8px 0; }}
    .bar-track {{ height: 10px; background: #edf1f5; border-radius: 999px; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 999px; }}
    @media (max-width: 980px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .cards, .layout {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Fold -1 Model Comparison</h1>
    <div class="subtitle">Static comparison of TotalSegmentator and postprocessed SegResNet myocardium masks for shared fold -1 samples.</div>
  </header>
  <main>
    <section class="grid cards">
      <div class="card"><div class="label">Shared Patients</div><div class="value">{len(rows)}</div><div class="pair">intersection of available metrics</div></div>
      <div class="card"><div class="label">Mean Dice</div><div class="value">{_fmt(segresnet_dice)} / {_fmt(totalseg_dice)}</div><div class="pair">SegResNet / TotalSegmentator</div></div>
      <div class="card"><div class="label">Mean HD95</div><div class="value">{_fmt(segresnet_hd95, 2)} / {_fmt(totalseg_hd95, 2)} mm</div><div class="pair">lower is better</div></div>
      <div class="card"><div class="label">Dice Wins</div><div class="value">{dice_wins.get(MODEL_SEGRESNET, 0)} / {dice_wins.get(MODEL_TOTALSEG, 0)}</div><div class="pair">SegResNet / TotalSegmentator</div></div>
    </section>
    <section class="grid layout">
      <aside class="panel">
        <h2>Patient Selection</h2>
        <select id="patientSelect" aria-label="Patient selector">{options}</select>
        <div class="model-key">
          <span class="pill seg">SegResNet</span>
          <span class="pill tot">TotalSegmentator</span>
          <span class="pill gt">Ground truth</span>
        </div>
        <div id="patientMetrics" class="metric-list"></div>
      </aside>
      <section class="panel viewer">
        <h2 id="patientTitle">Mask Comparison</h2>
        <div id="sliceControl" class="slice-control">
          <div class="slice-control-header"><span>Axial slice</span><strong id="sliceLabel">0 / 0</strong></div>
          <input id="sliceSlider" type="range" min="0" max="0" value="0" aria-label="Axial slice">
        </div>
        <img id="comparisonImage" alt="Selected patient mask comparison">
        <div id="missingImage" class="missing">No generated comparison image is available for this patient.</div>
      </section>
    </section>
    <section class="panel" style="margin-top: 18px;">
      <h2>Per-Patient Metrics</h2>
      <div id="diceBars"></div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Patient</th><th>SegResNet Dice</th><th>TotalSegmentator Dice</th><th>Dice Delta</th>
              <th>SegResNet HD95</th><th>TotalSegmentator HD95</th><th>Best Dice</th>
            </tr>
          </thead>
          <tbody>{''.join(table_rows)}</tbody>
        </table>
      </div>
    </section>
  </main>
  <script>
    window.COMPARISON_ROWS = {json_rows};
    const rows = window.COMPARISON_ROWS;
    const select = document.getElementById('patientSelect');
    const metrics = document.getElementById('patientMetrics');
    const image = document.getElementById('comparisonImage');
    const missing = document.getElementById('missingImage');
    const title = document.getElementById('patientTitle');
    const sliceControl = document.getElementById('sliceControl');
    const sliceSlider = document.getElementById('sliceSlider');
    const sliceLabel = document.getElementById('sliceLabel');
    let currentRow = null;
    function fmt(value, digits = 3) {{
      const number = Number(value);
      return Number.isFinite(number) ? number.toFixed(digits) : 'n/a';
    }}
    function metricRow(label, value) {{
      return `<div class="metric-row"><span>${{label}}</span><strong>${{value}}</strong></div>`;
    }}
    function sliceImagesFor(row) {{
      if (Array.isArray(row.comparison_slice_images) && row.comparison_slice_images.length) {{
        return row.comparison_slice_images;
      }}
      return row.comparison_image ? [row.comparison_image] : [];
    }}
    function renderSlice(index) {{
      if (!currentRow) return;
      const images = sliceImagesFor(currentRow);
      if (!images.length) {{
        image.removeAttribute('src');
        image.style.display = 'none';
        missing.style.display = 'block';
        sliceControl.style.display = 'none';
        return;
      }}
      const clamped = Math.max(0, Math.min(images.length - 1, Number(index) || 0));
      image.src = images[clamped];
      image.style.display = 'block';
      missing.style.display = 'none';
      sliceControl.style.display = images.length > 1 ? 'grid' : 'none';
      sliceSlider.max = String(images.length - 1);
      sliceSlider.value = String(clamped);
      sliceLabel.textContent = `${{clamped + 1}} / ${{images.length}}`;
    }}
    function renderPatient(patientId) {{
      const row = rows.find(item => item.patient_id === patientId) || rows[0];
      if (!row) return;
      currentRow = row;
      title.textContent = `${{row.patient_id}} Mask Comparison`;
      metrics.innerHTML = [
        metricRow('SegResNet Dice', fmt(row.segresnet_dice)),
        metricRow('TotalSegmentator Dice', fmt(row.totalseg_dice)),
        metricRow('Dice delta', fmt(row.dice_delta_segresnet_minus_totalseg)),
        metricRow('SegResNet HD95', `${{fmt(row.segresnet_hd95_mm, 2)}} mm`),
        metricRow('TotalSegmentator HD95', `${{fmt(row.totalseg_hd95_mm, 2)}} mm`),
        metricRow('Best Dice', row.best_dice_model || 'n/a'),
        metricRow('SegResNet volume error', `${{fmt(row.segresnet_volume_error_ml, 1)}} ml`),
        metricRow('TotalSegmentator volume error', `${{fmt(row.totalseg_volume_error_ml, 1)}} ml`)
      ].join('');
      renderSlice(0);
    }}
    function renderBars() {{
      const target = document.getElementById('diceBars');
      const sorted = [...rows].sort((a, b) => Number(b.dice_delta_segresnet_minus_totalseg) - Number(a.dice_delta_segresnet_minus_totalseg)).slice(0, 24);
      target.innerHTML = sorted.map(row => {{
        const seg = Math.max(0, Math.min(100, Number(row.segresnet_dice) * 100));
        const tot = Math.max(0, Math.min(100, Number(row.totalseg_dice) * 100));
        return `<div class="bar-row">
          <strong>${{row.patient_id}}</strong>
          <div>
            <div class="bar-track"><div class="bar-fill seg" style="width: ${{seg}}%"></div></div>
            <div class="bar-track" style="margin-top: 3px;"><div class="bar-fill tot" style="width: ${{tot}}%"></div></div>
          </div>
          <span>${{fmt(row.dice_delta_segresnet_minus_totalseg)}}</span>
        </div>`;
      }}).join('');
    }}
    select.addEventListener('change', event => renderPatient(event.target.value));
    sliceSlider.addEventListener('input', event => renderSlice(event.target.value));
    renderBars();
    renderPatient(select.value);
  </script>
</body>
</html>
"""
    output_html.write_text(html)


def _prefix_comparison_image_paths(comparison: pd.DataFrame, prefix: str) -> pd.DataFrame:
    comparison = comparison.copy()

    def prefixed(value: str) -> str:
        if not value:
            return ""
        return f"{prefix}/{value}"

    comparison["comparison_image"] = comparison["comparison_image"].map(prefixed)
    comparison["comparison_slice_images"] = comparison["comparison_slice_images"].map(
        lambda value: json.dumps([prefixed(path) for path in json.loads(value or "[]")])
    )
    return comparison


def _fold_label(fold: int) -> str:
    return "Fold -1 model" if fold == -1 else f"Fold {fold} model"


def _summary_payload(rows: pd.DataFrame) -> dict[str, object]:
    dice_wins = rows["best_dice_model"].value_counts().to_dict() if len(rows) else {}
    return {
        "patientCount": int(len(rows)),
        "segresnetDice": _fmt(_summary_stat(rows, "segresnet_dice")),
        "totalsegDice": _fmt(_summary_stat(rows, "totalseg_dice")),
        "segresnetHd95": _fmt(_summary_stat(rows, "segresnet_hd95_mm"), 2),
        "totalsegHd95": _fmt(_summary_stat(rows, "totalseg_hd95_mm"), 2),
        "segresnetWins": int(dice_wins.get(MODEL_SEGRESNET, 0)),
        "totalsegWins": int(dice_wins.get(MODEL_TOTALSEG, 0)),
    }


def render_multi_fold_html(fold_comparisons: dict[int, pd.DataFrame], output_html: Path) -> None:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    sorted_folds = sorted(fold_comparisons)
    if not sorted_folds:
        raise ValueError("At least one fold comparison is required.")

    fold_payload = {
        str(fold): {
            "label": _fold_label(fold),
            "summary": _summary_payload(comparison),
            "rows": _records_for_html(comparison),
        }
        for fold, comparison in sorted(fold_comparisons.items())
    }
    initial_fold = str(sorted_folds[0])
    fold_options = "\n".join(
        f'<option value="{fold}">{escape(_fold_label(fold))}</option>' for fold in sorted_folds
    )
    payload_json = json.dumps(fold_payload, ensure_ascii=True)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fold Model Comparison</title>
  <style>
    :root {{
      --bg: #f5f7fa; --panel: #ffffff; --ink: #17202a; --muted: #5f6f82;
      --line: #dbe2ea; --seg: #1f7a8c; --tot: #9b5de5; --ok: #2a9d8f;
      --shadow: 0 12px 34px rgba(30, 42, 56, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); line-height: 1.45; }}
    header {{ padding: 24px 32px 18px; background: #fff; border-bottom: 1px solid var(--line); }}
    .header-row {{ display: flex; justify-content: space-between; align-items: start; gap: 18px; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; letter-spacing: 0; }}
    .subtitle {{ color: var(--muted); max-width: 980px; }}
    .fold-control {{ min-width: 190px; display: grid; gap: 6px; color: var(--muted); font-size: 13px; }}
    main {{ padding: 24px 32px 36px; max-width: 1500px; margin: 0 auto; }}
    .grid {{ display: grid; gap: 16px; }}
    .cards {{ grid-template-columns: repeat(4, minmax(180px, 1fr)); margin-bottom: 18px; }}
    .card, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--shadow); }}
    .card {{ padding: 16px; }}
    .label {{ color: var(--muted); font-size: 13px; margin-bottom: 6px; }}
    .value {{ font-size: 26px; font-weight: 700; }}
    .pair {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
    .layout {{ grid-template-columns: 360px minmax(0, 1fr); align-items: start; }}
    .panel {{ padding: 18px; }}
    h2 {{ margin: 0 0 14px; font-size: 18px; }}
    select {{ width: 100%; min-height: 40px; border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; background: #fff; color: var(--ink); font-size: 14px; }}
    input[type="range"] {{ width: 100%; accent-color: var(--seg); }}
    .slice-control {{ display: grid; gap: 8px; margin: 12px 0 4px; padding: 12px; border: 1px solid var(--line); border-radius: 8px; background: #fbfcfd; }}
    .slice-control-header {{ display: flex; justify-content: space-between; gap: 12px; color: var(--muted); font-size: 13px; }}
    .metric-list {{ display: grid; gap: 10px; margin-top: 16px; }}
    .metric-row {{ display: grid; grid-template-columns: 1fr auto; gap: 12px; border-bottom: 1px solid var(--line); padding-bottom: 9px; }}
    .metric-row:last-child {{ border-bottom: 0; padding-bottom: 0; }}
    .model-key {{ display: flex; gap: 10px; margin: 14px 0; flex-wrap: wrap; }}
    .pill {{ border-radius: 999px; padding: 5px 10px; color: #fff; font-size: 12px; font-weight: 700; }}
    .seg {{ background: var(--seg); }} .tot {{ background: var(--tot); }} .gt {{ background: var(--ok); }}
    .viewer img {{ width: 100%; display: block; border: 1px solid var(--line); border-radius: 8px; background: #101820; }}
    .missing {{ display: none; padding: 22px; border: 1px dashed var(--line); border-radius: 8px; color: var(--muted); background: #fbfcfd; }}
    .table-wrap {{ overflow-x: auto; margin-top: 18px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: #fff; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 10px; text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child, th:last-child, td:last-child {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 700; background: #f9fbfd; position: sticky; top: 0; }}
    .bar-row {{ display: grid; grid-template-columns: 130px 1fr 55px; gap: 10px; align-items: center; margin: 8px 0; }}
    .bar-track {{ height: 10px; background: #edf1f5; border-radius: 999px; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 999px; }}
    @media (max-width: 980px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .header-row, .cards, .layout {{ grid-template-columns: 1fr; display: grid; }}
      .fold-control {{ min-width: 0; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="header-row">
      <div>
        <h1>Fold Model Comparison</h1>
        <div class="subtitle">Static comparison of TotalSegmentator and postprocessed SegResNet myocardium masks. Use the fold selector to switch between trained SegResNet fold models.</div>
      </div>
      <label class="fold-control">Model fold<select id="modelFoldSelect" aria-label="Model fold selector">{fold_options}</select></label>
    </div>
  </header>
  <main>
    <section class="grid cards">
      <div class="card"><div class="label">Shared Patients</div><div id="patientCount" class="value">0</div><div class="pair">intersection of available metrics</div></div>
      <div class="card"><div class="label">Mean Dice</div><div id="meanDice" class="value">n/a</div><div class="pair">SegResNet / TotalSegmentator</div></div>
      <div class="card"><div class="label">Mean HD95</div><div id="meanHd95" class="value">n/a</div><div class="pair">lower is better</div></div>
      <div class="card"><div class="label">Dice Wins</div><div id="diceWins" class="value">0 / 0</div><div class="pair">SegResNet / TotalSegmentator</div></div>
    </section>
    <section class="grid layout">
      <aside class="panel">
        <h2>Patient Selection</h2>
        <select id="patientSelect" aria-label="Patient selector"></select>
        <div class="model-key"><span class="pill seg">SegResNet</span><span class="pill tot">TotalSegmentator</span><span class="pill gt">Ground truth</span></div>
        <div id="patientMetrics" class="metric-list"></div>
      </aside>
      <section class="panel viewer">
        <h2 id="patientTitle">Mask Comparison</h2>
        <div id="sliceControl" class="slice-control">
          <div class="slice-control-header"><span>Axial slice</span><strong id="sliceLabel">0 / 0</strong></div>
          <input id="sliceSlider" type="range" min="0" max="0" value="0" aria-label="Axial slice">
        </div>
        <img id="comparisonImage" alt="Selected patient mask comparison">
        <div id="missingImage" class="missing">No generated comparison image is available for this patient.</div>
      </section>
    </section>
    <section class="panel" style="margin-top: 18px;">
      <h2>Per-Patient Metrics</h2>
      <label class="metric-control">Metric comparison
        <select id="metricSelect" aria-label="Metric comparison selector">
          <option value="dice">Dice</option>
          <option value="hd95">HD95</option>
          <option value="surface_hd95">Surface-only HD95</option>
          <option value="iou">IoU</option>
          <option value="volume">Volume</option>
        </select>
      </label>
      <div id="diceBars"></div>
      <div class="table-wrap">
        <table><thead id="metricsTableHead"></thead><tbody id="metricsTableBody"></tbody></table>
      </div>
    </section>
  </main>
  <script>
    window.FOLD_COMPARISONS = {payload_json};
    window.COMPARISON_ROWS = window.FOLD_COMPARISONS["{initial_fold}"].rows;
    const foldSelect = document.getElementById('modelFoldSelect');
    const select = document.getElementById('patientSelect');
    const metrics = document.getElementById('patientMetrics');
    const image = document.getElementById('comparisonImage');
    const missing = document.getElementById('missingImage');
    const title = document.getElementById('patientTitle');
    const sliceControl = document.getElementById('sliceControl');
    const sliceSlider = document.getElementById('sliceSlider');
    const sliceLabel = document.getElementById('sliceLabel');
    const metricSelect = document.getElementById('metricSelect');
    const tableHead = document.getElementById('metricsTableHead');
    const tableBody = document.getElementById('metricsTableBody');
    let rows = window.COMPARISON_ROWS;
    let currentRow = null;
    let selectedMetric = 'dice';
    const METRIC_CONFIGS = {{
      dice: {{
        label: 'Dice',
        segCol: 'segresnet_dice',
        totCol: 'totalseg_dice',
        deltaCol: 'dice_delta_segresnet_minus_totalseg',
        digits: 3,
        unit: ''
      }},
      hd95: {{
        label: 'HD95',
        segCol: 'segresnet_hd95_mm',
        totCol: 'totalseg_hd95_mm',
        deltaCol: 'hd95_delta_segresnet_minus_totalseg_mm',
        digits: 2,
        unit: ' mm'
      }},
      surface_hd95: {{
        label: 'Surface-only HD95',
        segCol: 'segresnet_surface_hd95_mm',
        totCol: 'totalseg_surface_hd95_mm',
        deltaCol: 'surface_hd95_delta_segresnet_minus_totalseg_mm',
        digits: 2,
        unit: ' mm'
      }},
      iou: {{
        label: 'IoU',
        segCol: 'segresnet_iou',
        totCol: 'totalseg_iou',
        deltaCol: 'iou_delta_segresnet_minus_totalseg',
        digits: 3,
        unit: ''
      }},
      volume: {{
        label: 'Volume',
        segCol: 'segresnet_volume_ml',
        totCol: 'totalseg_volume_ml',
        gtCol: 'ground_truth_volume_ml',
        deltaCol: 'volume_delta_segresnet_minus_totalseg_ml',
        digits: 1,
        unit: ' ml'
      }}
    }};
    function fmt(value, digits = 3) {{ const number = Number(value); return Number.isFinite(number) ? number.toFixed(digits) : 'n/a'; }}
    function fmtMetric(value, cfg) {{ const formatted = fmt(value, cfg.digits); return formatted === 'n/a' ? formatted : `${{formatted}}${{cfg.unit}}`; }}
    function metricRow(label, value) {{ return `<div class="metric-row"><span>${{label}}</span><strong>${{value}}</strong></div>`; }}
    function sliceImagesFor(row) {{ return Array.isArray(row.comparison_slice_images) && row.comparison_slice_images.length ? row.comparison_slice_images : (row.comparison_image ? [row.comparison_image] : []); }}
    function renderSlice(index) {{
      if (!currentRow) return;
      const images = sliceImagesFor(currentRow);
      if (!images.length) {{ image.removeAttribute('src'); image.style.display = 'none'; missing.style.display = 'block'; sliceControl.style.display = 'none'; return; }}
      const clamped = Math.max(0, Math.min(images.length - 1, Number(index) || 0));
      image.src = images[clamped]; image.style.display = 'block'; missing.style.display = 'none';
      sliceControl.style.display = images.length > 1 ? 'grid' : 'none';
      sliceSlider.max = String(images.length - 1); sliceSlider.value = String(clamped);
      sliceLabel.textContent = `${{clamped + 1}} / ${{images.length}}`;
    }}
    function renderPatient(patientId) {{
      const row = rows.find(item => item.patient_id === patientId) || rows[0];
      if (!row) return;
      currentRow = row;
      title.textContent = `${{row.patient_id}} Mask Comparison`;
      metrics.innerHTML = [
        metricRow('SegResNet Dice', fmt(row.segresnet_dice)),
        metricRow('TotalSegmentator Dice', fmt(row.totalseg_dice)),
        metricRow('Dice delta', fmt(row.dice_delta_segresnet_minus_totalseg)),
        metricRow('SegResNet HD95', `${{fmt(row.segresnet_hd95_mm, 2)}} mm`),
        metricRow('TotalSegmentator HD95', `${{fmt(row.totalseg_hd95_mm, 2)}} mm`),
        metricRow('Best Dice', row.best_dice_model || 'n/a'),
        metricRow('SegResNet volume error', `${{fmt(row.segresnet_volume_error_ml, 1)}} ml`),
        metricRow('TotalSegmentator volume error', `${{fmt(row.totalseg_volume_error_ml, 1)}} ml`)
      ].join('');
      renderSlice(0);
    }}
    function renderBars() {{
      const target = document.getElementById('diceBars');
      const cfg = METRIC_CONFIGS[selectedMetric] || METRIC_CONFIGS.dice;
      const sorted = [...rows].sort((a, b) => Math.abs(Number(b[cfg.deltaCol]) || 0) - Math.abs(Number(a[cfg.deltaCol]) || 0)).slice(0, 24);
      const maxValue = Math.max(1, ...sorted.flatMap(row => [
        Number(row[cfg.segCol]) || 0,
        Number(row[cfg.totCol]) || 0,
        cfg.gtCol ? Number(row[cfg.gtCol]) || 0 : 0
      ]));
      target.innerHTML = sorted.map(row => {{
        const seg = Math.max(0, Math.min(100, (Number(row[cfg.segCol]) || 0) / maxValue * 100));
        const tot = Math.max(0, Math.min(100, (Number(row[cfg.totCol]) || 0) / maxValue * 100));
        const gt = cfg.gtCol ? Math.max(0, Math.min(100, (Number(row[cfg.gtCol]) || 0) / maxValue * 100)) : null;
        const gtBar = cfg.gtCol ? `<div class="bar-track" style="margin-top: 3px;"><div class="bar-fill gt" style="width: ${{gt}}%"></div></div>` : '';
        return `<div class="bar-row"><strong>${{row.patient_id}}</strong><div><div class="bar-track"><div class="bar-fill seg" style="width: ${{seg}}%"></div></div><div class="bar-track" style="margin-top: 3px;"><div class="bar-fill tot" style="width: ${{tot}}%"></div></div>${{gtBar}}</div><span>${{fmtMetric(row[cfg.deltaCol], cfg)}}</span></div>`;
      }}).join('');
    }}
    function renderTable() {{
      const cfg = METRIC_CONFIGS[selectedMetric] || METRIC_CONFIGS.dice;
      const sorted = [...rows].sort((a, b) => Math.abs(Number(b[cfg.deltaCol]) || 0) - Math.abs(Number(a[cfg.deltaCol]) || 0));
      const gtHead = cfg.gtCol ? `<th>Ground truth ${{cfg.label}}</th>` : '';
      tableHead.innerHTML = `<tr><th>Patient</th><th>SegResNet ${{cfg.label}}</th><th>TotalSegmentator ${{cfg.label}}</th>${{gtHead}}<th>${{cfg.label}} Delta</th></tr>`;
      tableBody.innerHTML = sorted.map(row => {{
        const gtCell = cfg.gtCol ? `<td>${{fmtMetric(row[cfg.gtCol], cfg)}}</td>` : '';
        return `<tr><td>${{row.patient_id}}</td><td>${{fmtMetric(row[cfg.segCol], cfg)}}</td><td>${{fmtMetric(row[cfg.totCol], cfg)}}</td>${{gtCell}}<td>${{fmtMetric(row[cfg.deltaCol], cfg)}}</td></tr>`;
      }}).join('');
    }}
    function renderSummary(summary) {{
      document.getElementById('patientCount').textContent = summary.patientCount;
      document.getElementById('meanDice').textContent = `${{summary.segresnetDice}} / ${{summary.totalsegDice}}`;
      document.getElementById('meanHd95').textContent = `${{summary.segresnetHd95}} / ${{summary.totalsegHd95}} mm`;
      document.getElementById('diceWins').textContent = `${{summary.segresnetWins}} / ${{summary.totalsegWins}}`;
    }}
    function renderFold(fold) {{
      const payload = window.FOLD_COMPARISONS[String(fold)];
      rows = payload.rows;
      window.COMPARISON_ROWS = rows;
      renderSummary(payload.summary);
      select.innerHTML = rows.map(row => `<option value="${{row.patient_id}}">${{row.patient_id}}</option>`).join('');
      renderBars();
      renderTable();
      renderPatient(select.value);
    }}
    foldSelect.value = "{initial_fold}";
    foldSelect.addEventListener('change', event => renderFold(event.target.value));
    select.addEventListener('change', event => renderPatient(event.target.value));
    sliceSlider.addEventListener('input', event => renderSlice(event.target.value));
    metricSelect.addEventListener('change', event => {{
      selectedMetric = event.target.value;
      renderBars();
      renderTable();
    }});
    renderFold(foldSelect.value);
  </script>
</body>
</html>
"""
    output_html.write_text(html)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a static TotalSegmentator vs SegResNet fold-model comparison report."
    )
    parser.add_argument("--segresnet-metrics", type=Path, default=DEFAULT_SEGRESNET_METRICS)
    parser.add_argument("--totalseg-metrics", type=Path, default=DEFAULT_TOTALSEG_METRICS)
    parser.add_argument("--segresnet-mask-root", type=Path, default=DEFAULT_SEGRESNET_MASK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model-folds",
        default="-1,0,1,2,3,4",
        help="Comma-separated SegResNet model folds to include. Use -1 for the full/left-out model.",
    )
    parser.add_argument("--skip-images", action="store_true")
    return parser


def _add_comparison_images(
    comparison: pd.DataFrame,
    output_dir: Path,
    skip_images: bool,
) -> pd.DataFrame:
    comparison = comparison.copy()
    if skip_images:
        comparison["comparison_image"] = ""
        comparison["comparison_slice_images"] = "[]"
        return comparison

    image_dir = output_dir / "images"
    relative_paths = []
    slice_lists = []
    for _, row in comparison.iterrows():
        patient_id = str(row["patient_id"])
        slice_paths = generate_patient_slice_comparison_images(row, image_dir / patient_id)
        if slice_paths:
            relative_paths.append(slice_paths[0])
            slice_lists.append(json.dumps(slice_paths))
        else:
            relative_paths.append("")
            slice_lists.append("[]")
    comparison["comparison_image"] = relative_paths
    comparison["comparison_slice_images"] = slice_lists
    return comparison


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    model_folds = _parse_model_folds(args.model_folds)
    specs = build_model_fold_specs(model_folds)

    if len(specs) == 1 and specs[0].fold == -1:
        comparison = load_comparison_rows(
            args.segresnet_metrics,
            args.totalseg_metrics,
            segresnet_mask_root=args.segresnet_mask_root,
            model_fold=-1,
        )
        comparison = _add_comparison_images(comparison, args.output_dir, args.skip_images)
        write_comparison_csv(comparison, args.output_dir / "comparison_metrics.csv")
        render_html(comparison, args.output_dir / "index.html")
        return 0

    fold_comparisons = {}
    for spec in specs:
        comparison = load_comparison_rows(
            spec.metrics_csv,
            args.totalseg_metrics,
            segresnet_mask_root=spec.mask_root,
            model_fold=spec.fold,
        )
        fold_output_dir = args.output_dir / spec.output_dir_name
        comparison = _add_comparison_images(comparison, fold_output_dir, args.skip_images)
        write_comparison_csv(comparison, fold_output_dir / "comparison_metrics.csv")
        fold_comparisons[spec.fold] = _prefix_comparison_image_paths(
            comparison,
            spec.output_dir_name,
        )
    render_multi_fold_html(fold_comparisons, args.output_dir / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
