import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts import generate_fold_minus1_totalseg_vs_segresnet_report as report


class FoldMinus1TotalSegVsSegResNetReportTests(unittest.TestCase):
    def _write_csv(self, path, rows):
        pd.DataFrame(rows).to_csv(path, index=False)

    def _write_minimal_metric_inputs(self, root, fold=-1, patient_id="TAVI_001"):
        segresnet_csv = root / f"segresnet_{fold}.csv"
        totalseg_csv = root / "totalseg.csv"
        self._write_csv(
            segresnet_csv,
            [
                {
                    "patient_id": patient_id,
                    "fold": fold,
                    "mask_type": "postprocessed",
                    "mask_path": f"/seg/{patient_id}/segmentation_model_postprocessed.nii.gz",
                    "dice": 0.82,
                    "iou": 0.70,
                    "surface_hausdorff95_mm": 2.0,
                    "hausdorff95_mm": 2.4,
                    "pred_volume_ml": 104.0,
                    "ground_truth_volume_ml": 100.0,
                }
            ],
        )
        self._write_csv(
            totalseg_csv,
            [
                {
                    "patient_id": patient_id,
                    "dice": 0.78,
                    "iou": 0.64,
                    "hd95_surface_mm": 3.0,
                    "hd95_full_mask_mm": 3.5,
                    "totalsegmentator_volume_ml": 98.0,
                    "ground_truth_volume_ml": 100.0,
                    "totalsegmentator_mask": f"/totalseg/{patient_id}.nii.gz",
                    "ground_truth_mask": f"/gt/{patient_id}.nii.gz",
                }
            ],
        )
        return segresnet_csv, totalseg_csv

    def test_load_comparison_rows_uses_shared_patients_and_postprocessed_segresnet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segresnet_csv = root / "segresnet.csv"
            totalseg_csv = root / "totalseg.csv"
            self._write_csv(
                segresnet_csv,
                [
                    {
                        "patient_id": "TAVI_001",
                        "fold": -1,
                        "mask_type": "raw",
                        "mask_path": "/seg/raw_001.nii.gz",
                        "dice": 0.70,
                        "iou": 0.54,
                        "surface_hausdorff95_mm": 4.5,
                        "hausdorff95_mm": 5.5,
                        "pred_volume_ml": 101.0,
                        "ground_truth_volume_ml": 100.0,
                    },
                    {
                        "patient_id": "TAVI_001",
                        "fold": -1,
                        "mask_type": "postprocessed",
                        "mask_path": "/seg/post_001.nii.gz",
                        "dice": 0.82,
                        "iou": 0.70,
                        "surface_hausdorff95_mm": 2.0,
                        "hausdorff95_mm": 2.4,
                        "pred_volume_ml": 104.0,
                        "ground_truth_volume_ml": 100.0,
                    },
                    {
                        "patient_id": "TAVI_003",
                        "fold": -1,
                        "mask_type": "postprocessed",
                        "mask_path": "/seg/post_003.nii.gz",
                        "dice": 0.90,
                        "iou": 0.82,
                        "surface_hausdorff95_mm": 1.5,
                        "hausdorff95_mm": 1.9,
                        "pred_volume_ml": 119.0,
                        "ground_truth_volume_ml": 120.0,
                    },
                ],
            )
            self._write_csv(
                totalseg_csv,
                [
                    {
                        "patient_id": "TAVI_001",
                        "dice": 0.78,
                        "iou": 0.64,
                        "hd95_surface_mm": 3.0,
                        "hd95_full_mask_mm": 3.5,
                        "totalsegmentator_volume_ml": 98.0,
                        "ground_truth_volume_ml": 100.0,
                        "totalsegmentator_mask": "/totalseg/001.nii.gz",
                        "ground_truth_mask": "/gt/001.nii.gz",
                    },
                    {
                        "patient_id": "TAVI_002",
                        "dice": 0.88,
                        "iou": 0.79,
                        "hd95_surface_mm": 2.5,
                        "hd95_full_mask_mm": 2.8,
                        "totalsegmentator_volume_ml": 105.0,
                        "ground_truth_volume_ml": 100.0,
                        "totalsegmentator_mask": "/totalseg/002.nii.gz",
                        "ground_truth_mask": "/gt/002.nii.gz",
                    },
                ],
            )

            comparison = report.load_comparison_rows(segresnet_csv, totalseg_csv)

            self.assertEqual(comparison["patient_id"].tolist(), ["TAVI_001"])
            row = comparison.iloc[0]
            self.assertEqual(row["segresnet_mask_type"], "postprocessed")
            self.assertEqual(row["segresnet_mask_path"], "/seg/post_001.nii.gz")
            self.assertEqual(row["totalseg_mask_path"], "/totalseg/001.nii.gz")
            self.assertAlmostEqual(row["dice_delta_segresnet_minus_totalseg"], 0.04)
            self.assertAlmostEqual(row["hd95_delta_segresnet_minus_totalseg_mm"], -1.1)
            self.assertAlmostEqual(row["surface_hd95_delta_segresnet_minus_totalseg_mm"], -1.0)
            self.assertEqual(row["best_dice_model"], "SegResNet")
            self.assertEqual(row["best_hd95_model"], "SegResNet")

    def test_load_comparison_rows_accepts_requested_non_minus_one_model_fold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segresnet_csv, totalseg_csv = self._write_minimal_metric_inputs(
                root,
                fold=3,
                patient_id="TAVI_003",
            )

            comparison = report.load_comparison_rows(
                segresnet_csv,
                totalseg_csv,
                model_fold=3,
            )

            self.assertEqual(comparison["patient_id"].tolist(), ["TAVI_003"])

    def test_write_comparison_csv_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            comparison = pd.DataFrame(
                [
                    {
                        "patient_id": "TAVI_001",
                        "segresnet_dice": 0.82,
                        "totalseg_dice": 0.78,
                    }
                ]
            )
            output_csv = Path(tmp) / "nested" / "comparison.csv"

            report.write_comparison_csv(comparison, output_csv)

            written = pd.read_csv(output_csv)
            self.assertEqual(written["patient_id"].tolist(), ["TAVI_001"])
            self.assertAlmostEqual(written.loc[0, "segresnet_dice"], 0.82)

    def test_render_html_writes_static_dashboard_with_embedded_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_html = Path(tmp) / "index.html"
            comparison = pd.DataFrame(
                [
                    {
                        "patient_id": "TAVI_001",
                        "segresnet_dice": 0.82,
                        "totalseg_dice": 0.78,
                        "dice_delta_segresnet_minus_totalseg": 0.04,
                        "best_dice_model": "SegResNet",
                        "segresnet_hd95_mm": 2.0,
                        "totalseg_hd95_mm": 3.0,
                        "hd95_delta_segresnet_minus_totalseg_mm": -1.0,
                        "segresnet_surface_hd95_mm": 2.0,
                        "totalseg_surface_hd95_mm": 3.0,
                        "surface_hd95_delta_segresnet_minus_totalseg_mm": -1.0,
                        "best_hd95_model": "SegResNet",
                        "segresnet_volume_ml": 104.0,
                        "totalseg_volume_ml": 98.0,
                        "ground_truth_volume_ml": 100.0,
                        "segresnet_volume_error_ml": 4.0,
                        "totalseg_volume_error_ml": -2.0,
                        "comparison_image": "images/TAVI_001_comparison.png",
                        "comparison_slice_images": '["images/TAVI_001/slice_000.png","images/TAVI_001/slice_001.png"]',
                    }
                ]
            )

            report.render_html(comparison, output_html)

            html = output_html.read_text()
            self.assertIn("Fold -1 Model Comparison", html)
            self.assertIn("TotalSegmentator", html)
            self.assertIn("SegResNet", html)
            self.assertIn("<select id=\"patientSelect\"", html)
            self.assertIn("window.COMPARISON_ROWS", html)
            self.assertIn("TAVI_001", html)
            self.assertIn("images/TAVI_001_comparison.png", html)
            self.assertIn("images/TAVI_001/slice_001.png", html)
            self.assertIn("<input id=\"sliceSlider\"", html)
            self.assertIn("<table", html)

    def test_render_multi_fold_html_adds_metric_selector_for_per_patient_visualization(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_html = Path(tmp) / "index.html"
            comparison = pd.DataFrame(
                [
                    {
                        "patient_id": "TAVI_001",
                        "segresnet_dice": 0.82,
                        "totalseg_dice": 0.78,
                        "dice_delta_segresnet_minus_totalseg": 0.04,
                        "best_dice_model": "SegResNet",
                        "segresnet_hd95_mm": 2.4,
                        "totalseg_hd95_mm": 3.5,
                        "hd95_delta_segresnet_minus_totalseg_mm": -1.1,
                        "segresnet_surface_hd95_mm": 2.0,
                        "totalseg_surface_hd95_mm": 3.0,
                        "surface_hd95_delta_segresnet_minus_totalseg_mm": -1.0,
                        "best_hd95_model": "SegResNet",
                        "segresnet_iou": 0.70,
                        "totalseg_iou": 0.64,
                        "iou_delta_segresnet_minus_totalseg": 0.06,
                        "segresnet_volume_ml": 104.0,
                        "totalseg_volume_ml": 98.0,
                        "ground_truth_volume_ml": 100.0,
                        "segresnet_volume_error_ml": 4.0,
                        "totalseg_volume_error_ml": -2.0,
                        "volume_delta_segresnet_minus_totalseg_ml": 6.0,
                        "comparison_image": "fold_0/images/TAVI_001/slice_000.png",
                        "comparison_slice_images": '["fold_0/images/TAVI_001/slice_000.png"]',
                    }
                ]
            )

            report.render_multi_fold_html({0: comparison}, output_html)

            html = output_html.read_text()
            self.assertIn("id=\"metricSelect\"", html)
            self.assertIn('<option value="dice">Dice</option>', html)
            self.assertIn('<option value="hd95">HD95</option>', html)
            self.assertIn('<option value="surface_hd95">Surface-only HD95</option>', html)
            self.assertIn('<option value="iou">IoU</option>', html)
            self.assertIn('<option value="volume">Volume</option>', html)
            self.assertIn("METRIC_CONFIGS", html)
            self.assertIn("selectedMetric", html)

    def test_render_multi_fold_html_adds_ground_truth_bar_for_volume_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_html = Path(tmp) / "index.html"
            comparison = pd.DataFrame(
                [
                    {
                        "patient_id": "TAVI_001",
                        "segresnet_dice": 0.82,
                        "totalseg_dice": 0.78,
                        "dice_delta_segresnet_minus_totalseg": 0.04,
                        "best_dice_model": "SegResNet",
                        "segresnet_hd95_mm": 2.4,
                        "totalseg_hd95_mm": 3.5,
                        "hd95_delta_segresnet_minus_totalseg_mm": -1.1,
                        "segresnet_surface_hd95_mm": 2.0,
                        "totalseg_surface_hd95_mm": 3.0,
                        "surface_hd95_delta_segresnet_minus_totalseg_mm": -1.0,
                        "best_hd95_model": "SegResNet",
                        "segresnet_iou": 0.70,
                        "totalseg_iou": 0.64,
                        "iou_delta_segresnet_minus_totalseg": 0.06,
                        "segresnet_volume_ml": 104.0,
                        "totalseg_volume_ml": 98.0,
                        "ground_truth_volume_ml": 100.0,
                        "segresnet_volume_error_ml": 4.0,
                        "totalseg_volume_error_ml": -2.0,
                        "volume_delta_segresnet_minus_totalseg_ml": 6.0,
                        "comparison_image": "fold_0/images/TAVI_001/slice_000.png",
                        "comparison_slice_images": '["fold_0/images/TAVI_001/slice_000.png"]',
                    }
                ]
            )

            report.render_multi_fold_html({0: comparison}, output_html)

            html = output_html.read_text()
            self.assertIn("gtCol: 'ground_truth_volume_ml'", html)
            self.assertIn("bar-fill gt", html)
            self.assertIn("Ground truth", html)

    def test_render_multi_fold_html_adds_top_right_fold_selector(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_html = Path(tmp) / "index.html"
            comparison = pd.DataFrame(
                [
                    {
                        "patient_id": "TAVI_001",
                        "segresnet_dice": 0.82,
                        "totalseg_dice": 0.78,
                        "dice_delta_segresnet_minus_totalseg": 0.04,
                        "best_dice_model": "SegResNet",
                        "segresnet_hd95_mm": 2.0,
                        "totalseg_hd95_mm": 3.0,
                        "hd95_delta_segresnet_minus_totalseg_mm": -1.0,
                        "segresnet_surface_hd95_mm": 2.0,
                        "totalseg_surface_hd95_mm": 3.0,
                        "surface_hd95_delta_segresnet_minus_totalseg_mm": -1.0,
                        "best_hd95_model": "SegResNet",
                        "segresnet_volume_ml": 104.0,
                        "totalseg_volume_ml": 98.0,
                        "ground_truth_volume_ml": 100.0,
                        "segresnet_volume_error_ml": 4.0,
                        "totalseg_volume_error_ml": -2.0,
                        "comparison_image": "fold_0/images/TAVI_001/slice_000.png",
                        "comparison_slice_images": '["fold_0/images/TAVI_001/slice_000.png"]',
                    }
                ]
            )

            report.render_multi_fold_html({0: comparison, 1: comparison}, output_html)

            html = output_html.read_text()
            self.assertIn("id=\"modelFoldSelect\"", html)
            self.assertIn('<option value="0">Fold 0 model</option>', html)
            self.assertIn('<option value="1">Fold 1 model</option>', html)
            self.assertIn("window.FOLD_COMPARISONS", html)
            self.assertIn("renderFold", html)

    def test_panel_titles_put_totalsegmentator_error_before_segresnet_error(self):
        self.assertEqual(
            report.PANEL_TITLES,
            [
                "Ground truth",
                "TotalSegmentator",
                "SegResNet",
                "Model overlap",
                "TotalSegmentator error",
                "SegResNet error",
            ],
        )

    def test_slice_indices_skip_first_three_and_last_three_when_possible(self):
        self.assertEqual(report.slice_indices_to_render(10), [3, 4, 5, 6])
        self.assertEqual(report.slice_indices_to_render(6), [])
        self.assertEqual(report.slice_indices_to_render(5), [])

    def test_parser_defaults_to_requested_project_outputs(self):
        args = report.build_arg_parser().parse_args([])

        self.assertEqual(
            args.segresnet_metrics,
            Path("notebooks/output/models/dice_focal/masks/fold_minus1_metrics.csv"),
        )
        self.assertEqual(
            args.totalseg_metrics,
            Path("notebooks/total_segmentator_heart_myocardium_metrics.csv"),
        )
        self.assertEqual(
            args.output_dir,
            Path("notebooks/output/fold_minus1_totalseg_vs_segresnet"),
        )
        self.assertEqual(args.model_folds, "-1,0,1,2,3,4")

    def test_generate_patient_comparison_image_returns_false_for_missing_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = pd.Series(
                {
                    "patient_id": "TAVI_001",
                    "segresnet_mask_path": "/missing/seg.nii.gz",
                    "totalseg_mask_path": "/missing/totalseg.nii.gz",
                    "ground_truth_mask_path": "/missing/gt.nii.gz",
                }
            )

            created = report.generate_patient_comparison_image(
                row,
                Path(tmp) / "TAVI_001_comparison.png",
            )

            self.assertFalse(created)

    def test_main_writes_csv_and_html_when_images_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segresnet_csv, totalseg_csv = self._write_minimal_metric_inputs(root)
            output_dir = root / "report"

            exit_code = report.main(
                [
                    "--segresnet-metrics",
                    str(segresnet_csv),
                    "--totalseg-metrics",
                    str(totalseg_csv),
                    "--output-dir",
                    str(output_dir),
                    "--model-folds",
                    "-1",
                    "--skip-images",
                ]
            )

            self.assertEqual(exit_code, 0)
            comparison_csv = output_dir / "comparison_metrics.csv"
            index_html = output_dir / "index.html"
            self.assertTrue(comparison_csv.exists())
            self.assertTrue(index_html.exists())
            comparison = pd.read_csv(comparison_csv, keep_default_na=False)
            self.assertEqual(comparison["comparison_image"].tolist(), [""])
            self.assertEqual(comparison["comparison_slice_images"].tolist(), ["[]"])
            self.assertIn("TAVI_001", index_html.read_text())

    def test_build_model_fold_specs_maps_requested_fold_directories(self):
        specs = report.build_model_fold_specs([0, 4])

        self.assertEqual([spec.fold for spec in specs], [0, 4])
        self.assertEqual(
            specs[0].metrics_csv,
            Path("notebooks/output/models/dice_focal_fold_0/masks/fold_minus1_metrics.csv"),
        )
        self.assertEqual(
            specs[1].mask_root,
            Path("notebooks/output/models/dice_focal_fold_4/masks"),
        )

    def test_add_comparison_images_populates_representative_and_slice_list(self):
        original = report.generate_patient_slice_comparison_images
        try:
            def fake_generate(row, output_dir):
                self.assertEqual(row["patient_id"], "TAVI_001")
                self.assertEqual(output_dir.name, "TAVI_001")
                return ["images/TAVI_001/slice_000.png", "images/TAVI_001/slice_001.png"]

            report.generate_patient_slice_comparison_images = fake_generate
            with tempfile.TemporaryDirectory() as tmp:
                comparison = pd.DataFrame([{"patient_id": "TAVI_001"}])

                updated = report._add_comparison_images(comparison, Path(tmp), skip_images=False)

                self.assertEqual(updated.loc[0, "comparison_image"], "images/TAVI_001/slice_000.png")
                self.assertEqual(
                    updated.loc[0, "comparison_slice_images"],
                    '["images/TAVI_001/slice_000.png", "images/TAVI_001/slice_001.png"]',
                )
        finally:
            report.generate_patient_slice_comparison_images = original


if __name__ == "__main__":
    unittest.main()
