import csv
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import generate_fold_minus1_totalsegmentator_metrics as totalseg_metrics


class GenerateFoldMinus1TotalSegmentatorMetricsTests(unittest.TestCase):
    def _write_split(self, path: Path) -> None:
        with path.open("w", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=["patient_id", "fold"])
            writer.writeheader()
            writer.writerows(
                [
                    {"patient_id": "TAVI_001", "fold": "-1"},
                    {"patient_id": "TAVI_002", "fold": "0"},
                    {"patient_id": "TAVI_003", "fold": "-1"},
                ]
            )

    def _write_mask(self, path: Path, data: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image = nib.Nifti1Image(data.astype(np.uint8), affine=np.diag([*spacing, 1.0]))
        nib.save(image, path)

    def test_select_cases_keeps_fold_minus_one_cases_with_totalseg_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "nifti_data"
            split_csv = Path(tmp) / "cv_splits_qc.csv"
            self._write_split(split_csv)

            for patient_id in ("TAVI_001", "TAVI_002"):
                self._write_mask(root / patient_id / "registration_mask.nii.gz", np.ones((2, 2, 2)))
                self._write_mask(
                    root
                    / patient_id
                    / "TotalSegmentator"
                    / "CT_LATE"
                    / "heartchambers_highres"
                    / "heart_myocardium.nii.gz",
                    np.ones((2, 2, 2)),
                )
            self._write_mask(root / "TAVI_003" / "registration_mask.nii.gz", np.ones((2, 2, 2)))

            cases, skipped = totalseg_metrics.select_totalsegmentator_cases(
                root,
                split_csv,
                test_fold=-1,
            )

            self.assertEqual([case.patient_id for case in cases], ["TAVI_001"])
            self.assertEqual(skipped[0]["patient_id"], "TAVI_003")
            self.assertEqual(skipped[0]["reason"], "missing_totalsegmentator_mask")

    def test_evaluate_case_writes_example_shaped_patient_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "nifti_data"
            pred = np.zeros((3, 3, 3), dtype=np.uint8)
            gt = np.zeros((3, 3, 3), dtype=np.uint8)
            pred[0:2, 0:2, 0:2] = 1
            gt[1:3, 0:2, 0:2] = 1

            gt_path = root / "TAVI_001" / "registration_mask.nii.gz"
            pred_path = (
                root
                / "TAVI_001"
                / "TotalSegmentator"
                / "CT_LATE"
                / "heartchambers_highres"
                / "heart_myocardium.nii.gz"
            )
            self._write_mask(gt_path, gt, spacing=(2.0, 1.0, 1.0))
            self._write_mask(pred_path, pred, spacing=(2.0, 1.0, 1.0))

            case = totalseg_metrics.TotalSegmentatorCase(
                patient_id="TAVI_001",
                fold=-1,
                totalsegmentator_mask=pred_path,
                ground_truth_mask=gt_path,
                output_dir=root / "TAVI_001" / "TotalSegmentator",
            )

            row = totalseg_metrics.evaluate_case(case, write_patient_csv=True)

            self.assertEqual(row["patient_id"], "TAVI_001")
            self.assertEqual(row["fold"], -1)
            self.assertEqual(row["mask_type"], "heart_myocardium")
            self.assertEqual(row["mask_path"], str(pred_path))
            self.assertAlmostEqual(row["dice"], 0.5)
            self.assertAlmostEqual(row["iou"], 1 / 3)
            self.assertAlmostEqual(row["hausdorff_mm"], 2.0)
            self.assertAlmostEqual(row["surface_hausdorff_mm"], 2.0)
            self.assertEqual(row["totalsegmentator_voxels"], 8)
            self.assertEqual(row["ground_truth_voxels"], 8)
            self.assertEqual(row["voxel_spacing_mm"], "2x1x1")

            with (root / "TAVI_001" / "TotalSegmentator" / "metrics.csv").open(newline="") as file_obj:
                written = list(csv.DictReader(file_obj))
            self.assertEqual(len(written), 1)
            self.assertEqual(written[0]["patient_id"], "TAVI_001")

    def test_run_evaluation_writes_cohort_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "nifti_data"
            split_csv = Path(tmp) / "cv_splits_qc.csv"
            output_csv = Path(tmp) / "fold_minus1_totalseg.csv"
            self._write_split(split_csv)

            mask = np.ones((2, 2, 2), dtype=np.uint8)
            gt_path = root / "TAVI_001" / "registration_mask.nii.gz"
            pred_path = (
                root
                / "TAVI_001"
                / "TotalSegmentator"
                / "CT_LATE"
                / "heartchambers_highres"
                / "heart_myocardium.nii.gz"
            )
            self._write_mask(gt_path, mask)
            self._write_mask(pred_path, mask)

            rows, skipped = totalseg_metrics.run_evaluation(
                input_data=root,
                split_csv=split_csv,
                test_fold=-1,
                output_csv=output_csv,
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(skipped[0]["patient_id"], "TAVI_003")
            with output_csv.open(newline="") as file_obj:
                written = list(csv.DictReader(file_obj))
            self.assertEqual(written[0]["patient_id"], "TAVI_001")
            self.assertEqual(written[0]["dice"], "1.0")
            self.assertTrue(output_csv.with_name("fold_minus1_totalseg_summary.csv").exists())
            self.assertTrue(output_csv.with_name("fold_minus1_totalseg_skipped.csv").exists())


if __name__ == "__main__":
    unittest.main()
