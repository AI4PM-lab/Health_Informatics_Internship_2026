import csv
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import nibabel as nib

from scripts import run_local_fold_minus1_inference as local_inference


class LocalFoldMinus1InferenceTests(unittest.TestCase):
    def test_parser_defaults_target_requested_checkpoint_and_output_dir(self):
        args = local_inference.build_arg_parser().parse_args([])

        self.assertEqual(
            str(args.checkpoint),
            "notebooks/output/totalsegmentator_heart_myocardium_visualizations/models/dice_focal/best_metric_model.pth",
        )
        self.assertEqual(
            str(args.output_dir),
            "notebooks/output/totalsegmentator_heart_myocardium_visualizations/models/dice_focal/masks",
        )
        self.assertEqual(args.test_fold, -1)

    def test_select_fold_cases_uses_only_existing_fold_minus_one_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "nifti_data"
            split_csv = Path(tmp) / "cv_splits_qc.csv"
            rows = [
                {"patient_id": "TAVI_TEST_A", "fold": "-1"},
                {"patient_id": "TAVI_TRAIN", "fold": "0"},
                {"patient_id": "TAVI_TEST_MISSING", "fold": "-1"},
                {"patient_id": "TAVI_TEST_B", "fold": "-1"},
            ]
            with split_csv.open("w", newline="") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=["patient_id", "fold"])
                writer.writeheader()
                writer.writerows(rows)

            for patient_id in ("TAVI_TEST_A", "TAVI_TRAIN", "TAVI_TEST_B"):
                patient_dir = root / patient_id
                patient_dir.mkdir(parents=True)
                (patient_dir / "CT_LATE.nii.gz").write_bytes(b"image")
                (patient_dir / "registration_mask.nii.gz").write_bytes(b"label")

            cases = local_inference.select_cases_from_split(root, split_csv, test_fold=-1)

            self.assertEqual([case.patient_id for case in cases], ["TAVI_TEST_A", "TAVI_TEST_B"])

    def test_binary_metrics_identical_masks_are_perfect(self):
        mask = np.zeros((5, 5, 5), dtype=np.uint8)
        mask[1:4, 1:4, 1:4] = 1

        metrics = local_inference.binary_mask_metrics(mask, mask, spacing=(1.0, 1.0, 1.0))

        self.assertEqual(metrics["dice"], 1.0)
        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["surface_hausdorff_mm"], 0.0)
        self.assertEqual(metrics["hausdorff_mm"], 0.0)
        self.assertEqual(metrics["pred_volume_ml"], 0.027)
        self.assertEqual(metrics["ground_truth_volume_ml"], 0.027)

    def test_binary_metrics_shifted_masks_have_expected_overlap(self):
        pred = np.zeros((5, 5, 5), dtype=np.uint8)
        gt = np.zeros((5, 5, 5), dtype=np.uint8)
        pred[1:3, 1:3, 1:3] = 1
        gt[2:4, 1:3, 1:3] = 1

        metrics = local_inference.binary_mask_metrics(pred, gt, spacing=(2.0, 1.0, 1.0))

        self.assertAlmostEqual(metrics["dice"], 0.5)
        self.assertAlmostEqual(metrics["iou"], 1 / 3)
        self.assertAlmostEqual(metrics["hausdorff_mm"], 2.0)
        self.assertAlmostEqual(metrics["surface_hausdorff_mm"], 2.0)

    def test_save_mask_like_reference_accepts_numpy_qform_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference_path = Path(tmp) / "reference.nii.gz"
            output_path = Path(tmp) / "mask.nii.gz"
            reference = nib.Nifti1Image(
                np.zeros((3, 3, 3), dtype=np.uint8),
                affine=np.diag([2.0, 3.0, 4.0, 1.0]),
            )
            reference.set_qform(reference.affine, code=1)
            reference.set_sform(reference.affine, code=1)
            nib.save(reference, reference_path)

            mask = np.ones((3, 3, 3), dtype=np.uint8)
            local_inference.save_mask_like_reference(mask, reference_path, output_path)

            saved = nib.load(output_path)
            self.assertEqual(saved.header.get_data_dtype(), np.dtype("uint8"))
            self.assertEqual(int(saved.header["qform_code"]), 1)
            self.assertEqual(int(saved.header["sform_code"]), 1)

    def test_requested_mps_fails_early_when_unavailable(self):
        fake_torch = types.SimpleNamespace(
            device=lambda value: value,
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(
                    is_built=lambda: True,
                    is_available=lambda: False,
                )
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "MPS was requested"):
            local_inference._select_device(fake_torch, "mps")

    def test_filter_readable_cases_skips_invalid_nifti_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good_image = root / "good_image.nii.gz"
            good_label = root / "good_label.nii.gz"
            bad_image = root / "bad_image.nii.gz"
            bad_label = root / "bad_label.nii.gz"
            valid = nib.Nifti1Image(np.zeros((3, 3, 3), dtype=np.uint8), affine=np.eye(4))
            nib.save(valid, good_image)
            nib.save(valid, good_label)
            bad_image.write_bytes(b"not gzip")
            nib.save(valid, bad_label)

            cases = [
                local_inference.PatientCase("TAVI_GOOD", good_image, good_label, -1),
                local_inference.PatientCase("TAVI_BAD", bad_image, bad_label, -1),
            ]

            readable, skipped = local_inference.filter_readable_cases(cases)

            self.assertEqual([case.patient_id for case in readable], ["TAVI_GOOD"])
            self.assertEqual(skipped[0]["patient_id"], "TAVI_BAD")
            self.assertIn("image_error", skipped[0])


if __name__ == "__main__":
    unittest.main()
