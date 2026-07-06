import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts import submit_swin_unetr_full_train_job as submitter


class SwinUNETRFullTrainSubmitterTests(unittest.TestCase):
    def test_defaults_to_qc_full_training_parameters(self):
        args = submitter.parse_args(["--dry_run"])

        self.assertEqual(args.train_folds, [0, 1, 2, 3, 4])
        self.assertEqual(args.split_csv, "data/cv_splits_qc.csv")
        self.assertEqual(args.learning_rate, 0.0005)
        self.assertEqual(args.loss, "dice_focal")
        self.assertEqual(args.roi_size, 96)
        self.assertEqual(args.weight_decay, 1e-5)
        self.assertEqual(args.azure_compute, "azureml:clusterprdwe-g-t2-vzhst6")

    def test_training_command_uses_full_training_and_requested_hyperparameters(self):
        args = submitter.parse_args(["--dry_run"])

        command = submitter.training_command(args)

        self.assertIn("python scripts/swin_UNETR.py", command)
        self.assertIn("--split_csv data/cv_splits_qc.csv", command)
        self.assertIn("--full_train", command)
        self.assertIn("--train_folds 0 1 2 3 4", command)
        self.assertIn("--output_fold 0", command)
        self.assertIn("training.lr=0.0005", command)
        self.assertIn("training.loss=dice_focal", command)
        self.assertIn("training.weight_decay=1e-05", command)
        self.assertIn("transforms.roi_size=[96,96,96]", command)
        self.assertIn("--pretrained_swin_encoder weights/model_swinvit.pt", command)

    def test_script_help_works_when_executed_by_path(self):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/submit_swin_unetr_full_train_job.py",
                "--help",
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("full SwinUNETR training", result.stdout)

    def test_dry_run_writes_azure_job_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = submitter.parse_args(["--dry_run", "--output_dir", tmp])

            job_yaml = submitter.submit_job(args)
            job_text = Path(job_yaml).read_text()

            self.assertIn("swin_unetr_full_train", job_text)
            self.assertIn("--full_train", job_text)
            self.assertIn("--train_folds 0 1 2 3 4", job_text)
            self.assertIn("training.loss=dice_focal", job_text)


if __name__ == "__main__":
    unittest.main()
