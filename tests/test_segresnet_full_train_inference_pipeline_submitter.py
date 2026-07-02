import tempfile
import unittest
import subprocess
import sys
from pathlib import Path

from scripts import submit_segresnet_full_train_inference_pipeline as pipeline


class SegResNetFullTrainInferencePipelineSubmitterTests(unittest.TestCase):
    def test_defaults_to_full_train_folds_and_qc_split(self):
        args = pipeline.parse_args(["--dry_run"])

        self.assertEqual(args.train_folds, [0, 1, 2, 3, 4])
        self.assertEqual(args.test_fold, -1)
        self.assertEqual(args.split_csv, "data/cv_splits_qc.csv")
        self.assertEqual(args.azure_compute, "azureml:clusterprdwe-g-t2-vzhst6")

    def test_script_help_works_when_executed_by_path(self):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/submit_segresnet_full_train_inference_pipeline.py",
                "--help",
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("full SegResNet training", result.stdout)

    def test_training_command_uses_full_training_mode(self):
        args = pipeline.parse_args(["--dry_run"])

        command = pipeline.training_command(args)

        self.assertIn("python scripts/train_job.py", command)
        self.assertIn("--full_train", command)
        self.assertIn("--fold 0", command)
        self.assertIn("--train_folds 0 1 2 3 4", command)
        self.assertIn("--split_csv data/cv_splits_qc.csv", command)

    def test_inference_command_uses_single_checkpoint_and_reserved_test_fold(self):
        args = pipeline.parse_args(["--dry_run"])

        command = pipeline.inference_command(args)

        self.assertIn("python jobs/inference_job.py", command)
        self.assertIn("--fold_count 1", command)
        self.assertIn("--test_fold -1", command)
        self.assertIn("--checkpoints_root ${{inputs.trained_model}}", command)
        self.assertIn("--output_dir ${{outputs.output_dir}}", command)

    def test_prepares_minimal_code_bundle_for_full_train_and_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = pipeline.parse_args(["--dry_run", "--output_dir", tmp])

            code_dir = pipeline.prepare_code_bundle(args)

            self.assertEqual(code_dir, Path(tmp) / "azure_code")
            self.assertTrue((code_dir / "scripts" / "train_job.py").exists())
            self.assertTrue((code_dir / "jobs" / "inference_job.py").exists())
            self.assertTrue((code_dir / "config" / "train_config.yaml").exists())
            self.assertTrue((code_dir / "data" / "cv_splits_qc.csv").exists())
            self.assertFalse((code_dir / "notebooks").exists())
            self.assertFalse((code_dir / "outputs").exists())

    def test_dry_run_pipeline_wires_inference_after_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = pipeline.parse_args(["--dry_run", "--output_dir", tmp])

            pipeline_yaml = pipeline.submit_pipeline(args)
            job_text = Path(pipeline_yaml).read_text()

            self.assertIn(
                "${{parent.jobs.segresnet_full_train.outputs.completion_marker}}",
                job_text,
            )
            self.assertIn(
                "${{parent.jobs.segresnet_full_train.outputs.output_model}}",
                job_text,
            )
            self.assertIn("fold_minus_one_masks", job_text)


if __name__ == "__main__":
    unittest.main()
