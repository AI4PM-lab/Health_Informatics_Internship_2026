import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts import submit_segresnet_full_train_surface_hd_inference_pipeline as pipeline


class SegResNetFullTrainSurfaceHDPipelineSubmitterTests(unittest.TestCase):
    def test_training_command_uses_configurable_train_job_and_surface_hd_overrides(self):
        args = pipeline.parse_args(["--dry_run"])

        command = pipeline.training_command(args)

        self.assertIn("python scripts/train_job_only_dice.py", command)
        self.assertIn("--full_train", command)
        self.assertIn("--train_folds 0 1 2 3 4", command)
        self.assertIn("training.loss=surface_hausdorff", command)
        self.assertIn("training.hausdorff_validation_loss=dice", command)

    def test_user_overrides_are_appended_after_surface_hd_defaults(self):
        args = pipeline.parse_args(["--dry_run", "training.lr=0.0001"])

        command = pipeline.training_command(args)

        self.assertLess(
            command.index("training.loss=surface_hausdorff"),
            command.index("training.lr=0.0001"),
        )

    def test_script_help_works_when_executed_by_path(self):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/submit_segresnet_full_train_surface_hd_inference_pipeline.py",
                "--help",
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("surface-only Hausdorff", result.stdout)

    def test_dry_run_pipeline_uses_distinct_output_and_experiment_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = pipeline.parse_args(["--dry_run", "--output_dir", tmp])

            pipeline_yaml = pipeline.submit_pipeline(args)
            job_text = Path(pipeline_yaml).read_text()

            self.assertIn("surface_hausdorff", job_text)
            self.assertIn("segresnet_full_train_surface_hd", job_text)
            self.assertIn("fold_minus_one_masks", job_text)


if __name__ == "__main__":
    unittest.main()
