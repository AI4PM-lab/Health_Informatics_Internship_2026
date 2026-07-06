import json
import unittest
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/segresnet_cv_fold_mps_inference.ipynb")


class SegResNetCvFoldInferenceNotebookTests(unittest.TestCase):
    def _notebook_source(self) -> str:
        notebook = json.loads(NOTEBOOK_PATH.read_text())
        return "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
        )

    def test_cv_fold_inference_notebook_exists_and_targets_requested_paths(self):
        self.assertTrue(NOTEBOOK_PATH.exists())

        source = self._notebook_source()

        self.assertIn("data/cv_splits_qc.csv", source)
        self.assertIn("notebooks/output/models/dice_focal_fold_{fold}/best_metric_model.pth", source)
        self.assertIn("notebooks/output/models/dice_focal_fold_{fold}/masks", source)
        self.assertIn("FOLD_IDS = [0, 1, 2, 3, 4]", source)

    def test_cv_fold_inference_notebook_calls_fold_specific_cli_arguments(self):
        source = self._notebook_source()

        self.assertIn("run_local_fold_minus1_inference.main", source)
        self.assertIn('"--checkpoint"', source)
        self.assertIn('"--output_dir"', source)
        self.assertIn('"--test_fold"', source)
        self.assertIn('"--skip_existing"', source)
        self.assertIn("str(fold)", source)


if __name__ == "__main__":
    unittest.main()
