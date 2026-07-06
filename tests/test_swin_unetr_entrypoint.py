import unittest

from scripts import swin_UNETR as swin


class SwinUNETREntrypointTests(unittest.TestCase):
    def test_parser_accepts_full_training_arguments(self):
        parser = swin.build_arg_parser()

        args = parser.parse_args(
            [
                "--input_data",
                "/data",
                "--output_model",
                "/outputs",
                "--split_csv",
                "data/cv_splits_qc.csv",
                "--fold",
                "0",
                "--full_train",
                "--train_folds",
                "0",
                "1",
                "2",
                "3",
                "4",
                "--output_fold",
                "0",
            ]
        )

        self.assertTrue(args.full_train)
        self.assertEqual(args.train_folds, [0, 1, 2, 3, 4])
        self.assertEqual(args.output_fold, 0)

    def test_applies_swin_defaults_from_roi_size(self):
        model_cfg = {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 2,
            "init_filters": 16,
            "blocks_down": [1, 2, 2, 4],
            "blocks_up": [1, 1, 1],
        }

        result = swin.with_swin_defaults(model_cfg, [128, 128, 128])

        self.assertEqual(result["feature_size"], 48)
        self.assertEqual(result["use_checkpoint"], True)
        self.assertEqual(result["img_size"], [128, 128, 128])
        self.assertEqual(result["in_channels"], 1)
        self.assertEqual(result["out_channels"], 2)

    def test_filters_segresnet_only_keys_from_swin_kwargs(self):
        model_cfg = swin.with_swin_defaults(
            {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 2,
                "init_filters": 16,
                "blocks_down": [1, 2, 2, 4],
                "blocks_up": [1, 1, 1],
                "feature_size": 24,
                "use_checkpoint": False,
                "dropout_path_rate": 0.1,
            },
            [96, 96, 96],
        )

        result = swin.swin_kwargs_for_signature(
            model_cfg,
            {
                "in_channels",
                "out_channels",
                "feature_size",
                "use_checkpoint",
                "dropout_path_rate",
                "spatial_dims",
            },
        )

        self.assertEqual(
            result,
            {
                "in_channels": 1,
                "out_channels": 2,
                "feature_size": 24,
                "use_checkpoint": False,
                "dropout_path_rate": 0.1,
                "spatial_dims": 3,
            },
        )
        self.assertNotIn("init_filters", result)
        self.assertNotIn("blocks_down", result)
        self.assertNotIn("blocks_up", result)

    def test_uses_legacy_img_size_when_signature_needs_it(self):
        model_cfg = swin.with_swin_defaults(
            {
                "in_channels": 1,
                "out_channels": 2,
            },
            [96, 96, 96],
        )

        result = swin.swin_kwargs_for_signature(
            model_cfg,
            {"img_size", "in_channels", "out_channels", "feature_size"},
        )

        self.assertEqual(result["img_size"], (96, 96, 96))


if __name__ == "__main__":
    unittest.main()
