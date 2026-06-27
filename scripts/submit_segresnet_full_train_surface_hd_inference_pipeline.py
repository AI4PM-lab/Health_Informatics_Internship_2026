import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import submit_segresnet_cv_pipeline as cv_pipeline


DEFAULT_TRAIN_FOLDS = [0, 1, 2, 3, 4]
DEFAULT_TEST_FOLD = -1
DEFAULT_OUTPUT_FOLD = 0
DEFAULT_OUTPUT_DIR = "outputs/segresnet_full_train_surface_hd_inference_pipeline"
DEFAULT_EXPERIMENT_NAME = "myocardium_segresnet_full_train_surface_hd_inference"
DEFAULT_SURFACE_HD_OVERRIDES = [
    "training.loss=surface_hausdorff",
    "training.hausdorff_validation_loss=dice",
    "training.hausdorff_downsample=2",
    "training.hausdorff_lambda=1.0",
    "training.hausdorff_lambda_dice=0.0",
    "training.train_batch_size=1",
    "training.accumulation_steps=4",
]

DEFAULT_AZURE_COMPUTE = cv_pipeline.DEFAULT_AZURE_COMPUTE
DEFAULT_AZURE_ENVIRONMENT = cv_pipeline.DEFAULT_AZURE_ENVIRONMENT
DEFAULT_AZURE_INPUT_DATA = cv_pipeline.DEFAULT_AZURE_INPUT_DATA
DEFAULT_AZURE_TENANT_ID = cv_pipeline.DEFAULT_AZURE_TENANT_ID
DEFAULT_AZURE_SUBSCRIPTION_ID = cv_pipeline.DEFAULT_AZURE_SUBSCRIPTION_ID
DEFAULT_AZURE_RESOURCE_GROUP = cv_pipeline.DEFAULT_AZURE_RESOURCE_GROUP
DEFAULT_AZURE_WORKSPACE_NAME = cv_pipeline.DEFAULT_AZURE_WORKSPACE_NAME
DEFAULT_CONFIG = cv_pipeline.DEFAULT_CONFIG
DEFAULT_SPLIT_CSV = cv_pipeline.DEFAULT_SPLIT_CSV

azure_compute_name = cv_pipeline.azure_compute_name
safe_azure_name = cv_pipeline.safe_azure_name
prepare_code_bundle = cv_pipeline.prepare_code_bundle
get_azure_ml_client = cv_pipeline.get_azure_ml_client
serialize_azure_job = cv_pipeline.serialize_azure_job


def training_command(args):
    train_folds = " ".join(str(fold) for fold in args.train_folds)
    parts = [
        "python scripts/train_job_only_dice.py",
        "--input_data ${{inputs.input_data}}",
        "--output_model ${{outputs.output_model}}",
        f"--split_csv {args.split_csv}",
        f"--fold {args.output_fold}",
        f"--config {args.config}",
        "--full_train",
        f"--train_folds {train_folds}",
        f"--output_fold {args.output_fold}",
        *DEFAULT_SURFACE_HD_OVERRIDES,
        *args.extra_overrides,
    ]
    return " ".join(parts)


def inference_command(args):
    parts = [
        "python jobs/inference_job.py",
        "--input_data ${{inputs.input_data}}",
        "--output_dir ${{outputs.output_dir}}",
        "--fold_count 1",
        "--checkpoints_root ${{inputs.trained_model}}",
        f"--split_csv {args.split_csv}",
        f"--test_fold {args.test_fold}",
        f"--config {args.config}",
    ]
    return " ".join(parts)


def make_training_component(args):
    from azure.ai.ml import Input, Output, command

    command_text = (
        training_command(args)
        + " && mkdir -p ${{outputs.completion_marker}}"
        + " && printf done > ${{outputs.completion_marker}}/done.txt"
    )

    return command(
        name=safe_azure_name("segresnet_full_train_surface_hd"),
        display_name="Myocardium SegResNet Full Training Surface HD",
        description=(
            "Train one SegResNet model on all QC split folds 0-4 using the "
            "surface-only Hausdorff loss variant. Fold -1 remains reserved "
            "for held-out inference."
        ),
        code=str(args.code_dir),
        command=command_text,
        environment=args.azure_environment,
        environment_variables={
            "PYTHONPATH": ".",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        compute=azure_compute_name(args.azure_compute),
        inputs={"input_data": Input(type="uri_folder")},
        outputs={
            "output_model": Output(type="uri_folder"),
            "completion_marker": Output(type="uri_folder"),
        },
        distribution={
            "type": "pytorch",
            "process_count_per_instance": args.process_count_per_instance,
        },
        instance_count=args.azure_instance_count,
        shm_size=args.shm_size,
        is_deterministic=False,
    )


def make_inference_component(args):
    from azure.ai.ml import Input, Output, command

    command_text = (
        "test -f ${{inputs.final_marker}}/done.txt && " + inference_command(args)
    )

    return command(
        name=safe_azure_name("segresnet_surface_hd_fold_minus_one_inference"),
        display_name="Myocardium SegResNet Surface HD Fold -1 Inference",
        description=(
            "Run inference on every fold -1 case using the single full-training "
            "surface-HD checkpoint and write downloadable predicted masks."
        ),
        code=str(args.code_dir),
        command=command_text,
        environment=args.azure_environment,
        environment_variables={"PYTHONPATH": "."},
        compute=azure_compute_name(args.azure_compute),
        inputs={
            "input_data": Input(type="uri_folder"),
            "trained_model": Input(type="uri_folder"),
            "final_marker": Input(type="uri_folder"),
        },
        outputs={"output_dir": Output(type="uri_folder")},
        instance_count=1,
        shm_size=args.shm_size,
        is_deterministic=False,
    )


def build_azure_pipeline(args):
    from azure.ai.ml import Input
    from azure.ai.ml.dsl import pipeline

    compute_name = azure_compute_name(args.azure_compute)

    @pipeline(
        display_name=args.pipeline_display_name,
        description=(
            "Single SegResNet full-training job on folds 0-4 from "
            "cv_splits_qc.csv using surface-only Hausdorff loss, followed by "
            "inference on held-out fold -1."
        ),
        experiment_name=args.azure_experiment_name,
        default_compute=compute_name,
    )
    def segresnet_full_train_surface_hd_inference_pipeline(input_data):
        training_component = make_training_component(args)
        training_node = training_component(input_data=input_data)
        training_node.name = safe_azure_name("segresnet_full_train_surface_hd")
        training_node.tags = {
            "model": "segresnet",
            "loss": "surface_hausdorff",
            "split_csv": args.split_csv,
            "train_folds": ",".join(str(fold) for fold in args.train_folds),
            "stage": "full_training",
        }

        inference_component = make_inference_component(args)
        inference_node = inference_component(
            input_data=input_data,
            trained_model=training_node.outputs.output_model,
            final_marker=training_node.outputs.completion_marker,
        )
        inference_node.name = safe_azure_name("segresnet_surface_hd_fold_minus_one_inference")
        inference_node.tags = {
            "model": "segresnet",
            "loss": "surface_hausdorff",
            "split_csv": args.split_csv,
            "test_fold": str(args.test_fold),
            "stage": "inference",
        }
        return {"fold_minus_one_masks": inference_node.outputs.output_dir}

    return segresnet_full_train_surface_hd_inference_pipeline(
        input_data=Input(type="uri_folder", path=args.azure_input_data, mode="ro_mount")
    )


def submit_pipeline(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.code_dir = prepare_code_bundle(args)

    pipeline_job = build_azure_pipeline(args)
    pipeline_yaml = out_dir / "segresnet_full_train_surface_hd_inference_pipeline_job.yml"
    pipeline_job.dump(pipeline_yaml)
    print(f"Azure pipeline YAML: {pipeline_yaml}", flush=True)

    if args.dry_run:
        print("Dry run only. Azure pipeline not submitted.", flush=True)
        return pipeline_yaml

    ml_client = get_azure_ml_client(args)
    returned_job = ml_client.jobs.create_or_update(pipeline_job)
    payload = serialize_azure_job(returned_job)
    submission_json = out_dir / "segresnet_full_train_surface_hd_inference_submission.json"
    submission_json.write_text(json.dumps(payload, indent=2))

    print(f"Submitted pipeline: {payload['name']}", flush=True)
    if payload["studio_url"]:
        print(f"Studio URL: {payload['studio_url']}", flush=True)
    return submission_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Submit one Azure ML parent pipeline with full SegResNet training "
            "on folds 0-4 using surface-only Hausdorff loss and queued "
            "inference on fold -1."
        )
    )
    parser.add_argument("--train_folds", nargs="+", type=int, default=list(DEFAULT_TRAIN_FOLDS))
    parser.add_argument("--test_fold", type=int, default=DEFAULT_TEST_FOLD)
    parser.add_argument("--output_fold", type=int, default=DEFAULT_OUTPUT_FOLD)
    parser.add_argument("--split_csv", default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--pipeline_display_name",
        default="Myocardium SegResNet Surface HD Full Train + Fold -1 Inference",
    )
    parser.add_argument(
        "--azure_auth_mode",
        choices=["default", "browser"],
        default="default",
    )
    parser.add_argument("--azure_compute", default=DEFAULT_AZURE_COMPUTE)
    parser.add_argument("--azure_environment", default=DEFAULT_AZURE_ENVIRONMENT)
    parser.add_argument("--azure_input_data", default=DEFAULT_AZURE_INPUT_DATA)
    parser.add_argument("--azure_tenant_id", default=DEFAULT_AZURE_TENANT_ID)
    parser.add_argument("--azure_subscription_id", default=DEFAULT_AZURE_SUBSCRIPTION_ID)
    parser.add_argument("--azure_resource_group", default=DEFAULT_AZURE_RESOURCE_GROUP)
    parser.add_argument("--azure_workspace_name", default=DEFAULT_AZURE_WORKSPACE_NAME)
    parser.add_argument("--azure_experiment_name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--azure_instance_count", type=int, default=1)
    parser.add_argument("--process_count_per_instance", type=int, default=1)
    parser.add_argument("--shm_size", default="8g")

    args, extra_overrides = parser.parse_known_args(argv)
    args.extra_overrides = extra_overrides

    if not args.train_folds:
        parser.error("--train_folds must contain at least one fold")
    if len(set(args.train_folds)) != len(args.train_folds):
        parser.error("--train_folds cannot contain duplicate fold values")
    if any(fold < 0 for fold in args.train_folds):
        parser.error("--train_folds must be non-negative; fold -1 is reserved for testing")
    if args.test_fold in args.train_folds:
        parser.error("--test_fold must not be one of the training folds")
    if args.output_fold < 0:
        parser.error("--output_fold must be non-negative")

    return args


if __name__ == "__main__":
    submit_pipeline(parse_args())
