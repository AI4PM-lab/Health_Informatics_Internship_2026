import argparse
import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import submit_segresnet_cv_pipeline as cv_pipeline


DEFAULT_TRAIN_FOLDS = [0, 1, 2, 3, 4]
DEFAULT_OUTPUT_FOLD = 0
DEFAULT_OUTPUT_DIR = "outputs/swin_unetr_full_train_job"
DEFAULT_EXPERIMENT_NAME = "myocardium_swin_unetr_full_train_lr_3e-04"
DEFAULT_CONFIG = "config/train_config.yaml"
DEFAULT_SPLIT_CSV = "data/cv_splits_qc.csv"
DEFAULT_PRETRAINED_SWIN_ENCODER = "weights/model_swinvit.pt"
DEFAULT_LEARNING_RATE = 0.0003
DEFAULT_LOSS = "dice_focal"
DEFAULT_ROI_SIZE = 96
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_OVERRIDES = [
    "training.train_batch_size=1",
    "training.accumulation_steps=4",
    "inference.sw_batch_size=1",
]

DEFAULT_AZURE_COMPUTE = cv_pipeline.DEFAULT_AZURE_COMPUTE
DEFAULT_AZURE_ENVIRONMENT = cv_pipeline.DEFAULT_AZURE_ENVIRONMENT
DEFAULT_AZURE_INPUT_DATA = cv_pipeline.DEFAULT_AZURE_INPUT_DATA
DEFAULT_AZURE_TENANT_ID = cv_pipeline.DEFAULT_AZURE_TENANT_ID
DEFAULT_AZURE_SUBSCRIPTION_ID = cv_pipeline.DEFAULT_AZURE_SUBSCRIPTION_ID
DEFAULT_AZURE_RESOURCE_GROUP = cv_pipeline.DEFAULT_AZURE_RESOURCE_GROUP
DEFAULT_AZURE_WORKSPACE_NAME = cv_pipeline.DEFAULT_AZURE_WORKSPACE_NAME

azure_compute_name = cv_pipeline.azure_compute_name
safe_azure_name = cv_pipeline.safe_azure_name
get_azure_ml_client = cv_pipeline.get_azure_ml_client
serialize_azure_job = cv_pipeline.serialize_azure_job


def _project_relative_path(path):
    candidate = (PROJECT_ROOT / path).resolve()
    try:
        return candidate.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path must be inside the project: {path}") from exc


def _copy_tree(src, dst):
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        dirs_exist_ok=True,
    )


def prepare_code_bundle(args):
    code_dir = Path(args.output_dir) / "azure_code"
    if code_dir.exists():
        shutil.rmtree(code_dir)
    code_dir.mkdir(parents=True, exist_ok=True)

    for directory in ("scripts", "jobs", "lems_ct"):
        _copy_tree(PROJECT_ROOT / directory, code_dir / directory)

    for file_path in (
        args.config,
        args.split_csv,
        args.pretrained_swin_encoder,
    ):
        relative_path = _project_relative_path(file_path)
        destination = code_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / relative_path, destination)

    return code_dir


def roi_size_override(roi_size):
    return f"transforms.roi_size=[{roi_size},{roi_size},{roi_size}]"


def hyperparameter_overrides(args):
    return [
        f"training.lr={args.learning_rate}",
        f"training.loss={args.loss}",
        f"training.weight_decay={args.weight_decay}",
        roi_size_override(args.roi_size),
    ]


def training_command(args):
    train_folds = " ".join(str(fold) for fold in args.train_folds)
    parts = [
        "python scripts/swin_UNETR.py",
        "--input_data ${{inputs.input_data}}",
        "--output_model ${{outputs.output_model}}",
        f"--split_csv {args.split_csv}",
        f"--fold {args.output_fold}",
        f"--config {args.config}",
        "--device cuda",
        "--full_train",
        f"--train_folds {train_folds}",
        f"--output_fold {args.output_fold}",
        f"--pretrained_swin_encoder {args.pretrained_swin_encoder}",
        *DEFAULT_OVERRIDES,
        *hyperparameter_overrides(args),
        *args.extra_overrides,
    ]
    return " ".join(parts)


def build_azure_job(args):
    from azure.ai.ml import Input, Output, command

    return command(
        name=safe_azure_name("swin_unetr_full_train_lr_3e-04"),
        display_name=args.display_name,
        description=(
            "Full SwinUNETR training on all QC split folds 0-4 from "
            "data/cv_splits_qc.csv. Fold -1 remains excluded."
        ),
        code=str(args.code_dir),
        command=training_command(args),
        environment=args.azure_environment,
        environment_variables={
            "PYTHONPATH": ".",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        compute=azure_compute_name(args.azure_compute),
        inputs={
            "input_data": Input(
                type="uri_folder",
                path=args.azure_input_data,
                mode="ro_mount",
            )
        },
        outputs={"output_model": Output(type="uri_folder")},
        distribution={
            "type": "pytorch",
            "process_count_per_instance": args.process_count_per_instance,
        },
        instance_count=args.azure_instance_count,
        shm_size=args.shm_size,
        is_deterministic=False,
        experiment_name=args.azure_experiment_name,
        tags={
            "model": "swin_unetr",
            "split_csv": args.split_csv,
            "train_folds": ",".join(str(fold) for fold in args.train_folds),
            "loss": args.loss,
            "learning_rate": str(args.learning_rate),
            "weight_decay": str(args.weight_decay),
            "roi_size": f"{args.roi_size}x{args.roi_size}x{args.roi_size}",
            "stage": "full_training",
        },
    )


def submit_job(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.code_dir = prepare_code_bundle(args)

    job = build_azure_job(args)
    job_yaml = out_dir / "swin_unetr_full_train_job.yml"
    job.dump(job_yaml)
    print(f"Azure job YAML: {job_yaml}", flush=True)

    if args.dry_run:
        print("Dry run only. Azure job not submitted.", flush=True)
        return job_yaml

    ml_client = get_azure_ml_client(args)
    returned_job = ml_client.jobs.create_or_update(job)
    payload = serialize_azure_job(returned_job)
    submission_json = out_dir / "swin_unetr_full_train_submission.json"
    submission_json.write_text(json.dumps(payload, indent=2))

    print(f"Submitted job: {payload['name']}", flush=True)
    if payload["studio_url"]:
        print(f"Studio URL: {payload['studio_url']}", flush=True)
    return submission_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Submit one Azure ML command job for full SwinUNETR training on "
            "folds 0-4 from data/cv_splits_qc.csv."
        )
    )
    parser.add_argument(
        "--train_folds",
        nargs="+",
        type=int,
        default=list(DEFAULT_TRAIN_FOLDS),
    )
    parser.add_argument("--output_fold", type=int, default=DEFAULT_OUTPUT_FOLD)
    parser.add_argument("--split_csv", default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--pretrained_swin_encoder",
        default=DEFAULT_PRETRAINED_SWIN_ENCODER,
    )
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--loss", default=DEFAULT_LOSS)
    parser.add_argument("--roi_size", type=int, default=DEFAULT_ROI_SIZE)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--display_name",
        default="Myocardium SwinUNETR Full Training Folds 0-4 LR=0.0003",
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
        parser.error(
            "--train_folds must be non-negative; fold -1 is reserved for testing"
        )
    if args.output_fold < 0:
        parser.error("--output_fold must be non-negative")
    if args.roi_size <= 0:
        parser.error("--roi_size must be a positive integer")

    return args


def main(argv=None):
    args = parse_args(argv)
    submit_job(args)


if __name__ == "__main__":
    main(sys.argv[1:])
