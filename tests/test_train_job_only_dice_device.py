import os
import sys
import types
import unittest
from unittest import mock

import torch


def stub_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


dependency_stubs = {
    "mlflow": stub_module(
        "mlflow",
        system_metrics=types.SimpleNamespace(
            set_system_metrics_node_id=lambda *args, **kwargs: None
        ),
        log_metric=lambda *args, **kwargs: None,
        log_params=lambda *args, **kwargs: None,
        set_tag=lambda *args, **kwargs: None,
    ),
    "omegaconf": stub_module(
        "omegaconf",
        OmegaConf=types.SimpleNamespace(),
    ),
    "monai.utils": stub_module("monai.utils", set_determinism=lambda *args, **kwargs: None),
    "monai.losses": stub_module(
        "monai.losses",
        DiceLoss=object,
        DiceCELoss=object,
        DiceFocalLoss=object,
        HausdorffDTLoss=object,
        TverskyLoss=object,
    ),
    "monai.inferers": stub_module(
        "monai.inferers", sliding_window_inference=lambda *args, **kwargs: None
    ),
    "monai.data": stub_module(
        "monai.data",
        PersistentDataset=object,
        DataLoader=object,
        decollate_batch=lambda *args, **kwargs: None,
    ),
    "monai.transforms": stub_module("monai.transforms", AsDiscrete=object),
    "lems_ct.src.models.model": stub_module(
        "lems_ct.src.models.model", get_segresnet=lambda *args, **kwargs: None
    ),
    "lems_ct.src.utils.transforms": stub_module(
        "lems_ct.src.utils.transforms", get_transforms=lambda *args, **kwargs: None
    ),
    "lems_ct.src.utils.data": stub_module(
        "lems_ct.src.utils.data",
        get_files_from_csv=lambda *args, **kwargs: ([], []),
        get_training_files_from_csv=lambda *args, **kwargs: [],
        resolve_data_root=lambda path: path,
        resolve_local_path=lambda path: path,
    ),
    "lems_ct.src.utils.misc": stub_module(
        "lems_ct.src.utils.misc",
        update_ema_variables=lambda *args, **kwargs: None,
        exp_lr_scheduler_with_warmup=lambda *args, **kwargs: None,
        concat_all_gather=lambda tensor: tensor,
        train_collate_fn=lambda data: data,
    ),
    "lems_ct.src.metrics.utils": stub_module(
        "lems_ct.src.metrics.utils", calculate_dice_split=lambda *args, **kwargs: None
    ),
}

with mock.patch.dict(sys.modules, dependency_stubs):
    from scripts import train_job_only_dice as train_job


class SelectDeviceTests(unittest.TestCase):
    def test_cuda_request_without_index_uses_local_rank_zero(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            torch.cuda, "is_available", return_value=True
        ), mock.patch.object(torch.cuda, "set_device") as set_device:
            device = train_job.select_device("cuda", distributed=False)

        self.assertEqual(device, torch.device("cuda:0"))
        set_device.assert_called_once_with(0)

    def test_cuda_request_without_index_uses_local_rank(self):
        with mock.patch.dict(os.environ, {"LOCAL_RANK": "2"}, clear=True), mock.patch.object(
            torch.cuda, "is_available", return_value=True
        ), mock.patch.object(torch.cuda, "set_device") as set_device:
            device = train_job.select_device("cuda", distributed=True)

        self.assertEqual(device, torch.device("cuda:2"))
        set_device.assert_called_once_with(2)


class FakeDevice:
    def __init__(self, device_type):
        self.type = device_type


class FakeTensor:
    def __init__(self, device_type):
        self.shape = (1, 1, 4, 4, 4)
        self.device = FakeDevice(device_type)

    def float(self):
        return self

    def cpu(self):
        return FakeTensor("cpu")


class FakeLossValue:
    def __init__(self):
        self.to_device = None

    def __rmul__(self, _other):
        return self

    def to(self, device):
        self.to_device = device
        return self


class HausdorffLossDeviceTests(unittest.TestCase):
    def test_hausdorff_distance_transform_runs_on_cpu_for_cuda_inputs(self):
        observed_devices = []

        class RecordingHausdorffLoss:
            def __init__(self, **_kwargs):
                pass

            def __call__(self, input_tensor, target_tensor):
                observed_devices.append((input_tensor.device.type, target_tensor.device.type))
                return FakeLossValue()

        with mock.patch.object(train_job, "HausdorffDTLoss", RecordingHausdorffLoss):
            loss = train_job.DownsampledHausdorffDTLoss(downsample_factor=1)
            loss(FakeTensor("cuda"), FakeTensor("cuda"))

        self.assertEqual(observed_devices, [("cpu", "cpu")])


if __name__ == "__main__":
    unittest.main()
