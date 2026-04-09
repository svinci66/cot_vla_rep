"""
VILA-U Action Prediction Training (Memory-efficient version)
Based on VILA-U's train_mem.py framework
"""

from unittest import mock
from vila_u.train.train_action_prediction_main import train
from vila_u.train.accelerate_compat import patch_accelerator_init_for_old_versions
from vila_u.train.transformer_normalize_monkey_patch import patched_normalize


def __len__(self):
    return len(self.batch_sampler)


def __iter__(self):
    return self.batch_sampler.__iter__()


if __name__ == "__main__":
    patch_accelerator_init_for_old_versions()
    with (
        mock.patch('transformers.image_processing_utils.normalize', new=patched_normalize),
        mock.patch('accelerate.data_loader.BatchSamplerShard.__len__', new=__len__),
        mock.patch('accelerate.data_loader.BatchSamplerShard.__iter__', new=__iter__)
    ):
        train()
