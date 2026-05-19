import json
import os
import sys
import tempfile

import h5py
import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)


class DummyImageProcessor:
    def preprocess(self, image, return_tensors=None, do_resize=True, size=None):
        import torch

        height = size["height"]
        width = size["width"]
        return {"pixel_values": torch.zeros(1, 3, height, width)}


def write_demo_file(path, demo_names):
    with h5py.File(path, "w") as h5_file:
        data_group = h5_file.create_group("data")
        data_group.attrs["problem_info"] = json.dumps(
            {"language_instruction": "debug task"}
        )
        for demo_name in demo_names:
            demo = data_group.create_group(demo_name)
            actions = np.zeros((10, 7), dtype=np.float32)
            actions[:, 6] = 1.0
            demo.create_dataset("actions", data=actions)
            obs_group = demo.create_group("obs")
            obs_group.create_dataset(
                "agentview_rgb",
                data=np.zeros((10, 8, 8, 3), dtype=np.uint8),
            )


def test_gripper_only_actions_are_not_pause():
    from vila_u.data.libero_dataset_v2 import LiberoGoalDataset

    dataset = object.__new__(LiberoGoalDataset)
    dataset.pause_threshold = 0.01
    dataset.gripper_pause_threshold = 1e-6

    gripper_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    no_op_action = np.zeros(7)

    assert not dataset._is_pause(gripper_action, no_op_action)
    assert dataset._is_pause(gripper_action, gripper_action)
    assert dataset._is_pause(no_op_action, no_op_action)


def test_natural_demo_order_and_inclusive_chunks():
    from vila_u.data.libero_dataset_v2 import LiberoGoalDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        write_demo_file(
            os.path.join(tmpdir, "debug_task_demo.hdf5"),
            ["demo_10", "demo_2", "demo_0", "demo_1"],
        )

        dataset = LiberoGoalDataset(
            data_root=tmpdir,
            image_processor=DummyImageProcessor(),
            tokenizer=None,
            action_chunk_size=10,
            remove_pause_intervals=False,
            max_task_files=1,
            max_demos_per_task=3,
        )

        assert [sample["demo"] for sample in dataset.samples] == [
            "demo_0",
            "demo_1",
            "demo_2",
        ]
        assert len(dataset) == 3


def test_explicit_task_file_selection():
    from vila_u.data.libero_dataset_v2 import LiberoGoalDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        write_demo_file(os.path.join(tmpdir, "task_10_demo.hdf5"), ["demo_0"])
        write_demo_file(os.path.join(tmpdir, "task_2_demo.hdf5"), ["demo_0"])

        dataset = LiberoGoalDataset(
            data_root=tmpdir,
            image_processor=DummyImageProcessor(),
            tokenizer=None,
            action_chunk_size=10,
            remove_pause_intervals=False,
            task_file="task_10_demo.hdf5",
        )

        assert len(dataset) == 1
        assert dataset.samples[0]["file"].endswith("task_10_demo.hdf5")


if __name__ == "__main__":
    test_gripper_only_actions_are_not_pause()
    test_natural_demo_order_and_inclusive_chunks()
    test_explicit_task_file_selection()
    print("✓ LIBERO dataset v2 filters verified")
