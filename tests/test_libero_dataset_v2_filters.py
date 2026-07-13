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


def write_demo_file(path, demo_names, num_frames=10):
    with h5py.File(path, "w") as h5_file:
        data_group = h5_file.create_group("data")
        data_group.attrs["problem_info"] = json.dumps(
            {"language_instruction": "debug task"}
        )
        for demo_name in demo_names:
            demo = data_group.create_group(demo_name)
            actions = np.zeros((num_frames, 7), dtype=np.float32)
            actions[:, 0] = 0.02
            actions[:, 6] = 1.0
            demo.create_dataset("actions", data=actions)
            obs_group = demo.create_group("obs")
            obs_group.create_dataset(
                "agentview_rgb",
                data=np.zeros((num_frames, 8, 8, 3), dtype=np.uint8),
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


def test_train_and_eval_demo_ranges_are_disjoint():
    from vila_u.data.libero_dataset_v2 import LiberoGoalDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        demo_names = [f"demo_{index}" for index in range(50)]
        write_demo_file(
            os.path.join(tmpdir, "debug_task_demo.hdf5"),
            demo_names,
        )
        common_args = dict(
            data_root=tmpdir,
            image_processor=DummyImageProcessor(),
            tokenizer=None,
            action_chunk_size=10,
            remove_pause_intervals=False,
        )
        train_dataset = LiberoGoalDataset(
            **common_args,
            demo_start_index=0,
            demo_end_index=40,
        )
        eval_dataset = LiberoGoalDataset(
            **common_args,
            demo_start_index=40,
            demo_end_index=50,
        )

        train_demos = {sample["demo"] for sample in train_dataset.samples}
        eval_demos = {sample["demo"] for sample in eval_dataset.samples}
        assert train_demos == {f"demo_{index}" for index in range(40)}
        assert eval_demos == {f"demo_{index}" for index in range(40, 50)}
        assert train_demos.isdisjoint(eval_demos)


def test_fixed_t10_is_real_and_action_chunks_are_complete():
    from vila_u.data.libero_dataset_v2 import LiberoGoalDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        write_demo_file(
            os.path.join(tmpdir, "debug_task_demo.hdf5"),
            ["demo_40"],
            num_frames=15,
        )
        dataset = LiberoGoalDataset(
            data_root=tmpdir,
            image_processor=DummyImageProcessor(),
            tokenizer=None,
            action_chunk_size=10,
            remove_pause_intervals=True,
            include_subgoal_image=True,
            subgoal_min_offset=10,
            subgoal_max_offset=10,
            subgoal_sampling_strategy="fixed",
            demo_start_index=40,
            demo_end_index=50,
        )

        assert len(dataset) == 5
        for index, sample in enumerate(dataset.samples):
            assert sample["filtered_timestep"] == index
            item = dataset[index]
            assert item["subgoal_filtered_timestep"] == index + 10
            assert item["subgoal_timestep"] == index + 10
            assert tuple(item["action_labels"].shape) == (10, 7)

        invalid_sample = dict(dataset.samples[-1])
        invalid_sample["filtered_timestep"] = 5
        invalid_sample["timestep"] = 5
        try:
            dataset._sample_subgoal_timestep(invalid_sample)
        except IndexError:
            pass
        else:
            raise AssertionError("t+10 must raise instead of clamping past the trajectory")


def test_demo_names_are_strictly_parsed():
    from vila_u.data.libero_dataset_v2 import LiberoGoalDataset

    assert LiberoGoalDataset.demo_index("demo_49") == 49
    for invalid_name in ("demo49", "demo_49_extra", "other_49"):
        try:
            LiberoGoalDataset.demo_index(invalid_name)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid demo name to fail: {invalid_name}")


def test_raw_libero_gripper_is_converted_to_model_space():
    from vila_u.data.libero_dataset_v2 import LiberoGoalDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "debug_task_demo.hdf5")
        with h5py.File(path, "w") as h5_file:
            data_group = h5_file.create_group("data")
            data_group.attrs["problem_info"] = json.dumps(
                {"language_instruction": "debug task"}
            )
            demo = data_group.create_group("demo_0")
            actions = np.zeros((10, 7), dtype=np.float32)
            actions[:5, 6] = -1.0  # LIBERO raw open
            actions[5:, 6] = 1.0   # LIBERO raw close
            demo.create_dataset("actions", data=actions)
            obs_group = demo.create_group("obs")
            obs_group.create_dataset(
                "agentview_rgb",
                data=np.zeros((10, 8, 8, 3), dtype=np.uint8),
            )

        dataset = LiberoGoalDataset(
            data_root=tmpdir,
            image_processor=DummyImageProcessor(),
            tokenizer=None,
            action_chunk_size=10,
            remove_pause_intervals=False,
            max_task_files=1,
        )

        labels = dataset[0]["action_labels"].numpy()
        assert np.all(labels[:5, 6] == 1.0)
        assert np.all(labels[5:, 6] == -1.0)


if __name__ == "__main__":
    test_gripper_only_actions_are_not_pause()
    test_natural_demo_order_and_inclusive_chunks()
    test_explicit_task_file_selection()
    test_train_and_eval_demo_ranges_are_disjoint()
    test_fixed_t10_is_real_and_action_chunks_are_complete()
    test_demo_names_are_strictly_parsed()
    test_raw_libero_gripper_is_converted_to_model_space()
    print("✓ LIBERO dataset v2 filters verified")
