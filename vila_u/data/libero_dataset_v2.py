"""
LIBERO Goal Dataset V2 - Enhanced version with pause removal and proper preprocessing

Features:
1. Remove pause intervals from trajectories
2. Standardize image resolution to 256×256 pixels
3. Use VILA-U's image_processor for preprocessing
4. Proper action normalization
"""

import os
import json
import re
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional
from vila_u.utils.libero_image import rotate_libero_image_180
from vila_u.utils.libero_action import libero_raw_actions_to_model_actions


class LiberoGoalDataset(Dataset):
    """
    LIBERO Goal Dataset with enhanced preprocessing

    Features:
    - Removes pause intervals (low-motion segments)
    - Standardizes images to 256x256
    - Uses VILA-U's image_processor
    """

    def __init__(
        self,
        data_root: str,
        image_processor,
        tokenizer,
        action_chunk_size: int = 10,
        image_size: int = 256,
        remove_pause_intervals: bool = True,
        pause_threshold: float = 0.01,
        include_subgoal_image: bool = False,
        subgoal_min_offset: int = 1,
        subgoal_max_offset: Optional[int] = 10,
        subgoal_sampling_strategy: str = "fixed",
        max_task_files: Optional[int] = None,
        max_demos_per_task: Optional[int] = None,
        demo_start_index: int = 0,
        demo_end_index: Optional[int] = None,
        task_file: Optional[str] = None,
        task_file_pattern: Optional[str] = None,
        gripper_pause_threshold: float = 1e-6,
    ):
        """
        Args:
            data_root: LIBERO dataset root directory
            image_processor: VILA-U image processor
            tokenizer: VILA-U tokenizer
            action_chunk_size: Number of action steps to predict
            image_size: Target image size (will be 256x256)
            remove_pause_intervals: Whether to remove pause intervals
            pause_threshold: Threshold for detecting pause (L2 norm of action)
            include_subgoal_image: Whether to return a future frame as subgoal image
            subgoal_min_offset: Minimum future-frame offset for subgoal sampling
            subgoal_max_offset: Maximum future-frame offset for subgoal sampling. Defaults to 10
            subgoal_sampling_strategy: "fixed" uses subgoal_max_offset, "uniform" samples a random offset
            max_task_files: Optional limit on the number of HDF5 task files to load
            max_demos_per_task: Optional limit on demonstrations loaded per task file
            demo_start_index: Inclusive numeric demo index to load
            demo_end_index: Exclusive numeric demo index to load, or all remaining demos
            task_file: Optional exact HDF5 task filename to load
            task_file_pattern: Optional substring/regex used to select task files
            gripper_pause_threshold: Threshold for gripper action change when detecting pauses
        """
        self.data_root = data_root
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.action_chunk_size = action_chunk_size
        self.image_size = image_size
        self.remove_pause_intervals = remove_pause_intervals
        self.pause_threshold = pause_threshold
        self.include_subgoal_image = include_subgoal_image
        self.subgoal_min_offset = max(1, int(subgoal_min_offset))
        self.subgoal_max_offset = int(subgoal_max_offset if subgoal_max_offset is not None else 10)
        if self.subgoal_max_offset < self.subgoal_min_offset:
            raise ValueError(
                "subgoal_max_offset must be greater than or equal to subgoal_min_offset"
            )
        if subgoal_sampling_strategy not in {"uniform", "fixed"}:
            raise ValueError(
                "subgoal_sampling_strategy must be either 'uniform' or 'fixed'"
            )
        self.subgoal_sampling_strategy = subgoal_sampling_strategy
        self.max_task_files = max_task_files
        self.max_demos_per_task = max_demos_per_task
        self.demo_start_index = int(demo_start_index)
        self.demo_end_index = None if demo_end_index is None else int(demo_end_index)
        if self.demo_start_index < 0:
            raise ValueError("demo_start_index must be non-negative")
        if self.demo_end_index is not None and self.demo_end_index <= self.demo_start_index:
            raise ValueError("demo_end_index must be greater than demo_start_index")
        self.task_file = task_file
        self.task_file_pattern = task_file_pattern
        self.gripper_pause_threshold = gripper_pause_threshold

        # Build dataset index
        self.samples = self._build_index()

        print(f"[LiberoGoalDataset] Loaded {len(self.samples)} samples from {data_root}")
        if remove_pause_intervals:
            print(f"  - Pause removal enabled (threshold={pause_threshold})")
        if include_subgoal_image:
            print(
                "  - Future-frame subgoals enabled "
                f"(offset={self.subgoal_min_offset}-{self.subgoal_max_offset}, "
                f"strategy={self.subgoal_sampling_strategy})"
            )
        if max_task_files is not None:
            print(f"  - Limited to first {max_task_files} task file(s)")
        if max_demos_per_task is not None:
            print(f"  - Limited to first {max_demos_per_task} demo(s) per task")
        demo_end_label = "all" if self.demo_end_index is None else str(self.demo_end_index)
        print(f"  - Demo Range: [{self.demo_start_index}, {demo_end_label})")
        task_demo_counts = {
            os.path.basename(task_file): set(demo_names)
            for task_file, demo_names in self.loaded_demo_names_by_task.items()
        }
        task_sample_counts = {
            os.path.basename(task_file): 0
            for task_file in self.loaded_demo_names_by_task
        }
        for sample in self.samples:
            task_name = os.path.basename(sample["file"])
            task_sample_counts[task_name] = task_sample_counts.get(task_name, 0) + 1
        demo_counts = [len(demos) for demos in task_demo_counts.values()]
        if demo_counts and len(set(demo_counts)) == 1:
            demos_per_task = str(demo_counts[0])
        elif demo_counts:
            demos_per_task = f"{min(demo_counts)}-{max(demo_counts)}"
        else:
            demos_per_task = "0"
        print(f"  - Tasks: {len(task_sample_counts)}")
        print(f"  - Demos per task: {demos_per_task}")
        print(f"  - Valid samples per task: {task_sample_counts}")
        print(f"  - Total valid samples: {len(self.samples)}")
        if task_file is not None:
            print(f"  - Task file: {task_file}")
        if task_file_pattern is not None:
            print(f"  - Task file pattern: {task_file_pattern}")

    @staticmethod
    def _natural_key(value: str):
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", value)
        ]

    @staticmethod
    def demo_index(name: str) -> int:
        match = re.fullmatch(r"demo_(\d+)", name)
        if match is None:
            raise ValueError(f"Invalid demo name: {name}")
        return int(match.group(1))

    def _is_pause(self, action: np.ndarray, previous_action: Optional[np.ndarray] = None) -> bool:
        """
        Check whether an action is a no-op.

        A no-op has near-zero translation/rotation and does not change the
        gripper command/state relative to the previous action.

        Args:
            action: [7] action vector
            previous_action: Previous [7] action vector for gripper-change checks
        Returns:
            True if action is a no-op action
        """
        # Calculate L2 norm of position and rotation changes.
        # action[:3] = position delta, action[3:6] = rotation delta,
        # action[6] = gripper command/state.
        position_norm = np.linalg.norm(action[:3])
        rotation_norm = np.linalg.norm(action[3:6])
        if action.shape[0] > 6 and previous_action is not None and previous_action.shape[0] > 6:
            gripper_delta = abs(float(action[6]) - float(previous_action[6]))
        else:
            gripper_delta = abs(float(action[6])) if action.shape[0] > 6 else 0.0

        # Only filter no-ops: low arm motion and unchanged gripper state.
        return (position_norm < self.pause_threshold and
                rotation_norm < self.pause_threshold and
                gripper_delta < self.gripper_pause_threshold)

    def _non_pause_indices(self, actions: np.ndarray) -> list[int]:
        indices = []
        previous_action = None
        for timestep, action in enumerate(actions):
            if not self._is_pause(action, previous_action):
                indices.append(timestep)
            previous_action = action
        return indices

    def _remove_pauses(self, actions: np.ndarray) -> np.ndarray:
        """
        Remove pause intervals from action sequence

        Args:
            actions: [T, 7] action sequence
        Returns:
            filtered_actions: [T', 7] action sequence without pauses
        """
        return actions[self._non_pause_indices(actions)]

    def _build_index(self):
        """Build dataset index with pause removal"""
        samples = []
        self.loaded_demo_names_by_task = {}

        # Traverse all .hdf5 files
        filenames = [
            filename
            for filename in sorted(os.listdir(self.data_root), key=self._natural_key)
            if filename.endswith('.hdf5')
        ]
        if self.task_file is not None:
            requested = os.path.basename(self.task_file)
            filenames = [filename for filename in filenames if filename == requested]
            if not filenames:
                raise FileNotFoundError(
                    f"Task file {requested!r} not found under {self.data_root}"
                )
        if self.task_file_pattern is not None:
            pattern = re.compile(self.task_file_pattern)
            filenames = [
                filename
                for filename in filenames
                if self.task_file_pattern in filename or pattern.search(filename)
            ]
            if not filenames:
                raise FileNotFoundError(
                    f"No HDF5 task files under {self.data_root} matched "
                    f"pattern {self.task_file_pattern!r}"
                )
        if self.max_task_files is not None:
            filenames = filenames[: int(self.max_task_files)]

        for filename in filenames:

            filepath = os.path.join(self.data_root, filename)

            with h5py.File(filepath, 'r') as f:
                # Get language instruction
                problem_info = json.loads(f['data'].attrs['problem_info'])
                instruction = problem_info['language_instruction']

                # Traverse all demonstrations
                demo_names = sorted(f['data'].keys(), key=self.demo_index)
                demo_names = [
                    name
                    for name in demo_names
                    if self.demo_start_index <= self.demo_index(name)
                    and (
                        self.demo_end_index is None
                        or self.demo_index(name) < self.demo_end_index
                    )
                ]
                if self.max_demos_per_task is not None:
                    demo_names = demo_names[: int(self.max_demos_per_task)]
                self.loaded_demo_names_by_task[filepath] = tuple(demo_names)
                for demo_name in demo_names:
                    demo = f['data'][demo_name]

                    # Load all actions for this demo
                    all_actions = demo['actions'][:]  # [T, 7]
                    num_frames = len(all_actions)

                    if self.remove_pause_intervals:
                        # Remove pauses
                        non_pause_indices = self._non_pause_indices(all_actions)
                        num_samples = len(non_pause_indices)
                    else:
                        num_samples = len(all_actions)
                        non_pause_indices = list(range(num_samples))

                    # Require a complete action chunk and, when requested, a real
                    # future subgoal rather than clamping to the final frame.
                    required_future_delta = self.action_chunk_size - 1
                    if self.include_subgoal_image:
                        required_future_delta = max(
                            required_future_delta,
                            self.subgoal_max_offset,
                        )
                    num_valid_starts = max(0, num_samples - required_future_delta)
                    for filtered_t in range(num_valid_starts):
                        # Get original timestep indices
                        original_t = non_pause_indices[filtered_t]

                        samples.append({
                            'file': filepath,
                            'demo': demo_name,
                            'timestep': original_t,
                            'filtered_timestep': filtered_t,
                            'instruction': instruction,
                            'non_pause_indices': non_pause_indices,
                            'num_frames': num_frames,
                        })

        return samples

    def __len__(self):
        return len(self.samples)

    def _preprocess_rgb(self, rgb: np.ndarray) -> torch.Tensor:
        rgb = rotate_libero_image_180(rgb)
        image = Image.fromarray(rgb.astype(np.uint8))
        return self.image_processor.preprocess(
            image,
            return_tensors='pt',
            do_resize=True,
            size={'height': self.image_size, 'width': self.image_size},
        )['pixel_values'].squeeze(0)

    def _sample_subgoal_indices(self, sample) -> tuple[int, int]:
        if self.subgoal_sampling_strategy == "fixed":
            offset = self.subgoal_max_offset
        else:
            offset = int(np.random.randint(self.subgoal_min_offset, self.subgoal_max_offset + 1))

        if self.remove_pause_intervals:
            non_pause_indices = sample['non_pause_indices']
            filtered_t = sample['filtered_timestep']
            target_filtered_t = filtered_t + offset
            if target_filtered_t >= len(non_pause_indices):
                raise IndexError(
                    "Subgoal filtered timestep is outside the trajectory: "
                    f"start={filtered_t}, offset={offset}, "
                    f"filtered_length={len(non_pause_indices)}"
                )
            return int(non_pause_indices[target_filtered_t]), int(target_filtered_t)

        target_timestep = sample['timestep'] + offset
        if target_timestep >= sample['num_frames']:
            raise IndexError(
                "Subgoal timestep is outside the trajectory: "
                f"start={sample['timestep']}, offset={offset}, "
                f"num_frames={sample['num_frames']}"
            )
        return int(target_timestep), int(sample['filtered_timestep'] + offset)

    def _sample_subgoal_timestep(self, sample) -> int:
        subgoal_timestep, _ = self._sample_subgoal_indices(sample)
        return subgoal_timestep

    def __getitem__(self, idx):
        sample = self.samples[idx]

        with h5py.File(sample['file'], 'r') as f:
            demo = f['data'][sample['demo']]

            # Load observation image at original timestep
            t = sample['timestep']
            obs_rgb = demo['obs/agentview_rgb'][t]  # [H, W, 3]
            obs_tensor = self._preprocess_rgb(obs_rgb)  # [3, 256, 256]

            subgoal_tensor = None
            subgoal_timestep = None
            subgoal_filtered_timestep = None
            if self.include_subgoal_image:
                subgoal_timestep, subgoal_filtered_timestep = self._sample_subgoal_indices(sample)
                subgoal_rgb = demo['obs/agentview_rgb'][subgoal_timestep]
                subgoal_tensor = self._preprocess_rgb(subgoal_rgb)  # [3, 256, 256]

            # Load action sequence
            if self.remove_pause_intervals:
                # Get filtered action indices
                non_pause_indices = sample['non_pause_indices']
                filtered_t = sample['filtered_timestep']

                # Get next action_chunk_size actions (from filtered sequence)
                action_indices = non_pause_indices[
                    filtered_t : filtered_t + self.action_chunk_size
                ]
                actions = demo['actions'][action_indices]  # [chunk, 7]
                if filtered_t > 0:
                    previous_action = demo['actions'][non_pause_indices[filtered_t - 1]]
                else:
                    previous_action = actions[0]
            else:
                # Get actions directly
                actions = demo['actions'][t : t + self.action_chunk_size]  # [chunk, 7]
                previous_action = demo['actions'][t - 1] if t > 0 else actions[0]

            if len(actions) != self.action_chunk_size:
                raise IndexError(
                    "Incomplete action chunk escaped dataset indexing: "
                    f"expected={self.action_chunk_size}, got={len(actions)}, "
                    f"file={sample['file']}, demo={sample['demo']}, "
                    f"filtered_timestep={sample['filtered_timestep']}"
                )

            # Convert LIBERO raw gripper convention (-1=open, +1=close) to
            # OpenVLA-style model action convention (+1=open, -1=close).
            actions = libero_raw_actions_to_model_actions(actions)
            previous_action = libero_raw_actions_to_model_actions(previous_action)
            action_tensor = torch.from_numpy(actions).float()
            previous_action_tensor = torch.from_numpy(previous_action).float()

        item = {
            'observations': obs_tensor,  # [3, 256, 256]
            'instructions': sample['instruction'],  # str
            'action_labels': action_tensor,  # [chunk_size, 7]
            'previous_action_label': previous_action_tensor,  # [7]
            'file': sample['file'],
            'demo': sample['demo'],
            'timestep': int(sample['timestep']),
            'filtered_timestep': int(sample['filtered_timestep']),
        }
        if self.include_subgoal_image:
            item['subgoal_images'] = subgoal_tensor
            item['subgoal_timestep'] = subgoal_timestep
            item['subgoal_filtered_timestep'] = subgoal_filtered_timestep
        return item


def collate_fn(batch):
    """
    Custom collate function for action prediction

    Args:
        batch: List of samples from __getitem__
    Returns:
        Batched data dictionary
    """
    observations = torch.stack([item['observations'] for item in batch])
    instructions = [item['instructions'] for item in batch]
    action_labels = torch.stack([item['action_labels'] for item in batch])

    output = {
        'observations': observations,  # [B, 3, 256, 256]
        'instructions': instructions,  # List[str] of length B
        'action_labels': action_labels,  # [B, chunk_size, 7]
    }
    if 'subgoal_images' in batch[0]:
        output['subgoal_images'] = torch.stack([item['subgoal_images'] for item in batch])
        output['subgoal_timesteps'] = torch.tensor(
            [item['subgoal_timestep'] for item in batch],
            dtype=torch.long,
        )
        if 'subgoal_filtered_timestep' in batch[0]:
            output['subgoal_filtered_timesteps'] = torch.tensor(
                [item['subgoal_filtered_timestep'] for item in batch],
                dtype=torch.long,
            )
    return output


def compute_dataset_statistics(
    data_root: str,
    pause_threshold: float = 0.01,
    gripper_pause_threshold: float = 1e-6,
):
    """
    Compute statistics about the dataset

    Args:
        data_root: LIBERO dataset root directory
    Returns:
        Dictionary with statistics
    """
    total_samples = 0
    total_pauses = 0
    action_norms = []

    for filename in sorted(os.listdir(data_root), key=LiberoGoalDataset._natural_key):
        if not filename.endswith('.hdf5'):
            continue

        filepath = os.path.join(data_root, filename)

        with h5py.File(filepath, 'r') as f:
            for demo_name in sorted(f['data'].keys(), key=LiberoGoalDataset._natural_key):
                demo = f['data'][demo_name]
                actions = demo['actions'][:]  # [T, 7]

                total_samples += len(actions)

                # Count pauses
                previous_action = None
                for action in actions:
                    norm = np.linalg.norm(action[:6])  # position + rotation
                    action_norms.append(norm)
                    if action.shape[0] > 6 and previous_action is not None and previous_action.shape[0] > 6:
                        gripper_delta = abs(float(action[6]) - float(previous_action[6]))
                    else:
                        gripper_delta = abs(float(action[6])) if action.shape[0] > 6 else 0.0
                    if norm < pause_threshold and gripper_delta < gripper_pause_threshold:
                        total_pauses += 1
                    previous_action = action

    stats = {
        'total_samples': total_samples,
        'total_pauses': total_pauses,
        'pause_ratio': total_pauses / total_samples if total_samples > 0 else 0,
        'action_norm_mean': np.mean(action_norms),
        'action_norm_std': np.std(action_norms),
        'action_norm_min': np.min(action_norms),
        'action_norm_max': np.max(action_norms),
    }

    return stats
