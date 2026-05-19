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
        subgoal_max_offset: Optional[int] = None,
        subgoal_sampling_strategy: str = "uniform",
        max_task_files: Optional[int] = None,
        max_demos_per_task: Optional[int] = None,
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
            subgoal_max_offset: Maximum future-frame offset for subgoal sampling. Defaults to action_chunk_size
            subgoal_sampling_strategy: "uniform" samples a random offset, "fixed" uses subgoal_max_offset
            max_task_files: Optional limit on the number of HDF5 task files to load
            max_demos_per_task: Optional limit on demonstrations loaded per task file
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
        self.subgoal_max_offset = int(subgoal_max_offset or action_chunk_size)
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
                demo_names = sorted(f['data'].keys(), key=self._natural_key)
                if self.max_demos_per_task is not None:
                    demo_names = demo_names[: int(self.max_demos_per_task)]
                for demo_name in demo_names:
                    demo = f['data'][demo_name]

                    # Load all actions for this demo
                    all_actions = demo['actions'][:]  # [T, 7]
                    num_frames = len(all_actions)

                    if self.remove_pause_intervals:
                        # Remove pauses
                        filtered_actions = self._remove_pauses(all_actions)
                        num_samples = len(filtered_actions)

                        # Build mapping from filtered index to original index
                        non_pause_indices = self._non_pause_indices(all_actions)
                    else:
                        num_samples = len(all_actions)
                        non_pause_indices = list(range(num_samples))

                    # Create samples for valid starting positions
                    for filtered_t in range(num_samples - self.action_chunk_size + 1):
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
        image = Image.fromarray(rgb.astype(np.uint8))
        return self.image_processor.preprocess(
            image,
            return_tensors='pt',
            do_resize=True,
            size={'height': self.image_size, 'width': self.image_size},
        )['pixel_values'].squeeze(0)

    def _sample_subgoal_timestep(self, sample) -> int:
        if self.subgoal_sampling_strategy == "fixed":
            offset = self.subgoal_max_offset
        else:
            offset = int(np.random.randint(self.subgoal_min_offset, self.subgoal_max_offset + 1))

        final_timestep = max(0, sample['num_frames'] - 1)
        return min(sample['timestep'] + offset, final_timestep)

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
            if self.include_subgoal_image:
                subgoal_timestep = self._sample_subgoal_timestep(sample)
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
            else:
                # Get actions directly
                actions = demo['actions'][t : t + self.action_chunk_size]  # [chunk, 7]

            # Normalize actions to [-1, 1]
            # LIBERO actions are typically already normalized, but we clip to be safe
            actions = np.clip(actions, -1.0, 1.0)
            action_tensor = torch.from_numpy(actions).float()

        item = {
            'observations': obs_tensor,  # [3, 256, 256]
            'instructions': sample['instruction'],  # str
            'action_labels': action_tensor,  # [chunk_size, 7]
        }
        if self.include_subgoal_image:
            item['subgoal_images'] = subgoal_tensor
            item['subgoal_timestep'] = subgoal_timestep
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
