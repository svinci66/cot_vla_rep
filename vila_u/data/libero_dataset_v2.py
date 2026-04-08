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
        """
        self.data_root = data_root
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.action_chunk_size = action_chunk_size
        self.image_size = image_size
        self.remove_pause_intervals = remove_pause_intervals
        self.pause_threshold = pause_threshold

        # Build dataset index
        self.samples = self._build_index()

        print(f"[LiberoGoalDataset] Loaded {len(self.samples)} samples from {data_root}")
        if remove_pause_intervals:
            print(f"  - Pause removal enabled (threshold={pause_threshold})")

    def _is_pause(self, action: np.ndarray) -> bool:
        """
        Check if an action represents a pause

        Args:
            action: [7] action vector
        Returns:
            True if action is a pause (low motion)
        """
        # Calculate L2 norm of position and rotation changes
        # action[:3] = position delta, action[3:6] = rotation delta
        position_norm = np.linalg.norm(action[:3])
        rotation_norm = np.linalg.norm(action[3:6])

        # Consider it a pause if both position and rotation changes are small
        return (position_norm < self.pause_threshold and
                rotation_norm < self.pause_threshold)

    def _remove_pauses(self, actions: np.ndarray) -> np.ndarray:
        """
        Remove pause intervals from action sequence

        Args:
            actions: [T, 7] action sequence
        Returns:
            filtered_actions: [T', 7] action sequence without pauses
        """
        # Find non-pause indices
        non_pause_mask = np.array([
            not self._is_pause(action) for action in actions
        ])

        # Filter out pauses
        filtered_actions = actions[non_pause_mask]

        return filtered_actions

    def _build_index(self):
        """Build dataset index with pause removal"""
        samples = []

        # Traverse all .hdf5 files
        for filename in sorted(os.listdir(self.data_root)):
            if not filename.endswith('.hdf5'):
                continue

            filepath = os.path.join(self.data_root, filename)

            with h5py.File(filepath, 'r') as f:
                # Get language instruction
                problem_info = json.loads(f['data'].attrs['problem_info'])
                instruction = problem_info['language_instruction']

                # Traverse all demonstrations
                for demo_name in f['data'].keys():
                    demo = f['data'][demo_name]

                    # Load all actions for this demo
                    all_actions = demo['actions'][:]  # [T, 7]

                    if self.remove_pause_intervals:
                        # Remove pauses
                        filtered_actions = self._remove_pauses(all_actions)
                        num_samples = len(filtered_actions)

                        # Build mapping from filtered index to original index
                        non_pause_indices = []
                        for t in range(len(all_actions)):
                            if not self._is_pause(all_actions[t]):
                                non_pause_indices.append(t)
                    else:
                        num_samples = len(all_actions)
                        non_pause_indices = list(range(num_samples))

                    # Create samples for valid starting positions
                    for filtered_t in range(num_samples - self.action_chunk_size):
                        # Get original timestep indices
                        original_t = non_pause_indices[filtered_t]

                        samples.append({
                            'file': filepath,
                            'demo': demo_name,
                            'timestep': original_t,
                            'filtered_timestep': filtered_t,
                            'instruction': instruction,
                            'non_pause_indices': non_pause_indices,
                        })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        with h5py.File(sample['file'], 'r') as f:
            demo = f['data'][sample['demo']]

            # Load observation image at original timestep
            t = sample['timestep']
            obs_rgb = demo['obs/agentview_rgb'][t]  # [H, W, 3]

            # Convert to PIL Image
            obs_image = Image.fromarray(obs_rgb.astype(np.uint8))

            # Resize to 256x256 and preprocess using VILA-U's image_processor
            # The image_processor will handle normalization
            obs_tensor = self.image_processor.preprocess(
                obs_image,
                return_tensors='pt',
                do_resize=True,
                size={'height': self.image_size, 'width': self.image_size},
            )['pixel_values'].squeeze(0)  # [3, 256, 256]

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

        return {
            'observations': obs_tensor,  # [3, 256, 256]
            'instructions': sample['instruction'],  # str
            'action_labels': action_tensor,  # [chunk_size, 7]
        }


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

    return {
        'observations': observations,  # [B, 3, 256, 256]
        'instructions': instructions,  # List[str] of length B
        'action_labels': action_labels,  # [B, chunk_size, 7]
    }


def compute_dataset_statistics(data_root: str):
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

    for filename in sorted(os.listdir(data_root)):
        if not filename.endswith('.hdf5'):
            continue

        filepath = os.path.join(data_root, filename)

        with h5py.File(filepath, 'r') as f:
            for demo_name in f['data'].keys():
                demo = f['data'][demo_name]
                actions = demo['actions'][:]  # [T, 7]

                total_samples += len(actions)

                # Count pauses
                for action in actions:
                    norm = np.linalg.norm(action[:6])  # position + rotation
                    action_norms.append(norm)
                    if norm < 0.01:
                        total_pauses += 1

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
