"""
LIBERO Goal Dataset Reader

This module provides functionality to read and process LIBERO Goal dataset files.
LIBERO Goal datasets are stored in HDF5 format with the following structure:
- data/demo_X/: Contains demonstration episodes
  - obs/: Observations (images, robot states, etc.)
  - actions: Action sequences
  - rewards: Reward values
  - dones: Episode termination flags
- mask/: Optional filter keys for data subsets
- data attributes: problem_info (language instruction), env_args (environment metadata)

Usage:
    reader = LiberoGoalReader(dataset_path)
    data = reader.load_dataset()
    reader.save_dataset(data, output_path)
"""

import h5py
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple, Any


class LiberoGoalReader:
    """Reader for LIBERO Goal dataset files in HDF5 format."""

    def __init__(self, dataset_path: str):
        """
        Initialize the reader with a dataset path.

        Args:
            dataset_path: Path to the HDF5 dataset file
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        self.dataset_path = dataset_path
        self._file = None

    def __enter__(self):
        """Context manager entry."""
        self._file = h5py.File(self.dataset_path, "r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._file is not None:
            self._file.close()

    def get_dataset_info(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset.

        Args:
            verbose: Whether to print detailed information

        Returns:
            Dictionary containing dataset statistics and metadata
        """
        with h5py.File(self.dataset_path, "r") as f:
            # Get demonstration list
            demos = sorted(list(f["data"].keys()))

            # Extract trajectory lengths and action statistics
            traj_lengths = []
            action_min = np.inf
            action_max = -np.inf

            for ep in demos:
                traj_lengths.append(f["data/{}/actions".format(ep)].shape[0])
                action_min = min(action_min, np.min(f["data/{}/actions".format(ep)][()]))
                action_max = max(action_max, np.max(f["data/{}/actions".format(ep)][()]))

            traj_lengths = np.array(traj_lengths)

            # Get metadata
            problem_info = json.loads(f["data"].attrs["problem_info"])
            language_instruction = "".join(problem_info["language_instruction"]).strip('"')
            env_meta = json.loads(f["data"].attrs["env_args"])

            # Get filter keys if available
            all_filter_keys = {}
            if "mask" in f:
                for fk in f["mask"]:
                    fk_demos = sorted(
                        [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(fk)])]
                    )
                    all_filter_keys[fk] = fk_demos

            info = {
                "total_transitions": int(np.sum(traj_lengths)),
                "total_trajectories": len(traj_lengths),
                "traj_length_mean": float(np.mean(traj_lengths)),
                "traj_length_std": float(np.std(traj_lengths)),
                "traj_length_min": int(np.min(traj_lengths)),
                "traj_length_max": int(np.max(traj_lengths)),
                "action_min": float(action_min),
                "action_max": float(action_max),
                "language_instruction": language_instruction,
                "filter_keys": all_filter_keys,
                "env_meta": env_meta,
                "demos": demos
            }

            if verbose:
                print("\n" + "="*60)
                print(f"Dataset: {os.path.basename(self.dataset_path)}")
                print("="*60)
                print(f"Total transitions: {info['total_transitions']}")
                print(f"Total trajectories: {info['total_trajectories']}")
                print(f"Trajectory length - mean: {info['traj_length_mean']:.2f}, "
                      f"std: {info['traj_length_std']:.2f}")
                print(f"Trajectory length - min: {info['traj_length_min']}, "
                      f"max: {info['traj_length_max']}")
                print(f"Action range: [{info['action_min']:.4f}, {info['action_max']:.4f}]")
                print(f"Language instruction: {info['language_instruction']}")
                print(f"\nFilter keys: {list(info['filter_keys'].keys()) if info['filter_keys'] else 'None'}")
                print("="*60 + "\n")

            return info

    def load_dataset(self, filter_key: Optional[str] = None,
                     demo_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Load the complete dataset or a filtered subset.

        Args:
            filter_key: Optional filter key to load specific demonstrations
            demo_indices: Optional list of demonstration indices to load

        Returns:
            Dictionary containing all dataset information
        """
        with h5py.File(self.dataset_path, "r") as f:
            # Determine which demos to load
            if filter_key is not None:
                demos = sorted(
                    [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)])]
                )
            else:
                demos = sorted(list(f["data"].keys()))

            if demo_indices is not None:
                demos = [demos[i] for i in demo_indices if i < len(demos)]

            # Load metadata
            problem_info = json.loads(f["data"].attrs["problem_info"])
            env_args = json.loads(f["data"].attrs["env_args"])

            # Load all demonstrations
            demonstrations = []
            for demo_name in demos:
                demo_data = self._load_single_demo(f, demo_name)
                demonstrations.append(demo_data)

            dataset = {
                "problem_info": problem_info,
                "env_args": env_args,
                "demonstrations": demonstrations,
                "num_demos": len(demonstrations)
            }

            return dataset

    def _load_single_demo(self, f: h5py.File, demo_name: str) -> Dict[str, Any]:
        """Load a single demonstration episode."""
        demo_group = f["data/{}".format(demo_name)]

        demo_data = {
            "name": demo_name,
            "num_samples": demo_group.attrs["num_samples"],
            "actions": demo_group["actions"][()],
            "rewards": demo_group["rewards"][()],
            "dones": demo_group["dones"][()],
            "obs": {},
            "next_obs": {}
        }

        # Load observations
        if "obs" in demo_group:
            for obs_key in demo_group["obs"].keys():
                demo_data["obs"][obs_key] = demo_group["obs/{}".format(obs_key)][()]

        # Load next observations
        if "next_obs" in demo_group:
            for obs_key in demo_group["next_obs"].keys():
                demo_data["next_obs"][obs_key] = demo_group["next_obs/{}".format(obs_key)][()]

        return demo_data

    def save_dataset(self, dataset: Dict[str, Any], output_path: str,
                     compression: str = "gzip"):
        """
        Save dataset to a new HDF5 file.

        Args:
            dataset: Dataset dictionary (from load_dataset)
            output_path: Path for the output HDF5 file
            compression: Compression method ('gzip', 'lzf', or None)
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                    exist_ok=True)

        with h5py.File(output_path, "w") as f:
            # Save metadata as attributes
            data_group = f.create_group("data")
            data_group.attrs["problem_info"] = json.dumps(dataset["problem_info"])
            data_group.attrs["env_args"] = json.dumps(dataset["env_args"])

            # Save each demonstration
            for demo_data in dataset["demonstrations"]:
                demo_name = demo_data["name"]
                demo_group = data_group.create_group(demo_name)
                demo_group.attrs["num_samples"] = demo_data["num_samples"]

                # Save actions, rewards, dones
                demo_group.create_dataset("actions", data=demo_data["actions"],
                                         compression=compression)
                demo_group.create_dataset("rewards", data=demo_data["rewards"],
                                         compression=compression)
                demo_group.create_dataset("dones", data=demo_data["dones"],
                                         compression=compression)

                # Save observations
                obs_group = demo_group.create_group("obs")
                for obs_key, obs_data in demo_data["obs"].items():
                    obs_group.create_dataset(obs_key, data=obs_data,
                                            compression=compression)

                # Save next observations
                if demo_data["next_obs"]:
                    next_obs_group = demo_group.create_group("next_obs")
                    for obs_key, obs_data in demo_data["next_obs"].items():
                        next_obs_group.create_dataset(obs_key, data=obs_data,
                                                     compression=compression)

        print(f"Dataset saved to: {output_path}")


def main():
    """Example usage of LiberoGoalReader."""
    import argparse

    parser = argparse.ArgumentParser(description="Read and process LIBERO Goal datasets")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input HDF5 dataset file")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to output HDF5 file (optional)")
    parser.add_argument("--info", action="store_true",
                       help="Print dataset information")
    parser.add_argument("--filter-key", type=str, default=None,
                       help="Filter key to load specific demonstrations")
    parser.add_argument("--demo-indices", type=int, nargs="+", default=None,
                       help="Specific demonstration indices to load")

    args = parser.parse_args()

    # Create reader
    reader = LiberoGoalReader(args.input)

    # Print info if requested
    if args.info:
        reader.get_dataset_info(verbose=True)

    # Load and optionally save dataset
    print("Loading dataset...")
    dataset = reader.load_dataset(filter_key=args.filter_key,
                                  demo_indices=args.demo_indices)
    print(f"Loaded {dataset['num_demos']} demonstrations")

    if args.output:
        print(f"Saving dataset to {args.output}...")
        reader.save_dataset(dataset, args.output)
        print("Done!")


if __name__ == "__main__":
    main()
