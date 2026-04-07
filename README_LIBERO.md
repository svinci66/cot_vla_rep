# LIBERO Goal Dataset Reader

This directory contains tools for reading and processing LIBERO Goal datasets.

## Files

- `libero_goal_reader.py`: Main reader class for LIBERO Goal HDF5 datasets
- `test_libero_reader.py`: Test script and usage examples

## Quick Start

### 1. Get Dataset Information

```python
from libero_goal_reader import LiberoGoalReader

reader = LiberoGoalReader("path/to/dataset.hdf5")
info = reader.get_dataset_info(verbose=True)
```

### 2. Load Dataset

```python
# Load full dataset
dataset = reader.load_dataset()

# Load specific demonstrations
dataset = reader.load_dataset(demo_indices=[0, 1, 2])

# Load with filter key
dataset = reader.load_dataset(filter_key="train")
```

### 3. Access Data

```python
# Get metadata
language_instruction = dataset['problem_info']['language_instruction']
env_args = dataset['env_args']

# Iterate through demonstrations
for demo in dataset['demonstrations']:
    actions = demo['actions']  # Shape: (T, action_dim)
    observations = demo['obs']  # Dict of observation arrays
    rewards = demo['rewards']  # Shape: (T,)

    # Access specific observations
    rgb_image = demo['obs']['agentview_rgb']  # Shape: (T, H, W, 3)
    robot_state = demo['obs']['robot0_eef_pos']  # Shape: (T, 3)
```

### 4. Save Dataset

```python
reader.save_dataset(dataset, "output.hdf5")
```

## Command Line Usage

```bash
# Show dataset information
python libero_goal_reader.py --input dataset.hdf5 --info

# Load and save dataset
python libero_goal_reader.py --input dataset.hdf5 --output new_dataset.hdf5

# Load specific demonstrations
python libero_goal_reader.py --input dataset.hdf5 --demo-indices 0 1 2 --output subset.hdf5

# Use filter key
python libero_goal_reader.py --input dataset.hdf5 --filter-key train --output train_data.hdf5
```

## Testing

```bash
# Run tests with a dataset file
python test_libero_reader.py /path/to/dataset.hdf5

# Show example usage
python test_libero_reader.py
```

## Dataset Structure

LIBERO Goal datasets are stored in HDF5 format with the following structure:

```
dataset.hdf5
├── data/
│   ├── demo_0/
│   │   ├── actions          # (T, 7) - robot actions
│   │   ├── rewards          # (T,) - reward values
│   │   ├── dones            # (T,) - episode termination flags
│   │   ├── obs/
│   │   │   ├── agentview_rgb       # (T, H, W, 3) - RGB images
│   │   │   ├── robot0_eef_pos      # (T, 3) - end-effector position
│   │   │   ├── robot0_eef_quat     # (T, 4) - end-effector orientation
│   │   │   └── ...
│   │   └── next_obs/
│   │       └── ...
│   ├── demo_1/
│   └── ...
└── mask/                    # Optional filter keys
    ├── train
    └── valid
```

## Requirements

- h5py
- numpy

## Notes

- All LIBERO Goal tasks have language instructions describing the task
- Actions are normalized to [-1, 1] range
- Observations include RGB images, depth images, and robot proprioceptive states
- Each demonstration is a complete episode trajectory
