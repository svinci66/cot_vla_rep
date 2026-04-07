"""
Minimal example demonstrating LiberoGoalReader usage without dependencies.

This shows the API and expected behavior when h5py is available.
"""

def example_without_running():
    """
    Example code showing how to use LiberoGoalReader.
    This doesn't actually run but shows the API.
    """

    print("="*70)
    print("LiberoGoalReader Usage Examples")
    print("="*70)

    print("""
# Installation
pip install h5py numpy

# Basic Usage
from libero_goal_reader import LiberoGoalReader

# 1. Initialize reader
reader = LiberoGoalReader("path/to/libero_goal/task_demo.hdf5")

# 2. Get dataset information
info = reader.get_dataset_info(verbose=True)
# Output:
# - Total transitions, trajectories
# - Trajectory length statistics
# - Action value ranges
# - Language instruction
# - Available filter keys

# 3. Load complete dataset
dataset = reader.load_dataset()
print(f"Loaded {dataset['num_demos']} demonstrations")
print(f"Task: {dataset['problem_info']['language_instruction']}")

# 4. Access demonstration data
for i, demo in enumerate(dataset['demonstrations']):
    print(f"Demo {i}: {demo['name']}")
    print(f"  Steps: {demo['num_samples']}")
    print(f"  Actions shape: {demo['actions'].shape}")  # (T, 7)
    print(f"  Observations: {list(demo['obs'].keys())}")

    # Access specific observations
    if 'agentview_rgb' in demo['obs']:
        rgb = demo['obs']['agentview_rgb']  # (T, H, W, 3)
        print(f"  RGB images shape: {rgb.shape}")

    if 'robot0_eef_pos' in demo['obs']:
        eef_pos = demo['obs']['robot0_eef_pos']  # (T, 3)
        print(f"  End-effector positions shape: {eef_pos.shape}")

# 5. Load subset of demonstrations
subset = reader.load_dataset(demo_indices=[0, 1, 2])
print(f"Loaded {subset['num_demos']} demos")

# 6. Save to new file
reader.save_dataset(subset, "output.hdf5")

# 7. Use with context manager
with LiberoGoalReader("dataset.hdf5") as reader:
    data = reader.load_dataset()
    # Process data...

# Command Line Interface
# ----------------------

# Show info only
$ python libero_goal_reader.py --input dataset.hdf5 --info

# Load and save
$ python libero_goal_reader.py --input dataset.hdf5 --output new.hdf5

# Load specific demos
$ python libero_goal_reader.py --input dataset.hdf5 --demo-indices 0 1 2 --output subset.hdf5

# Use filter key
$ python libero_goal_reader.py --input dataset.hdf5 --filter-key train --output train.hdf5
    """)

    print("\n" + "="*70)
    print("Dataset Structure")
    print("="*70)

    print("""
LIBERO Goal HDF5 Structure:

dataset.hdf5
├── data/                           # Main data group
│   ├── attributes:
│   │   ├── problem_info           # JSON: language instruction, task info
│   │   └── env_args               # JSON: environment configuration
│   │
│   ├── demo_0/                    # First demonstration
│   │   ├── attributes:
│   │   │   └── num_samples        # Number of timesteps
│   │   ├── actions                # (T, 7) - robot actions
│   │   ├── rewards                # (T,) - reward values
│   │   ├── dones                  # (T,) - done flags
│   │   ├── obs/                   # Observations at each timestep
│   │   │   ├── agentview_rgb      # (T, H, W, 3) - RGB camera
│   │   │   ├── eye_in_hand_rgb    # (T, H, W, 3) - wrist camera
│   │   │   ├── robot0_eef_pos     # (T, 3) - end-effector position
│   │   │   ├── robot0_eef_quat    # (T, 4) - end-effector quaternion
│   │   │   ├── robot0_gripper_qpos # (T, 2) - gripper joint positions
│   │   │   └── robot0_joint_pos   # (T, 7) - joint positions
│   │   └── next_obs/              # Next observations (same structure)
│   │       └── ...
│   │
│   ├── demo_1/                    # Second demonstration
│   │   └── ...
│   └── ...
│
└── mask/                          # Optional: filter keys for data splits
    ├── train                      # Training demo names
    └── valid                      # Validation demo names

Key Points:
- T = number of timesteps in trajectory (varies per demo)
- H, W = image height/width (typically 128x128 or 256x256)
- Actions are 7-DOF: 3 position + 3 rotation + 1 gripper
- Actions normalized to [-1, 1] range
- Each task has a language instruction describing the goal
    """)

    print("\n" + "="*70)
    print("Integration with VILA-U")
    print("="*70)

    print("""
To integrate LIBERO Goal data with VILA-U:

1. Load LIBERO dataset:
   reader = LiberoGoalReader("libero_goal/task_demo.hdf5")
   dataset = reader.load_dataset()

2. Extract language instruction:
   instruction = dataset['problem_info']['language_instruction']

3. Process demonstrations:
   for demo in dataset['demonstrations']:
       # Get visual observations
       images = demo['obs']['agentview_rgb']  # (T, H, W, 3)

       # Get actions
       actions = demo['actions']  # (T, 7)

       # Convert to VILA-U format as needed
       # ...

4. Save in VILA-U compatible format:
   reader.save_dataset(processed_dataset, "vila_u_format.hdf5")
    """)


if __name__ == "__main__":
    example_without_running()
