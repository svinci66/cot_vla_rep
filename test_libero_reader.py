"""
Test script for LiberoGoalReader

This script demonstrates how to use the LiberoGoalReader to:
1. Read LIBERO Goal dataset files
2. Extract dataset information
3. Load demonstrations
4. Save to a new file
"""

import os
import sys
from libero_goal_reader import LiberoGoalReader


def test_reader(dataset_path: str):
    """Test the LiberoGoalReader with a sample dataset."""

    print("="*70)
    print("Testing LiberoGoalReader")
    print("="*70)

    # Check if file exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        print("\nPlease provide a valid path to a LIBERO Goal dataset file.")
        print("Example: python test_libero_reader.py /path/to/libero_goal/task_demo.hdf5")
        return

    # Test 1: Get dataset info
    print("\n[Test 1] Getting dataset information...")
    reader = LiberoGoalReader(dataset_path)
    info = reader.get_dataset_info(verbose=True)

    # Test 2: Load full dataset
    print("\n[Test 2] Loading full dataset...")
    dataset = reader.load_dataset()
    print(f"Successfully loaded {dataset['num_demos']} demonstrations")
    print(f"Language instruction: {dataset['problem_info']['language_instruction']}")

    # Test 3: Show structure of first demonstration
    if dataset['demonstrations']:
        print("\n[Test 3] Structure of first demonstration:")
        demo = dataset['demonstrations'][0]
        print(f"  Demo name: {demo['name']}")
        print(f"  Number of samples: {demo['num_samples']}")
        print(f"  Actions shape: {demo['actions'].shape}")
        print(f"  Rewards shape: {demo['rewards'].shape}")
        print(f"  Observation keys: {list(demo['obs'].keys())}")

        # Show observation shapes
        print("\n  Observation shapes:")
        for obs_key, obs_data in demo['obs'].items():
            print(f"    {obs_key}: {obs_data.shape}")

    # Test 4: Load subset of demonstrations
    print("\n[Test 4] Loading subset of demonstrations (first 2)...")
    subset_dataset = reader.load_dataset(demo_indices=[0, 1])
    print(f"Successfully loaded {subset_dataset['num_demos']} demonstrations")

    # Test 5: Save to new file
    output_path = "/tmp/test_libero_output.hdf5"
    print(f"\n[Test 5] Saving dataset to {output_path}...")
    reader.save_dataset(subset_dataset, output_path)

    # Verify saved file
    print("\n[Test 6] Verifying saved file...")
    reader2 = LiberoGoalReader(output_path)
    info2 = reader2.get_dataset_info(verbose=False)
    print(f"Verified: {info2['total_trajectories']} trajectories in saved file")

    # Clean up
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Cleaned up temporary file: {output_path}")

    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)


def example_usage():
    """Show example usage patterns."""
    print("\n" + "="*70)
    print("Example Usage Patterns")
    print("="*70)

    print("""
# Example 1: Get dataset information
from libero_goal_reader import LiberoGoalReader

reader = LiberoGoalReader("path/to/dataset.hdf5")
info = reader.get_dataset_info(verbose=True)

# Example 2: Load and process dataset
dataset = reader.load_dataset()
for demo in dataset['demonstrations']:
    actions = demo['actions']
    observations = demo['obs']
    # Process your data here...

# Example 3: Load specific demonstrations
dataset = reader.load_dataset(demo_indices=[0, 1, 2])

# Example 4: Save to new file
reader.save_dataset(dataset, "output.hdf5")

# Example 5: Use as context manager
with LiberoGoalReader("path/to/dataset.hdf5") as reader:
    dataset = reader.load_dataset()
    # Process data...

# Example 6: Command line usage
# python libero_goal_reader.py --input dataset.hdf5 --info
# python libero_goal_reader.py --input dataset.hdf5 --output new_dataset.hdf5
# python libero_goal_reader.py --input dataset.hdf5 --demo-indices 0 1 2 --output subset.hdf5
    """)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        test_reader(dataset_path)
    else:
        print("Usage: python test_libero_reader.py <path_to_dataset.hdf5>")
        print("\nNo dataset path provided. Showing example usage instead...")
        example_usage()

        print("\n" + "="*70)
        print("To run tests, provide a dataset path:")
        print("  python test_libero_reader.py /path/to/libero_goal/task_demo.hdf5")
        print("="*70)
