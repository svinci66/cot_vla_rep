"""
Training profiling utilities for performance analysis
"""
import time
import torch
from contextlib import contextmanager
from collections import defaultdict
from typing import Dict, List


class TrainingProfiler:
    """
    Lightweight profiler for training loop performance analysis.

    Usage:
        profiler = TrainingProfiler()

        with profiler.profile("data_loading"):
            batch = next(dataloader)

        with profiler.profile("forward"):
            outputs = model(batch)

        # Print summary
        profiler.print_summary()
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.counts: Dict[str, int] = defaultdict(int)
        self.cuda_available = torch.cuda.is_available()

    @contextmanager
    def profile(self, name: str):
        """Context manager for timing a code block"""
        if not self.enabled:
            yield
            return

        # Synchronize CUDA before timing
        if self.cuda_available:
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        try:
            yield
        finally:
            # Synchronize CUDA after timing
            if self.cuda_available:
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start_time
            self.timings[name].append(elapsed)
            self.counts[name] += 1

    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a specific timing"""
        if name not in self.timings or not self.timings[name]:
            return {}

        times = self.timings[name]
        return {
            'count': len(times),
            'total': sum(times),
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
        }

    def print_summary(self, top_k: int = None):
        """Print timing summary"""
        if not self.timings:
            print("No profiling data collected")
            return

        print("\n" + "="*80)
        print("Training Performance Profile")
        print("="*80)

        # Calculate total time
        total_time = sum(sum(times) for times in self.timings.values())

        # Sort by total time
        sorted_names = sorted(
            self.timings.keys(),
            key=lambda x: sum(self.timings[x]),
            reverse=True
        )

        if top_k:
            sorted_names = sorted_names[:top_k]

        # Print header
        print(f"{'Operation':<30} {'Count':>8} {'Total(s)':>10} {'Mean(ms)':>10} {'%':>8}")
        print("-"*80)

        # Print each operation
        for name in sorted_names:
            stats = self.get_stats(name)
            percentage = (stats['total'] / total_time * 100) if total_time > 0 else 0

            print(f"{name:<30} {stats['count']:>8} {stats['total']:>10.3f} "
                  f"{stats['mean']*1000:>10.2f} {percentage:>7.1f}%")

        print("-"*80)
        print(f"{'TOTAL':<30} {'':<8} {total_time:>10.3f} {'':<10} {'100.0%':>8}")
        print("="*80 + "\n")

    def reset(self):
        """Reset all timing data"""
        self.timings.clear()
        self.counts.clear()

    def get_summary_dict(self) -> Dict[str, Dict[str, float]]:
        """Get summary as dictionary"""
        return {name: self.get_stats(name) for name in self.timings.keys()}


class StepProfiler:
    """
    Detailed profiler for a single training step.
    Breaks down forward, backward, and optimizer steps.
    """

    def __init__(self):
        self.profiler = TrainingProfiler()
        self.step_count = 0

    def profile_step(self, model, inputs, optimizer, scaler=None):
        """
        Profile a complete training step with detailed breakdown.

        Returns:
            loss: The computed loss
            timings: Dictionary of timing information
        """
        self.step_count += 1

        # 1. Data transfer to GPU
        with self.profiler.profile("1_data_to_gpu"):
            if torch.cuda.is_available():
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                         for k, v in inputs.items()}

        # 2. Forward pass breakdown
        with self.profiler.profile("2_forward_total"):

            with self.profiler.profile("2a_prepare_inputs"):
                # This includes multimodal input preparation
                pass

            with self.profiler.profile("2b_vision_encoding"):
                # Vision tower forward
                pass

            with self.profiler.profile("2c_llm_forward"):
                # LLM forward pass
                pass

            with self.profiler.profile("2d_loss_computation"):
                # Loss calculation
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(**inputs)
                        loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                else:
                    outputs = model(**inputs)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs

        # 3. Backward pass
        with self.profiler.profile("3_backward"):
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        # 4. Optimizer step
        with self.profiler.profile("4_optimizer_step"):
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        # 5. Zero gradients
        with self.profiler.profile("5_zero_grad"):
            optimizer.zero_grad()

        return loss, self.profiler.get_summary_dict()

    def print_summary(self):
        """Print detailed step summary"""
        self.profiler.print_summary()


def add_profiling_to_trainer(trainer, num_profile_steps: int = 10):
    """
    Add profiling hooks to a Hugging Face Trainer.

    Args:
        trainer: The Trainer instance
        num_profile_steps: Number of steps to profile
    """
    profiler = TrainingProfiler()

    original_training_step = trainer.training_step
    step_count = [0]  # Use list to allow modification in closure

    def profiled_training_step(model, inputs):
        step_count[0] += 1

        if step_count[0] <= num_profile_steps:
            with profiler.profile("training_step_total"):
                with profiler.profile("compute_loss"):
                    loss = original_training_step(model, inputs)

            # Print summary after profiling period
            if step_count[0] == num_profile_steps:
                print(f"\n{'='*80}")
                print(f"Profiling Summary (first {num_profile_steps} steps)")
                print(f"{'='*80}")
                profiler.print_summary()
        else:
            loss = original_training_step(model, inputs)

        return loss

    trainer.training_step = profiled_training_step
    return profiler


def profile_dataloader(dataloader, num_batches: int = 10):
    """
    Profile dataloader performance.

    Args:
        dataloader: The DataLoader to profile
        num_batches: Number of batches to profile
    """
    profiler = TrainingProfiler()

    print(f"\nProfiling DataLoader (first {num_batches} batches)...")

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        with profiler.profile("dataloader_iteration"):
            # Simulate processing
            if isinstance(batch, dict):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        _ = v.shape

    profiler.print_summary()
    return profiler
