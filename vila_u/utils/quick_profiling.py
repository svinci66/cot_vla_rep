#!/usr/bin/env python3
"""
Quick performance profiling for training - integrated version.

This adds minimal overhead profiling to the actual training loop.
"""

import torch
import time
from collections import defaultdict
from typing import Dict


class QuickProfiler:
    """Minimal overhead profiler for production training"""

    def __init__(self, log_interval: int = 50):
        self.log_interval = log_interval
        self.step_count = 0
        self.timings = defaultdict(list)
        self.last_log_step = 0

    def record(self, name: str, elapsed: float):
        """Record a timing"""
        self.timings[name].append(elapsed)

    def step(self):
        """Call this at the end of each training step"""
        self.step_count += 1

        if self.step_count % self.log_interval == 0:
            self.print_summary()
            self.reset_timings()

    def print_summary(self):
        """Print timing summary"""
        if not self.timings:
            return

        steps = self.step_count - self.last_log_step
        print(f"\n{'='*80}")
        print(f"Performance Summary (steps {self.last_log_step+1}-{self.step_count})")
        print(f"{'='*80}")

        total_time = sum(sum(times) for times in self.timings.values())
        avg_step_time = total_time / steps if steps > 0 else 0

        print(f"Average step time: {avg_step_time*1000:.2f}ms")
        print(f"Throughput: {1/avg_step_time:.2f} steps/sec" if avg_step_time > 0 else "N/A")
        print(f"\n{'Component':<30} {'Avg(ms)':>10} {'%':>8}")
        print("-"*80)

        # Sort by average time
        sorted_names = sorted(
            self.timings.keys(),
            key=lambda x: sum(self.timings[x]) / len(self.timings[x]),
            reverse=True
        )

        for name in sorted_names:
            times = self.timings[name]
            avg_time = sum(times) / len(times)
            percentage = (avg_time / avg_step_time * 100) if avg_step_time > 0 else 0
            print(f"{name:<30} {avg_time*1000:>10.2f} {percentage:>7.1f}%")

        print("="*80 + "\n")
        self.last_log_step = self.step_count

    def reset_timings(self):
        """Reset timing data"""
        self.timings.clear()


def add_profiling_hooks(trainer, log_interval: int = 50):
    """
    Add profiling to Hugging Face Trainer with minimal overhead.

    Usage:
        trainer = Trainer(...)
        profiler = add_profiling_hooks(trainer, log_interval=50)
    """
    profiler = QuickProfiler(log_interval=log_interval)

    # Hook into training_step
    original_training_step = trainer.training_step

    def profiled_training_step(model, inputs):
        """Profiled training step"""
        start = time.perf_counter()

        # Original training step
        loss = original_training_step(model, inputs)

        # Record timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        profiler.record("training_step", elapsed)
        profiler.step()

        return loss

    trainer.training_step = profiled_training_step

    # Hook into get_train_dataloader to profile data loading
    original_get_train_dataloader = trainer.get_train_dataloader

    def profiled_get_train_dataloader():
        """Profile dataloader creation"""
        start = time.perf_counter()
        dataloader = original_get_train_dataloader()
        elapsed = time.perf_counter() - start
        print(f"DataLoader creation time: {elapsed:.2f}s")
        return dataloader

    trainer.get_train_dataloader = profiled_get_train_dataloader

    return profiler


# Standalone timing utilities
class Timer:
    """Simple timer context manager"""

    def __init__(self, name: str = "", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.elapsed = 0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start
        if self.verbose:
            print(f"{self.name}: {self.elapsed*1000:.2f}ms")


def profile_function(func):
    """Decorator to profile a function"""
    def wrapper(*args, **kwargs):
        with Timer(func.__name__):
            return func(*args, **kwargs)
    return wrapper
