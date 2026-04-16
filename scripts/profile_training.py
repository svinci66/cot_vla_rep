#!/usr/bin/env python3
"""
Performance profiling script for VILA-U action prediction training.

This script profiles the training loop to identify performance bottlenecks.

Usage:
    python scripts/profile_training.py --model_path /path/to/model --data_root /path/to/data
"""

import os
import sys
import time
import torch
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vila_u.utils.profiling import TrainingProfiler
from vila_u.model.builder import load_pretrained_model
from vila_u.data.libero_dataset_v2 import LiberoGoalDataset
from vila_u.train.train_action_prediction_main import (
    ActionPredictionTrainer,
    DiscreteActionPredictionDataCollator,
)
from vila_u.utils.action_tokenizer import select_action_token_ids
from vila_u.constants import ACTION_NUM_BINS
from torch.utils.data import DataLoader


def profile_data_loading(data_root, image_processor, tokenizer, num_batches=20):
    """Profile data loading performance"""
    print("\n" + "="*80)
    print("PROFILING DATA LOADING")
    print("="*80)

    profiler = TrainingProfiler()

    # Create dataset
    with profiler.profile("dataset_creation"):
        dataset = LiberoGoalDataset(
            data_root=data_root,
            image_processor=image_processor,
            tokenizer=tokenizer,
            action_chunk_size=10,
            image_size=256,
            remove_pause_intervals=True,
            pause_threshold=0.01,
            mm_use_im_start_end=False,
            action_token_ids=select_action_token_ids(tokenizer, ACTION_NUM_BINS),
            use_discrete_action_prediction=True,
        )

    print(f"Dataset size: {len(dataset)} samples")

    # Test single sample loading
    print("\nTesting single sample loading...")
    for i in range(5):
        with profiler.profile("single_sample_load"):
            sample = dataset[i]

    # Create dataloader
    collate_fn = DiscreteActionPredictionDataCollator(
        tokenizer=tokenizer,
        model_max_length=512,
        mm_use_im_start_end=False,
        action_token_ids=select_action_token_ids(tokenizer, ACTION_NUM_BINS),
        action_slot_token_id=tokenizer.convert_tokens_to_ids("<act>"),
        action_chunk_size=10,
        action_dim=7,
        use_hybrid_attention=True,
    )

    # Test different worker configurations
    for num_workers in [0, 4, 8, 16]:
        print(f"\n--- Testing with {num_workers} workers ---")

        dataloader = DataLoader(
            dataset,
            batch_size=20,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True if num_workers > 0 else False,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None,
        )

        # Warmup
        iterator = iter(dataloader)
        next(iterator)

        # Profile
        batch_times = []
        for i in range(min(num_batches, len(dataloader))):
            start = time.perf_counter()
            batch = next(iterator)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.perf_counter() - start
            batch_times.append(elapsed)

        avg_time = sum(batch_times) / len(batch_times)
        print(f"  Average batch time: {avg_time*1000:.2f}ms")
        print(f"  Throughput: {20/avg_time:.1f} samples/sec")

    profiler.print_summary()
    return profiler


def profile_model_forward(model, batch, num_iterations=10):
    """Profile model forward pass"""
    print("\n" + "="*80)
    print("PROFILING MODEL FORWARD PASS")
    print("="*80)

    profiler = TrainingProfiler()

    # Move batch to GPU
    if torch.cuda.is_available():
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}
        model = model.cuda()

    model.eval()

    # Warmup
    with torch.no_grad():
        _ = model(**batch)

    # Profile forward pass breakdown
    print(f"\nProfiling {num_iterations} forward passes...")

    for i in range(num_iterations):
        with profiler.profile("forward_total"):

            with profiler.profile("1_prepare_inputs"):
                # This happens inside model forward
                pass

            with profiler.profile("2_model_forward"):
                with torch.no_grad():
                    outputs = model(**batch)

    profiler.print_summary()
    return profiler


def profile_training_step(model, batch, optimizer, num_iterations=10):
    """Profile complete training step"""
    print("\n" + "="*80)
    print("PROFILING TRAINING STEP (Forward + Backward + Optimizer)")
    print("="*80)

    profiler = TrainingProfiler()

    # Move to GPU
    if torch.cuda.is_available():
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}
        model = model.cuda()

    model.train()

    print(f"\nProfiling {num_iterations} training steps...")

    for i in range(num_iterations):
        with profiler.profile("step_total"):

            # Forward
            with profiler.profile("1_forward"):
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs

            # Backward
            with profiler.profile("2_backward"):
                loss.backward()

            # Optimizer
            with profiler.profile("3_optimizer_step"):
                optimizer.step()

            # Zero grad
            with profiler.profile("4_zero_grad"):
                optimizer.zero_grad()

    profiler.print_summary()
    return profiler


def profile_attention_mask(batch_size=20, seq_len=512, num_action_tokens=70, num_iterations=100):
    """Profile hybrid attention mask construction"""
    print("\n" + "="*80)
    print("PROFILING HYBRID ATTENTION MASK CONSTRUCTION")
    print("="*80)

    from vila_u.utils.hybrid_attention import build_hybrid_attention_mask

    profiler = TrainingProfiler()

    # Create dummy attention mask
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    if torch.cuda.is_available():
        attention_mask = attention_mask.cuda()

    print(f"\nProfiling {num_iterations} mask constructions...")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Action tokens: {num_action_tokens}")

    for i in range(num_iterations):
        with profiler.profile("build_hybrid_attention_mask"):
            mask = build_hybrid_attention_mask(
                attention_mask,
                num_action_tokens,
                dtype=torch.float32,
            )

    profiler.print_summary()

    # Calculate throughput
    stats = profiler.get_stats("build_hybrid_attention_mask")
    if stats:
        print(f"\nThroughput: {1000/stats['mean']:.1f} masks/sec")
        print(f"Per-batch overhead: {stats['mean']*1000:.2f}ms")

    return profiler


def main():
    parser = argparse.ArgumentParser(description="Profile VILA-U training performance")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to pretrained model")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Path to LIBERO dataset")
    parser.add_argument("--profile_data", action="store_true",
                       help="Profile data loading")
    parser.add_argument("--profile_forward", action="store_true",
                       help="Profile model forward pass")
    parser.add_argument("--profile_training", action="store_true",
                       help="Profile training step")
    parser.add_argument("--profile_attention", action="store_true",
                       help="Profile attention mask construction")
    parser.add_argument("--profile_all", action="store_true",
                       help="Profile everything")
    parser.add_argument("--num_batches", type=int, default=20,
                       help="Number of batches to profile")

    args = parser.parse_args()

    # If no specific profiling selected, do all
    if not any([args.profile_data, args.profile_forward,
                args.profile_training, args.profile_attention]):
        args.profile_all = True

    print("="*80)
    print("VILA-U Training Performance Profiler")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_root}")
    print("="*80)

    # Profile attention mask (lightweight, always do this)
    if args.profile_all or args.profile_attention:
        profile_attention_mask()

    # Load model (needed for other profiling)
    if args.profile_all or args.profile_forward or args.profile_training or args.profile_data:
        print("\nLoading model...")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=args.model_path,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        print("Model loaded successfully")

    # Profile data loading
    if args.profile_all or args.profile_data:
        profile_data_loading(
            args.data_root,
            image_processor,
            tokenizer,
            num_batches=args.num_batches
        )

    # Get a sample batch for forward/training profiling
    if args.profile_all or args.profile_forward or args.profile_training:
        print("\nCreating sample batch...")
        dataset = LiberoGoalDataset(
            data_root=args.data_root,
            image_processor=image_processor,
            tokenizer=tokenizer,
            action_chunk_size=10,
            image_size=256,
            remove_pause_intervals=True,
            action_token_ids=select_action_token_ids(tokenizer, ACTION_NUM_BINS),
            use_discrete_action_prediction=True,
        )

        collate_fn = DiscreteActionPredictionDataCollator(
            tokenizer=tokenizer,
            model_max_length=512,
            mm_use_im_start_end=False,
            action_token_ids=select_action_token_ids(tokenizer, ACTION_NUM_BINS),
            action_slot_token_id=tokenizer.convert_tokens_to_ids("<act>"),
            action_chunk_size=10,
            action_dim=7,
            use_hybrid_attention=True,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=20,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

        batch = next(iter(dataloader))

    # Profile forward pass
    if args.profile_all or args.profile_forward:
        profile_model_forward(model, batch, num_iterations=10)

    # Profile training step
    if args.profile_all or args.profile_training:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        profile_training_step(model, batch, optimizer, num_iterations=10)

    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)
    print("\nRecommendations will be printed above.")
    print("Look for operations taking >10% of total time.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
