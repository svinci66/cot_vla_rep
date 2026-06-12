from collections import Counter

from vila_u.train.vila_u_trainer import VILADistributedSampler


class DummyDemoDataset:
    def __init__(self, num_demos=50, samples_per_demo=96):
        self.samples = []
        for demo_idx in range(num_demos):
            for _ in range(samples_per_demo):
                self.samples.append({"demo": f"demo_{demo_idx}"})

    def __len__(self):
        return len(self.samples)


def collect_rank_indices(dataset, rank, rank_slice_after_shuffle=True):
    sampler = VILADistributedSampler(
        dataset,
        num_replicas=4,
        rank=rank,
        seed=123,
        batch_size=16,
        sample_len_list=[len(dataset)],
        rank_slice_after_shuffle=rank_slice_after_shuffle,
    )
    return list(iter(sampler))


def test_rank_slice_after_shuffle_covers_all_demos_per_rank_without_duplicates():
    dataset = DummyDemoDataset(num_demos=50, samples_per_demo=96)

    rank_indices = [collect_rank_indices(dataset, rank) for rank in range(4)]
    for indices in rank_indices:
        demo_counts = Counter(dataset.samples[index]["demo"] for index in indices)
        assert len(indices) == 1200
        assert len(demo_counts) == 50
        assert min(demo_counts.values()) > 0

    union = [index for indices in rank_indices for index in indices]
    assert len(union) == 4800
    assert len(set(union)) == 4800


def test_legacy_contiguous_rank_slice_can_miss_later_demos():
    dataset = DummyDemoDataset(num_demos=50, samples_per_demo=96)
    rank0_indices = collect_rank_indices(dataset, rank=0, rank_slice_after_shuffle=False)
    rank0_demos = {dataset.samples[index]["demo"] for index in rank0_indices}

    assert len(rank0_indices) == 1200
    assert len(rank0_demos) < 50
