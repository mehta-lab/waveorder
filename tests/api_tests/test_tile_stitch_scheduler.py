"""Scheduler tests — coverage-greedy ordering + co-occurrence batching."""

from __future__ import annotations

import pytest

from waveorder.tile_stitch.scheduler import (
    bundle_inputs_by_cooccurrence,
    output_to_batches_map,
    schedule_coverage_greedy,
)


# --- schedule_coverage_greedy ---


def test_schedule_returns_permutation():
    inputs = [10, 20, 30, 40]
    out_to_in = {0: [10, 20], 1: [20, 30], 2: [30, 40]}
    schedule = schedule_coverage_greedy(inputs, out_to_in)
    assert sorted(schedule) == sorted(inputs)


def test_schedule_picks_last_dep_first():
    """If picking input X completes output O's deps, X scores higher than
    an unpopular tile whose remaining deps all live elsewhere."""
    # Output 0 needs only [10]; picking 10 immediately completes O0.
    # Output 1 needs [20, 30, 40]; no single pick completes it.
    inputs = [10, 20, 30, 40]
    out_to_in = {0: [10], 1: [20, 30, 40]}
    schedule = schedule_coverage_greedy(inputs, out_to_in)
    assert schedule[0] == 10


def test_schedule_tiebreaks_by_popularity():
    """When two inputs are tied on last-dep count, the more popular wins."""
    inputs = [1, 2, 3]
    # Both 1 and 2 are last dep of one output each; 1 is popular (2 outputs),
    # 2 has only 1 output total.
    out_to_in = {0: [1], 1: [2], 2: [1, 3]}
    schedule = schedule_coverage_greedy(inputs, out_to_in)
    # 1 should land before 2 because of higher popularity (broke the tie)
    assert schedule.index(1) < schedule.index(2)


def test_schedule_handles_disconnected_inputs():
    """Inputs that no output depends on still appear in the schedule."""
    inputs = [1, 2, 3, 99]  # 99 has no dependents
    out_to_in = {0: [1, 2], 1: [2, 3]}
    schedule = schedule_coverage_greedy(inputs, out_to_in)
    assert 99 in schedule


def test_schedule_empty():
    assert schedule_coverage_greedy([], {}) == []


# --- bundle_inputs_by_cooccurrence ---


def test_batch_size_one_returns_singletons():
    inputs = [1, 2, 3]
    out_to_in = {0: [1, 2], 1: [2, 3]}
    batches = bundle_inputs_by_cooccurrence(inputs, out_to_in, batch_size=1)
    assert all(len(b) == 1 for b in batches)
    assert sorted(b[0] for b in batches) == [1, 2, 3]


def test_batch_size_zero_rejected():
    with pytest.raises(ValueError, match="batch_size must be"):
        bundle_inputs_by_cooccurrence([1, 2], {}, batch_size=0)


def test_batch_groups_inputs_serving_same_outputs():
    """Inputs 1 and 2 both serve outputs {A, B}; they should batch together."""
    inputs = [1, 2, 3]
    # 1, 2 both serve outputs 0+1; 3 serves output 2 only.
    out_to_in = {0: [1, 2], 1: [1, 2], 2: [3]}
    batches = bundle_inputs_by_cooccurrence(inputs, out_to_in, batch_size=2)
    paired = next((b for b in batches if 1 in b), None)
    assert paired is not None
    assert 2 in paired


def test_batch_partition_is_complete():
    """All inputs land in exactly one batch."""
    inputs = [1, 2, 3, 4, 5]
    out_to_in = {0: [1, 2], 1: [2, 3], 2: [4, 5]}
    batches = bundle_inputs_by_cooccurrence(inputs, out_to_in, batch_size=3)
    flat = sorted(t for b in batches for t in b)
    assert flat == sorted(inputs)


def test_batch_stops_early_on_zero_overlap():
    """No remaining ungrouped input shares deps → batch stays a singleton."""
    inputs = [1, 2]
    # Outputs are completely disjoint
    out_to_in = {0: [1], 1: [2]}
    batches = bundle_inputs_by_cooccurrence(inputs, out_to_in, batch_size=4)
    assert len(batches) == 2  # both stay singleton (no overlap to merge on)


# --- output_to_batches_map ---


def test_output_to_batches_map_basic():
    batches = [[1, 2], [3, 4]]
    out_to_in = {0: [1, 3], 1: [2], 2: [4]}
    mapping = output_to_batches_map(batches, out_to_in)
    assert mapping[0] == [0, 1]  # output 0 needs both batches
    assert mapping[1] == [0]
    assert mapping[2] == [1]


def test_output_to_batches_map_skips_inputs_outside_batches():
    """Inputs not present in any batch are silently dropped from the map."""
    batches = [[1]]
    out_to_in = {0: [1, 99]}  # 99 is not in any batch
    mapping = output_to_batches_map(batches, out_to_in)
    assert mapping[0] == [0]


def test_output_to_batches_map_deduplicates():
    """Same batch listed once per output, even if multiple inputs in it."""
    batches = [[1, 2, 3]]
    out_to_in = {0: [1, 2, 3]}
    mapping = output_to_batches_map(batches, out_to_in)
    assert mapping[0] == [0]
