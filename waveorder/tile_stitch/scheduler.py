"""Input-tile scheduling for pipelined Stage A → Stage B execution.

The driver schedules input tile reconstructions so stitch tasks become
eligible early. Two orderings are precomputed at plan-build time:

* ``schedule_coverage_greedy`` returns a permutation of input tile ids
  used as a dask priority hint — tiles earlier in the schedule run
  first. Greedy by "last-remaining-dep" count: the input that completes
  the most output tiles' dependencies wins.
* ``bundle_inputs_by_cooccurrence`` partitions input tiles into batches
  whose dependent-output sets overlap. A batch becomes one Stage A
  task; finishing it unblocks a tight cluster of output tiles instead
  of a smear across the output grid.

Both functions are pure / graph-derived (no axis assumptions), so they
work for any bipartite dependency graph regardless of dataset shape or
partitioning.
"""

from __future__ import annotations

from collections import defaultdict


def schedule_coverage_greedy(
    input_tile_ids: list[int],
    output_to_inputs: dict[int, list[int]],
) -> list[int]:
    """Order input tiles to greedily maximise the rate at which output
    tiles become ready (all their deps scheduled).

    At each step picks the input tile that is the **last remaining dep**
    for the most output tiles. Tiebreaker: the one with the highest
    static popularity (most output tiles depending on it).

    Pure function. Runs in O(N²·M) worst case — for N≈10³, M≈10³ that's
    ~10⁹ ops which is too slow; if we ever scale that high we'd switch
    to a heap-based incremental priority. For our 200×180 graph this is
    well under a second.

    Returns
    -------
    ordered : list[int]
        Permutation of ``input_tile_ids`` (same length, same set, new order).
    """
    inputs_to_outputs: dict[int, set[int]] = defaultdict(set)
    for ot_id, deps in output_to_inputs.items():
        for tid in deps:
            inputs_to_outputs[tid].add(ot_id)

    popularity = {tid: len(inputs_to_outputs[tid]) for tid in input_tile_ids}

    remaining: dict[int, set[int]] = {
        ot_id: set(deps) for ot_id, deps in output_to_inputs.items()
    }

    pending = set(input_tile_ids)
    schedule: list[int] = []

    while pending:
        best_score = -1
        best_pop = -1
        best_tid = -1
        for tid in pending:
            last_count = sum(
                1
                for ot_id in inputs_to_outputs[tid]
                if remaining[ot_id] == {tid}
            )
            pop = popularity[tid]
            if (last_count, pop) > (best_score, best_pop):
                best_score = last_count
                best_pop = pop
                best_tid = tid

        schedule.append(best_tid)
        pending.discard(best_tid)
        for ot_id in inputs_to_outputs[best_tid]:
            remaining[ot_id].discard(best_tid)

    return schedule


def bundle_inputs_by_cooccurrence(
    input_tile_ids: list[int],
    output_to_inputs: dict[int, list[int]],
    batch_size: int,
) -> list[list[int]]:
    """Group input tiles whose dependent-output sets overlap maximally.

    A batch becomes one Stage A task. Inputs serving the SAME output
    tiles co-locate so finishing the batch unblocks those outputs
    simultaneously, instead of dispatching N stitches each waiting on
    one straggler.

    Algorithm:
      1. Build inputs_to_outputs reverse index.
      2. Sort inputs by descending |dependent outputs| (popular tiles
         seed first).
      3. For each ungrouped input (seed), grow a batch by adding the
         ungrouped input that maximises overlap of its dependent outputs
         with the current batch's output union. Stop early on 0 overlap.
      4. Return list of batches.

    With ``batch_size=1`` returns one singleton batch per input (== no
    batching).

    Returns
    -------
    list[list[int]]
        Permutation of ``input_tile_ids`` partitioned into batches.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    inputs_to_outputs: dict[int, set[int]] = defaultdict(set)
    for ot_id, deps in output_to_inputs.items():
        for tid in deps:
            inputs_to_outputs[tid].add(ot_id)

    ungrouped: set[int] = set(input_tile_ids)
    seed_order = sorted(input_tile_ids, key=lambda t: -len(inputs_to_outputs[t]))

    batches: list[list[int]] = []
    for seed in seed_order:
        if seed not in ungrouped:
            continue
        batch: list[int] = [seed]
        union: set[int] = set(inputs_to_outputs[seed])
        ungrouped.discard(seed)

        while len(batch) < batch_size and ungrouped:
            best_overlap = 0
            best_cand = -1
            for cand in ungrouped:
                overlap = len(inputs_to_outputs[cand] & union)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_cand = cand
            if best_overlap == 0:
                break
            batch.append(best_cand)
            union |= inputs_to_outputs[best_cand]
            ungrouped.discard(best_cand)

        batches.append(batch)

    return batches


def output_to_batches_map(
    input_batches: list[list[int]],
    output_to_inputs: dict[int, list[int]],
) -> dict[int, list[int]]:
    """For each output tile, which input batches must complete before it
    can stitch?

    Returns dict[output_tile_id, sorted list of batch indices]. The
    driver tracks this as ``remaining_batches`` and dispatches the
    stitch when all listed batches have completed.
    """
    tile_to_batch: dict[int, int] = {}
    for bidx, batch in enumerate(input_batches):
        for tid in batch:
            tile_to_batch[tid] = bidx

    out: dict[int, list[int]] = {}
    for ot_id, deps in output_to_inputs.items():
        bset = {tile_to_batch[tid] for tid in deps if tid in tile_to_batch}
        out[ot_id] = sorted(bset)
    return out
