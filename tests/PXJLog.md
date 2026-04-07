# PXJ: Parallel Prefix Join

## What PXJ Is

PXJ is a **distribution framework** for computing Transitive Closure across
multiple GPUs. It is NOT a specific join parenthesization strategy — it is the
**execution harness** that enables different strategies to be distributed.

The core insight: computing TC requires evaluating a chain of joins, and there
are many ways to parenthesize that chain (extend one hop at a time, square
repeatedly, hybrid approaches, extend by arbitrary chunk sizes, etc.). On a
single node, any parenthesization works. But **distributing** the work across
ranks requires a contract for who computes what — and that's what PXJ provides.

## PXJ's Contract

Given `k = total_rank` (must be power of 2), PXJ assigns each rank `p`
responsibility for all paths of length `p+1, p+1+k, p+1+2k, ...`. It achieves
this in three phases:

1. **Phase 1** (comm-free, fixed): Each rank computes its seed paths and `g^k`
2. **Phase 2** (distributed, **pluggable**): Compute TC(g^k) — all paths whose
   length is a multiple of k
3. **Phase 3** (distributed, fixed): Extend seeds by TC(g^k), global reduce

Phase 1 and Phase 3 are the same regardless of strategy. Phase 2 is the
**strategy slot** — the only requirement is that it produces TC(g^k). How it
gets there is not PXJ's concern.

## PXJ as a Benchmarking Constraint

The value of PXJ is not that any specific parenthesization is optimal. Rather,
PXJ **constrains** how a parenthesization plan can be executed in a distributed
setting. A planner (cost-based optimizer, heuristic, or manual choice) decides
the strategy; PXJ provides the execution framework that makes distribution
possible.

Different Phase 2 strategies to benchmark:
- **Logarithmic squaring** (PXJLog): self-join to double path lengths each
  iteration. O(log(diameter/k)) iterations, but hash table rebuilt every
  iteration.
- **Linear extension** (PXJLin): extend by k each iteration using a static
  hash table on g^k. O(diameter/k) iterations, but no hash table rebuild.
- **Hybrid**: square for a few iterations then switch to linear. Or extend by
  some chunk size other than k.
- Other strategies as needed.

Phase 1 + Phase 3 cost is fixed across all strategies. The question of which
Phase 2 strategy wins depends on the graph (diameter, density) and the system
(MPI latency, GPU memory).

---

# PXJLog: PXJ with Logarithmic Squaring

PXJLog is an instantiation of PXJ where Phase 2 uses repeated squaring.

## Overview

Given `k = total_rank` (must be power of 2), each rank `p` computes all paths
of length `p+1, p+1+k, p+1+2k, ...`. Three phases:

1. **Phase 1** (comm-free): Each rank computes its seed and `g^k` via parallel
   prefix
2. **Phase 2** (distributed): TC of `g^k` via repeated squaring (Approach 1b)
3. **Phase 3** (distributed): Extend seeds by TC(g^k), global reduce

---

## Phase 1: Parallel Prefix Seed Computation

**Goal:** Each rank `p` needs both its seed (paths of length `p+1`) and `g^k`
(paths of exactly length `k`) for Phase 2. The parallel prefix / binary lifting
structure computes both with the minimum number of multiplications: `log2(k)`
joins instead of `k`.

**Setup:**
- `MPI_Allgatherv` replicates the full edge set on every rank
- `g` = forward edges (key=col0, value=col1), hash table built on this
- `l` = reverse edges (key=col1, value=col0), deduplicated
- `g_rev` = copy of `l` (used for squaring `g`)

**Loop** (`log2(k)` steps, purely local, no MPI):

```
for step = 0 to log2(k)-1:
    if rank has bit `step` set:
        l = join(g_hash_table, l)      // extend seed by current g_power
        l_power += g_power

    // Square g: join g_rev against g_hash_table
    g_rev = join(g_hash_table, g_rev)
    g = reverse(g_rev)                 // convert back to forward
    rebuild g_hash_table on new g
    g_power *= 2
```

This is exponentiation by squaring applied to the rank index. Rank `p`'s binary
representation determines which powers of `g` get composed into its seed. For
example with `k=8`:
- Rank 0 (000): no extensions, seed = original edges (length 1)
- Rank 3 (011): extends at step 0 (g^1) and step 1 (g^2), seed = length 4
- Rank 5 (101): extends at step 0 (g^1) and step 2 (g^4), seed = length 6
- Rank 7 (111): extends at all steps, seed = length 8

**After the loop:**
- `l` = seed for this rank (paths of length `rank+1`), different per rank
- `g` = forward `g^k`, identical on all ranks
- `g_rev` = reverse `g^k`, identical on all ranks

**Transition to Phase 2:**
- Every rank has the same `g^k`, so each rank drops tuples that don't belong
  to its partition using `thrust::remove_if` with predicate
  `get_rank(key, total_rank) != rank`. No MPI needed.
- This produces `gk_fwd` (forward) and `gk_rev` (reverse), both filtered to
  this rank's partition.
- Seeds stay local (not redistributed)

---

## Phase 2: TC of g^k via Squaring (Approach 1b)

**Goal:** Compute all paths whose length is a multiple of `k`: `k, 2k, 3k, ...`

**Input:** Each rank has `gk_fwd` and `gk_rev` — its partition of g^k (forward
and reverse), where `get_rank(entity.key, total_rank) == rank` for both. The
partition is obtained by a local `thrust::remove_if` filter (no MPI).

**Why forward and reverse?** The codebase convention is:
- `full` (reverse, key=col1, value=col0) — the probe side and canonical copy
  for `subtract_known` / `merge_delta`
- Hash table built on forward (key=col0) — so probing reverse `(b, a)` against
  forward hash table finds entries `(b, c)` and produces `(c, a)`, a reversed
  extended path. The reverse convention is self-consistent through the join.

**Key difference from normal TC:** The hash table must be rebuilt every iteration
because we are joining the relation with itself (squaring), so as `full` grows,
the hash table side grows too.

**Setup:**
- `full` = `gk_rev`, deduplicated — the growing relation (reverse)
- `delta` = copy of `full` — tuples to probe each iteration
- `hash_table` = built on `gk_fwd` (forward), then `gk_fwd` freed (hash table
  has its own allocation)

**Why two redistributions per iteration:** In standard TC, the hash table is on
a fixed relation (forward edges) partitioned once by col0. The probe side
(reverse) is partitioned by col1. The join matches probe.key (col1) against
hash_table.key (col0) — co-partitioned by the same hash function.

In squaring, we join `full` with itself, but the two sides need different
partitionings: the probe side needs partition by col1 (reverse key), the hash
table side needs partition by col0 (forward key). Since both derive from the
same `full`, we must repartition one of them. This means two `get_split_relation`
calls per iteration:
1. Redistribute join result (reverse) by col1 — for probe / subtract / merge
2. Redistribute full_fwd (forward) by col0 — for hash table

**Loop:**

```
while true:
    join_result  = get_local_join(hash_table, delta)
    cudaFree(delta)

    // Redistribute by col1 (reverse key) for probe side
    delta        = get_split_relation(join_result)
    cudaFree(join_result)

    delta        = deduplicate(delta)
    delta        = subtract_known(delta, full)
    if delta is empty: break
    full         = merge_delta(full, delta)

    // Rebuild hash table: repartition by col0 (forward key)
    cudaFree(hash_table)
    full_fwd          = reverse_entity_ar(full)
    full_fwd_part     = get_split_relation(full_fwd)
    cudaFree(full_fwd)
    hash_table        = get_hash_table(full_fwd_part)
    cudaFree(full_fwd_part)

    iterations++
```

**After the loop:** `full` contains TC(g^k), and `hash_table` is built on
the forward version repartitioned by col0. Both are needed for Phase 3.

---

## Phase 3: Extend Seeds and Reduce

**Goal:** Each rank joins its seed (paths of length `r = rank+1`) with
TC(g^k) (all multiples of `k`) to produce paths of length `r+k, r+2k, r+3k,
...`

**Key design choice:** Rather than allgathering TC(g^k) to every rank (memory
heavy — every rank holds the full closure), we distribute the seeds instead.
Seeds are small (one path length per rank), while TC(g^k) can be large. TC(g^k)
is already partitioned from Phase 2, so we keep it in place and bring the seeds
to it.

**Partitioning correctness:** The join matches `seed.key` against TC(g^k) hash
table key (forward, col0). TC(g^k) is partitioned by `get_rank(col0)`.
`get_split_relation(seed)` routes by `get_rank(seed.key)`, landing seed tuples
on the rank that has the matching TC entries. The join is then fully local.

**Flow:**

```
// TC(g^k) hash table is kept from Phase 2 (built on forward, partitioned by col0)

// Distribute seeds to match TC(g^k) partition
seed_dist       = get_split_relation(seed)

// Single local join: extend seeds by all multiples of k
extended        = get_local_join(hash_table, seed_dist)
cudaFree(hash_table)

// Union seed + extended, then redistribute and dedup for final TC
result          = concat(seed_dist, extended)
result          = get_split_relation(result)
result          = deduplicate(result)
```

**What this produces:** The complete TC, partitioned across ranks. Union across
all `k` ranks covers every path length: rank 0 contributes lengths
`1, 1+k, 1+2k, ...`, rank 1 contributes `2, 2+k, 2+2k, ...`, etc.

**Why a single join suffices:** TC(g^k) contains paths of length `k, 2k, 3k,
...`. Composing a seed path of length `r` with any of these gives
`r+k, r+2k, r+3k, ...` — all in one join. Combined with the seed itself
(length r), this covers every path length assigned to that rank.

---

## Known Limitations

- **u32 size limit:** All sizes use `unsigned int` (max ~4.3B). The join
  internally uses 64-bit offsets (`get_local_join_ll` available for callers
  that need >4B results), but most callers use the u32 wrapper which aborts
  cleanly if the result exceeds u32.

- **thrust::unique >2B bug:** `thrust::unique` silently returns 0 for >2B
  elements. The `deduplicate` function works around this by splitting into
  two halves when size > 2B.

---

## Complexity Summary (PXJLog)

| Phase | Iterations | Communication |
|-------|-----------|---------------|
| Phase 1 | `log2(k)` | None (allgather once at start) |
| Phase 2 | `O(log(diameter/k))` | 2x `get_split_relation` per iteration |
| Phase 3 | 1 join | `get_split_relation` twice (seeds in, result out) |

The total iteration count is `log2(k) + O(log(diameter/k)) + 1`, compared to
`O(diameter)` for standard semi-naive TC. The win is fewer communication
rounds, especially on high-latency systems where MPI synchronization barriers
dominate. Phase 1's log2(k) joins are entirely comm-free.
