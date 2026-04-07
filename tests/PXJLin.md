# PXJLin: PXJ with Linear Extension

PXJLin is an instantiation of PXJ where Phase 2 uses standard semi-naive
linear extension instead of repeated squaring.

## Overview

Given `k = total_rank` (must be power of 2), each rank `p` computes all paths
of length `p+1, p+1+k, p+1+2k, ...`. Three phases:

1. **Phase 1** (comm-free): Each rank computes its seed and `g^k` via parallel
   prefix — identical to PXJLog
2. **Phase 2** (distributed): TC of `g^k` via linear semi-naive extension
3. **Phase 3** (distributed): Extend seeds by TC(g^k), global reduce —
   uses allgather + flipped join to avoid large hash table and TC redistribution

---

## Phase 1: Parallel Prefix Seed Computation

Identical to PXJLog. See `PXJLog.md` for details.

**After Phase 1:**
- `seed` = paths of length `rank+1` (different per rank)
- `gk_fwd` = forward `g^k`, filtered to this rank's partition
- `gk_rev` = reverse `g^k`, filtered to this rank's partition

---

## Phase 2: TC of g^k via Linear Extension

**Key difference from PXJLog:** The hash table is built once on `gk_fwd` and
never rebuilt. Each iteration extends by `k` hops using this static hash table,
identical to the standard TC semi-naive loop.

**Setup:**
- `full` = `gk_rev`, deduplicated — the growing relation (reverse)
- `delta` = copy of `full` — tuples to probe each iteration
- `hash_table` = built on `gk_fwd` (forward) — **static, never rebuilt**

**Loop:**

```
while true:
    join_result  = get_local_join(hash_table, delta)     // local GPU only
    cudaFree(delta)

    delta        = get_split_relation(join_result)        // one comm per iter
    cudaFree(join_result)

    delta        = deduplicate(delta)
    delta        = subtract_known(delta, full)
    if delta_size > 0:
        full     = merge_delta(full, delta)

    fixpoint?    = get_total_size(full_size) unchanged → break
```

**Comparison with PXJLog Phase 2:**

| | PXJLog (squaring) | PXJLin (linear) |
|---|---|---|
| Hash table | Rebuilt every iteration | Built once |
| `get_split_relation` per iteration | 2 (delta + full_fwd) | 1 (delta only) |
| Iterations | O(log(diameter/k)) | O(diameter/k) |
| Per-iteration cost | Higher (HT rebuild + 2x comm) | Lower (no rebuild, 1x comm) |

PXJLin trades more iterations for cheaper iterations. It is better suited for
graphs with small diameter/k ratio, or when GPU memory is limited (no need to
hold both forward and reverse copies of the growing relation for HT rebuild).

**After the loop:** `full` contains TC(g^k) — all paths whose length is a
multiple of k — partitioned across ranks by `get_rank(col1)` (reverse key).

---

## Phase 3: Extend Seeds and Reduce

**Goal:** Each rank joins its seed (paths of length `r = rank+1`) with TC(g^k)
to produce paths of length `r+k, r+2k, r+3k, ...`

### The partitioning problem

TC(g^k) comes out of Phase 2 in reverse convention, hash-partitioned by col1.
The join between seeds and TC(g^k) matches on col0 (the value field in reverse
convention). This means TC(g^k) is **not** partitioned by the join column.

**Naive approach (PXJLog):** Build a large hash table on TC(g^k) forward
(repartitioned by col0). This requires redistributing all of TC(g^k) and
building a hash table on it — both expensive when TC(g^k) is large.

### Allgather + flipped join

Seeds are small (one path length per rank). Instead of moving TC(g^k) to
the seeds, we replicate seeds to every rank:

```
all_seeds       = MPI_Allgatherv(seed)          // tiny: ~total_edges total
seed_ht         = get_hash_table(all_seeds)     // tiny hash table

full_fwd        = reverse(full)                 // local, no comm
                                                // still on same ranks

extended        = get_local_join(seed_ht, full_fwd)   // fully local
                  // seed_ht(col1=b, col0=a) x full_fwd(col0=b, col1=c)
                  // output: (a, c) forward convention
```

**Why this works:** After allgather, every rank has all seeds in its hash
table. Each rank reverses its local TC(g^k) partition to forward `(col0=b,
col1=c)` and probes against the seed hash table on col0=b. The join is fully
local — no redistribution of TC(g^k) needed.

**Join semantics:** The join kernel matches `probe.key` against `ht.key`:
- HT entry: seed `(key=col1=b, value=col0=a)` — path `a→...→b`
- Probe: TC_fwd `(key=col0=b, value=col1=c)` — path `b→...→c`
- Match on `b`
- Output: `(ht.value=a, probe.value=c)` = `(col0=a, col1=c)` forward
- Reverse to `(col1=c, col0=a)` — path `a→...→c` in reverse convention

### Memory-efficient redistribution

The Phase 3 join can produce very large results (often >2B tuples per rank)
because each TC(g^k) tuple may match multiple seeds. The implementation
avoids holding multiple large copies simultaneously:

1. **Free `full` and `delta`** immediately after the join — TC(g^k) is no
   longer needed
2. **Reverse `extended` in-place** via `thrust::for_each` (swaps key↔value)
   — no separate `extended_rev` allocation needed
3. **Custom GPU→host→MPI→GPU split** for extended redistribution:
   - Sort extended by destination rank on GPU (in-place)
   - Copy to host, **free GPU buffer** (extended is now only on host)
   - Use `MPI_Alltoallv_c` (MPI-4 large-count) for redistribution
     (standard `MPI_Alltoallv` overflows `int` displacements at >2B tuples)
   - Allocate fresh GPU receive buffer and upload
   - This avoids having input + send_buffer + receive_buffer on GPU
     simultaneously (~3× the data size)
4. **Seed split uses library `get_split_relation`** — seeds are tiny

5. **Concat via `cudaMemcpy`** instead of kernel — handles >u32 sizes
6. **Dedup handles >2B elements** via split-unique workaround
   (thrust::unique has a bug that returns 0 for >2B elements)

### Communication cost

| Operation | Volume |
|-----------|--------|
| Allgather seeds | ~total_edges (trivial) |
| Extended redistribution | ~TC size × fan-out / k per rank |
| Seed redistribution | ~total_edges / k (trivial) |

The expensive TC(g^k) redistribution from the naive approach is eliminated
entirely. The extended redistribution is large but unavoidable — it's the
actual TC result being hash-partitioned for the final dedup.

---

## Known Limitations

- **u32 join result limit:** The join uses `unsigned int` for per-rank result
  size. With `get_local_join_ll`, the scan and write kernel use 64-bit offsets
  internally, so results >4B tuples are correctly computed. However, callers
  that use `get_local_join` (u32 wrapper) abort if the result exceeds u32.
  Phase 3 uses `get_local_join_ll` directly to avoid this limit.

- **thrust::unique >2B bug:** `thrust::unique` silently returns the begin
  iterator (0 unique elements) when called on >2B elements. Worked around
  by splitting into two halves, uniquifying each, then re-uniquifying the
  boundary.

- **MPI_Alltoallv_c required:** Phase 3's extended redistribution uses MPI-4
  `MPI_Alltoallv_c` for large counts/displacements. Requires an MPI
  implementation that supports MPI-4 (e.g., MPICH on Perlmutter).

---

## Complexity Summary (PXJLin)

| Phase | Iterations | Communication |
|-------|-----------|---------------|
| Phase 1 | `log2(k)` | Allgather once at start |
| Phase 2 | `O(diameter/k)` | 1x `get_split_relation` per iteration |
| Phase 3 | 1 join | Allgather seeds + custom split for extended + 1x `get_split_relation` for seeds |

Compared to PXJLog:
- Phase 2 has more iterations but each is cheaper (no HT rebuild, half the
  comm)
- Phase 3 avoids the large TC(g^k) redistribution and hash table build
