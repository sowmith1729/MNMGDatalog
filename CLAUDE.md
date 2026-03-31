# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MNMGDatalog is a multi-node multi-GPU Datalog engine in CUDA/C++ with MPI. It evaluates Datalog programs using semi-naive evaluation distributed across GPUs. Each MPI rank maps to one GPU — parallelism is per-GPU, not per-node.

## Repository Structure

```
include/           # Header-only library (.cuh) — all shared logic lives here
tests/
  benchmarks/      # Full benchmark programs with timing instrumentation
  tc.cu            # Stripped test versions (no timing, no output formatting)
  single_join.cu
  PXJLog.cu
misc/
  binary_file_utils.py   # bin↔txt converter for data files
  graphs/                # Sample graph data (e.g. fe_body)
```

All test and benchmark files include a single umbrella header:
```cpp
#include "mnmg.cuh"
```

## Build

### CMake (preferred)

```bash
mkdir build && cd build
cmake ..
make tc          # Transitive Closure        (tests/benchmarks/tc.cu)
make sg          # Same Generation           (tests/benchmarks/sg.cu)
make wcc         # Weakly Connected Components (tests/benchmarks/wcc.cu)
make single_join # Single Join               (tests/benchmarks/single_join.cu)
make tc_nl       # TC with nested-loop join  (tests/benchmarks/tc_nl.cu)
make sg_nl       # SG with nested-loop join  (tests/benchmarks/sg_nl.cu)
make wcc_debug   # WCC with DEBUG macro
make test_tc     # Stripped TC test           (tests/tc.cu)
make test_join   # Stripped single-join test  (tests/single_join.cu)
```

CMake options:
- `-DENABLE_TIMESTAMPS=ON/OFF` (default ON)
- `-DENABLE_DEBUG=ON/OFF` (default OFF) — compiles with `-g -DDEBUG`
- `-DENABLE_POLARIS=ON/OFF` (default OFF) — links Cray GTL CUDA library

### Running

```bash
mpirun -np <NPROCS> ./<target>.out <data_file> <cuda_aware_mpi> <method> [job_run]
# cuda_aware_mpi: 0=traditional MPI, 1=CUDA-Aware-MPI
# method:         0=two-pass buffer prep, 1=sort-based buffer prep
# job_run:        0=write output file, 1=skip file write
```

`NPROCS` = number of GPUs. For local multi-GPU runs, `cudaSetDevice(rank % num_devices)` must be called after `MPI_Comm_rank` — the code does not do this automatically, so without it all ranks default to GPU 0.

### Makefile (legacy — references flat file paths, may not work with current layout)

```bash
make runtc DATA_FILE=data/data_7035.bin NPROCS=3 CUDA_AWARE_MPI=0 METHOD=0
make runsg DATA_FILE=data/data_7035.bin NPROCS=8 CUDA_AWARE_MPI=0 METHOD=0
make runwcc DATA_FILE=data/dummy.bin NPROCS=8 CUDA_AWARE_MPI=0 METHOD=0
make runsinglejoin DATA_FILE=10000000 NPROCS=4 CUDA_AWARE_MPI=0 METHOD=0 RAND_RANGE=1000000
```

## Library Modules (`include/`)

| File | Purpose |
|------|---------|
| `mnmg.cuh` | Umbrella header — include this and nothing else |
| `utils.cuh` | `Entity {int key, int value}`, `Output`, `KernelTimer`, hash functions, debug helpers |
| `error_handler.cuh` | `checkCuda()` macro |
| `kernels.cuh` | CUDA kernels: hash table build/probe, data routing (`get_send_count`, `get_rank_data`), entity array manipulation |
| `hash_table.cuh` | `get_hash_table()` — open-addressing GPU hash table, 0.6 load factor, power-of-2 size |
| `join.cuh` | `get_join()` (hash join), `get_join_nl()` (nested-loop/binary-search join) |
| `comm.cuh` | `get_split_relation()` — all-to-all redistribution, contains all MPI data movement |
| `parallel_io.cuh` | `parallel_read()`, `parallel_write()`, `parallel_generate()` |
| `timestamp_util.h` | `LOG_TIMESTAMP` macro, active when `ENABLE_TIMESTAMPS` is defined |

## Execution Pattern

Every benchmark follows the same structure. Pre-loop:

1. Read/generate data — each rank gets a contiguous chunk
2. Upload to GPU, reshape into `Entity` arrays (forward + reverse)
3. `get_split_relation` twice — distribute forward relation (hash table side) and reverse relation (probe side) so that for each rank, `hash(key) % total_rank == rank`
4. Deduplicate the probe side
5. Build hash table on the forward relation — **built once, never rebuilt**

Iteration loop (semi-naive):

```
join_result  = get_join(hash_table, t_delta)         // local GPU only
t_delta      = get_split_relation(join_result)        // all MPI here
               deduplicate(t_delta)                   // local, after redistribution
t_delta     -= set_difference(t_delta, t_full)        // remove known facts
t_full       = merge(t_full, t_delta)                 // local
fixpoint?    = MPI_Allreduce(t_full_size)             // only other MPI call
```

MPI outside `get_split_relation` is only: `MPI_Allreduce` for the fixpoint check each iteration, and `MPI_Allgather` once at the end for write offsets.

## Key Design Details

**Hash table** — open-addressing, linear probing, `key == -1` sentinel for empty slots. Multiple rows sharing the same key occupy consecutive slots (or nearby with gaps from other keys). Lookup scans forward until it hits a `-1`. The 0.6 load factor guarantees termination and short probe chains. Built with `atomicCAS` on the key field as an ownership gate.

**Data routing** — both `get_position` (hash table slot) and `get_rank` (MPI destination) use the same Murmur3-style mix: `key ^= key >> 16; key *= 0x85ebca6b; ...`. The hash table probes and the MPI routing are intentionally the same function so co-partitioned data always joins locally.

**`get_split_relation` methods**:
- Two-pass (METHOD=0): `get_send_count` kernel counts per-rank destinations, `get_rank_data` packs the send buffer
- Sort-based (METHOD=1): `thrust::stable_sort_by_key` by rank mapping, then `thrust::reduce_by_key` for counts
- Both call `MPI_Alltoall` (counts) then `MPI_Alltoallv` (data). CUDA-Aware MPI passes device pointers directly; traditional MPI copies through host.

**Deduplication** always happens *after* redistribution, never before. Pre-redistribution duplicates are scattered across ranks so they can't be locally resolved; after redistribution, all tuples with the same key are on the same rank.

**Data format** — binary files store `int` pairs (4 bytes each). Use `python3 misc/binary_file_utils.py txt_to_bin` / `bin_to_txt` to convert.

## Code Style

- `.clang-format`: LLVM style, 80-column limit, 4-space indent, left pointer alignment
- CUDA kernels use stride loops: `for (int i = index; i < n; i += stride)`
- Debug output gated behind `#ifdef DEBUG`
- `tests/` stripped versions use `double _t = 0.0` as a throwaway for function signature `double*` params that originally captured timing
