# MNMGDatalog Library Reference

Header-only CUDA/MPI library for distributed Datalog evaluation across multiple GPUs. Include `mnmg.cuh` and nothing else.

## Include Order

```
mnmg.cuh
  -> system headers, MPI, Thrust
  -> utils.cuh           (data types, comparators, hash function, debug helpers)
  -> error_handler.cuh   (checkCuda macro)
  -> kernels.cuh         (CUDA kernels, make_entity_array)
  -> comm.cuh            (MPI redistribution, get_total_size)
  -> hash_table.cuh      (get_hash_table)
  -> join.cuh            (joins, dedup, set ops, merge)
  -> parallel_io.cuh     (parallel_read, parallel_write, parallel_generate)
  -> timestamp_util.h    (conditional, when ENABLE_TIMESTAMPS defined)
```

Order matters due to dependencies between headers. The `clang-format off` block in `mnmg.cuh` preserves it.

---

## Data Types

### `Entity`
```cpp
struct Entity {
    int key;
    int value;
};
```
Fundamental data unit. Stored on GPU, transmitted via MPI as `MPI_UINT64_T` (8 bytes = two `int`s). The "forward" entity has `(col0, col1)` as `(key, value)`; the "reverse" swaps them to `(col1, col0)`.

### `Output`
```cpp
struct Output { ... };
```
Holds benchmark results: grid/block sizes, input/output counts, and all timing breakdown fields. Used only in benchmark programs for CSV output.

### `KernelTimer`
```cpp
struct KernelTimer {
    void start_timer();
    void stop_timer();
    float get_spent_time();  // returns seconds
};
```
CUDA event-based timer. Wraps `cudaEventRecord`/`cudaEventElapsedTime`. Returns elapsed time in **seconds** (internally divides ms by 1000). Use for GPU kernel timing when `MPI_Wtime()` would not capture async kernel execution accurately.

### Comparators and Predicates

| Name | Type | Description |
|------|------|-------------|
| `set_cmp` | strict weak ordering | Sorts by `(key, value)` lexicographically. Used as the universal sort comparator. |
| `is_equal` | binary predicate | True if both `key` and `value` match. Used for full-tuple deduplication. |
| `is_equal_key` | binary predicate | True if `key` matches (ignores `value`). Used for key-only deduplication (WCC). |
| `is_key_equal_value` | unary predicate | True if `key == value`. Used to filter self-loops (SG base case). |
| `cmp` | comparator | Like `set_cmp` but returns `true` on equal elements (not strict). Avoid for `thrust::sort`; use `set_cmp` instead. |
| `minimum_by_value` | binary op | Returns the Entity with the smaller `value`. Used with `thrust::reduce_by_key` in WCC. |
| `minimum_value` | binary op | Returns the smaller `int`. |
| `get_key` | unary op | Extracts `key` from Entity. Used with `thrust::transform_iterator`. |

---

## Core Functions

### Hash Function

```cpp
__device__ int get_position(int key, int hash_table_row_size);
```
Murmur3-style mix: `key ^= key >> 16; key *= 0x85ebca6b; key ^= key >> 13; key *= 0xc2b2ae35; key ^= key >> 16`. Returns `key & (hash_table_row_size - 1)`. Requires power-of-2 table size.

```cpp
__host__ __device__ int get_rank(int key, int total_rank);
```
Same Murmur3 mix, returns `key % total_rank`. **Intentionally the same hash** as `get_position` so that co-partitioned data (same key -> same rank) always joins locally.

### Error Checking

```cpp
#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
```
Wraps any CUDA API call. Prints file/line on error and calls `exit()`.

---

## Entity Array Construction (`kernels.cuh`)

### `make_entity_array`
```cpp
Entity* make_entity_array(int grid_size, int block_size,
                          int* data_device, int row_size, bool reverse);
```
Allocates a GPU `Entity` array and populates it from a flat `int*` array of `(col0, col1)` pairs.

- **`data_device`** — device pointer to `row_size * 2` ints, laid out as `[col0_0, col1_0, col0_1, col1_1, ...]`
- **`reverse`** — if `false`: `key=col0, value=col1`; if `true`: `key=col1, value=col0`
- **Returns** — newly allocated `Entity*` on device. Caller must `cudaFree`.

### Conversion Kernels

| Kernel | Direction | Layout |
|--------|-----------|--------|
| `create_entity_ar` | `int* -> Entity*` | `key=data[2i], value=data[2i+1]` |
| `create_entity_ar_reverse` | `int* -> Entity*` | `key=data[2i+1], value=data[2i]` |
| `get_int_ar_from_entity_ar` | `Entity* -> int*` | `data[2i]=key, data[2i+1]=value` |
| `get_reverse_int_ar_from_entity_ar` | `Entity* -> int*` | `data[2i]=value, data[2i+1]=key` |
| `reverse_t_full` | `Entity* -> int*` | Same as `get_reverse_int_ar_from_entity_ar` |
| `reverse_entity_ar` | `Entity* -> Entity*` | Swaps key and value into a new array |
| `same_key_value_entity_ar` | `Entity* -> Entity*` | Output `key=input.key, value=input.key` (self-loops) |
| `replace_key_by_value` | `Entity* -> Entity*` | Output `key=input.value, value=0` |
| `duplicate_entity_with_reverse` | `Entity* -> Entity*` | 1 row -> 2 rows: original + reversed |
| `create_entity_ar_with_offset` | `int* -> Entity*` | Like `create_entity_ar` but writes at `output[i+offset]` |
| `concat_entity_ar` | `2x Entity* -> Entity*` | Concatenates two arrays into one |

All kernels use stride loops: `for (int i = index; i < n; i += stride)`.

---

## Hash Table (`hash_table.cuh`)

### `get_hash_table`
```cpp
Entity* get_hash_table(int grid_size, int block_size,
                       Entity* edge, int edge_size,
                       int* hash_table_size, double* compute_time);
```
Builds an open-addressing hash table on the GPU.

- **`edge`** — device `Entity*` array to index
- **`edge_size`** — number of entities
- **`hash_table_size`** [out] — actual table size (power-of-2, with 0.6 load factor)
- **`compute_time`** [out] — wall-clock time spent (seconds, via `MPI_Wtime()`)
- **Returns** — device `Entity*` hash table. Caller must `cudaFree`.

**Design:**
- Load factor: 0.6. Table size = next power-of-2 >= `ceil(edge_size / 0.6)`.
- Empty sentinel: `key == -1`. Table initialized with `thrust::fill` of `{-1, -1}`.
- Insertion: `atomicCAS` on key field as ownership gate. Linear probing on collision.
- Lookup: scan forward from `get_position(key)` until `key == -1`.
- Multiple rows with the same key occupy consecutive/nearby slots.

**Important:** The hash table is built **once** before the iteration loop and **never rebuilt**. It indexes the static base relation (forward edges).

---

## Join (`join.cuh`)

### `get_local_join`
```cpp
Entity* get_local_join(int grid_size, int block_size,
                       Entity* hash_table, int hash_table_size,
                       Entity* relation, int relation_size,
                       int* join_result_size, double* compute_time);
```
GPU-local hash join. Probes `relation` against `hash_table`.

- **Join semantics:** For each `relation[i]` with key `k` and value `v`, finds all hash table entries with key `k`. For each match with hash table value `w`, produces output `Entity{key=w, value=v}`.
- **`join_result_size`** [out] — number of result entities
- **`compute_time`** [out] — wall-clock time
- **Returns** — newly allocated result on device. Caller must `cudaFree`.

**Two-pass approach:**
1. `get_join_result_size_entity` — counts matches per probe row
2. `thrust::exclusive_scan` — computes write offsets
3. `get_join_result_entity` — materializes results

Returns `nullptr` and size 0 if `hash_table_size == 0`.

### `get_join_nl`
```cpp
Entity* get_join_nl(int grid_size, int block_size,
                    Entity* input_relation, int input_relation_size,
                    Entity* relation, int relation_size,
                    int* join_result_size, double* compute_time);
```
Nested-loop join with binary search. Same semantics as `get_local_join` but probes a **sorted** `Entity*` array instead of a hash table. **Requires `input_relation` to be sorted by key** (use `deduplicate` first, which sorts).

Same two-pass approach but uses binary search to find matching key ranges.

### `get_global_join`
```cpp
Entity* get_global_join(int rank, int total_rank,
                        int grid_size, int block_size,
                        Entity* hash_table, int hash_table_size,
                        Entity* probe, int probe_size,
                        int total_columns, int cuda_aware_mpi,
                        int comm_method, int iterations,
                        int* result_size, double* join_time,
                        double* buffer_preparation_time = nullptr,
                        double* communication_time = nullptr,
                        double* buffer_memory_clear_time = nullptr);
```
**Join + redistribute in one call.** Performs `get_local_join`, then if `total_rank > 1`, calls `get_split_relation` on the result and frees the intermediate join result.

- **Optional timing params** — if `nullptr`, internal throwaway `double _t` is used. If non-null, time is **accumulated** (`+=`) into the pointed-to value.
- **Returns** — redistributed result (or local result if single rank). Caller must `cudaFree`.

**When NOT to use:** If you need to transform the join result before redistribution (e.g., reversing entities in SG, or custom merge in WCC), use `get_local_join` + manual `get_split_relation` instead.

### `deduplicate`
```cpp
int deduplicate(Entity* ar, int size, double* time = nullptr);
```
Sorts by `(key, value)` using `set_cmp`, then removes consecutive duplicates using `is_equal`.

- **In-place** on `ar` (sorts the array, removes dupes at the tail)
- **`time`** [optional] — if non-null, elapsed time is accumulated (`+=`)
- **Returns** — new (smaller or equal) size

**Important:** Always call **after** `get_split_relation`, never before. Pre-redistribution duplicates are scattered across ranks and can't be locally resolved.

### `subtract_known`
```cpp
int subtract_known(Entity* delta, int delta_size,
                   Entity* full, long long full_size,
                   double* time = nullptr);
```
Set difference: removes from `delta` any entity present in `full`. Both must be **sorted** by `set_cmp` (which `deduplicate` ensures).

- **In-place** on `delta` — result written to front of `delta` array
- **Returns** — new size of delta after subtraction

### `merge_delta`
```cpp
Entity* merge_delta(Entity* t_full, long long t_full_size,
                    Entity* t_delta, int t_delta_size,
                    long long* new_size, double* time = nullptr);
```
Merges `t_full` and `t_delta` (both sorted by `set_cmp`) into a new allocation.

- **Frees `t_full`** internally. The returned pointer replaces it.
- **Does NOT free `t_delta`.**
- **Returns** — newly allocated merged array. Assign to `t_full`:
  ```cpp
  t_full = merge_delta(t_full, t_full_size, t_delta, t_delta_size, &t_full_size);
  ```

---

## Communication (`comm.cuh`)

### `get_split_relation`
```cpp
Entity* get_split_relation(int rank, Entity* data_device, int data_size,
                           int total_columns, int total_rank,
                           int grid_size, int block_size,
                           int cuda_aware_mpi, int* size, int method,
                           double* buffer_preparation_time,
                           double* communication_time,
                           double* buffer_memory_clear_time,
                           int iterations);
```
All-to-all redistribution. Routes each entity to rank `get_rank(entity.key, total_rank)` so that all entities with the same key end up on the same rank.

- **`method`** — `0` = two-pass (kernel-based), `1` = sort-based (thrust)
- **`cuda_aware_mpi`** — `0` = copy through host buffers, `1` = pass device pointers to MPI
- **`size`** [out] — number of entities received
- **Timing params** — **set** (not accumulated). The function writes the total prep/comm/clear time for this call.
- **`iterations`** — controls debug output (printed only on iteration 0)
- **Returns** — newly allocated `Entity*` on device. Caller must `cudaFree`.

**Two-pass method (0):**
1. `get_send_count` kernel counts per-rank destinations using `atomicAdd`
2. `thrust::exclusive_scan` computes displacements
3. `get_rank_data` kernel packs send buffer using `atomicAdd` on displacements
4. `MPI_Alltoall` exchanges counts
5. `MPI_Alltoallv` exchanges data

**Sort-based method (1):**
1. `thrust::transform` maps each entity to its destination rank
2. `thrust::stable_sort_by_key` groups by rank
3. `thrust::reduce_by_key` computes per-rank counts
4. `MPI_Alltoall` + `MPI_Alltoallv` as above

Both methods free all internal buffers before returning.

### `get_split_relation_sort_method` / `get_split_relation_pass_method`
Direct access to the two implementations. Same signature minus the `method` parameter. Prefer `get_split_relation` which dispatches.

### `get_total_size`
```cpp
long long get_total_size(long long local_size, int total_rank,
                         double* time = nullptr);
```
Returns the global sum of `local_size` across all ranks via `MPI_Allreduce`. If `total_rank == 1`, returns `local_size` directly. Used for fixpoint checks in iterative programs.

---

## Parallel I/O (`parallel_io.cuh`)

### `parallel_read`
```cpp
int* parallel_read(int rank, int total_rank, const char* input_file,
                   int total_columns, int* row_count,
                   int* total_rows_count, double* compute_time);
```
Reads a binary file in parallel. Each rank reads a contiguous chunk using `MPI_File_read_at`.

- **Binary format** — packed `int` pairs, 4 bytes each. File size / (sizeof(int) * total_columns) = total rows.
- **Partitioning** — uses `BLOCK_START`/`BLOCK_SIZE` macros for non-uniform distribution (larger blocks to earlier ranks when not evenly divisible).
- **`row_count`** [out] — rows read by this rank
- **`total_rows_count`** [out] — total rows in file
- **Returns** — host `int*` array of size `row_count * total_columns`. Caller must `free`.

### `parallel_write`
```cpp
void parallel_write(int rank, int total_rank, const char* output_file_name,
                    int* ar_host, int* displacement, int total_columns,
                    int row_size, double* compute_time);
```
Writes results to a binary file in parallel. Each rank writes at its displacement offset using `MPI_File_write_at`.

- **`displacement`** — array of per-rank displacements (in number of ints), computed via `MPI_Allgather` of row counts.

### `parallel_generate`
```cpp
int* parallel_generate(int total_rank, int rank, int total_rows,
                       int total_columns, int rand_range,
                       long long* row_count, double* compute_time);
```
Generates random data on the host. Each rank produces `total_rows / total_rank` rows (if `total_rows > 10M`) or `total_rows` rows. Values are random in `[1, rand_range]`. Seed is `rank + 1` for reproducibility.

- **Returns** — host `int*`. Caller must `free`.

---

## Timestamp Utility (`timestamp_util.h`)

Conditionally included when `ENABLE_TIMESTAMPS` is defined (CMake: `-DENABLE_TIMESTAMPS=ON`).

```cpp
#define LOG_TIMESTAMP(label) ...
```
Prints `<nanoseconds>,<label>,<rank>` to stdout. Expects `rank` to be in scope as a local variable.

---

## Timing Convention

All timing parameters follow this pattern:
- Functions that **set** the time (assign, not accumulate): `get_split_relation`, `get_hash_table`, `get_local_join`, `get_join_nl`, `parallel_read`, `parallel_write`, `parallel_generate`.
- Functions that **accumulate** the time (`+=`): `deduplicate`, `subtract_known`, `merge_delta`, `get_total_size`, `get_global_join` (for its optional split-relation timing).
- Optional params (`double* time = nullptr`): when `nullptr`, no timing is recorded. Used in `src/` test files that pass `&_t` (throwaway) for required params and `nullptr` for optional ones.

---

## Common Initialization Pattern

Every program follows this boilerplate:

```cpp
MPI_Init(&argc, &argv);
MPI_Barrier(MPI_COMM_WORLD);
int total_rank, rank;
MPI_Comm_size(MPI_COMM_WORLD, &total_rank);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

int device_id, number_of_sm, num_devices;
cudaGetDeviceCount(&num_devices);
cudaSetDevice(rank % num_devices);        // one rank per GPU
cudaGetDevice(&device_id);
cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount, device_id);

int block_size = 512;
int grid_size = 32 * number_of_sm;
```

---

## Execution Patterns

### Transitive Closure (TC)

```
input_relation  = get_split_relation(forward_data)
t_delta         = get_split_relation(reverse_data)
t_delta         = deduplicate(t_delta)
t_full          = copy of t_delta
hash_table      = get_hash_table(input_relation)

while (fixpoint not reached):
    t_delta     = get_global_join(hash_table, t_delta)    // join + redistribute
    t_delta     = deduplicate(t_delta)
    t_delta     = subtract_known(t_delta, t_full)
    t_full      = merge_delta(t_full, t_delta)
    check fixpoint via get_total_size(t_full_size)
```

### Same Generation (SG)

Cannot use `get_global_join` because each iteration requires two joins with an intermediate reverse step:

```
while (fixpoint not reached):
    // Join 1: hash_table x t_delta
    join1       = get_local_join(hash_table, t_delta)
    join1_rev   = reverse_entity_ar(join1)                // swap key/value
    join1_dist  = get_split_relation(join1_rev)

    // Join 2: hash_table x join1_dist
    join2       = get_local_join(hash_table, join1_dist)
    t_delta     = get_split_relation(join2)
    t_delta     = deduplicate(t_delta)
    t_delta     = subtract_known(t_delta, t_full)
    t_full      = merge_delta(t_full, t_delta)
```

SG base case also filters self-loops with `thrust::remove_if(is_key_equal_value())`.

### Weakly Connected Components (WCC)

Uses `get_local_join` directly. Key differences from TC:
- Key-only deduplication (`is_equal_key` + `minimum_by_value`) instead of full-tuple dedup
- Custom merge: concatenate cc + join_result, sort, reduce_by_key to keep minimum value per key
- Set difference into a **new buffer** (not in-place) since the merge produces a new cc array
- Cannot use `merge_delta` or `subtract_known` due to different merge semantics
