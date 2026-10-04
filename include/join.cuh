__global__ void get_join_result_size_entity(Entity* hash_table,
                                            int hash_table_size,
                                            Entity* t_delta,
                                            unsigned int t_delta_size,
                                            int* join_result_size) {
    unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= t_delta_size)
        return;

    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < t_delta_size; i += stride) {
        int key = t_delta[i].key;
        int current_size = 0;
        int position = get_position(key, hash_table_size);
        while (true) {
            if (hash_table[position].key == key) {
                current_size++;
            } else if (hash_table[position].key == -1) {
                break;
            }
            position = (position + 1) & (hash_table_size - 1);
        }
        join_result_size[i] = current_size;
    }
}

// `base_offset` lets a caller materialize one slice of a larger join: the
// offsets in `offset` stay absolute, but writes land at offset[i] -
// base_offset so a per-slice output buffer can be sized to that slice alone.
// Zero (the default) reproduces the whole-relation behaviour.
__global__ void get_join_result_entity(Entity* hash_table, int hash_table_size,
                                       Entity* t_delta,
                                       unsigned int t_delta_size,
                                       unsigned long long* offset,
                                       Entity* join_result,
                                       unsigned long long base_offset = 0) {
    unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= t_delta_size)
        return;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = index; i < t_delta_size; i += stride) {
        int key = t_delta[i].key;
        int value = t_delta[i].value;
        unsigned long long start_index = offset[i] - base_offset;
        int position = get_position(key, hash_table_size);
        while (true) {
            if (hash_table[position].key == key) {
                join_result[start_index].key = hash_table[position].value;
                join_result[start_index].value = value;
                start_index++;
            } else if (hash_table[position].key == -1) {
                break;
            }
            position = (position + 1) & (hash_table_size - 1);
        }
    }
}

// Core join implementation — returns result size as unsigned long long
Entity* get_local_join_ll(int grid_size, int block_size, Entity* hash_table,
                          int hash_table_size, Entity* relation,
                          unsigned int relation_size,
                          unsigned long long* join_result_size,
                          double* compute_time) {
    double start_time, end_time, elapsed_time;
    start_time = MPI_Wtime();
    Entity* join_result = nullptr;
    if (hash_table_size == 0 || relation_size == 0) {
        *join_result_size = 0;
        end_time = MPI_Wtime();
        elapsed_time = end_time - start_time;
        *compute_time = elapsed_time;
        return join_result;
    }
    // Count matches per probe row (int is fine — no single row overflows)
    int* join_count;
    checkCuda(cudaMalloc((void**)&join_count,
                         (size_t)relation_size * sizeof(int)));
    checkCuda(cudaMemset(join_count, 0, (size_t)relation_size * sizeof(int)));
    get_join_result_size_entity<<<grid_size, block_size>>>(
        hash_table, hash_table_size, relation, relation_size, join_count);
    checkCuda(cudaDeviceSynchronize());

    // Read last count before scan
    int last_count;
    cudaMemcpy(&last_count, join_count + relation_size - 1, sizeof(int),
               cudaMemcpyDeviceToHost);

    // Exclusive scan: int counts → unsigned long long offsets (no overflow)
    unsigned long long* join_offset;
    checkCuda(cudaMalloc((void**)&join_offset,
                         (size_t)relation_size * sizeof(unsigned long long)));
    thrust::exclusive_scan(thrust::device, join_count,
                           join_count + relation_size, join_offset, 0ULL,
                           thrust::plus<unsigned long long>());
    cudaFree(join_count);

    // Total = last offset + last count (64-bit, no overflow)
    unsigned long long last_offset;
    cudaMemcpy(&last_offset, join_offset + relation_size - 1,
               sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    unsigned long long result_size_ll = last_offset + last_count;

    // Allocate and materialize join result
    cudaError_t alloc_err = cudaMalloc((void**)&join_result,
                                       result_size_ll * sizeof(Entity));
    if (alloc_err != cudaSuccess) {
        size_t _free, _total;
        cudaMemGetInfo(&_free, &_total);
        fprintf(stderr,
                "OOM in join: need %llu tuples (%.1fGB), "
                "free=%.1fGB/%.1fGB\n",
                result_size_ll,
                result_size_ll * sizeof(Entity) / 1e9,
                _free / 1e9, _total / 1e9);
        fflush(stderr);
    }
    checkCuda(alloc_err);
    get_join_result_entity<<<grid_size, block_size>>>(
        hash_table, hash_table_size, relation, relation_size, join_offset,
        join_result);
    checkCuda(cudaDeviceSynchronize());
    cudaFree(join_offset);

    *join_result_size = result_size_ll;
    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;
    *compute_time = elapsed_time;
    return join_result;
}

// Wrapper that returns unsigned int — aborts if result exceeds u32
Entity* get_local_join(int grid_size, int block_size, Entity* hash_table,
                       int hash_table_size, Entity* relation,
                       unsigned int relation_size,
                       unsigned int* join_result_size,
                       double* compute_time) {
    unsigned long long result_ll = 0;
    Entity* result = get_local_join_ll(
        grid_size, block_size, hash_table, hash_table_size, relation,
        relation_size, &result_ll, compute_time);
    if (result_ll > UINT_MAX) {
        fprintf(stderr,
                "ERROR: Join result %llu tuples exceeds u32 max. "
                "Reduce GPU count or use a smaller graph.\n",
                result_ll);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    *join_result_size = (unsigned int)result_ll;
    return result;
}

__global__ void get_nl_join_result_size_entity(Entity* input_relation,
                                               int input_relation_size,
                                               Entity* t_delta,
                                               int t_delta_size,
                                               int* join_result_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < t_delta_size; i += stride) {
        int key = t_delta[i].key;
        int count = 0;
        // Binary search lower bound in input_relation for key
        int left = 0, right = input_relation_size;
        while (left < right) {
            int mid = (left + right) / 2;
            if (input_relation[mid].key < key)
                left = mid + 1;
            else
                right = mid;
        }
        int j = left;
        // Count matching keys
        while (j < input_relation_size && input_relation[j].key == key) {
            count++;
            j++;
        }
        join_result_size[i] = count;
    }
}

__global__ void get_nl_join_result_entity(Entity* input_relation,
                                          int input_relation_size,
                                          Entity* t_delta, int t_delta_size,
                                          int* offset, Entity* join_result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < t_delta_size; i += stride) {
        int key = t_delta[i].key;
        int t_value = t_delta[i].value;
        int pos = offset[i];

        // Binary search lower bound in input_relation for key
        int left = 0, right = input_relation_size;
        while (left < right) {
            int mid = (left + right) / 2;
            if (input_relation[mid].key < key)
                left = mid + 1;
            else
                right = mid;
        }
        int j = left;
        // Output matching pairs
        while (j < input_relation_size && input_relation[j].key == key) {
            join_result[pos].key = input_relation[j].value;
            join_result[pos].value = t_value;
            pos++;
            j++;
        }
    }
}

Entity* get_join_nl(int grid_size, int block_size, Entity* hash_table,
                    int hash_table_size, Entity* relation, int relation_size,
                    int* join_result_size, double* compute_time) {
    double start_time, end_time, elapsed_time;
    start_time = MPI_Wtime();
    Entity* join_result = nullptr;
    if (hash_table_size == 0 || relation_size == 0) {
        *join_result_size = 0;
        end_time = MPI_Wtime();
        elapsed_time = end_time - start_time;
        *compute_time = elapsed_time;
        return join_result;
    }
    int result_size;
    int* join_offset;
    checkCuda(cudaMalloc((void**)&join_offset, relation_size * sizeof(int)));
    checkCuda(cudaMemset(join_offset, 0, relation_size * sizeof(int)));

    get_nl_join_result_size_entity<<<grid_size, block_size>>>(
        hash_table, hash_table_size, relation, relation_size, join_offset);
    checkCuda(cudaDeviceSynchronize());

    result_size =
        thrust::reduce(thrust::device, join_offset, join_offset + relation_size,
                       0, thrust::plus<int>());

    thrust::exclusive_scan(thrust::device, join_offset,
                           join_offset + relation_size, join_offset);
#ifdef DEBUG
    cout << "result_size * sizeof(Entity): " << result_size * sizeof(Entity)
         << endl;
#endif
    checkCuda(cudaMalloc((void**)&join_result, result_size * sizeof(Entity)));
    get_nl_join_result_entity<<<grid_size, block_size>>>(
        hash_table, hash_table_size, relation, relation_size, join_offset,
        join_result);
    cudaFree(join_offset);
    *join_result_size = result_size;
    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;
    *compute_time = elapsed_time;
    return join_result;
}

unsigned int deduplicate(Entity* ar, unsigned int size,
                         double* time = nullptr) {
    double start = 0.0, end = 0.0;
    if (time)
        start = MPI_Wtime();
    thrust::sort(thrust::device, ar, ar + size, set_cmp());
    unsigned int new_size;
    if (size > 2000000000u) {
        // thrust::unique has a bug with >2B elements (returns begin
        // iterator). Work around by uniquifying two halves, then
        // compacting and uniquifying the boundary overlap.
        unsigned int half = size / 2;
        // Find a split point where ar[half-1] != ar[half] to avoid
        // cutting a run of duplicates (or just unique each half and
        // handle overlap at the boundary)
        Entity* mid_end =
            thrust::unique(thrust::device, ar, ar + half, is_equal());
        unsigned int first_half = mid_end - ar;
        Entity* second_end = thrust::unique(thrust::device, ar + half,
                                             ar + size, is_equal());
        unsigned int second_half = second_end - (ar + half);
        // Move second unique result right after first
        if (first_half < half) {
            cudaMemcpy(ar + first_half, ar + half,
                       (size_t)second_half * sizeof(Entity),
                       cudaMemcpyDeviceToDevice);
        }
        unsigned int combined = first_half + second_half;
        // Re-unique the boundary (last of first half may == first of
        // second half). The combined array is already sorted.
        Entity* final_end = thrust::unique(thrust::device, ar,
                                            ar + combined, is_equal());
        new_size = final_end - ar;
    } else {
        new_size =
            (thrust::unique(thrust::device, ar, ar + size, is_equal())) -
            ar;
    }
    if (time) {
        end = MPI_Wtime();
        *time += end - start;
    }
    return new_size;
}

// Join `probe` against `hash_table` in slices, deduplicating each slice and
// folding it into a running sorted set.
//
// get_local_join_ll materializes one tuple per matching (probe row, hash
// table row) pair. When many distinct paths connect the same endpoints --
// dense meshes, and PXJ Phase 3 in particular -- that multiset is orders of
// magnitude larger than the set of distinct pairs it reduces to, so it
// dominates peak memory and any redistribution that follows it.
//
// This variant caps the materialized slice at `budget` tuples and removes
// duplicates before the whole multiset ever exists. It is sound because
//     dedup(A U B) == dedup(dedup(A) U dedup(B))
// -- removing exact duplicates locally cannot change the final set. Callers
// that redistribute afterwards still need a post-redistribution dedup for
// duplicates that land on different ranks.
//
// Slice boundaries come from the offset array rather than from an even split
// of the probe: hot keys make the per-row match count very uneven, so equal
// probe slices would give wildly unequal output. Returns a sorted,
// duplicate-free result. `budget == 0` means one slice (no cap).
Entity* get_local_join_dedup(int grid_size, int block_size, Entity* hash_table,
                             int hash_table_size, Entity* probe,
                             unsigned int probe_size,
                             unsigned long long budget,
                             unsigned int* result_size,
                             unsigned long long* raw_size = nullptr,
                             double* compute_time = nullptr) {
    double start = MPI_Wtime();
    *result_size = 0;
    if (raw_size)
        *raw_size = 0;
    if (hash_table_size == 0 || probe_size == 0) {
        if (compute_time)
            *compute_time += MPI_Wtime() - start;
        return nullptr;
    }
    // deduplicate() takes an unsigned int, so a slice can never exceed that
    // regardless of what the caller asked for. 0 means "as large as is safe".
    if (budget == 0 || budget > 0xFFFFFFFFull)
        budget = 0xFFFFFFFFull;

    // Count matches per probe row, then exclusive-scan to absolute offsets.
    // Same as get_local_join_ll, but the offsets are kept so slice
    // boundaries can be found in them.
    int* join_count;
    checkCuda(cudaMalloc((void**)&join_count,
                         (size_t)probe_size * sizeof(int)));
    checkCuda(cudaMemset(join_count, 0, (size_t)probe_size * sizeof(int)));
    get_join_result_size_entity<<<grid_size, block_size>>>(
        hash_table, hash_table_size, probe, probe_size, join_count);
    checkCuda(cudaDeviceSynchronize());

    int last_count;
    cudaMemcpy(&last_count, join_count + probe_size - 1, sizeof(int),
               cudaMemcpyDeviceToHost);

    unsigned long long* join_offset;
    checkCuda(cudaMalloc((void**)&join_offset,
                         (size_t)probe_size * sizeof(unsigned long long)));
    thrust::exclusive_scan(thrust::device, join_count, join_count + probe_size,
                           join_offset, 0ULL,
                           thrust::plus<unsigned long long>());
    cudaFree(join_count);

    unsigned long long last_offset;
    cudaMemcpy(&last_offset, join_offset + probe_size - 1,
               sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    unsigned long long total = last_offset + last_count;
    if (raw_size)
        *raw_size = total;

    Entity* accum = nullptr;
    unsigned int accum_size = 0;
    unsigned int lo = 0;
    unsigned long long lo_off = 0;

    while (lo < probe_size) {
        // Find the largest slice whose output stays within budget. The
        // offsets are non-decreasing, so lower_bound gives the first probe
        // row whose output starts at or past the target.
        unsigned int hi;
        if (lo_off + budget >= total) {
            hi = probe_size;
        } else {
            unsigned long long target = lo_off + budget;
            hi = (unsigned int)(thrust::lower_bound(thrust::device, join_offset,
                                                    join_offset + probe_size,
                                                    target) -
                                join_offset);
            // A single probe row can exceed the budget on its own; always
            // make progress.
            if (hi <= lo)
                hi = lo + 1;
        }

        unsigned long long hi_off;
        if (hi >= probe_size) {
            hi_off = total;
        } else {
            cudaMemcpy(&hi_off, join_offset + hi, sizeof(unsigned long long),
                       cudaMemcpyDeviceToHost);
        }
        unsigned long long slice_out = hi_off - lo_off;

        if (slice_out > 0) {
            Entity* part;
            checkCuda(cudaMalloc((void**)&part,
                                 (size_t)slice_out * sizeof(Entity)));
            get_join_result_entity<<<grid_size, block_size>>>(
                hash_table, hash_table_size, probe + lo, hi - lo,
                join_offset + lo, part, lo_off);
            checkCuda(cudaDeviceSynchronize());
            checkCuda(cudaGetLastError());

            unsigned int part_size =
                deduplicate(part, (unsigned int)slice_out);

            if (accum == nullptr) {
                accum = part;
                accum_size = part_size;
            } else {
                // Both ranges are sorted and unique, so set_union merges and
                // drops duplicates in one pass.
                Entity* merged;
                checkCuda(cudaMalloc(
                    (void**)&merged,
                    (size_t)(accum_size + part_size) * sizeof(Entity)));
                unsigned int merged_size =
                    thrust::set_union(thrust::device, accum,
                                      accum + accum_size, part,
                                      part + part_size, merged, set_cmp()) -
                    merged;
                cudaFree(accum);
                cudaFree(part);
                accum = merged;
                accum_size = merged_size;
            }
        }

        lo = hi;
        lo_off = hi_off;
    }

    cudaFree(join_offset);
    *result_size = accum_size;
    if (compute_time)
        *compute_time += MPI_Wtime() - start;
    return accum;
}

unsigned int subtract_known(Entity* delta, unsigned int delta_size,
                            Entity* full, unsigned int full_size,
                            double* time = nullptr) {
    double start = 0.0, end = 0.0;
    if (time)
        start = MPI_Wtime();
    unsigned int new_size =
        thrust::set_difference(thrust::device, delta, delta + delta_size, full,
                               full + full_size, delta, set_cmp()) -
        delta;
    if (time) {
        end = MPI_Wtime();
        *time += end - start;
    }
    return new_size;
}

Entity* merge_delta(Entity* t_full, unsigned int t_full_size, Entity* t_delta,
                    unsigned int t_delta_size, unsigned int* new_size,
                    double* time = nullptr) {
    double start = 0.0, end = 0.0;
    if (time)
        start = MPI_Wtime();
    unsigned int merged_size = t_delta_size + t_full_size;
    Entity* merged;
    checkCuda(
        cudaMalloc((void**)&merged, (size_t)merged_size * sizeof(Entity)));
    thrust::merge(thrust::device, t_full, t_full + t_full_size, t_delta,
                  t_delta + t_delta_size, merged, set_cmp());
    cudaFree(t_full);
    *new_size = merged_size;
    if (time) {
        end = MPI_Wtime();
        *time += end - start;
    }
    return merged;
}

Entity* get_global_join(int rank, int total_rank, int grid_size, int block_size,
                        Entity* hash_table, int hash_table_size, Entity* probe,
                        unsigned int probe_size, int total_columns,
                        int cuda_aware_mpi, int comm_method, int iterations,
                        unsigned int* result_size, double* join_time,
                        double* buffer_preparation_time = nullptr,
                        double* communication_time = nullptr,
                        double* buffer_memory_clear_time = nullptr,
                        unsigned long long dedup_budget = 0) {
    double _t = 0.0;
    unsigned int join_result_size = 0;
    Entity* join_result;
    if (dedup_budget > 0) {
        // Collapse duplicate pairs before they are redistributed. The join
        // emits one tuple per matching (probe row, hash table row) pair, so
        // on graphs with many distinct paths between the same endpoints the
        // multiset handed to get_split_relation is far larger than the set
        // it reduces to. Callers still deduplicate after redistribution --
        // that is what removes duplicates living on different ranks.
        //
        // Applied at total_rank == 1 too: with nothing partitioned away the
        // multiset is at its largest there, and slicing is what keeps peak
        // memory bounded and every slice inside the u32 index limit. Single
        // GPU runs are the ones that overflow first, not last.
        join_result = get_local_join_dedup(
            grid_size, block_size, hash_table, hash_table_size, probe,
            probe_size, dedup_budget, &join_result_size, nullptr, join_time);
    } else {
        join_result =
            get_local_join(grid_size, block_size, hash_table, hash_table_size,
                           probe, probe_size, &join_result_size, join_time);
    }
    if (total_rank == 1) {
        *result_size = join_result_size;
        return join_result;
    }
    Entity* distributed = get_split_relation(
        rank, join_result, join_result_size, total_columns, total_rank,
        grid_size, block_size, cuda_aware_mpi, result_size, comm_method,
        buffer_preparation_time ? buffer_preparation_time : &_t,
        communication_time ? communication_time : &_t,
        buffer_memory_clear_time ? buffer_memory_clear_time : &_t, iterations);
    cudaFree(join_result);
    return distributed;
}
