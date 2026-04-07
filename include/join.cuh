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

__global__ void get_join_result_entity(Entity* hash_table, int hash_table_size,
                                       Entity* t_delta,
                                       unsigned int t_delta_size,
                                       unsigned long long* offset,
                                       Entity* join_result) {
    unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= t_delta_size)
        return;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = index; i < t_delta_size; i += stride) {
        int key = t_delta[i].key;
        int value = t_delta[i].value;
        unsigned long long start_index = offset[i];
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
    checkCuda(cudaMalloc((void**)&join_result,
                         result_size_ll * sizeof(Entity)));
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
                        double* buffer_memory_clear_time = nullptr) {
    double _t = 0.0;
    unsigned int join_result_size = 0;
    Entity* join_result =
        get_local_join(grid_size, block_size, hash_table, hash_table_size,
                       probe, probe_size, &join_result_size, join_time);
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
