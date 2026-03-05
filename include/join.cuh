__global__ void get_join_result_size_entity(Entity* hash_table,
                                            int hash_table_size,
                                            Entity* t_delta, int t_delta_size,
                                            int* join_result_size) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= t_delta_size)
        return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < t_delta_size; i += stride) {
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
                                       Entity* t_delta, int t_delta_size,
                                       int* offset, Entity* join_result) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= t_delta_size)
        return;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < t_delta_size; i += stride) {
        int key = t_delta[i].key;
        int value = t_delta[i].value;
        int start_index = offset[i];
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

Entity* get_local_join(int grid_size, int block_size, Entity* hash_table,
                       int hash_table_size, Entity* relation, int relation_size,
                       int* join_result_size, double* compute_time) {
    double start_time, end_time, elapsed_time;
    start_time = MPI_Wtime();
    Entity* join_result = nullptr;
    if (hash_table_size == 0) {
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

    get_join_result_size_entity<<<grid_size, block_size>>>(
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
    get_join_result_entity<<<grid_size, block_size>>>(
        hash_table, hash_table_size, relation, relation_size, join_offset,
        join_result);
    cudaFree(join_offset);
    *join_result_size = result_size;
    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;
    *compute_time = elapsed_time;
    return join_result;
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
    if (hash_table_size == 0) {
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

int deduplicate(Entity* ar, int size) {
    thrust::sort(thrust::device, ar, ar + size, set_cmp());
    return (thrust::unique(thrust::device, ar, ar + size, is_equal())) - ar;
}

int subtract_known(Entity* delta, int delta_size, Entity* full,
                   long long full_size) {
    return thrust::set_difference(thrust::device, delta, delta + delta_size,
                                  full, full + full_size, delta, set_cmp()) -
           delta;
}

Entity* merge_delta(Entity* t_full, long long t_full_size, Entity* t_delta,
                    int t_delta_size, long long* new_size) {
    long long merged_size = (long long)t_delta_size + t_full_size;
    Entity* merged;
    checkCuda(cudaMalloc((void**)&merged, merged_size * sizeof(Entity)));
    thrust::merge(thrust::device, t_full, t_full + t_full_size, t_delta,
                  t_delta + t_delta_size, merged, set_cmp());
    cudaFree(t_full);
    *new_size = merged_size;
    return merged;
}

Entity* get_global_join(int rank, int total_rank, int grid_size, int block_size,
                        Entity* hash_table, int hash_table_size, Entity* probe,
                        int probe_size, int total_columns, int cuda_aware_mpi,
                        int comm_method, int iterations, int* result_size,
                        double* compute_time) {
    int join_result_size = 0;
    Entity* join_result =
        get_local_join(grid_size, block_size, hash_table, hash_table_size,
                       probe, probe_size, &join_result_size, compute_time);
    if (total_rank == 1) {
        *result_size = join_result_size;
        return join_result;
    }
    Entity* distributed = get_split_relation(
        rank, join_result, join_result_size, total_columns, total_rank,
        grid_size, block_size, cuda_aware_mpi, result_size, comm_method,
        compute_time, compute_time, compute_time, iterations);
    cudaFree(join_result);
    return distributed;
}
