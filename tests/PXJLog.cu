#include "mnmg.cuh"

using namespace std;

struct not_my_partition {
    int rank, total_rank;
    __host__ __device__ bool operator()(const Entity& e) {
        return get_rank(e.key, total_rank) != rank;
    }
};

void benchmark(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Barrier(MPI_COMM_WORLD);
    int total_rank, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &total_rank);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int device_id;
    int number_of_sm;
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    cudaSetDevice(rank % num_devices);
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount,
                           device_id);
    int block_size, grid_size;
    block_size = 512;
    grid_size = 32 * number_of_sm;
    setlocale(LC_ALL, "");
    double _t = 0.0;

    int iterations = 0;
    const char* input_file;
    int comm_method = 0;
    int cuda_aware_mpi = 0;

    if (argc >= 4) {
        input_file = argv[1];
        cuda_aware_mpi = atoi(argv[2]);
        comm_method = atoi(argv[3]);
    } else if (argc == 3) {
        input_file = argv[1];
        cuda_aware_mpi = atoi(argv[2]);
    } else if (argc == 2) {
        input_file = argv[1];
    } else {
        input_file = "hipc_2019.bin";
    }

    int total_columns = 2;
    int row_size = 0;
    int total_rows = 0;

    // ─── Phase 1: Parallel prefix seed computation ───────────────────────
    double phase1_start = MPI_Wtime();

    int* local_data_host =
        parallel_read(rank, total_rank, input_file, total_columns, &row_size,
                      &total_rows, &_t);
    int local_count = row_size * total_columns;

    // Allgather full graph to every rank for comm-free Phase 1
    int* all_row_counts = (int*)malloc(total_rank * sizeof(int));
    MPI_Allgather(&row_size, 1, MPI_INT, all_row_counts, 1, MPI_INT,
                  MPI_COMM_WORLD);
    int* recv_counts = (int*)malloc(total_rank * sizeof(int));
    int* recv_displs = (int*)calloc(total_rank, sizeof(int));
    for (int r = 0; r < total_rank; r++) {
        recv_counts[r] = all_row_counts[r] * total_columns;
    }
    for (int r = 1; r < total_rank; r++) {
        recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];
    }
    int full_count = total_rows * total_columns;
    int* full_data_host = (int*)malloc(full_count * sizeof(int));
    MPI_Allgatherv(local_data_host, local_count, MPI_INT, full_data_host,
                   recv_counts, recv_displs, MPI_INT, MPI_COMM_WORLD);
    free(local_data_host);
    free(all_row_counts);
    free(recv_counts);
    free(recv_displs);

    int* full_data_device;
    checkCuda(
        cudaMalloc((void**)&full_data_device, full_count * sizeof(int)));
    cudaMemcpy(full_data_device, full_data_host, full_count * sizeof(int),
               cudaMemcpyHostToDevice);
    free(full_data_host);

    Entity* g = make_entity_array(grid_size, block_size, full_data_device,
                                  total_rows, false);
    Entity* l = make_entity_array(grid_size, block_size, full_data_device,
                                  total_rows, true);
    int g_size = total_rows;
    int l_size = total_rows;
    cudaFree(full_data_device);

    l_size = deduplicate(l, l_size);

    Entity* g_rev;
    int g_rev_size = l_size;
    checkCuda(cudaMalloc((void**)&g_rev, l_size * sizeof(Entity)));
    cudaMemcpy(g_rev, l, l_size * sizeof(Entity), cudaMemcpyDeviceToDevice);

    int g_hash_table_rows = 0;
    Entity* g_hash_table = get_hash_table(grid_size, block_size, g, g_size,
                                          &g_hash_table_rows, &_t);

    assert((total_rank > 0 && (total_rank & (total_rank - 1)) == 0) &&
           "total_rank should be power of 2");
    int l_power = 1;
    int g_power = 1;
    int log_steps = log2(total_rank);

    for (int step = 0; step < log_steps; step++) {
        if (rank & (1 << step)) {
            int temp_l_size;
            Entity* temp_l =
                get_local_join(grid_size, block_size, g_hash_table,
                               g_hash_table_rows, l, l_size, &temp_l_size, &_t);
            cudaFree(l);
            l_size = deduplicate(temp_l, temp_l_size);
            l = temp_l;
            l_power += g_power;
        }
        int temp_g_rev_size;
        Entity* temp_g_rev =
            get_local_join(grid_size, block_size, g_hash_table,
                           g_hash_table_rows, g_rev, g_rev_size, &temp_g_rev_size, &_t);

        cudaFree(g_rev);
        g_rev = temp_g_rev;
        g_rev_size = deduplicate(g_rev, temp_g_rev_size);

        int* g_arr;
        checkCuda(cudaMalloc((void**)&g_arr, g_rev_size * 2 * sizeof(int)));
        reverse_t_full<<<grid_size, block_size>>>(g_arr, g_rev_size, g_rev);
        cudaDeviceSynchronize();
        cudaFree(g);
        cudaFree(g_hash_table);
        g = make_entity_array(grid_size, block_size, g_arr, g_rev_size, false);
        cudaFree(g_arr);
        g_size = g_rev_size;
        g_hash_table = get_hash_table(grid_size, block_size, g, g_size,
                                      &g_hash_table_rows, &_t);
        g_power += g_power;
    }
    cudaFree(g_hash_table);
    Entity* seed = l;
    int seed_size = l_size;

    // Drop tuples not in this rank's partition
    not_my_partition pred = {rank, total_rank};
    Entity* g_end = thrust::remove_if(thrust::device, g, g + g_size, pred);
    int gk_fwd_size = g_end - g;
    Entity* gk_fwd = g;

    Entity* g_rev_end =
        thrust::remove_if(thrust::device, g_rev, g_rev + g_rev_size, pred);
    int gk_rev_size = g_rev_end - g_rev;
    Entity* gk_rev = g_rev;

    double phase1_end = MPI_Wtime();
    double phase1_time = phase1_end - phase1_start;
    printf("Rank %d Phase 1: %.4fs | seed_size=%d (length %d), g^k_size=%d\n",
           rank, phase1_time, seed_size, l_power, g_size);

    // ─── Phase 2: TC of g^k using squaring (Approach 1b) ────────────────
    double phase2_start = MPI_Wtime();
    double p2_join_time = 0.0;
    double p2_comm_time = 0.0;
    double p2_dedup_time = 0.0;
    double p2_subtract_time = 0.0;
    double p2_merge_time = 0.0;
    double p2_hashtable_time = 0.0;
    double p2_reverse_comm_time = 0.0;

    Entity* full = gk_rev;
    long long full_size = gk_rev_size;
    full_size = deduplicate(full, full_size);

    Entity* delta;
    int delta_size = full_size;
    checkCuda(cudaMalloc((void**)&delta, full_size * sizeof(Entity)));
    cudaMemcpy(delta, full, full_size * sizeof(Entity),
               cudaMemcpyDeviceToDevice);

    // Build initial hash table on forward g^k
    int ht_rows = 0;
    Entity* hash_table = get_hash_table(grid_size, block_size, gk_fwd,
                                        gk_fwd_size, &ht_rows,
                                        &p2_hashtable_time);
    cudaFree(gk_fwd);

    long long global_full_size = get_total_size(full_size, total_rank);

    while (true) {
        // Local join
        double t0 = MPI_Wtime();
        int join_result_size = 0;
        Entity* join_result =
            get_local_join(grid_size, block_size, hash_table, ht_rows,
                           delta, delta_size, &join_result_size, &_t);
        cudaFree(delta);
        double t1 = MPI_Wtime();
        p2_join_time += t1 - t0;

        // Redistribute join result (comm)
        double buf_t = 0.0, comm_t = 0.0, clear_t = 0.0;
        delta = get_split_relation(
            rank, join_result, join_result_size, total_columns, total_rank,
            grid_size, block_size, cuda_aware_mpi, &delta_size,
            comm_method, &buf_t, &comm_t, &clear_t, iterations);
        cudaFree(join_result);
        p2_comm_time += buf_t + comm_t + clear_t;

        // Deduplicate
        t0 = MPI_Wtime();
        delta_size = deduplicate(delta, delta_size);
        t1 = MPI_Wtime();
        p2_dedup_time += t1 - t0;

        // Subtract known
        t0 = MPI_Wtime();
        delta_size = subtract_known(delta, delta_size, full, full_size);
        t1 = MPI_Wtime();
        p2_subtract_time += t1 - t0;

        // Merge
        if (delta_size > 0) {
            t0 = MPI_Wtime();
            full = merge_delta(full, full_size, delta, delta_size,
                               &full_size);
            t1 = MPI_Wtime();
            p2_merge_time += t1 - t0;
        }

        // Rebuild hash table: reverse full, repartition by col0
        // NOTE: Currently repartitions ALL of full_fwd each iteration.
        // Optimization: only repartition the new delta (reversed), merge
        // into a persistent full_fwd_part. Same number of comms but much
        // less volume — only delta is shipped instead of all of full.
        t0 = MPI_Wtime();
        cudaFree(hash_table);
        Entity* full_fwd;
        checkCuda(
            cudaMalloc((void**)&full_fwd, full_size * sizeof(Entity)));
        reverse_entity_ar<<<grid_size, block_size>>>(full, full_size,
                                                     full_fwd);
        cudaDeviceSynchronize();
        t1 = MPI_Wtime();
        p2_hashtable_time += t1 - t0;

        buf_t = 0.0; comm_t = 0.0; clear_t = 0.0;
        int full_fwd_part_size = 0;
        Entity* full_fwd_part = get_split_relation(
            rank, full_fwd, full_size, total_columns, total_rank,
            grid_size, block_size, cuda_aware_mpi, &full_fwd_part_size,
            comm_method, &buf_t, &comm_t, &clear_t, iterations);
        cudaFree(full_fwd);
        p2_reverse_comm_time += buf_t + comm_t + clear_t;

        t0 = MPI_Wtime();
        hash_table = get_hash_table(grid_size, block_size, full_fwd_part,
                                    full_fwd_part_size, &ht_rows, &_t);
        cudaFree(full_fwd_part);
        t1 = MPI_Wtime();
        p2_hashtable_time += t1 - t0;

        long long old_global_full_size = global_full_size;
        global_full_size = get_total_size(full_size, total_rank);
        iterations++;
        if (old_global_full_size == global_full_size) {
            break;
        }
    }

    double phase2_end = MPI_Wtime();
    double phase2_time = phase2_end - phase2_start;
    long long global_tc_gk_size = get_total_size(full_size, total_rank);
    if (rank == 0) {
        printf("Phase 2: %.4fs | TC(g^k) = %lld tuples, %d iterations\n",
               phase2_time, global_tc_gk_size, iterations);
        printf("  Join: %.4fs | Dedup: %.4fs | Subtract: %.4fs | "
               "Merge: %.4fs | HT build: %.4fs\n",
               p2_join_time, p2_dedup_time, p2_subtract_time,
               p2_merge_time, p2_hashtable_time);
        printf("  Comm (delta): %.4fs | Comm (HT repartition): %.4fs | "
               "Total comm: %.4fs\n",
               p2_comm_time, p2_reverse_comm_time,
               p2_comm_time + p2_reverse_comm_time);
    }

    // ─── Phase 3: seed ⋈ TC(g^k), then global reduce/dedup ─────────────
    double phase3_start = MPI_Wtime();

    // Distribute seeds to match TC(g^k) partition
    int seed_dist_size = 0;
    Entity* seed_dist = get_split_relation(
        rank, seed, seed_size, total_columns, total_rank, grid_size,
        block_size, cuda_aware_mpi, &seed_dist_size, comm_method, &_t, &_t,
        &_t, 0);
    cudaFree(seed);

    // Extend seeds by TC(g^k): single local join
    int extended_size = 0;
    Entity* extended =
        get_local_join(grid_size, block_size, hash_table, ht_rows,
                       seed_dist, seed_dist_size, &extended_size, &_t);
    cudaFree(hash_table);

    // Concat seed + extended
    int combined_size = seed_dist_size + extended_size;
    Entity* combined;
    checkCuda(cudaMalloc((void**)&combined, combined_size * sizeof(Entity)));
    concat_entity_ar<<<grid_size, block_size>>>(
        seed_dist, seed_dist_size, extended, extended_size, combined,
        combined_size);
    cudaDeviceSynchronize();
    cudaFree(seed_dist);
    cudaFree(extended);

    // Final redistribution and dedup for complete TC
    int result_size = 0;
    Entity* result = get_split_relation(
        rank, combined, combined_size, total_columns, total_rank, grid_size,
        block_size, cuda_aware_mpi, &result_size, comm_method, &_t, &_t,
        &_t, 0);
    cudaFree(combined);
    result_size = deduplicate(result, result_size);

    double phase3_end = MPI_Wtime();
    double phase3_time = phase3_end - phase3_start;

    long long total_tc_size =
        get_total_size((long long)result_size, total_rank);

    double total_time = phase1_time + phase2_time + phase3_time;
    double max_total_time;
    MPI_Allreduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Phase 3: %.4fs\n", phase3_time);
        printf("─────────────────────────────────────\n");
        printf("Total: %.4fs | TC size: %lld\n", max_total_time, total_tc_size);
        printf("Phase 1: %.4fs | Phase 2: %.4fs | Phase 3: %.4fs\n",
               phase1_time, phase2_time, phase3_time);
    }

    cudaFree(result);
    cudaFree(full);
    cudaFree(delta);

    MPI_Finalize();
}

int main(int argc, char** argv) {
    benchmark(argc, argv);
    return 0;
}
