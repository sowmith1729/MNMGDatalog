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
    printf("R%d [P1] parallel_read done: row_size=%d total_rows=%d\n",
           rank, row_size, total_rows);
    fflush(stdout);

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
    printf("R%d [P1] allgather done\n", rank);
    fflush(stdout);

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
    unsigned int g_size = total_rows;
    unsigned int l_size = total_rows;
    cudaFree(full_data_device);

    l_size = deduplicate(l, l_size);
    printf("R%d [P1] entity arrays + dedup done: g_size=%u l_size=%u\n",
           rank, g_size, l_size);
    fflush(stdout);

    Entity* g_rev;
    unsigned int g_rev_size = l_size;
    checkCuda(cudaMalloc((void**)&g_rev, l_size * sizeof(Entity)));
    cudaMemcpy(g_rev, l, l_size * sizeof(Entity), cudaMemcpyDeviceToDevice);

    int g_hash_table_rows = 0;
    Entity* g_hash_table = get_hash_table(grid_size, block_size, g, g_size,
                                          &g_hash_table_rows, &_t);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());
    printf("R%d [P1] initial HT built: ht_rows=%d\n", rank, g_hash_table_rows);
    fflush(stdout);

    assert((total_rank > 0 && (total_rank & (total_rank - 1)) == 0) &&
           "total_rank should be power of 2");
    int l_power = 1;
    int g_power = 1;
    int log_steps = log2(total_rank);

    for (int step = 0; step < log_steps; step++) {
        if (rank & (1 << step)) {
            unsigned int temp_l_size;
            Entity* temp_l =
                get_local_join(grid_size, block_size, g_hash_table,
                               g_hash_table_rows, l, l_size, &temp_l_size,
                               &_t);
            checkCuda(cudaDeviceSynchronize());
            checkCuda(cudaGetLastError());
            printf("R%d [P1] step %d: seed join done, temp_l_size=%u\n",
                   rank, step, temp_l_size);
            fflush(stdout);
            cudaFree(l);
            l_size = deduplicate(temp_l, temp_l_size);
            l = temp_l;
            l_power += g_power;
            printf("R%d [P1] step %d: seed dedup done, l_size=%u\n",
                   rank, step, l_size);
            fflush(stdout);
        }
        unsigned int temp_g_rev_size;
        Entity* temp_g_rev =
            get_local_join(grid_size, block_size, g_hash_table,
                           g_hash_table_rows, g_rev, g_rev_size,
                           &temp_g_rev_size, &_t);
        checkCuda(cudaDeviceSynchronize());
        checkCuda(cudaGetLastError());
        printf("R%d [P1] step %d: g_rev join done, temp_g_rev_size=%u\n",
               rank, step, temp_g_rev_size);
        fflush(stdout);

        cudaFree(g_rev);
        g_rev = temp_g_rev;
        g_rev_size = deduplicate(g_rev, temp_g_rev_size);
        printf("R%d [P1] step %d: g_rev dedup done, g_rev_size=%u\n",
               rank, step, g_rev_size);
        fflush(stdout);

        int* g_arr;
        checkCuda(cudaMalloc((void**)&g_arr, g_rev_size * 2 * sizeof(int)));
        reverse_t_full<<<grid_size, block_size>>>(g_arr, g_rev_size, g_rev);
        checkCuda(cudaDeviceSynchronize());
        checkCuda(cudaGetLastError());
        cudaFree(g);
        cudaFree(g_hash_table);
        g = make_entity_array(grid_size, block_size, g_arr, g_rev_size, false);
        cudaFree(g_arr);
        g_size = g_rev_size;
        g_hash_table = get_hash_table(grid_size, block_size, g, g_size,
                                      &g_hash_table_rows, &_t);
        checkCuda(cudaDeviceSynchronize());
        checkCuda(cudaGetLastError());
        printf("R%d [P1] step %d: g squared, g_size=%u ht_rows=%d\n",
               rank, step, g_size, g_hash_table_rows);
        fflush(stdout);
        g_power += g_power;
    }
    cudaFree(g_hash_table);
    Entity* seed = l;
    int seed_size = l_size;

    // Drop tuples not in this rank's partition
    not_my_partition pred = {rank, total_rank};
    Entity* g_end = thrust::remove_if(thrust::device, g, g + g_size, pred);
    unsigned int gk_fwd_size = g_end - g;
    Entity* gk_fwd = g;

    Entity* g_rev_end =
        thrust::remove_if(thrust::device, g_rev, g_rev + g_rev_size, pred);
    unsigned int gk_rev_size = g_rev_end - g_rev;
    Entity* gk_rev = g_rev;
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());
    printf("R%d [P1] partition filter done: gk_fwd=%u gk_rev=%u\n",
           rank, gk_fwd_size, gk_rev_size);
    fflush(stdout);

    double phase1_end = MPI_Wtime();
    double phase1_time = phase1_end - phase1_start;
    printf("Rank %d Phase 1: %.4fs | seed_size=%d (length %d), g^k_size=%u\n",
           rank, phase1_time, seed_size, l_power, g_size);
    fflush(stdout);

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
    unsigned int full_size = gk_rev_size;
    full_size = deduplicate(full, full_size);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());
    printf("R%d [P2] dedup full done: full_size=%u\n", rank, full_size);
    fflush(stdout);

    Entity* delta;
    unsigned int delta_size = full_size;
    checkCuda(
        cudaMalloc((void**)&delta, (size_t)full_size * sizeof(Entity)));
    cudaMemcpy(delta, full, (size_t)full_size * sizeof(Entity),
               cudaMemcpyDeviceToDevice);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());
    printf("R%d [P2] delta copy done: delta_size=%u\n", rank, delta_size);
    fflush(stdout);

    // Build initial hash table on forward g^k
    int ht_rows = 0;
    Entity* hash_table = get_hash_table(grid_size, block_size, gk_fwd,
                                        gk_fwd_size, &ht_rows,
                                        &p2_hashtable_time);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());
    printf("R%d [P2] HT built: gk_fwd_size=%u ht_rows=%d\n",
           rank, gk_fwd_size, ht_rows);
    fflush(stdout);
    cudaFree(gk_fwd);

    long long global_full_size = get_total_size(full_size, total_rank);
    printf("R%d [P2] entering loop: global_full_size=%lld\n",
           rank, global_full_size);
    fflush(stdout);

    // PXJ_P2_CHUNK_TUPLES / PXJ_P3_CHUNK_TUPLES: when > 0, join in slices of
    // at most that many tuples and collapse duplicate pairs locally before
    // redistributing. Unset/0 keeps the original behaviour (redistribute the
    // raw multiset, deduplicate after it lands). Squaring grows the hash
    // table side every iteration, so the multiset here is especially large.
    unsigned long long p2_chunk = 0;
    if (const char* env = getenv("PXJ_P2_CHUNK_TUPLES"))
        p2_chunk = strtoull(env, nullptr, 10);
    unsigned long long p3_chunk = 0;
    if (const char* env = getenv("PXJ_P3_CHUNK_TUPLES"))
        p3_chunk = strtoull(env, nullptr, 10);

    while (true) {
        // Local join
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        printf("R%d [P2] iter %d: PRE-JOIN GPU mem: %.1fGB free / %.1fGB total"
               " | delta_size=%u full_size=%u\n",
               rank, iterations, free_mem / 1e9, total_mem / 1e9,
               delta_size, full_size);
        fflush(stdout);
        double t0 = MPI_Wtime();
        unsigned int join_result_size = 0;
        unsigned long long join_raw = 0;
        Entity* join_result;
        if (p2_chunk > 0) {
            join_result = get_local_join_dedup(
                grid_size, block_size, hash_table, ht_rows, delta, delta_size,
                p2_chunk, &join_result_size, &join_raw, &_t);
        } else {
            join_result =
                get_local_join(grid_size, block_size, hash_table, ht_rows,
                               delta, delta_size, &join_result_size, &_t);
            join_raw = join_result_size;
        }
        checkCuda(cudaDeviceSynchronize());
        checkCuda(cudaGetLastError());
        printf("R%d [P2] iter %d: join done, raw=%llu result_size=%u\n",
               rank, iterations, join_raw, join_result_size);
        fflush(stdout);
        cudaFree(delta);
        double t1 = MPI_Wtime();
        p2_join_time += t1 - t0;

        // Redistribute join result (comm)
        double buf_t = 0.0, comm_t = 0.0, clear_t = 0.0;
        delta = get_split_relation(
            rank, join_result, join_result_size, total_columns, total_rank,
            grid_size, block_size, cuda_aware_mpi, &delta_size,
            comm_method, &buf_t, &comm_t, &clear_t, iterations);
        checkCuda(cudaDeviceSynchronize());
        checkCuda(cudaGetLastError());
        printf("R%d [P2] iter %d: split done, delta_size=%u\n",
               rank, iterations, delta_size);
        fflush(stdout);
        cudaFree(join_result);
        p2_comm_time += buf_t + comm_t + clear_t;

        // Deduplicate
        t0 = MPI_Wtime();
        delta_size = deduplicate(delta, delta_size);
        checkCuda(cudaDeviceSynchronize());
        checkCuda(cudaGetLastError());
        t1 = MPI_Wtime();
        p2_dedup_time += t1 - t0;
        printf("R%d [P2] iter %d: dedup done, delta_size=%u\n",
               rank, iterations, delta_size);
        fflush(stdout);

        // Subtract known
        t0 = MPI_Wtime();
        delta_size = subtract_known(delta, delta_size, full, full_size);
        checkCuda(cudaDeviceSynchronize());
        checkCuda(cudaGetLastError());
        t1 = MPI_Wtime();
        p2_subtract_time += t1 - t0;
        printf("R%d [P2] iter %d: subtract done, delta_size=%u\n",
               rank, iterations, delta_size);
        fflush(stdout);

        // Merge
        if (delta_size > 0) {
            t0 = MPI_Wtime();
            full = merge_delta(full, full_size, delta, delta_size,
                               &full_size);
            checkCuda(cudaDeviceSynchronize());
            checkCuda(cudaGetLastError());
            t1 = MPI_Wtime();
            p2_merge_time += t1 - t0;
            printf("R%d [P2] iter %d: merge done, full_size=%u\n",
                   rank, iterations, full_size);
            fflush(stdout);
        }

        // Rebuild hash table: reverse full, repartition by col0
        t0 = MPI_Wtime();
        cudaFree(hash_table);
        Entity* full_fwd;
        checkCuda(cudaMalloc((void**)&full_fwd,
                             (size_t)full_size * sizeof(Entity)));
        reverse_entity_ar<<<grid_size, block_size>>>(full, full_size,
                                                     full_fwd);
        checkCuda(cudaDeviceSynchronize());
        checkCuda(cudaGetLastError());
        t1 = MPI_Wtime();
        p2_hashtable_time += t1 - t0;

        buf_t = 0.0; comm_t = 0.0; clear_t = 0.0;
        unsigned int full_fwd_part_size = 0;
        Entity* full_fwd_part = get_split_relation(
            rank, full_fwd, full_size, total_columns, total_rank,
            grid_size, block_size, cuda_aware_mpi, &full_fwd_part_size,
            comm_method, &buf_t, &comm_t, &clear_t, iterations);
        cudaFree(full_fwd);
        p2_reverse_comm_time += buf_t + comm_t + clear_t;
        printf("R%d [P2] iter %d: HT repartition done, fwd_part_size=%u\n",
               rank, iterations, full_fwd_part_size);
        fflush(stdout);

        t0 = MPI_Wtime();
        hash_table = get_hash_table(grid_size, block_size, full_fwd_part,
                                    full_fwd_part_size, &ht_rows, &_t);
        cudaFree(full_fwd_part);
        checkCuda(cudaDeviceSynchronize());
        checkCuda(cudaGetLastError());
        t1 = MPI_Wtime();
        p2_hashtable_time += t1 - t0;
        printf("R%d [P2] iter %d: HT rebuilt, ht_rows=%d\n",
               rank, iterations, ht_rows);
        fflush(stdout);

        long long old_global_full_size = global_full_size;
        global_full_size = get_total_size(full_size, total_rank);
        iterations++;
        printf("R%d [P2] iter %d done: global_full_size=%lld\n",
               rank, iterations, global_full_size);
        fflush(stdout);
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
    // At 1 GPU: Phase 2's full IS the complete TC. Skip Phase 3.
    if (total_rank == 1) {
        double phase3_start = MPI_Wtime();
        unsigned int result_size = full_size;
        Entity* result = full;
        double phase3_end = MPI_Wtime();
        double phase3_time = phase3_end - phase3_start;
        long long total_tc_size = result_size;
        double total_time = phase1_time + phase2_time + phase3_time;
        printf("Phase 1: %.4fs | seed_size=%d (length %d), g^k_size=%u\n",
               phase1_time, seed_size, l_power, g_size);
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
        printf("Phase 3: %.4fs (skipped — single GPU)\n", phase3_time);
        printf("─────────────────────────────────────\n");
        printf("Total: %.4fs | TC size: %lld\n", total_time, total_tc_size);
        cudaFree(result);
        cudaFree(delta);
        cudaFree(hash_table);
        cudaFree(seed);
        MPI_Finalize();
        return;
    }

    printf("R%d [P3] entering Phase 3: seed_size=%d full_size=%u\n",
           rank, seed_size, full_size);
    fflush(stdout);
    double phase3_start = MPI_Wtime();

    // Distribute seeds to match TC(g^k) partition
    unsigned int seed_dist_size = 0;
    Entity* seed_dist = get_split_relation(
        rank, seed, seed_size, total_columns, total_rank, grid_size,
        block_size, cuda_aware_mpi, &seed_dist_size, comm_method, &_t, &_t,
        &_t, 0);
    cudaFree(seed);
    printf("R%d [P3] seed distributed: seed_dist_size=%u\n",
           rank, seed_dist_size);
    fflush(stdout);

    // Extend seeds by TC(g^k): single local join
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("R%d [P3] PRE-JOIN GPU mem: %.1fGB free / %.1fGB total | "
           "probe_size=%u ht_rows=%d\n",
           rank, free_mem / 1e9, total_mem / 1e9, seed_dist_size, ht_rows);
    fflush(stdout);
    unsigned int extended_size = 0;
    unsigned long long extended_raw = 0;
    Entity* extended;
    if (p3_chunk > 0) {
        extended = get_local_join_dedup(
            grid_size, block_size, hash_table, ht_rows, seed_dist,
            seed_dist_size, p3_chunk, &extended_size, &extended_raw, &_t);
    } else {
        extended = get_local_join(grid_size, block_size, hash_table, ht_rows,
                                  seed_dist, seed_dist_size, &extended_size,
                                  &_t);
        extended_raw = extended_size;
    }
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());
    cudaFree(hash_table);
    printf("R%d [P3] join raw=%llu -> kept=%u (%.1fx collapse)\n", rank,
           extended_raw, extended_size,
           extended_size ? (double)extended_raw / extended_size : 1.0);
    fflush(stdout);

    // Concat seed + extended
    unsigned int combined_size = seed_dist_size + extended_size;
    Entity* combined;
    checkCuda(cudaMalloc((void**)&combined,
                         (size_t)combined_size * sizeof(Entity)));
    concat_entity_ar<<<grid_size, block_size>>>(
        seed_dist, seed_dist_size, extended, extended_size, combined,
        combined_size);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());
    cudaFree(seed_dist);
    cudaFree(extended);
    printf("R%d [P3] concat done: combined_size=%u (seed_dist=%u + extended=%u)\n",
           rank, combined_size, seed_dist_size, extended_size);
    fflush(stdout);

    // Final redistribution and dedup for complete TC
    unsigned int result_size = 0;
    Entity* result = get_split_relation(
        rank, combined, combined_size, total_columns, total_rank, grid_size,
        block_size, cuda_aware_mpi, &result_size, comm_method, &_t, &_t,
        &_t, 0);
    cudaFree(combined);
    printf("R%d [P3] split done: result_size=%u\n", rank, result_size);
    fflush(stdout);

    result_size = deduplicate(result, result_size);
    printf("R%d [P3] dedup done: result_size=%u\n", rank, result_size);
    fflush(stdout);

    double phase3_end = MPI_Wtime();
    double phase3_time = phase3_end - phase3_start;

    long long total_tc_size =
        get_total_size((long long)result_size, total_rank);

    double total_time = phase1_time + phase2_time + phase3_time;
    double max_total_time;
    MPI_Allreduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Phase 1: %.4fs | seed_size=%d (length %d), g^k_size=%u\n",
               phase1_time, seed_size, l_power, g_size);
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
        printf("Phase 3: %.4fs\n", phase3_time);
        printf("─────────────────────────────────────\n");
        printf("Total: %.4fs | TC size: %lld\n", max_total_time, total_tc_size);
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
