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

    // ─── Phase 2: TC of g^k using linear extension ──────────────────────
    double phase2_start = MPI_Wtime();
    double p2_join_time = 0.0;
    double p2_comm_time = 0.0;
    double p2_dedup_time = 0.0;
    double p2_subtract_time = 0.0;
    double p2_merge_time = 0.0;
    double p2_hashtable_time = 0.0;

    Entity* full = gk_rev;
    unsigned int full_size = gk_rev_size;
    full_size = deduplicate(full, full_size);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());
    printf("R%d [P2] dedup full done: full_size=%u\n", rank, full_size);
    fflush(stdout);

    Entity* delta;
    unsigned int delta_size = full_size;
    checkCuda(cudaMalloc((void**)&delta, full_size * sizeof(Entity)));
    cudaMemcpy(delta, full, full_size * sizeof(Entity),
               cudaMemcpyDeviceToDevice);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());
    printf("R%d [P2] delta copy done: delta_size=%u\n", rank, delta_size);
    fflush(stdout);

    // Static hash table on g^k forward — built once, never rebuilt
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
        Entity* join_result =
            get_local_join(grid_size, block_size, hash_table, ht_rows,
                           delta, delta_size, &join_result_size, &_t);
        checkCuda(cudaDeviceSynchronize());
        checkCuda(cudaGetLastError());
        printf("R%d [P2] iter %d: join done, result_size=%u\n",
               rank, iterations, join_result_size);
        fflush(stdout);
        cudaFree(delta);
        double t1 = MPI_Wtime();
        p2_join_time += t1 - t0;

        // Redistribute join result by col1 (reverse key)
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

        // Fixpoint check
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

    cudaFree(hash_table);

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
        printf("  Comm: %.4fs\n", p2_comm_time);
    }

    // ─── Phase 3: seed x TC(g^k), then global reduce/dedup ─────────────
    // Allgather seeds to every rank (tiny), build HT on seeds, probe
    // with local TC(g^k) forward. Avoids redistributing TC(g^k).
    //
    // Why allgather works: TC(g^k) is already hash-partitioned by col1
    // from Phase 2. The join column is col0 (= value in reverse), which
    // is NOT what TC is partitioned by. Rather than repartitioning the
    // large TC(g^k), we replicate the small seeds to every rank so the
    // join is fully local.
    // Memory accounting at Phase 2/3 boundary
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("R%d [P2→P3] GPU mem: %.1fGB free / %.1fGB total\n",
           rank, free_mem / 1e9, total_mem / 1e9);
    printf("R%d [P2→P3] Live allocations:\n"
           "  full (TC(g^k) rev): %u tuples = %.1fGB\n"
           "  delta:              %u tuples = %.1fGB\n"
           "  seed:               %d tuples = %.1fGB\n",
           rank,
           full_size, (size_t)full_size * sizeof(Entity) / 1e9,
           delta_size, (size_t)delta_size * sizeof(Entity) / 1e9,
           seed_size, (size_t)seed_size * sizeof(Entity) / 1e9);
    printf("R%d [P2→P3] Accounted: %.1fGB | Unaccounted: %.1fGB\n",
           rank,
           ((size_t)full_size + delta_size + seed_size) * sizeof(Entity) / 1e9,
           (total_mem - free_mem) / 1e9 -
               ((size_t)full_size + delta_size + seed_size) * sizeof(Entity) / 1e9);
    fflush(stdout);
    printf("R%d [P3] entering Phase 3: seed_size=%d full_size=%u\n",
           rank, seed_size, full_size);
    fflush(stdout);
    double phase3_start = MPI_Wtime();
    double p3_comm_time = 0.0;
    double p3_join_time = 0.0;
    double p3_hashtable_time = 0.0;
    double p3_dedup_time = 0.0;
    double p3_kernel_time = 0.0;

    // Allgather seeds to every rank
    double t0 = MPI_Wtime();
    int* all_seed_counts = (int*)malloc(total_rank * sizeof(int));
    MPI_Allgather(&seed_size, 1, MPI_INT, all_seed_counts, 1, MPI_INT,
                  MPI_COMM_WORLD);
    int* seed_displs = (int*)calloc(total_rank, sizeof(int));
    int total_seed_size = 0;
    for (int r = 0; r < total_rank; r++) {
        seed_displs[r] = total_seed_size;
        total_seed_size += all_seed_counts[r];
    }
    printf("R%d [P3] allgather counts done: total_seed_size=%d\n",
           rank, total_seed_size);
    fflush(stdout);
    Entity* seed_host = (Entity*)malloc(seed_size * sizeof(Entity));
    cudaMemcpy(seed_host, seed, seed_size * sizeof(Entity),
               cudaMemcpyDeviceToHost);
    Entity* all_seeds_host =
        (Entity*)malloc(total_seed_size * sizeof(Entity));
    MPI_Allgatherv(seed_host, seed_size, MPI_UINT64_T, all_seeds_host,
                   all_seed_counts, seed_displs, MPI_UINT64_T,
                   MPI_COMM_WORLD);
    free(seed_host);
    free(all_seed_counts);
    free(seed_displs);
    printf("R%d [P3] allgatherv seeds done\n", rank);
    fflush(stdout);

    Entity* all_seeds_device;
    checkCuda(cudaMalloc((void**)&all_seeds_device,
                         total_seed_size * sizeof(Entity)));
    cudaMemcpy(all_seeds_device, all_seeds_host,
               total_seed_size * sizeof(Entity), cudaMemcpyHostToDevice);
    free(all_seeds_host);
    p3_comm_time += MPI_Wtime() - t0;

    // Build HT on all seeds (tiny — total across all ranks)
    t0 = MPI_Wtime();
    int seed_ht_rows = 0;
    Entity* seed_ht = get_hash_table(grid_size, block_size, all_seeds_device,
                                     total_seed_size, &seed_ht_rows, &_t);
    cudaFree(all_seeds_device);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());
    p3_hashtable_time += MPI_Wtime() - t0;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("R%d [P3] seed HT built: total_seed_size=%d seed_ht_rows=%d | "
           "GPU mem: %.1fGB free\n"
           "  Live: full=%.1fGB + delta=%.1fGB + seed=%.1fGB + "
           "seed_ht=%.1fGB = %.1fGB | used=%.1fGB\n",
           rank, total_seed_size, seed_ht_rows, free_mem / 1e9,
           (size_t)full_size * sizeof(Entity) / 1e9,
           (size_t)delta_size * sizeof(Entity) / 1e9,
           (size_t)seed_size * sizeof(Entity) / 1e9,
           (size_t)seed_ht_rows * sizeof(Entity) / 1e9,
           ((size_t)full_size + delta_size + seed_size + seed_ht_rows) *
               sizeof(Entity) / 1e9,
           (total_mem - free_mem) / 1e9);
    fflush(stdout);

    // Reverse local TC(g^k) to forward (no comm — data stays on rank)
    t0 = MPI_Wtime();
    Entity* full_fwd;
    checkCuda(
        cudaMalloc((void**)&full_fwd, (size_t)full_size * sizeof(Entity)));
    reverse_entity_ar<<<grid_size, block_size>>>(full, full_size, full_fwd);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());
    p3_kernel_time += MPI_Wtime() - t0;
    printf("R%d [P3] reversed full to full_fwd: full_size=%u\n",
           rank, full_size);
    fflush(stdout);

    // Local join: TC_fwd(col0=b, col1=c) against seed_HT(col1=b, col0=a)
    // Output: (seed.value, TC_fwd.value) = (col0=a, col1=c) forward
    t0 = MPI_Wtime();
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("R%d [P3] PRE-JOIN GPU mem: %.1fGB free / %.1fGB total | "
           "probe_size=%u seed_ht_rows=%d\n",
           rank, free_mem / 1e9, total_mem / 1e9, full_size, seed_ht_rows);
    fflush(stdout);
    unsigned long long extended_size = 0;
    Entity* extended =
        get_local_join_ll(grid_size, block_size, seed_ht, seed_ht_rows,
                          full_fwd, full_size, &extended_size, &_t);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());
    cudaFree(seed_ht);
    cudaFree(full_fwd);
    p3_join_time += MPI_Wtime() - t0;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("R%d [P3] join done: extended_size=%llu (%.1fGB) | "
           "GPU mem: %.1fGB free\n",
           rank, extended_size, extended_size * sizeof(Entity) / 1e9,
           free_mem / 1e9);
    fflush(stdout);

    // Free TC(g^k) and Phase 2 delta — no longer needed
    cudaFree(full);
    cudaFree(delta);


    // Reverse extended in-place: (col0=a, col1=c) → (col1=c, col0=a)
    t0 = MPI_Wtime();
    thrust::for_each(
        thrust::device, extended, extended + extended_size,
        [] __device__(Entity & e) {
            int tmp = e.key;
            e.key = e.value;
            e.value = tmp;
        });
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());

    p3_kernel_time += MPI_Wtime() - t0;

    // Custom memory-efficient split for extended: sort on GPU, copy to
    // host, FREE GPU buffer, then receive into fresh GPU buffer.
    // This avoids having extended + send_data + receive_data on GPU at once.
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("R%d [P3] PRE-SPLIT GPU mem: %.1fGB free / %.1fGB total | "
           "extended_size=%llu (%.1fGB)\n",
           rank, free_mem / 1e9, total_mem / 1e9,
           extended_size, extended_size * sizeof(Entity) / 1e9);
    fflush(stdout);

    t0 = MPI_Wtime();
    // Sort extended by destination rank on GPU (in-place)
    thrust::device_vector<uint8_t> rank_map(extended_size);
    thrust::transform(
        thrust::device, extended, extended + extended_size, rank_map.begin(),
        [total_rank = total_rank] __device__(const Entity& e) -> uint8_t {
            return (uint8_t)(get_rank(e.key, total_rank));
        });
    thrust::stable_sort_by_key(thrust::device, rank_map.begin(),
                               rank_map.end(), extended);

    // Compute send counts from sorted rank mapping (use long long to
    // handle per-rank counts > 2B)
    thrust::device_vector<long long> d_unique_counts(total_rank);
    thrust::device_vector<uint8_t> d_unique_ranks(total_rank);
    auto range_end = thrust::reduce_by_key(
        thrust::device, rank_map.begin(), rank_map.end(),
        thrust::constant_iterator<long long>(1), d_unique_ranks.begin(),
        d_unique_counts.begin());
    int num_unique = range_end.first - d_unique_ranks.begin();
    rank_map.clear();
    rank_map.shrink_to_fit();

    thrust::host_vector<uint8_t> h_unique_ranks(d_unique_ranks.begin(),
                                                 d_unique_ranks.begin() + num_unique);
    thrust::host_vector<long long> h_unique_counts(d_unique_counts.begin(),
                                                    d_unique_counts.begin() + num_unique);
    d_unique_counts.clear(); d_unique_counts.shrink_to_fit();
    d_unique_ranks.clear(); d_unique_ranks.shrink_to_fit();
    MPI_Count* send_count_host =
        (MPI_Count*)calloc(total_rank, sizeof(MPI_Count));
    for (int r = 0; r < num_unique; r++)
        send_count_host[h_unique_ranks[r]] = h_unique_counts[r];

    // Copy sorted extended to host, then FREE GPU buffer
    Entity* ext_host =
        (Entity*)malloc((size_t)extended_size * sizeof(Entity));
    cudaMemcpy(ext_host, extended, (size_t)extended_size * sizeof(Entity),
               cudaMemcpyDeviceToHost);
    cudaFree(extended);
    p3_kernel_time += MPI_Wtime() - t0;

    cudaMemGetInfo(&free_mem, &total_mem);
    printf("R%d [P3] extended sorted & copied to host, GPU freed: "
           "%.1fGB free\n", rank, free_mem / 1e9);
    fflush(stdout);

    // Exchange counts (MPI_Count = long long, use MPI_LONG_LONG for alltoall)
    t0 = MPI_Wtime();
    MPI_Count* recv_count_host =
        (MPI_Count*)calloc(total_rank, sizeof(MPI_Count));
    MPI_Alltoall(send_count_host, 1, MPI_LONG_LONG,
                 recv_count_host, 1, MPI_LONG_LONG, MPI_COMM_WORLD);

    size_t total_recv = 0;
    for (int r = 0; r < total_rank; r++)
        total_recv += recv_count_host[r];

    // MPI-4 large-count alltoallv: MPI_Count for counts, MPI_Aint for
    // displacements — handles per-rank counts and cumulative sums > 2B
    MPI_Aint* send_displs_c = (MPI_Aint*)calloc(total_rank, sizeof(MPI_Aint));
    MPI_Aint* recv_displs_c = (MPI_Aint*)calloc(total_rank, sizeof(MPI_Aint));
    for (int r = 1; r < total_rank; r++) {
        send_displs_c[r] = send_displs_c[r - 1] + send_count_host[r - 1];
        recv_displs_c[r] = recv_displs_c[r - 1] + recv_count_host[r - 1];
    }

    // Allocate receive on GPU (extended is freed — plenty of room now)
    Entity* ext_split;
    checkCuda(cudaMalloc((void**)&ext_split,
                         total_recv * sizeof(Entity)));
    Entity* recv_host =
        (Entity*)malloc(total_recv * sizeof(Entity));
    MPI_Alltoallv_c(ext_host, send_count_host, send_displs_c,
                    MPI_UINT64_T, recv_host, recv_count_host,
                    recv_displs_c, MPI_UINT64_T, MPI_COMM_WORLD);
    free(send_count_host);
    free(recv_count_host);
    free(send_displs_c);
    free(recv_displs_c);
    free(ext_host);
    cudaMemcpy(ext_split, recv_host, total_recv * sizeof(Entity),
               cudaMemcpyHostToDevice);
    free(recv_host);
    size_t ext_split_size = total_recv;
    p3_comm_time += MPI_Wtime() - t0;
    printf("R%d [P3] extended split done: ext_split_size=%zu\n",
           rank, ext_split_size);
    fflush(stdout);

    // Split seed (tiny — use library function)
    double buf_t3 = 0.0, comm_t3 = 0.0, clear_t3 = 0.0;
    unsigned int seed_split_size = 0;
    Entity* seed_split = get_split_relation(
        rank, seed, seed_size, total_columns, total_rank, grid_size,
        block_size, cuda_aware_mpi, &seed_split_size, comm_method,
        &buf_t3, &comm_t3, &clear_t3, 0);
    cudaFree(seed);
    p3_comm_time += buf_t3 + comm_t3 + clear_t3;
    printf("R%d [P3] seed split done: seed_split_size=%u\n",
           rank, seed_split_size);
    fflush(stdout);

    // Concat redistributed results locally
    t0 = MPI_Wtime();
    size_t result_size_ll = ext_split_size + seed_split_size;
    Entity* result;
    checkCuda(cudaMalloc((void**)&result,
                         result_size_ll * sizeof(Entity)));
    // Copy seed_split then ext_split into result (concat via memcpy for
    // large sizes — concat_entity_ar kernel uses unsigned int bounds)
    cudaMemcpy(result, seed_split,
               (size_t)seed_split_size * sizeof(Entity),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(result + seed_split_size, ext_split,
               ext_split_size * sizeof(Entity),
               cudaMemcpyDeviceToDevice);
    checkCuda(cudaDeviceSynchronize());
    cudaFree(seed_split);
    cudaFree(ext_split);
    printf("R%d [P3] concat done: result_size=%zu (seed=%u + ext=%zu)\n",
           rank, result_size_ll, seed_split_size, ext_split_size);
    fflush(stdout);

    // Dedup handles >2B via split-unique workaround
    unsigned int result_size;
    if (result_size_ll > UINT_MAX) {
        // Sort the full range (thrust::sort handles >2B)
        thrust::sort(thrust::device, result, result + result_size_ll,
                     set_cmp());
        // Unique in two halves (thrust::unique bug at >2B)
        unsigned int half = (unsigned int)(result_size_ll / 2);
        Entity* mid_end =
            thrust::unique(thrust::device, result, result + half,
                           is_equal());
        unsigned int first_half = mid_end - result;
        Entity* second_end = thrust::unique(
            thrust::device, result + half,
            result + result_size_ll, is_equal());
        unsigned int second_half = second_end - (result + half);
        if (first_half < half) {
            cudaMemcpy(result + first_half, result + half,
                       (size_t)second_half * sizeof(Entity),
                       cudaMemcpyDeviceToDevice);
        }
        unsigned int combined = first_half + second_half;
        Entity* final_end = thrust::unique(thrust::device, result,
                                            result + combined, is_equal());
        result_size = final_end - result;
    } else {
        result_size = deduplicate(result, (unsigned int)result_size_ll);
    }
    p3_dedup_time += MPI_Wtime() - t0;
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
        printf("  Comm: %.4fs\n", p2_comm_time);
        printf("Phase 3: %.4fs\n", phase3_time);
        printf("  Join: %.4fs | HT build: %.4fs | Comm: %.4fs | "
               "Dedup: %.4fs | Kernels: %.4fs\n",
               p3_join_time, p3_hashtable_time, p3_comm_time,
               p3_dedup_time, p3_kernel_time);
        printf("─────────────────────────────────────\n");
        printf("Total: %.4fs | TC size: %lld\n", max_total_time, total_tc_size);
    }

    cudaFree(result);

    MPI_Finalize();
}

int main(int argc, char** argv) {
    benchmark(argc, argv);
    return 0;
}
