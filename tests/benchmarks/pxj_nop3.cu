#include "mnmg.cuh"

using namespace std;

void benchmark(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Barrier(MPI_COMM_WORLD);
    int total_rank, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &total_rank);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Output output;
    int device_id;
    int number_of_sm;
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    cudaSetDevice(rank % num_devices);
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount,
                           device_id);
    warm_up_kernel<<<1, 1>>>();
    int block_size, grid_size;
    block_size = 512;
    grid_size = 32 * number_of_sm;
    setlocale(LC_ALL, "");
    double _t = 0.0;

    int iterations = 0;
    const char* input_file;
    int comm_method = 0;
    int job_run = 0;
    int cuda_aware_mpi = 0;

    if (argc == 5) {
        input_file = argv[1];
        cuda_aware_mpi = atoi(argv[2]);
        comm_method = atoi(argv[3]);
        job_run = atoi(argv[4]);
    } else if (argc == 4) {
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
    // Same as PXJLin — every rank gets full g^k and its own seed.
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
    unsigned int g_size = total_rows;
    unsigned int l_size = total_rows;
    cudaFree(full_data_device);

    l_size = deduplicate(l, l_size);

    Entity* g_rev;
    unsigned int g_rev_size = l_size;
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
            unsigned int temp_l_size;
            Entity* temp_l =
                get_local_join(grid_size, block_size, g_hash_table,
                               g_hash_table_rows, l, l_size, &temp_l_size,
                               &_t);
            cudaFree(l);
            l_size = deduplicate(temp_l, temp_l_size);
            l = temp_l;
            l_power += g_power;
        }
        unsigned int temp_g_rev_size;
        Entity* temp_g_rev =
            get_local_join(grid_size, block_size, g_hash_table,
                           g_hash_table_rows, g_rev, g_rev_size,
                           &temp_g_rev_size, &_t);
        cudaFree(g_rev);
        g_rev = temp_g_rev;
        g_rev_size = deduplicate(g_rev, temp_g_rev_size);

        int* g_arr;
        checkCuda(cudaMalloc((void**)&g_arr, g_rev_size * 2 * sizeof(int)));
        reverse_t_full<<<grid_size, block_size>>>(g_arr, g_rev_size, g_rev);
        checkCuda(cudaDeviceSynchronize());
        cudaFree(g);
        cudaFree(g_hash_table);
        g = make_entity_array(grid_size, block_size, g_arr, g_rev_size, false);
        cudaFree(g_arr);
        g_size = g_rev_size;
        g_hash_table = get_hash_table(grid_size, block_size, g, g_size,
                                      &g_hash_table_rows, &_t);
        g_power += g_power;
    }

    // Keep full g^k (NOT partitioned) — each rank uses the entire g^k
    Entity* seed = l;
    unsigned int seed_size = l_size;

    double phase1_end = MPI_Wtime();
    double phase1_time = phase1_end - phase1_start;

    // ─── Phase 2: Comm-free TC of seed extended by g^k ──────────────────
    // Standard semi-naive, entirely local. No MPI in the loop.
    // HT on full g^k forward. Delta starts from seed (reverse).
    // Computes paths of length r+1, r+1+k, r+1+2k, ...
    double phase2_start = MPI_Wtime();
    double p2_join_time = 0.0;
    double p2_dedup_time = 0.0;
    double p2_subtract_time = 0.0;
    double p2_merge_time = 0.0;

    // HT already built on g^k forward from Phase 1 (g_hash_table)
    int ht_rows = g_hash_table_rows;
    Entity* hash_table = g_hash_table;

    // full and delta start from seed (reverse convention)
    // Seed is paths of length r+1. Dedup before starting.
    unsigned int full_size = seed_size;
    Entity* full;
    checkCuda(cudaMalloc((void**)&full,
                         (size_t)seed_size * sizeof(Entity)));
    cudaMemcpy(full, seed, (size_t)seed_size * sizeof(Entity),
               cudaMemcpyDeviceToDevice);
    full_size = deduplicate(full, full_size);

    unsigned int delta_size = full_size;
    Entity* delta;
    checkCuda(cudaMalloc((void**)&delta,
                         (size_t)full_size * sizeof(Entity)));
    cudaMemcpy(delta, full, (size_t)full_size * sizeof(Entity),
               cudaMemcpyDeviceToDevice);

    cudaFree(seed);
    cudaFree(g_rev);
    // Keep g (forward) alive — hash_table points into it? No, hash_table
    // is a separate allocation. Free g.
    cudaFree(g);

    while (true) {
        // Local join: delta (reverse) against g^k HT (forward)
        double t0 = MPI_Wtime();
        unsigned int join_result_size = 0;
        Entity* join_result =
            get_local_join(grid_size, block_size, hash_table, ht_rows,
                           delta, delta_size, &join_result_size, &_t);
        cudaFree(delta);
        double t1 = MPI_Wtime();
        p2_join_time += t1 - t0;

        // Dedup — no redistribution needed, fully local
        t0 = MPI_Wtime();
        delta_size = deduplicate(join_result, join_result_size);
        delta = join_result;
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

        // Fixpoint — local check, then global allreduce
        long long global_full_size = get_total_size(full_size, total_rank);
        iterations++;

        // Need to check if ANY rank still has new tuples
        unsigned int global_delta = 0;
        unsigned int local_delta = delta_size;
        MPI_Allreduce(&local_delta, &global_delta, 1, MPI_UNSIGNED,
                      MPI_SUM, MPI_COMM_WORLD);
        if (global_delta == 0) {
            break;
        }
    }

    cudaFree(hash_table);

    double phase2_end = MPI_Wtime();
    double phase2_time = phase2_end - phase2_start;

    // Ranks may have overlapping tuples — redistribute and dedup
    // to get the correct partitioned TC.
    double p3_start = MPI_Wtime();
    double buf_t3 = 0.0, comm_t3 = 0.0, clear_t3 = 0.0;
    unsigned int result_size = 0;
    Entity* result = get_split_relation(
        rank, full, full_size, total_columns, total_rank, grid_size,
        block_size, cuda_aware_mpi, &result_size, comm_method,
        &buf_t3, &comm_t3, &clear_t3, 0);
    cudaFree(full);
    full = result;
    full_size = deduplicate(full, result_size);
    double p3_time = MPI_Wtime() - p3_start;

    long long total_tc_size = get_total_size(full_size, total_rank);

    double total_time = phase1_time + phase2_time;
    double max_total_time;
    MPI_Allreduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    if (rank == 0) {
        output.input_rows = total_rows;
        output.total_rank = total_rank;
        output.iterations = iterations;
        output.output_size = total_tc_size;
        output.total_time = max_total_time;
        output.join_time = p2_join_time;
        output.deduplication_time = p2_dedup_time;
        output.merge_time = p2_merge_time + p2_subtract_time;
        output.hashtable_build_time = 0.0;
        output.communication_time = 0.0;
        output.buffer_preparation_time = 0.0;

        printf("# Input,# Process,# Iterations,# TC,Total Time,Join,"
               "Deduplication,Merge,Phase1,Phase2,Final Dedup\n");
        printf("%d,%d,%d,%lld,%.4lf,%.4lf,%.4lf,%.4lf,%.4lf,%.4lf,%.4lf\n",
               output.input_rows, output.total_rank, output.iterations,
               output.output_size, output.total_time,
               p2_join_time, p2_dedup_time,
               p2_merge_time + p2_subtract_time,
               phase1_time, phase2_time, p3_time);
    }

    cudaFree(full);
    cudaFree(delta);

    MPI_Finalize();
}

int main(int argc, char** argv) {
    benchmark(argc, argv);
    return 0;
}
