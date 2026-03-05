#include "mnmg.cuh"

using namespace std;

void benchmark(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Barrier(MPI_COMM_WORLD);
    int total_rank, rank;
    int i;
    MPI_Comm_size(MPI_COMM_WORLD, &total_rank);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int device_id;
    int number_of_sm;
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
    string output_file = string(input_file) + "_tc.bin";
    const char* output_file_name = output_file.c_str();

    int total_columns = 2;
    int row_size = 0;
    int total_rows = 0;
    int* local_data_host =
        parallel_read(rank, total_rank, input_file, total_columns, &row_size,
                      &total_rows, &_t);
    int local_count = row_size * total_columns;

    int* local_data_device;
    checkCuda(
        cudaMalloc((void**)&local_data_device, local_count * sizeof(int)));
    cudaMemcpy(local_data_device, local_data_host, local_count * sizeof(int),
               cudaMemcpyHostToDevice);
    Entity* local_data;
    checkCuda(cudaMalloc((void**)&local_data, row_size * sizeof(Entity)));
    Entity* local_data_reverse;
    checkCuda(
        cudaMalloc((void**)&local_data_reverse, row_size * sizeof(Entity)));
    create_entity_ar<<<grid_size, block_size>>>(local_data, row_size,
                                                local_data_device);
    create_entity_ar_reverse<<<grid_size, block_size>>>(
        local_data_reverse, row_size, local_data_device);

    int input_relation_size = 0;
    Entity* input_relation;
    if (total_rank == 1) {
        input_relation = local_data;
        input_relation_size = row_size;
    } else {
        input_relation = get_split_relation(
            rank, local_data, row_size, total_columns, total_rank, grid_size,
            block_size, cuda_aware_mpi, &input_relation_size, comm_method, &_t,
            &_t, &_t, iterations);
    }

    int t_delta_size = 0;
    Entity* t_delta;
    if (total_rank == 1) {
        t_delta = local_data_reverse;
        t_delta_size = row_size;
    } else {
        t_delta = get_split_relation(rank, local_data_reverse, row_size,
                                     total_columns, total_rank, grid_size,
                                     block_size, cuda_aware_mpi, &t_delta_size,
                                     comm_method, &_t, &_t, &_t, iterations);
    }

    thrust::sort(thrust::device, t_delta, t_delta + t_delta_size, set_cmp());
    t_delta_size = (thrust::unique(thrust::device, t_delta,
                                   t_delta + t_delta_size, is_equal())) -
                   t_delta;

    Entity* t_full;
    checkCuda(cudaMalloc((void**)&t_full, t_delta_size * sizeof(Entity)));
    cudaMemcpy(t_full, t_delta, t_delta_size * sizeof(Entity),
               cudaMemcpyDeviceToDevice);
#ifdef DEBUG
    cout << "t_full initialization done" << endl;
#endif
    long long global_t_full_size;
    long long t_full_size = t_delta_size;
    if (total_rank == 1) {
        global_t_full_size = t_full_size;
    } else {
        MPI_Allreduce(&t_full_size, &global_t_full_size, 1, MPI_LONG_LONG_INT,
                      MPI_SUM, MPI_COMM_WORLD);
    }

    int hash_table_rows = 0;
    Entity* hash_table =
        get_hash_table(grid_size, block_size, input_relation,
                       input_relation_size, &hash_table_rows, &_t);

    Entity* join_result;
    Entity* new_t_full;
    while (true) {
        int join_result_size = 0;
        join_result =
            get_join(grid_size, block_size, hash_table, hash_table_rows,
                     t_delta, t_delta_size, &join_result_size, &_t);

        cudaFree(t_delta);
        if (total_rank == 1) {
            t_delta = join_result;
            t_delta_size = join_result_size;
        } else {
            t_delta = get_split_relation(
                rank, join_result, join_result_size, total_columns, total_rank,
                grid_size, block_size, cuda_aware_mpi, &t_delta_size,
                comm_method, &_t, &_t, &_t, iterations);
        }

        thrust::sort(thrust::device, t_delta, t_delta + t_delta_size,
                     set_cmp());
        t_delta_size = (thrust::unique(thrust::device, t_delta,
                                       t_delta + t_delta_size, is_equal())) -
                       t_delta;

        t_delta_size = thrust::set_difference(
                           thrust::device, t_delta, t_delta + t_delta_size,
                           t_full, t_full + t_full_size, t_delta, set_cmp()) -
                       t_delta;

        long long new_t_full_size = t_delta_size + t_full_size;
        checkCuda(
            cudaMalloc((void**)&new_t_full, new_t_full_size * sizeof(Entity)));
        thrust::merge(thrust::device, t_full, t_full + t_full_size, t_delta,
                      t_delta + t_delta_size, new_t_full, set_cmp());

        cudaFree(t_full);
        t_full = new_t_full;
        t_full_size = new_t_full_size;

        if (total_rank == 1) {
            long long old_global_t_full_size = global_t_full_size;
            iterations++;
            global_t_full_size = t_full_size;
            if (old_global_t_full_size == t_full_size) {
                break;
            }
        } else {
            long long old_global_t_full_size = global_t_full_size;
            MPI_Allreduce(&t_full_size, &global_t_full_size, 1,
                          MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
            iterations++;
            if (old_global_t_full_size == global_t_full_size) {
                break;
            }
        }
    }

    int* t_full_ar;
    checkCuda(cudaMalloc((void**)&t_full_ar,
                         t_full_size * total_columns * sizeof(int)));
    reverse_t_full<<<grid_size, block_size>>>(t_full_ar, t_full_size, t_full);

    int* t_full_ar_host =
        (int*)malloc(t_full_size * total_columns * sizeof(int));
    cudaMemcpy(t_full_ar_host, t_full_ar,
               t_full_size * total_columns * sizeof(int),
               cudaMemcpyDeviceToHost);

    int* t_full_counts = (int*)calloc(total_rank, sizeof(int));
    MPI_Allgather(&t_full_size, 1, MPI_INT, t_full_counts, 1, MPI_INT,
                  MPI_COMM_WORLD);
    int* t_full_displacements = (int*)calloc(total_rank, sizeof(int));
    for (i = 1; i < total_rank; i++) {
        t_full_displacements[i] = t_full_displacements[i - 1] +
                                  (t_full_counts[i - 1] * total_columns);
    }

    if (job_run == 0) {
        parallel_write(rank, total_rank, output_file_name, t_full_ar_host,
                       t_full_displacements, total_columns, t_full_size, &_t);
        cout << "Rank " << rank
             << " wrote local tuples on file: " << output_file_name << endl;
    }

    cudaFree(local_data_device);
    cudaFree(input_relation);
    cudaFree(local_data);
    cudaFree(local_data_reverse);
    cudaFree(t_full);
    cudaFree(new_t_full);
    cudaFree(t_delta);
    cudaFree(t_full_ar);
    cudaFree(hash_table);

    free(t_full_ar_host);
    free(t_full_counts);
    free(t_full_displacements);
    free(local_data_host);

    MPI_Finalize();
}

int main(int argc, char** argv) {
    benchmark(argc, argv);
    return 0;
}
