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

    Entity* local_data = make_entity_array(grid_size, block_size,
                                           local_data_device, row_size, false);
    Entity* local_data_reverse = make_entity_array(
        grid_size, block_size, local_data_device, row_size, true);

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

    t_delta_size = deduplicate(t_delta, t_delta_size);

    Entity* t_full;
    checkCuda(cudaMalloc((void**)&t_full, t_delta_size * sizeof(Entity)));
    cudaMemcpy(t_full, t_delta, t_delta_size * sizeof(Entity),
               cudaMemcpyDeviceToDevice);
#ifdef DEBUG
    cout << "t_full initialization done" << endl;
#endif
    long long t_full_size = t_delta_size;
    long long global_t_full_size = get_total_size(t_full_size, total_rank);

    int hash_table_rows = 0;
    Entity* hash_table =
        get_hash_table(grid_size, block_size, input_relation,
                       input_relation_size, &hash_table_rows, &_t);

    while (true) {
        Entity* old_t_delta = t_delta;
        t_delta = get_global_join(rank, total_rank, grid_size, block_size,
                                  hash_table, hash_table_rows, old_t_delta,
                                  t_delta_size, total_columns, cuda_aware_mpi,
                                  comm_method, iterations, &t_delta_size, &_t);
        cudaFree(old_t_delta);

        t_delta_size = deduplicate(t_delta, t_delta_size);
        t_delta_size =
            subtract_known(t_delta, t_delta_size, t_full, t_full_size);
        t_full = merge_delta(t_full, t_full_size, t_delta, t_delta_size,
                             &t_full_size);

        long long old_global_t_full_size = global_t_full_size;
        global_t_full_size = get_total_size(t_full_size, total_rank);
        iterations++;
        if (rank == 0)
            printf("Number of global tuples in full is %ll\n",
                   global_t_full_size);
        if (old_global_t_full_size == global_t_full_size) {
            break;
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
    printf("the t_full_counts are %d\n", t_full_counts);
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
