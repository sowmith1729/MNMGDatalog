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
    int iterations = 1;
    const char* input_file;
    int comm_method = 0;
    int job_run = 1;
    int cuda_aware_mpi = 0;
    int rand_range = 1000000;
    if (argc == 5) {
        input_file = argv[1];
        cuda_aware_mpi = atoi(argv[2]);
        comm_method = atoi(argv[3]);
        rand_range = atoi(argv[4]);
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
    string output_file = string(input_file) + "_singlejoin.bin";
    const char* output_file_name = output_file.c_str();
    int total_rows = atoi(input_file);

    int total_columns = 2;
    long long row_size = 0;
    int* local_data_host =
        parallel_generate(total_rank, rank, total_rows, total_columns,
                          rand_range, &row_size, &_t);

    long long local_count = row_size * total_columns;
    long long global_row_size = 0;
    if (total_rank == 1) {
        global_row_size = row_size;
    } else {
        MPI_Allreduce(&row_size, &global_row_size, 1, MPI_LONG_LONG_INT,
                      MPI_SUM, MPI_COMM_WORLD);
    }

    int* local_data_device;
    checkCuda(
        cudaMalloc((void**)&local_data_device, local_count * sizeof(int)));
    cudaMemcpy(local_data_device, local_data_host, local_count * sizeof(int),
               cudaMemcpyHostToDevice);
    Entity* local_data =
        make_entity_array(grid_size, block_size, local_data_device, row_size,
                          false);
    Entity* local_data_reverse =
        make_entity_array(grid_size, block_size, local_data_device, row_size,
                          true);

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
#ifdef DEBUG
    cout << "Rank: " << rank << ", input_relation_size: " << input_relation_size
         << endl;
#endif
    input_relation_size = deduplicate(input_relation, input_relation_size);
#ifdef DEBUG
    cout << "Rank: " << rank
         << ", input_relation_size after deduplication: " << input_relation_size
         << endl;
#endif

    int reverse_relation_size = 0;
    Entity* reverse_relation;
    if (total_rank == 1) {
        reverse_relation = local_data_reverse;
        reverse_relation_size = row_size;
    } else {
        reverse_relation = get_split_relation(
            rank, local_data_reverse, row_size, total_columns, total_rank,
            grid_size, block_size, cuda_aware_mpi, &reverse_relation_size,
            comm_method, &_t, &_t, &_t, iterations);
    }
#ifdef DEBUG
    cout << "Rank: " << rank
         << ", reverse_relation_size: " << reverse_relation_size << endl;
#endif
    reverse_relation_size = deduplicate(reverse_relation, reverse_relation_size);
#ifdef DEBUG
    cout << "Rank: " << rank << ", reverse_relation_size after deduplication: "
         << reverse_relation_size << endl;
#endif

    int hash_table_rows = 0;
    Entity* hash_table =
        get_hash_table(grid_size, block_size, input_relation,
                       input_relation_size, &hash_table_rows, &_t);
#ifdef DEBUG
    cout << "Rank: " << rank << ", hash_table_rows: " << hash_table_rows
         << endl;
#endif

    int distributed_join_result_size = 0;
    Entity* distributed_join_result =
        get_global_join(rank, total_rank, grid_size, block_size, hash_table,
                        hash_table_rows, reverse_relation, reverse_relation_size,
                        total_columns, cuda_aware_mpi, comm_method, iterations,
                        &distributed_join_result_size, &_t);
#ifdef DEBUG
    cout << "Rank: " << rank
         << ", distributed_join_result_size: " << distributed_join_result_size
         << endl;
#endif

    distributed_join_result_size =
        deduplicate(distributed_join_result, distributed_join_result_size);
#ifdef DEBUG
    cout << "Rank: " << rank
         << ", distributed_join_result_size after deduplication: "
         << distributed_join_result_size << endl;
#endif

    long long global_join_result_size = 0;
    long long distributed_join_result_size_temp = distributed_join_result_size;
    if (total_rank == 1) {
        global_join_result_size = distributed_join_result_size_temp;
    } else {
        MPI_Allreduce(&distributed_join_result_size_temp,
                      &global_join_result_size, 1, MPI_LONG_LONG_INT, MPI_SUM,
                      MPI_COMM_WORLD);
    }

    int* distributed_join_result_ar;
    checkCuda(
        cudaMalloc((void**)&distributed_join_result_ar,
                   distributed_join_result_size * total_columns * sizeof(int)));
    get_reverse_int_ar_from_entity_ar<<<grid_size, block_size>>>(
        distributed_join_result, distributed_join_result_size,
        distributed_join_result_ar);

    int* distributed_join_result_ar_host = (int*)malloc(
        distributed_join_result_size * total_columns * sizeof(int));
    cudaMemcpy(distributed_join_result_ar_host, distributed_join_result_ar,
               distributed_join_result_size * total_columns * sizeof(int),
               cudaMemcpyDeviceToHost);

    int* join_result_counts = (int*)calloc(total_rank, sizeof(int));
    MPI_Allgather(&distributed_join_result_size, 1, MPI_INT, join_result_counts,
                  1, MPI_INT, MPI_COMM_WORLD);

    int* join_result_displacements = (int*)calloc(total_rank, sizeof(int));
    for (i = 1; i < total_rank; i++) {
        join_result_displacements[i] =
            join_result_displacements[i - 1] +
            (join_result_counts[i - 1] * total_columns);
    }

    if (job_run == 0) {
        parallel_write(rank, total_rank, output_file_name,
                       distributed_join_result_ar_host,
                       join_result_displacements, total_columns,
                       distributed_join_result_size, &_t);
        cout << "Generated " << output_file_name << endl;
    }

    cudaFree(local_data_device);
    cudaFree(input_relation);
    cudaFree(local_data);
    cudaFree(local_data_reverse);
    cudaFree(distributed_join_result);
    cudaFree(distributed_join_result_ar);
    cudaFree(hash_table);

    free(distributed_join_result_ar_host);
    free(join_result_counts);
    free(join_result_displacements);
    free(local_data_host);

    MPI_Finalize();
}

int main(int argc, char** argv) {
    benchmark(argc, argv);
    return 0;
}
