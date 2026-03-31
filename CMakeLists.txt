cmake_minimum_required(VERSION 3.18)
project(gpu_graph_algorithms LANGUAGES CUDA CXX)

# ─── Options ────────────────────────────────────────────────────────────────
option(ENABLE_TIMESTAMPS "Enable timestamp logging" ON)
option(ENABLE_DEBUG "Enable debug mode" OFF)

# ─── Global settings ───────────────────────────────────────────────────────
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

if(ENABLE_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -DDEBUG")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

if(ENABLE_TIMESTAMPS)
    add_compile_definitions(ENABLE_TIMESTAMPS)
endif()

# ─── Find MPI ──────────────────────────────────────────────────────────────
find_package(MPI REQUIRED)

# ─── Helper: define a CUDA+MPI target ─────────────────────────────────────
function(add_cuda_mpi_target target_name source_file)
    add_executable(${target_name} ${source_file})
    target_include_directories(${target_name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(${target_name} PRIVATE MPI::MPI_CXX m)
    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        OUTPUT_NAME "${target_name}.out"
    )
endfunction()

# ─── Targets ───────────────────────────────────────────────────────────────
add_cuda_mpi_target(tc          tests/benchmarks/tc.cu)
add_cuda_mpi_target(tc_nl       tests/benchmarks/tc_nl.cu)
add_cuda_mpi_target(sg          tests/benchmarks/sg.cu)
add_cuda_mpi_target(sg_nl       tests/benchmarks/sg_nl.cu)
add_cuda_mpi_target(wcc         tests/benchmarks/wcc.cu)
add_cuda_mpi_target(single_join tests/benchmarks/single_join.cu)


add_cuda_mpi_target(test_tc         tests/tc.cu)
add_cuda_mpi_target(test_join       tests/single_join.cu)
add_cuda_mpi_target(pxjlog          tests/PXJLog.cu)

# ─── Debug variant for wcc ─────────────────────────────────────────────────
add_cuda_mpi_target(wcc_debug tests/benchmarks/wcc.cu)
target_compile_definitions(wcc_debug PRIVATE DEBUG)

# ─── Cray support (Polaris, Perlmutter, etc.) ─────────────────────────────
option(ENABLE_CRAY "Link Cray GTL library for GPU-aware MPI" OFF)

if(ENABLE_CRAY)
    set(MPICH_GTL_LIB "/opt/cray/pe/mpich/8.1.28/gtl/lib/libmpi_gtl_cuda.so"
        CACHE FILEPATH "Path to MPICH GTL CUDA library")

    # Link GTL to all targets
    get_property(all_targets DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY BUILDSYSTEM_TARGETS)
    foreach(tgt ${all_targets})
        get_target_property(tgt_type ${tgt} TYPE)
        if(tgt_type STREQUAL "EXECUTABLE" AND EXISTS ${MPICH_GTL_LIB})
            target_link_libraries(${tgt} PRIVATE ${MPICH_GTL_LIB})
        endif()
    endforeach()
endif()

# ─── Clean helpers (optional convenience targets) ──────────────────────────
add_custom_target(cleanoutput
    COMMAND ${CMAKE_COMMAND} -E rm -f
        ${CMAKE_SOURCE_DIR}/data/*_tc.bin*
        ${CMAKE_SOURCE_DIR}/data/*_sg.bin*
        ${CMAKE_SOURCE_DIR}/data/*_cc.bin*
        ${CMAKE_SOURCE_DIR}/data/*_singlejoin.bin*
        ${CMAKE_SOURCE_DIR}/*_singlejoin.bin*
    COMMENT "Removing binary output files"
)
