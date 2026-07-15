cmake_minimum_required(VERSION 3.24)

project(cutlass_learning_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# This workstation is an Ada GPU (SM 8.9).  The scripts override this value
# after detecting the active device, so another machine can still reuse the lab.
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures" FORCE)
endif()

set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass" CACHE PATH
    "Path to a CUTLASS v4.5.3 checkout")

if(NOT EXISTS "${CUTLASS_DIR}/include/cutlass/version.h")
  message(FATAL_ERROR
    "CUTLASS was not found at ${CUTLASS_DIR}. Run scripts/bootstrap.sh first, "
    "or pass -DCUTLASS_DIR=/path/to/cutlass.")
endif()

file(READ "${CUTLASS_DIR}/include/cutlass/version.h" CUTLASS_VERSION_HEADER)
if(NOT CUTLASS_VERSION_HEADER MATCHES "#define CUTLASS_MAJOR 4" OR
   NOT CUTLASS_VERSION_HEADER MATCHES "#define CUTLASS_MINOR 5" OR
   NOT CUTLASS_VERSION_HEADER MATCHES "#define CUTLASS_PATCH 3")
  message(FATAL_ERROR "This lab is pinned to CUTLASS v4.5.3: ${CUTLASS_DIR}")
endif()

find_package(CUDAToolkit 11.4 REQUIRED)

add_library(lab_build_options INTERFACE)
target_include_directories(lab_build_options INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/common"
  "${CUTLASS_DIR}/include"
)
target_compile_options(lab_build_options INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-lineinfo>
)
target_link_libraries(lab_build_options INTERFACE CUDA::cudart)

function(add_cuda_lab target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE lab_build_options)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
endfunction()

add_cuda_lab(device_info 00_device_info/main.cu)
add_cuda_lab(vector_add 01_vector_add/main.cu)
add_cuda_lab(vector_add_advanced 01_vector_add/advanced.cu)
add_cuda_lab(tiled_gemm 02_tiled_gemm/main.cu)
add_cuda_lab(cutlass_sgemm 03_cutlass_sgemm/main.cu)
add_cuda_lab(cutlass_tensorop 03_cutlass_sgemm/tensorop.cu)
add_cuda_lab(cute_layout 04_cute_layout/main.cu)
target_link_libraries(cutlass_sgemm PRIVATE CUDA::cublas)
target_link_libraries(cutlass_tensorop PRIVATE CUDA::cublas)
if(TARGET CUDA::nvtx3)
  target_link_libraries(vector_add_advanced PRIVATE CUDA::nvtx3)
elseif(TARGET CUDA::nvToolsExt)
  target_link_libraries(vector_add_advanced PRIVATE CUDA::nvToolsExt)
else()
  message(FATAL_ERROR "Vector Add advanced requires an NVTX CMake target")
endif()

include(CTest)
if(BUILD_TESTING)
  add_test(NAME device_info COMMAND $<TARGET_FILE:device_info>)
  # Project 1 boundary matrix. A/B/C offsets are measured in float elements;
  # offset=1 deliberately makes a cudaMalloc pointer non-16-byte-aligned.
  add_test(NAME vector_add_n1
           COMMAND $<TARGET_FILE:vector_add> 1 2 0 0 0 64)
  add_test(NAME vector_add_n3
           COMMAND $<TARGET_FILE:vector_add> 3 2 0 0 0 64)
  add_test(NAME vector_add_n255
           COMMAND $<TARGET_FILE:vector_add> 255 2 0 0 0 64)
  add_test(NAME vector_add_n256
           COMMAND $<TARGET_FILE:vector_add> 256 2 0 0 0 128)
  add_test(NAME vector_add_n257
           COMMAND $<TARGET_FILE:vector_add> 257 2 0 0 0 256)
  add_test(NAME vector_add_n1000003_sweep
           COMMAND $<TARGET_FILE:vector_add> 1000003 3)
  add_test(NAME vector_add_unaligned_a
           COMMAND $<TARGET_FILE:vector_add> 257 2 1 0 0 256)
  add_test(NAME vector_add_unaligned_b
           COMMAND $<TARGET_FILE:vector_add> 257 2 0 1 0 256)
  add_test(NAME vector_add_unaligned_c
           COMMAND $<TARGET_FILE:vector_add> 257 2 0 0 1 256)
  add_test(NAME vector_add_unaligned_all
           COMMAND $<TARGET_FILE:vector_add> 1000003 2 1 1 1 512)
  add_test(NAME vector_add_advanced_aligned
           COMMAND $<TARGET_FILE:vector_add_advanced>
                   --n 257 --iterations 2 --rounds 2 --warmup 1
                   --min-block 32 --max-block 160 --step 64
                   --csv ${CMAKE_BINARY_DIR}/results/vector_add_advanced_aligned.csv)
  add_test(NAME vector_add_advanced_small_n
           COMMAND $<TARGET_FILE:vector_add_advanced>
                   --n 1 --iterations 2 --rounds 2 --warmup 1 --block 32
                   --types all --variants vector
                   --csv ${CMAKE_BINARY_DIR}/results/vector_add_advanced_small_n.csv)
  add_test(NAME vector_add_advanced_tail2
           COMMAND $<TARGET_FILE:vector_add_advanced>
                   --n 258 --iterations 2 --rounds 2 --warmup 1 --block 96
                   --types all --variants vector
                   --csv ${CMAKE_BINARY_DIR}/results/vector_add_advanced_tail2.csv)
  add_test(NAME vector_add_advanced_tail3
           COMMAND $<TARGET_FILE:vector_add_advanced>
                   --n 259 --iterations 2 --rounds 2 --warmup 1 --block 1024
                   --types all --variants vector
                   --csv ${CMAKE_BINARY_DIR}/results/vector_add_advanced_tail3.csv)
  add_test(NAME vector_add_advanced_unaligned
           COMMAND $<TARGET_FILE:vector_add_advanced>
                   --n 257 --iterations 2 --rounds 2 --warmup 1
                   --block 96 --offset 1
                   --csv ${CMAKE_BINARY_DIR}/results/vector_add_advanced_unaligned.csv)
  add_test(NAME vector_add_advanced_reject_nonwarp
           COMMAND $<TARGET_FILE:vector_add_advanced>
                   --n 257 --iterations 1 --rounds 1 --block 33
                   --csv ${CMAKE_BINARY_DIR}/results/should_not_exist.csv)
  set_tests_properties(vector_add_advanced_reject_nonwarp
                       PROPERTIES WILL_FAIL TRUE)
  add_test(NAME tiled_gemm_tail COMMAND $<TARGET_FILE:tiled_gemm> 129 131 127 5)
  add_test(NAME cutlass_sgemm_tail COMMAND $<TARGET_FILE:cutlass_sgemm> 129 131 127 5)
  add_test(NAME cutlass_tensorop_tail COMMAND $<TARGET_FILE:cutlass_tensorop> 136 152 120 5)
  add_test(NAME cute_layout_mapping COMMAND $<TARGET_FILE:cute_layout>)
  set(VECTOR_ADD_TESTS
      vector_add_n1 vector_add_n3 vector_add_n255 vector_add_n256
      vector_add_n257 vector_add_n1000003_sweep vector_add_unaligned_a
      vector_add_unaligned_b vector_add_unaligned_c vector_add_unaligned_all
      vector_add_advanced_aligned vector_add_advanced_small_n
      vector_add_advanced_tail2 vector_add_advanced_tail3
      vector_add_advanced_unaligned
      vector_add_advanced_reject_nonwarp)
  set_tests_properties(${VECTOR_ADD_TESTS}
                       PROPERTIES TIMEOUT 120 LABELS vector_add)
  set_tests_properties(device_info tiled_gemm_tail cutlass_sgemm_tail
                       cutlass_tensorop_tail cute_layout_mapping
                       PROPERTIES TIMEOUT 120)
endif()

message(STATUS "CUTLASS learning lab")
message(STATUS "  CUTLASS: ${CUTLASS_DIR} (required v4.5.3)")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
