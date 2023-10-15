#pragma once

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_ELEMENTS_PER_BLOCK (MAX_THREADS_PER_BLOCK * 2)

#include <tuple>

void warm_up();
float scan_cpu(int *data, int *prefix_sum, int N);
std::tuple<float, float> sequential_scan_gpu(int *data, int *prefix_sum, int N);
std::tuple<float, float> parallel_block_scan_gpu(int *data, int *prefix_sum, int N, bool bcao);
std::tuple<float, float> parallel_large_scan_gpu(int *data, int *prefix_sum, int N, bool bcao);