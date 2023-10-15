#include "scan.cuh"
#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <tuple>

int main(int argc, char **argv)
{
    warm_up();
    int nums[] = {1000, 2048, 100000, 10000000};
    int len = sizeof(nums) / sizeof(int);
    for (int i = 0; i < len; i++)
    {
        int N = nums[i];
        size_t arr_size = N * sizeof(int);
        int *data = (int *)malloc(arr_size);
        int *prefix_sum_cpu = (int *)malloc(arr_size);
        int *prefix_sum_gpu = (int *)malloc(arr_size);
        float total_cost, kernel_cost;
        data_init(data, N);
        printf("-------------------------- N = %d --------------------------\n", N);

        total_cost = scan_cpu(data, prefix_sum_cpu, N);
        printf("%35s - total: %10.5f ms\n", "scan_cpu", total_cost);

        std::tie(total_cost, kernel_cost) = sequential_scan_gpu(data, prefix_sum_gpu, N);
        results_check(prefix_sum_cpu, prefix_sum_gpu, N);
        printf("%35s - total: %10.5f ms    kernel: %10.5f ms\n", "sequential_scan_gpu", total_cost, kernel_cost);

        if (N <= MAX_ELEMENTS_PER_BLOCK)
        {
            std::tie(total_cost, kernel_cost) = parallel_block_scan_gpu(data, prefix_sum_gpu, N, false);
            results_check(prefix_sum_cpu, prefix_sum_gpu, N);
            printf("%35s - total: %10.5f ms    kernel: %10.5f ms\n", "parallel_block_scan_gpu", total_cost,
                   kernel_cost);

            std::tie(total_cost, kernel_cost) = parallel_block_scan_gpu(data, prefix_sum_gpu, N, true);
            results_check(prefix_sum_cpu, prefix_sum_gpu, N);
            printf("%35s - total: %10.5f ms    kernel: %10.5f ms\n", "parallel_block_scan_gpu with bcao", total_cost,
                   kernel_cost);
        }

        std::tie(total_cost, kernel_cost) = parallel_large_scan_gpu(data, prefix_sum_gpu, N, false);
        results_check(prefix_sum_cpu, prefix_sum_gpu, N);
        printf("%35s - total: %10.5f ms    kernel: %10.5f ms\n", "parallel_large_scan_gpu", total_cost, kernel_cost);

        std::tie(total_cost, kernel_cost) = parallel_large_scan_gpu(data, prefix_sum_gpu, N, true);
        results_check(prefix_sum_cpu, prefix_sum_gpu, N);
        printf("%35s - total: %10.5f ms    kernel: %10.5f ms\n", "parallel_large_scan_gpu with bcao", total_cost,
               kernel_cost);

        free(data);
        free(prefix_sum_cpu);
        free(prefix_sum_gpu);
        printf("\n");
    }
}