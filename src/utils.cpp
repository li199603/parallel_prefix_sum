#include "utils.h"
#include <cstdio>
#include <random>

void data_init(int *data, int N)
{
    std::uniform_int_distribution<> int_generator(-10, 10);
    std::default_random_engine rand_engine(time(nullptr));
    for (int i = 0; i < N; i++)
    {
        data[i] = int_generator(rand_engine);
    }
}

void results_check(int *a, int *b, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (a[i] != b[i])
        {
            printf("results_check fail\n");
            exit(1);
        }
    }
}

void print_int_arr(int *a, int N)
{
    for (int i = 0; i < N; i++)
    {
        printf("%d ", a[i]);
    }
    printf("\n");
}

int next_power_of_two(int x)
{
    int power = 1;
    while (power < x)
    {
        power *= 2;
    }
    return power;
}