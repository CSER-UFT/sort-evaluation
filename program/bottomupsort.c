#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <math.h>

static long min_long(long a, long b) {
    return (a < b) ? a : b;
}

/* -----------------------------------------------------------------------
 * Geracao de valores aleatorios em tres distribuicoes.
 * A seed e aplicada uma vez via srand() antes da chamada (ver main).
 * Troca de distribuicao = comentar/descomentar UMA linha no main.
 * --------------------------------------------------------------------- */

/* Amostra uniforme em (0,1), evitando 0 e 1 (seguro para log()). */
static double rand_unit(void) {
    return ((double) rand() + 1.0) / ((double) RAND_MAX + 2.0);
}

/* Uniforme: equivalente ao comportamento original (rand() puro),
 * preservando a reprodutibilidade das entradas ja existentes. */
static void fill_uniform(int *array, long n) {
    for (long idx = 0; idx < n; idx++) {
        array[idx] = rand();
    }
}

/* Normal (gaussiana) via Box-Muller. mu = media, sigma = desvio padrao. */
static void fill_normal(int *array, long n, double mu, double sigma) {
    for (long idx = 0; idx < n; idx++) {
        double u1 = rand_unit();
        double u2 = rand_unit();
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        array[idx] = (int) lround(mu + sigma * z);
    }
}

/* Exponencial via transformada inversa. lambda = taxa (>0), media = 1/lambda.
 * scale espalha os valores pela faixa inteira desejada. */
static void fill_exponential(int *array, long n, double lambda, double scale) {
    for (long idx = 0; idx < n; idx++) {
        double u = rand_unit();
        array[idx] = (int) lround(scale * (-log(u) / lambda));
    }
}

static void merge_blocks(const int *src, int *dst, long left, long mid, long right) {
    long i = left;
    long j = mid;
    long k = left;

    while (i < mid && j < right) {
        if (src[i] <= src[j]) {
            dst[k++] = src[i++];
        } else {
            dst[k++] = src[j++];
        }
    }

    while (i < mid) {
        dst[k++] = src[i++];
    }

    while (j < right) {
        dst[k++] = src[j++];
    }
}

static void bottom_up_merge_sort_serial(int *array, long n) {
    if (n <= 1) {
        return;
    }

    int *temp = (int *) malloc((size_t)n * sizeof(int));
    if (!temp) {
        perror("malloc temp");
        exit(1);
    }

    int *src = array;
    int *dst = temp;

    for (long width = 1; width < n; width *= 2) {
        for (long left = 0; left < n; left += 2 * width) {
            long mid = min_long(left + width, n);
            long right = min_long(left + 2 * width, n);
            merge_blocks(src, dst, left, mid, right);
        }

        int *swap = src;
        src = dst;
        dst = swap;
    }

    if (src != array) {
        for (long i = 0; i < n; i++) {
            array[i] = src[i];
        }
    }

    free(temp);
}

#ifdef PARALLEL
static void bottom_up_merge_sort_parallel(int *array, long n) {
    if (n <= 1) {
        return;
    }

    int *temp = (int *) malloc((size_t)n * sizeof(int));
    if (!temp) {
        perror("malloc temp");
        exit(1);
    }

    int *src = array;
    int *dst = temp;

    /*
     * Mantemos uma unica regiao paralela para reduzir o overhead de criar
     * threads a cada largura do Merge Sort. Em cada fase, os blocos sao
     * independentes e podem ser mesclados em paralelo.
     */
    #pragma omp parallel shared(src, dst, array, n)
    {
        for (long width = 1; width < n; width *= 2) {
            #pragma omp for schedule(static)
            for (long left = 0; left < n; left += 2 * width) {
                long mid = min_long(left + width, n);
                long right = min_long(left + 2 * width, n);
                merge_blocks(src, dst, left, mid, right);
            }

            #pragma omp single
            {
                int *swap = src;
                src = dst;
                dst = swap;
            }
        }

        if (src != array) {
            #pragma omp for schedule(static)
            for (long i = 0; i < n; i++) {
                array[i] = src[i];
            }
        }
    }

    free(temp);
}
#endif

static int is_sorted(const int *array, long n) {
    for (long i = 1; i < n; i++) {
        if (array[i - 1] > array[i]) {
            return 0;
        }
    }
    return 1;
}

static long long checksum_array(const int *array, long n) {
    long long checksum = 0;

    for (long i = 0; i < n; i++) {
        checksum += (long long) array[i] * (long long) ((i % 97) + 1);
    }

    return checksum;
}

static int read_input_file(const char *filename, int *seed, long *n) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("fopen");
        return 0;
    }

    if (fscanf(f, "%d", seed) != 1) {
        fprintf(stderr, "Erro ao ler seed do arquivo de entrada\n");
        fclose(f);
        return 0;
    }

    if (fscanf(f, "%ld", n) != 1) {
        fprintf(stderr, "Erro ao ler tamanho do vetor do arquivo de entrada\n");
        fclose(f);
        return 0;
    }

    fclose(f);
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s input.in <threads>\n", argv[0]);
        return 1;
    }

    int threads = atoi(argv[2]);
    if (threads < 1) {
        fprintf(stderr, "Invalid number of threads\n");
        return 1;
    }

#ifdef PARALLEL
    omp_set_num_threads(threads);
#else
    (void) threads;
#endif

    int seed;
    long n;
    if (!read_input_file(argv[1], &seed, &n)) {
        return 1;
    }

    if (n <= 0) {
        fprintf(stderr, "Invalid input size\n");
        return 1;
    }

    srand((unsigned int) seed);

    int *array = (int *) malloc((size_t)n * sizeof(int));
    if (!array) {
        perror("malloc array");
        return 1;
    }

    /* ---- Escolha da distribuicao: deixe apenas UMA linha ativa ---- */
    fill_uniform(array, n);
    /* fill_normal(array, n, 0.0, 1000000.0); */                  /* mu=0, sigma=1e6 */
    /* fill_exponential(array, n, 1.0, 1000000.0); */             /* lambda=1, scale=1e6 */

#ifdef PARALLEL
    bottom_up_merge_sort_parallel(array, n);
#else
    bottom_up_merge_sort_serial(array, n);
#endif

    if (!is_sorted(array, n)) {
        fprintf(stderr, "Erro: vetor nao foi ordenado corretamente\n");
        free(array);
        return 1;
    }

    long long result = checksum_array(array, n);
    printf("Result: %lld\n", result);

    free(array);
    return 0;
}
