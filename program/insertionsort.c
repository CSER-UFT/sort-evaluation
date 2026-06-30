#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

#define VALUE_LIMIT 1000
#define INSERTION_SORT_CUTOFF 32L
#define PARALLEL_MIN_ELEMENTS 32768L

static int parse_positive_int(const char *text, int *value) {
    char *end = NULL;

    errno = 0;
    long parsed = strtol(text, &end, 10);

    if (errno != 0 || end == text || *end != '\0' || parsed < 1 || parsed > INT_MAX) {
        return 0;
    }

    *value = (int)parsed;
    return 1;
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

/* Uniforme: mantem o comportamento original (rand() % VALUE_LIMIT),
 * preservando a reprodutibilidade das entradas ja existentes. */
static void fill_uniform(int *a, long n) {
    for (long idx = 0; idx < n; idx++) {
        a[idx] = rand() % VALUE_LIMIT;
    }
}

/* Normal (gaussiana) via Box-Muller. mu = media, sigma = desvio padrao. */
static void fill_normal(int *a, long n, double mu, double sigma) {
    for (long idx = 0; idx < n; idx++) {
        double u1 = rand_unit();
        double u2 = rand_unit();
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        a[idx] = (int) lround(mu + sigma * z);
    }
}

/* Exponencial via transformada inversa. lambda = taxa (>0), media = 1/lambda.
 * scale espalha os valores pela faixa inteira desejada. */
static void fill_exponential(int *a, long n, double lambda, double scale) {
    for (long idx = 0; idx < n; idx++) {
        double u = rand_unit();
        a[idx] = (int) lround(scale * (-log(u) / lambda));
    }
}

static void insertion_sort(int *a, long left, long right) {
    for (long i = left + 1; i <= right; i++) {
        int value = a[i];
        long j = i - 1;

        while (j >= left && a[j] > value) {
            a[j + 1] = a[j];
            j--;
        }

        a[j + 1] = value;
    }
}

static void merge_runs(int *restrict a, int *restrict tmp, long left, long mid, long right) {
    if (a[mid] <= a[mid + 1]) {
        return;
    }

    long left_count = mid - left + 1;
    memcpy(&tmp[left], &a[left], (size_t)left_count * sizeof(*a));

    long i = left;
    long j = mid + 1;
    long k = left;

    while (i <= mid && j <= right) {
        if (tmp[i] <= a[j]) {
            a[k++] = tmp[i++];
        } else {
            a[k++] = a[j++];
        }
    }

    if (i <= mid) {
        long remaining = mid - i + 1;
        memcpy(&a[k], &tmp[i], (size_t)remaining * sizeof(*a));
    }
}

static void merge_sort_serial(int *restrict a, int *restrict tmp, long left, long right) {
    if (right - left + 1 <= INSERTION_SORT_CUTOFF) {
        insertion_sort(a, left, right);
        return;
    }

    long mid = left + (right - left) / 2;

    merge_sort_serial(a, tmp, left, mid);
    merge_sort_serial(a, tmp, mid + 1, right);
    merge_runs(a, tmp, left, mid, right);
}

#ifdef PARALLEL
static void merge_sort_parallel(int *restrict a, int *restrict tmp, long left, long right, int depth) {
    long count = right - left + 1;

    if (count <= INSERTION_SORT_CUTOFF) {
        insertion_sort(a, left, right);
        return;
    }

    if (depth <= 0 || count < PARALLEL_MIN_ELEMENTS) {
        merge_sort_serial(a, tmp, left, right);
        return;
    }

    long mid = left + (right - left) / 2;

    #pragma omp task shared(a, tmp) firstprivate(left, mid, depth)
    merge_sort_parallel(a, tmp, left, mid, depth - 1);

    #pragma omp task shared(a, tmp) firstprivate(mid, right, depth)
    merge_sort_parallel(a, tmp, mid + 1, right, depth - 1);

    #pragma omp taskwait
    merge_runs(a, tmp, left, mid, right);
}

static int task_depth_for(int threads, long n) {
    int depth = 0;
    int tasks = 1;

    while (tasks < threads && n / tasks >= PARALLEL_MIN_ELEMENTS && depth < 30) {
        depth++;
        tasks <<= 1;
    }

    return depth;
}
#endif

static long checksum_and_count_errors(const int *a, long n, long *errors) {
    long sum = 0;
    long bad_pairs = 0;

#ifdef PARALLEL
    #pragma omp parallel for reduction(+:sum, bad_pairs) schedule(static) if(n >= PARALLEL_MIN_ELEMENTS)
#endif
    for (long i = 0; i < n; i++) {
        sum += a[i];

        if (i > 0 && a[i - 1] > a[i]) {
            bad_pairs++;
        }
    }

    *errors = bad_pairs;
    return sum;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Uso: %s input.in <threads>\n", argv[0]);
        return 1;
    }

    int threads = 0;

    if (!parse_positive_int(argv[2], &threads)) {
        fprintf(stderr, "Numero de threads invalido: %s\n", argv[2]);
        return 1;
    }

#ifdef PARALLEL
    omp_set_num_threads(threads);
#endif

    FILE *f = fopen(argv[1], "r");

    if (!f) {
        perror("fopen");
        return 1;
    }

    int seed = 0;
    long n = 0;

    if (fscanf(f, "%d", &seed) != 1) {
        fprintf(stderr, "Erro ao ler seed\n");
        fclose(f);
        return 1;
    }

    if (fscanf(f, "%ld", &n) != 1) {
        fprintf(stderr, "Erro ao ler n\n");
        fclose(f);
        return 1;
    }

    fclose(f);

    if (n <= 0) {
        fprintf(stderr, "Tamanho do vetor invalido: %ld\n", n);
        return 1;
    }

    if ((size_t)n > SIZE_MAX / sizeof(int)) {
        fprintf(stderr, "Entrada grande demais para alocar: %ld\n", n);
        return 1;
    }

    int *a = malloc((size_t)n * sizeof(*a));
    int *tmp = malloc((size_t)n * sizeof(*tmp));

    if (!a || !tmp) {
        perror("malloc");
        free(a);
        free(tmp);
        return 1;
    }

    srand(seed);

    /* ---- Escolha da distribuicao: deixe apenas UMA linha ativa ---- */
    fill_uniform(a, n);
    /* fill_normal(a, n, 500.0, 150.0); */               /* mu=500, sigma=150 */
    /* fill_exponential(a, n, 1.0, 200.0); */            /* lambda=1, scale=200 */

#ifdef PARALLEL
    int depth = task_depth_for(threads, n);

    #pragma omp parallel
    {
        #pragma omp single nowait
        merge_sort_parallel(a, tmp, 0, n - 1, depth);
    }
#else
    (void)threads;
    merge_sort_serial(a, tmp, 0, n - 1);
#endif

    long errors = 0;
    long sum = checksum_and_count_errors(a, n, &errors);

    printf("Sorted: %s\n", errors == 0 ? "yes" : "no");
    printf("Checksum: %ld\n", sum);
    printf("First element: %d\n", a[0]);
    printf("Last element: %d\n", a[n - 1]);
    printf("N: %ld\n", n);

#ifdef PARALLEL
    printf("Threads: %d\n", threads);
    printf("Task depth: %d\n", depth);
#else
    printf("Threads: 1\n");
#endif

    free(a);
    free(tmp);

    return errors == 0 ? 0 : 1;
}
