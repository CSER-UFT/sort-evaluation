#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <limits.h>

static double rand_unit(void) {
    return ((double) rand() + 1.0) / ((double) RAND_MAX + 2.0);
}
static void fill_uniform(int *a, long n) {
    for (long idx = 0; idx < n; idx++) a[idx] = rand() % 1000000;
}
static void fill_normal(int *a, long n, double mu, double sigma) {
    for (long idx = 0; idx < n; idx++) {
        double u1 = rand_unit(), u2 = rand_unit();
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        a[idx] = (int) lround(mu + sigma * z);
    }
}
static void fill_exponential(int *a, long n, double lambda, double scale) {
    for (long idx = 0; idx < n; idx++) {
        double u = rand_unit();
        a[idx] = (int) lround(scale * (-log(u) / lambda));
    }
}

/* Shell sort: insertion sort com gaps decrescentes (sequencia de Knuth:
 * 1, 4, 13, 40, ... h = 3h+1). Par da subfamilia "insercao" com o insertion.
 * Cada gap define 'gap' subsequencias intercaladas e INDEPENDENTES; o gap=1
 * final e uma insertion sort completa. */
#ifdef PARALLEL
/* Paraleliza sobre as subsequencias (offsets s em [0,gap)), que sao disjuntas
 * -> sem corrida. No gap=1 ha uma subsequencia so (efetivamente serial). */
static void shell_sort(int *a, long n) {
    long h = 1;
    while (h < n / 3) h = 3 * h + 1;
    for (; h >= 1; h = (h - 1) / 3) {
        #pragma omp parallel for schedule(static)
        for (long s = 0; s < h; s++) {
            for (long i = s + h; i < n; i += h) {
                int tmp = a[i];
                long j = i;
                while (j >= h && a[j - h] > tmp) { a[j] = a[j - h]; j -= h; }
                a[j] = tmp;
            }
        }
        if (h == 1) break;
    }
}
#else
static void shell_sort(int *a, long n) {
    long h = 1;
    while (h < n / 3) h = 3 * h + 1;
    for (; h >= 1; h = (h - 1) / 3) {
        for (long i = h; i < n; i++) {
            int tmp = a[i];
            long j = i;
            while (j >= h && a[j - h] > tmp) { a[j] = a[j - h]; j -= h; }
            a[j] = tmp;
        }
        if (h == 1) break;
    }
}
#endif

static int is_sorted(const int *a, long n) {
    for (long i = 1; i < n; i++) if (a[i - 1] > a[i]) return 0;
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc != 3) { fprintf(stderr, "Usage: %s input.in <threads>\n", argv[0]); return 1; }
    int threads = atoi(argv[2]);
#ifdef PARALLEL
    omp_set_num_threads(threads);
#else
    (void) threads;
#endif
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror("fopen"); return 1; }
    int seed; long n;
    if (fscanf(f, "%d", &seed) != 1) return 1;
    if (fscanf(f, "%ld", &n) != 1) return 1;
    fclose(f);
    srand(seed);
    int *a = malloc(n * sizeof(int));
    if (!a) { perror("malloc"); return 1; }

    /* ---- Escolha da distribuicao: deixe apenas UMA linha ativa ---- */
    fill_uniform(a, n);
    /* fill_normal(a, n, 500000.0, 150000.0); */
    /* fill_exponential(a, n, 1.0, 200000.0); */

#ifndef NOSORT
    double start = omp_get_wtime();
    shell_sort(a, n);
    double end = omp_get_wtime();
#else
    double start = 0.0, end = 0.0; (void) start; (void) end;
#endif

#ifdef DUMP
    /* Auditoria: soma e XOR sao invariantes sob permutacao -> validam que o
     * multiconjunto foi preservado (nenhum valor perdido/duplicado). */
    unsigned long long sum = 0, xr = 0;
    for (long i = 0; i < n; i++) { sum += (unsigned int) a[i]; xr ^= (unsigned int) a[i]; }
    printf("SUM %llu XOR %llu SORTED %d\n", sum, xr, is_sorted(a, n));
#else
    if (!is_sorted(a, n)) { fprintf(stderr, "Erro: vetor nao foi ordenado corretamente\n"); free(a); return 1; }
    printf("Result: %d (Sorted in %.4f seconds)\n", a[0], end - start);
#endif
    free(a);
    return 0;
}
