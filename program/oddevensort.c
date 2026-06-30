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

/* Odd-even mergesort (Batcher): rede de ordenacao O(n log^2 n). Como o bitonic,
 * requer n potencia de 2 (padding com INT_MAX). Par de "rede" com o bitonic:
 * outra forma de construir a mesma garantia de paralelismo sem dependencia de
 * dados. */
static long next_pow2(long n) { long m = 1; while (m < n) m <<= 1; return m; }

#ifdef PARALLEL
/* Para cada estagio (p,k), valores distintos de j atuam sobre faixas de indices
 * disjuntas -> o laco j paraleliza sem corrida. */
static void oddeven_sort(int *a, long n) {
    if (n < 2) return;
    long m = next_pow2(n);
    int *b = malloc(m * sizeof(int));
    if (!b) { perror("malloc"); exit(1); }
    for (long i = 0; i < n; i++) b[i] = a[i];
    for (long i = n; i < m; i++) b[i] = INT_MAX;

    for (long p = 1; p < m; p <<= 1)
        for (long k = p; k >= 1; k >>= 1) {
            #pragma omp parallel for schedule(static)
            for (long j = k % p; j < m - k; j += 2 * k)
                for (long i = 0; i < k; i++) {
                    long lo = i + j, hi = i + j + k;
                    if (hi < m && (lo / (2 * p)) == (hi / (2 * p))) {
                        if (b[lo] > b[hi]) { int t = b[lo]; b[lo] = b[hi]; b[hi] = t; }
                    }
                }
        }
    for (long i = 0; i < n; i++) a[i] = b[i];
    free(b);
}
#else
static void oddeven_sort(int *a, long n) {
    if (n < 2) return;
    long m = next_pow2(n);
    int *b = malloc(m * sizeof(int));
    if (!b) { perror("malloc"); exit(1); }
    for (long i = 0; i < n; i++) b[i] = a[i];
    for (long i = n; i < m; i++) b[i] = INT_MAX;

    for (long p = 1; p < m; p <<= 1)
        for (long k = p; k >= 1; k >>= 1)
            for (long j = k % p; j < m - k; j += 2 * k)
                for (long i = 0; i < k; i++) {
                    long lo = i + j, hi = i + j + k;
                    if (hi < m && (lo / (2 * p)) == (hi / (2 * p))) {
                        if (b[lo] > b[hi]) { int t = b[lo]; b[lo] = b[hi]; b[hi] = t; }
                    }
                }
    for (long i = 0; i < n; i++) a[i] = b[i];
    free(b);
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
    oddeven_sort(a, n);
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
