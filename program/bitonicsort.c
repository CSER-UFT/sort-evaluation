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

/* Bitonic sort (Batcher): rede de ordenacao O(n log^2 n). Requer n potencia de
 * 2, entao o vetor e copiado para um buffer de tamanho m = proxima potencia de
 * 2 e o excedente e preenchido com INT_MAX (vai para o fim e e descartado).
 * Subfamilia "rede": sequencia FIXA de compare-swaps, sem dependencia de dados
 * -> paraleliza por construcao. */
static long next_pow2(long n) { long m = 1; while (m < n) m <<= 1; return m; }

#ifdef PARALLEL
/* Para cada estagio (k,j), os compare-swaps do laco i sao pares disjuntos
 * (casamento perfeito), entao o laco i paraleliza sem corrida. k e j sao
 * sequenciais (cada estagio depende do anterior). */
static void bitonic_sort(int *a, long n) {
    if (n < 2) return;
    long m = next_pow2(n);
    int *b = malloc(m * sizeof(int));
    if (!b) { perror("malloc"); exit(1); }
    for (long i = 0; i < n; i++) b[i] = a[i];
    for (long i = n; i < m; i++) b[i] = INT_MAX;

    for (long k = 2; k <= m; k <<= 1)
        for (long j = k >> 1; j > 0; j >>= 1) {
            #pragma omp parallel for schedule(static)
            for (long i = 0; i < m; i++) {
                long l = i ^ j;
                if (l > i) {
                    int up = ((i & k) == 0);
                    if ((up && b[i] > b[l]) || (!up && b[i] < b[l])) {
                        int t = b[i]; b[i] = b[l]; b[l] = t;
                    }
                }
            }
        }
    for (long i = 0; i < n; i++) a[i] = b[i];
    free(b);
}
#else
static void bitonic_sort(int *a, long n) {
    if (n < 2) return;
    long m = next_pow2(n);
    int *b = malloc(m * sizeof(int));
    if (!b) { perror("malloc"); exit(1); }
    for (long i = 0; i < n; i++) b[i] = a[i];
    for (long i = n; i < m; i++) b[i] = INT_MAX;

    for (long k = 2; k <= m; k <<= 1)
        for (long j = k >> 1; j > 0; j >>= 1)
            for (long i = 0; i < m; i++) {
                long l = i ^ j;
                if (l > i) {
                    int up = ((i & k) == 0);
                    if ((up && b[i] > b[l]) || (!up && b[i] < b[l])) {
                        int t = b[i]; b[i] = b[l]; b[l] = t;
                    }
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
    bitonic_sort(a, n);
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
