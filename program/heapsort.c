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

/* Heapsort: max-heap binario. Par da subfamilia "selecao" com o smoothsort
 * (que usa heaps de Leonardo). */
static void siftdown(int *a, long root, long end) {
    while (2 * root + 1 < end) {
        long child = 2 * root + 1;
        if (child + 1 < end && a[child] < a[child + 1]) child++;
        if (a[root] < a[child]) {
            int t = a[root]; a[root] = a[child]; a[child] = t;
            root = child;
        } else break;
    }
}
#ifdef PARALLEL
/* A CONSTRUCAO do heap e paralelizavel por nivel: nos de um mesmo nivel tem
 * subarvores disjuntas, entao seus siftdowns sao independentes. A EXTRACAO e
 * inerentemente sequencial (cada retirada depende da anterior). Por isso
 * heapsort escala pouco -- resultado honesto para a bancada. */
static void heap_sort(int *a, long n) {
    long last_internal = n / 2 - 1;
    for (long d = 62; d >= 0; d--) {
        long lo = (1L << d) - 1;
        if (lo > last_internal) continue;
        long hi = (1L << (d + 1)) - 2;
        if (hi > last_internal) hi = last_internal;
        #pragma omp parallel for schedule(static)
        for (long i = lo; i <= hi; i++) siftdown(a, i, n);
    }
    for (long end = n - 1; end > 0; end--) {
        int t = a[0]; a[0] = a[end]; a[end] = t;
        siftdown(a, 0, end);
    }
}
#else
static void heap_sort(int *a, long n) {
    for (long i = n / 2 - 1; i >= 0; i--) siftdown(a, i, n);
    for (long end = n - 1; end > 0; end--) {
        int t = a[0]; a[0] = a[end]; a[end] = t;
        siftdown(a, 0, end);
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
    heap_sort(a, n);
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
