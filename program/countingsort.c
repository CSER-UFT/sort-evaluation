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

/* Counting sort: conta ocorrencias em [min,max] e reconstroi. O(n + k), onde
 * k = max-min+1. Subfamilia "distribuicao" -> nao compara elementos. */
#ifdef PARALLEL
/* Paralelo: histogramas locais por thread (sem contencao), merge por balde,
 * prefix sum serial e escrita dos baldes em paralelo (regioes disjuntas). */
static void counting_sort(int *a, long n) {
    if (n < 2) return;
    int mn = a[0], mx = a[0];
    #pragma omp parallel for reduction(min:mn) reduction(max:mx) schedule(static)
    for (long i = 0; i < n; i++) { if (a[i] < mn) mn = a[i]; if (a[i] > mx) mx = a[i]; }
    long K = (long) mx - mn + 1;

    int nt = omp_get_max_threads();
    long *hist = calloc((size_t) nt * K, sizeof(long));
    long *cnt  = malloc(K * sizeof(long));
    long *start = malloc((K + 1) * sizeof(long));
    if (!hist || !cnt || !start) { perror("calloc"); exit(1); }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long *h = hist + (long) tid * K;
        #pragma omp for schedule(static)
        for (long i = 0; i < n; i++) h[a[i] - mn]++;
    }
    #pragma omp parallel for schedule(static)
    for (long v = 0; v < K; v++) {
        long s = 0;
        for (int t = 0; t < nt; t++) s += hist[(long) t * K + v];
        cnt[v] = s;
    }
    start[0] = 0;
    for (long v = 0; v < K; v++) start[v + 1] = start[v] + cnt[v];
    #pragma omp parallel for schedule(static)
    for (long v = 0; v < K; v++)
        for (long j = start[v]; j < start[v + 1]; j++) a[j] = (int) (v + mn);

    free(hist); free(cnt); free(start);
}
#else
static void counting_sort(int *a, long n) {
    if (n < 2) return;
    int mn = a[0], mx = a[0];
    for (long i = 1; i < n; i++) { if (a[i] < mn) mn = a[i]; if (a[i] > mx) mx = a[i]; }
    long K = (long) mx - mn + 1;
    long *cnt = calloc(K, sizeof(long));
    if (!cnt) { perror("calloc"); exit(1); }
    for (long i = 0; i < n; i++) cnt[a[i] - mn]++;
    long idx = 0;
    for (long v = 0; v < K; v++)
        while (cnt[v]-- > 0) a[idx++] = (int) (v + mn);
    free(cnt);
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
    counting_sort(a, n);
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
