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

/* Radix sort LSD, base 256 (4 passadas para int de 32 bits). Usa counting
 * estavel por byte. Valores sao deslocados por -min para virar nao-negativos,
 * o que trata negativos e qualquer faixa. Par de "distribuicao" com o counting:
 * o radix supera a limitacao do k grande fazendo varias passadas por digito. */
#ifdef PARALLEL
/* Passada paralela ESTAVEL: cada thread histograma seu pedaco; os offsets por
 * (thread,digito) sao calculados de modo que cada thread escreva uma regiao
 * disjunta do destino, preservando a ordem (estabilidade) e sem corrida. */
static void radix_pass(unsigned *src, unsigned *dst, long n, int shift, int nt) {
    long (*hist)[256] = calloc((size_t) nt, sizeof(*hist));
    long (*tstart)[256] = calloc((size_t) nt, sizeof(*tstart));
    if (!hist || !tstart) { perror("calloc"); exit(1); }

    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num();
        long lo = (n * (long) tid) / nt, hi = (n * (long) (tid + 1)) / nt;
        long *h = hist[tid];
        for (long i = lo; i < hi; i++) h[(src[i] >> shift) & 0xFFu]++;
    }
    long s = 0;
    for (int d = 0; d < 256; d++)
        for (int t = 0; t < nt; t++) { tstart[t][d] = s; s += hist[t][d]; }
    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num();
        long lo = (n * (long) tid) / nt, hi = (n * (long) (tid + 1)) / nt;
        long *ts = tstart[tid];
        for (long i = lo; i < hi; i++) {
            unsigned key = (src[i] >> shift) & 0xFFu;
            dst[ts[key]++] = src[i];
        }
    }
    free(hist); free(tstart);
}
static void radix_sort(int *a, long n) {
    if (n < 2) return;
    int mn = a[0];
    #pragma omp parallel for reduction(min:mn) schedule(static)
    for (long i = 0; i < n; i++) if (a[i] < mn) mn = a[i];
    unsigned *buf = malloc(n * sizeof(unsigned));
    unsigned *tmp = malloc(n * sizeof(unsigned));
    if (!buf || !tmp) { perror("malloc"); exit(1); }
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; i++) buf[i] = (unsigned) ((long) a[i] - mn);
    int nt = omp_get_max_threads();
    for (int pass = 0; pass < 4; pass++) {
        radix_pass(buf, tmp, n, pass * 8, nt);
        unsigned *t = buf; buf = tmp; tmp = t;
    }
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; i++) a[i] = (int) ((long) buf[i] + mn);
    free(buf); free(tmp);
}
#else
static void radix_sort(int *a, long n) {
    if (n < 2) return;
    int mn = a[0];
    for (long i = 1; i < n; i++) if (a[i] < mn) mn = a[i];
    unsigned *buf = malloc(n * sizeof(unsigned));
    unsigned *tmp = malloc(n * sizeof(unsigned));
    if (!buf || !tmp) { perror("malloc"); exit(1); }
    for (long i = 0; i < n; i++) buf[i] = (unsigned) ((long) a[i] - mn);
    for (int pass = 0; pass < 4; pass++) {
        int shift = pass * 8;
        long count[256] = {0};
        for (long i = 0; i < n; i++) count[(buf[i] >> shift) & 0xFFu]++;
        long pos[256], s = 0;
        for (int d = 0; d < 256; d++) { pos[d] = s; s += count[d]; }
        for (long i = 0; i < n; i++) {
            unsigned key = (buf[i] >> shift) & 0xFFu;
            tmp[pos[key]++] = buf[i];
        }
        unsigned *t = buf; buf = tmp; tmp = t;
    }
    for (long i = 0; i < n; i++) a[i] = (int) ((long) buf[i] + mn);
    free(buf); free(tmp);
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
    radix_sort(a, n);
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
