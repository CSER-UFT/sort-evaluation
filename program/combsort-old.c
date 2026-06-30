#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

/* -----------------------------------------------------------------------
 * Geracao de valores aleatorios em tres distribuicoes.
 * A seed e aplicada uma vez via srand() antes da chamada (ver main).
 * Troca de distribuicao = comentar/descomentar UMA linha no main.
 * --------------------------------------------------------------------- */

/* Amostra uniforme em (0,1), evitando 0 e 1 (seguro para log()). */
static double rand_unit(void) {
    return ((double) rand() + 1.0) / ((double) RAND_MAX + 2.0);
}

/* Uniforme: mantem o comportamento original (rand() % 1000000),
 * preservando a reprodutibilidade das entradas ja existentes. */
static void fill_uniform(int *a, long n) {
    for (long idx = 0; idx < n; idx++) {
        a[idx] = rand() % 1000000;
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

#ifdef PARALLEL
/*
 * Comb sort PARALELO correto.
 * Cada classe de residuo r em [0, gap) define uma cadeia independente de
 * indices (r, r+gap, r+2gap, ...). Cadeias distintas nunca compartilham
 * posicoes do vetor, entao podem ser processadas em paralelo SEM corrida de
 * dados. Dentro de cada cadeia os compare-swaps permanecem sequenciais, o
 * que preserva exatamente a semantica do comb sort serial.
 *
 * Observacao: paralelizar sobre o indice linear i (com passo unitario, como
 * exige o loop unrolling) NAO e seguro, pois as iteracoes i e i+gap escrevem
 * ambas em a[i+gap] quando gap < n/2 -> corrida que corrompe o vetor.
 */
static void comb_sort(int *a, long n) {
    long gap = n;
    float shrink = 1.3f;
    int swapped = 1;

    while (gap > 1 || swapped) {
        gap = (long)(gap / shrink);
        if (gap < 1) gap = 1;
        swapped = 0;

        #pragma omp parallel for reduction(|:swapped) schedule(static)
        for (long r = 0; r < gap; r++) {
            int local_swap = 0;
            for (long j = r; j + gap < n; j += gap) {
                int condition = a[j] > a[j + gap];
                int mask = -condition;
                int t = (a[j] ^ a[j + gap]) & mask;
                a[j] ^= t;
                a[j + gap] ^= t;
                local_swap |= condition;
            }
            swapped |= local_swap;
        }
    }
}
#else
/*
 * Comb sort SERIAL com loop unrolling (fase V3, para aumentar o ILP).
 * Versao original, validada contra qsort.
 */
static void comb_sort(int *a, long n) {
    long gap = n;
    float shrink = 1.3f;
    int swapped = 1;

    while (gap > 1 || swapped) {
        gap = (long)(gap / shrink);
        if (gap < 1) gap = 1;
        swapped = 0;

        long limit = n - gap;
        long unroll_factor = 4;
        long unrolled_limit = limit - (limit % unroll_factor);

        for (long i = 0; i < unrolled_limit; i += unroll_factor) {
            int c1 = a[i] > a[i + gap];
            int t1 = (a[i] ^ a[i + gap]) & (-c1);
            a[i] ^= t1;
            a[i + gap] ^= t1;

            int c2 = a[i+1] > a[i+1 + gap];
            int t2 = (a[i+1] ^ a[i+1 + gap]) & (-c2);
            a[i+1] ^= t2;
            a[i+1 + gap] ^= t2;

            int c3 = a[i+2] > a[i+2 + gap];
            int t3 = (a[i+2] ^ a[i+2 + gap]) & (-c3);
            a[i+2] ^= t3;
            a[i+2 + gap] ^= t3;

            int c4 = a[i+3] > a[i+3 + gap];
            int t4 = (a[i+3] ^ a[i+3 + gap]) & (-c4);
            a[i+3] ^= t4;
            a[i+3 + gap] ^= t4;

            swapped |= c1 | c2 | c3 | c4;
        }

        for (long i = unrolled_limit; i < limit; i++) {
            int condition = a[i] > a[i + gap];
            int mask = -condition;
            int t = (a[i] ^ a[i + gap]) & mask;
            a[i] = a[i] ^ t;
            a[i + gap] = a[i + gap] ^ t;
            swapped |= condition;
        }
    }
}
#endif

static int is_sorted(const int *a, long n) {
    for (long i = 1; i < n; i++) {
        if (a[i - 1] > a[i]) {
            return 0;
        }
    }
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s input.in <threads>\n", argv[0]);
        return 1;
    }

    int threads = atoi(argv[2]);

#ifdef PARALLEL
    omp_set_num_threads(threads);
#else
    (void) threads;
#endif

    FILE *f = fopen(argv[1], "r");
    if (!f) {
        perror("fopen");
        return 1;
    }

    int seed;
    long n;
    if (fscanf(f, "%d", &seed) != 1) return 1;
    if (fscanf(f, "%ld", &n) != 1) return 1;
    fclose(f);

    srand(seed);

    int *a = malloc(n * sizeof(int));
    if (!a) {
        perror("malloc");
        return 1;
    }

    /* ---- Escolha da distribuicao: deixe apenas UMA linha ativa ---- */
    fill_uniform(a, n);
    /* fill_normal(a, n, 500000.0, 150000.0); */          /* mu=5e5, sigma=1.5e5 */
    /* fill_exponential(a, n, 1.0, 200000.0); */          /* lambda=1, scale=2e5 */

    double start = omp_get_wtime();
    comb_sort(a, n);
    double end = omp_get_wtime();

    if (!is_sorted(a, n)) {
        fprintf(stderr, "Erro: vetor nao foi ordenado corretamente\n");
        free(a);
        return 1;
    }

    printf("Result: %d (Sorted in %.4f seconds)\n", a[0], end - start);

    free(a);
    return 0;
}
