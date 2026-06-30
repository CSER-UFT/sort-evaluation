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

/* Uniforme: mantem o comportamento original (rand() % 1000),
 * preservando a reprodutibilidade das entradas ja existentes. */
static void fill_uniform(int *a, long n) {
    for (long idx = 0; idx < n; idx++) {
        a[idx] = rand() % 1000;
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

/* ===== MergeSort iterativo (bottom-up) com paralelizacao OpenMP =====
 *
 * merge() escreve em aux indexado pela POSICAO GLOBAL (aux[inicio..fim]),
 * nao a partir de zero. Isso e essencial para a versao paralela: como os
 * blocos [inicio,fim] de cada merge sao disjuntos, cada thread escreve em
 * uma faixa distinta de aux, eliminando a corrida de dados.
 */
static void merge(int v[], long inicio, long meio, long fim, int aux[]) {
    long i = inicio, j = meio + 1, k = inicio;

    while (i <= meio && j <= fim) {
        if (v[i] <= v[j]) {
            aux[k++] = v[i++];
        } else {
            aux[k++] = v[j++];
        }
    }
    while (i <= meio) {
        aux[k++] = v[i++];
    }
    while (j <= fim) {
        aux[k++] = v[j++];
    }

    for (k = inicio; k <= fim; k++) {
        v[k] = aux[k];
    }
}

static void mergeSort(int v[], long n, int aux[]) {
    /*
     * currSize = tamanho do subarray em cada nivel (1, 2, 4, 8, ...).
     * Em cada nivel, os merges de pares adjacentes sao independentes
     * (blocos disjuntos), entao o laco sobre start e paralelizado.
     */
    for (long currSize = 1; currSize < n; currSize *= 2) {
#ifdef PARALLEL
        #pragma omp parallel for schedule(static)
#endif
        for (long start = 0; start < n - currSize; start += 2 * currSize) {
            long mid = start + currSize - 1;
            long end = (start + 2 * currSize - 1 < n - 1)
                       ? start + 2 * currSize - 1
                       : n - 1;
            merge(v, start, mid, end, aux);
        }
    }
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

    FILE *f = fopen(argv[1], "r");
    if (!f) {
        perror("fopen");
        return 1;
    }

    int seed;
    long n;
    if (fscanf(f, "%d", &seed) != 1 || fscanf(f, "%ld", &n) != 1) {
        fprintf(stderr, "Erro ao ler seed/n do arquivo de entrada\n");
        fclose(f);
        return 1;
    }
    fclose(f);

    if (n <= 0) {
        fprintf(stderr, "Tamanho do vetor invalido: %ld\n", n);
        return 1;
    }

    srand(seed);

    int *a = malloc(n * sizeof(int));
    int *aux = malloc(n * sizeof(int));
    if (!a || !aux) {
        perror("malloc");
        free(a);
        free(aux);
        return 1;
    }

    /* ---- Escolha da distribuicao: deixe apenas UMA linha ativa ---- */
    fill_uniform(a, n);
    /* fill_normal(a, n, 500.0, 150.0); */               /* mu=500, sigma=150 */
    /* fill_exponential(a, n, 1.0, 200.0); */            /* lambda=1, scale=200 */

    mergeSort(a, n, aux);

    int is_sorted = 1;
    for (long i = 0; i < n - 1; i++) {
        if (a[i] > a[i + 1]) {
            is_sorted = 0;
            break;
        }
    }

    printf("Array esta ordenado: %s\n", is_sorted ? "SIM" : "NAO");
    printf("First element: %d\n", a[0]);
    printf("Last element: %d\n", a[n - 1]);
    printf("N: %ld\n", n);

    free(a);
    free(aux);
    return is_sorted ? 0 : 1;
}
