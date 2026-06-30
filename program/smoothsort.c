#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>

#if defined(PARALLEL) || defined(_OPENMP)
#include <omp.h>
#endif

#define LP_LIMIT 90

static inline void swap_int(int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

/* Numeros de Leonardo: L(0)=L(1)=1, L(k)=L(k-1)+L(k-2)+1 */
static unsigned long long leonardo_table[LP_LIMIT];

static void init_leonardo_table(void)
{
    leonardo_table[0] = 1;
    leonardo_table[1] = 1;
    for (int i = 2; i < LP_LIMIT; i++)
        leonardo_table[i] = leonardo_table[i - 1] + leonardo_table[i - 2] + 1ULL;
}

/* -----------------------------------------------------------------------
 * Geracao de valores aleatorios em tres distribuicoes.
 * A seed e aplicada uma vez via srand() antes da chamada (ver main).
 * Troca de distribuicao = comentar/descomentar UMA linha no main.
 * --------------------------------------------------------------------- */

/* Amostra uniforme em (0,1), evitando 0 e 1 (seguro para log()). */
static double rand_unit(void)
{
    return ((double) rand() + 1.0) / ((double) RAND_MAX + 2.0);
}

/* Uniforme: mantem o comportamento original (rand() % 1000000),
 * preservando a reprodutibilidade das entradas ja existentes. */
static void fill_uniform(int *a, size_t n)
{
    for (size_t idx = 0; idx < n; idx++)
        a[idx] = rand() % 1000000;
}

/* Normal (gaussiana) via Box-Muller. mu = media, sigma = desvio padrao. */
static void fill_normal(int *a, size_t n, double mu, double sigma)
{
    for (size_t idx = 0; idx < n; idx++) {
        double u1 = rand_unit();
        double u2 = rand_unit();
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        a[idx] = (int) lround(mu + sigma * z);
    }
}

/* Exponencial via transformada inversa. lambda = taxa (>0), media = 1/lambda.
 * scale espalha os valores pela faixa inteira desejada. */
static void fill_exponential(int *a, size_t n, double lambda, double scale)
{
    for (size_t idx = 0; idx < n; idx++) {
        double u = rand_unit();
        a[idx] = (int) lround(scale * (-log(u) / lambda));
    }
}

/* -----------------------------------------------------------------------
 * Smoothsort (Dijkstra). Implementacao baseada em uma pilha explicita das
 * ordens das arvores de Leonardo presentes na floresta, validada por teste
 * exaustivo (n = 2..3000, multiplas distribuicoes) contra qsort e com
 * AddressSanitizer, alem de n = 1M/5M.
 *
 * sift   : restaura a propriedade de heap DENTRO de uma arvore de ordem k.
 * trinkle: corrige a raiz da arvore mais a direita ATRAVESSANDO as raizes a
 *          esquerda (invariante das raizes ascendentes) e entao aplica sift.
 * --------------------------------------------------------------------- */

static void sift(int *m, size_t root, int k)
{
    int val = m[root];

    while (k >= 2) {
        size_t right_child = root - 1;
        size_t left_child = root - 1 - (size_t) leonardo_table[k - 2];

        size_t bigger;
        int bigger_order;

        if (m[left_child] > m[right_child]) {
            bigger = left_child;
            bigger_order = k - 1;
        } else {
            bigger = right_child;
            bigger_order = k - 2;
        }

        if (val >= m[bigger])
            break;

        m[root] = m[bigger];
        root = bigger;
        k = bigger_order;
    }

    m[root] = val;
}

static void trinkle(int *m, const int *orders, int idx, size_t root)
{
    int k = orders[idx];
    int val = m[root];

    while (idx > 0) {
        size_t left_root = root - (size_t) leonardo_table[k];

        if (m[left_root] <= val)
            break;

        if (k >= 2) {
            size_t right_child = root - 1;
            size_t left_child = root - 1 - (size_t) leonardo_table[k - 2];
            if (m[right_child] >= m[left_root] || m[left_child] >= m[left_root])
                break;
        }

        m[root] = m[left_root];
        root = left_root;
        idx--;
        k = orders[idx];
    }

    m[root] = val;
    sift(m, root, k);
}

static void smoothsort_seq(int *m, size_t n)
{
    if (n < 2)
        return;

    int orders[LP_LIMIT + 4];
    int sz = 0;

    /* Construcao da floresta de heaps de Leonardo. */
    for (size_t i = 0; i < n; i++) {
        if (sz >= 2 && orders[sz - 1] + 1 == orders[sz - 2]) {
            /* Funde as duas arvores mais a direita (ordens k e k+1) em k+2,
             * usando o elemento i como nova raiz. */
            int merged = orders[sz - 1] + 2;
            sz -= 2;
            orders[sz++] = merged;
        } else if (sz >= 1 && orders[sz - 1] == 1) {
            orders[sz++] = 0;
        } else {
            orders[sz++] = 1;
        }

        /* Se a arvore mais a direita ainda pode crescer, basta sift;
         * se ja esta em posicao final, trinkle para ajustar entre as raizes. */
        int r = orders[sz - 1];
        unsigned long long lr1 = (r - 1 >= 0) ? leonardo_table[r - 1] : 0ULL;

        if ((unsigned long long) i + lr1 + 1ULL < (unsigned long long) n)
            sift(m, i, r);
        else
            trinkle(m, orders, sz - 1, i);
    }

    /* Ajuste final da arvore mais a direita. */
    trinkle(m, orders, sz - 1, n - 1);

    /* Desmonte: remove a raiz (maior elemento) da arvore mais a direita,
     * dividindo-a em seus dois filhos quando a ordem e >= 2. */
    for (size_t i = n - 1; i > 0; i--) {
        int r = orders[sz - 1];

        if (r <= 1) {
            sz--;
        } else {
            sz--;
            size_t right_root = i - 1;
            size_t left_root = i - 1 - (size_t) leonardo_table[r - 2];

            orders[sz++] = r - 1;
            orders[sz++] = r - 2;

            trinkle(m, orders, sz - 2, left_root);
            trinkle(m, orders, sz - 1, right_root);
        }
    }
}

#ifdef PARALLEL
static void merge(int *arr, int *temp, size_t left, size_t mid, size_t right)
{
    size_t i = left;
    size_t j = mid + 1;
    size_t k = left;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }
    while (i <= mid)
        temp[k++] = arr[i++];
    while (j <= right)
        temp[k++] = arr[j++];

    for (i = left; i <= right; i++)
        arr[i] = temp[i];
}

static void parallel_merge_smoothsort_rec(int *arr, int *temp, size_t left, size_t right, int threads_available)
{
    if (left >= right)
        return;

    /* Caso base: ordena o bloco com smoothsort sequencial. */
    if (threads_available <= 1 || (right - left) < 10000) {
        smoothsort_seq(&arr[left], right - left + 1);
        return;
    }

    size_t mid = left + (right - left) / 2;
    int t_left = threads_available / 2;
    int t_right = threads_available - t_left;

    #pragma omp task shared(arr, temp)
    parallel_merge_smoothsort_rec(arr, temp, left, mid, t_left);

    #pragma omp task shared(arr, temp)
    parallel_merge_smoothsort_rec(arr, temp, mid + 1, right, t_right);

    #pragma omp taskwait
    merge(arr, temp, left, mid, right);
}

static void smoothsort_paralelo(int *arr, size_t n, int threads)
{
    if (n < 2)
        return;

    int *temp = malloc(n * sizeof(int));
    if (!temp) {
        smoothsort_seq(arr, n);
        return;
    }

    #pragma omp parallel
    {
        #pragma omp single
        {
            parallel_merge_smoothsort_rec(arr, temp, 0, n - 1, threads);
        }
    }

    free(temp);
}
#endif

int main(int argc, char *argv[])
{
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
    size_t n;

    if (fscanf(f, "%d", &seed) != 1 || fscanf(f, "%zu", &n) != 1) {
        fclose(f);
        fprintf(stderr, "Invalid input file format\n");
        return 1;
    }
    fclose(f);

    int *arr = malloc(n * sizeof(int));
    if (!arr) {
        perror("malloc");
        return 1;
    }

    init_leonardo_table();

    /* Geracao serial (identica para serial e paralelo, fora da regiao medida,
     * garantindo que ambos ordenem exatamente o mesmo vetor). */
    srand((unsigned) seed);

    /* ---- Escolha da distribuicao: deixe apenas UMA linha ativa ---- */
    fill_uniform(arr, n);
    /* fill_normal(arr, n, 500000.0, 150000.0); */       /* mu=5e5, sigma=1.5e5 */
    /* fill_exponential(arr, n, 1.0, 200000.0); */       /* lambda=1, scale=2e5 */

    double inicio_tempo = 0.0;
    double fim_tempo = 0.0;

#ifdef PARALLEL
    inicio_tempo = omp_get_wtime();
    smoothsort_paralelo(arr, n, threads);
    fim_tempo = omp_get_wtime();
#else
    #ifdef _OPENMP
        inicio_tempo = omp_get_wtime();
    #endif

    smoothsort_seq(arr, n);

    #ifdef _OPENMP
        fim_tempo = omp_get_wtime();
    #endif
#endif

    int is_sorted = 1;
    for (size_t i = 0; i + 1 < n; i++) {
        if (arr[i] > arr[i + 1]) {
            is_sorted = 0;
            break;
        }
    }

    if (!is_sorted) {
        fprintf(stderr, "ERRO: O algoritmo falhou na ordenacao.\n");
        free(arr);
        return 2;
    }

    printf("%f\n", fim_tempo - inicio_tempo);

    free(arr);
    return 0;
}
