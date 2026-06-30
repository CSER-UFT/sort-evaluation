#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef PARALLEL
#include <omp.h>
#endif

#define QUICKSORT_THRESHOLD 10000

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

static inline void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

static void dual_pivot_partition(int arr[], long low, long high, long *lp, long *rp) {
    if (arr[low] > arr[high])
        swap(&arr[low], &arr[high]);
    int pivot1 = arr[low];
    int pivot2 = arr[high];
    long i = low + 1;
    long lt = low + 1;
    long gt = high - 1;
    while (i <= gt) {
        if (arr[i] < pivot1) {
            swap(&arr[i], &arr[lt]);
            lt++;
        } else if (arr[i] > pivot2) {
            while (arr[gt] > pivot2 && i < gt)
                gt--;
            swap(&arr[i], &arr[gt]);
            gt--;
            if (arr[i] < pivot1) {
                swap(&arr[i], &arr[lt]);
                lt++;
            }
        }
        i++;
    }
    lt--;
    gt++;
    swap(&arr[low], &arr[lt]);
    swap(&arr[high], &arr[gt]);
    *lp = lt;
    *rp = gt;
}

static void dual_pivot_quicksort_seq(int arr[], long low, long high) {
    if (low >= high)
        return;
    long lp, rp;
    dual_pivot_partition(arr, low, high, &lp, &rp);

    /*
     * Guarda contra explosao de profundidade com valores repetidos:
     * so recursa na particao do meio se os pivos forem distintos. Se
     * pivot1 == pivot2, todos os elementos do meio sao iguais a eles e
     * ja estao em posicao. Sem isso, entradas com muitos duplicados
     * (ex.: distribuicao exponencial) geram recursao O(n) e estouram a
     * pilha. arr[lp] guarda pivot1 e arr[rp] guarda pivot2 apos a particao.
     */
    int pivots_differ = (arr[lp] < arr[rp]);

    dual_pivot_quicksort_seq(arr, low, lp - 1);
    if (pivots_differ)
        dual_pivot_quicksort_seq(arr, lp + 1, rp - 1);
    dual_pivot_quicksort_seq(arr, rp + 1, high);
}

#ifdef PARALLEL
static void dual_pivot_quicksort_parallel(int arr[], long low, long high) {
    if (low >= high)
        return;
    long size = high - low + 1;
    if (size < QUICKSORT_THRESHOLD) {
        dual_pivot_quicksort_seq(arr, low, high);
        return;
    }
    long lp, rp;
    dual_pivot_partition(arr, low, high, &lp, &rp);

    int pivots_differ = (arr[lp] < arr[rp]);

#pragma omp task shared(arr)
    dual_pivot_quicksort_parallel(arr, low, lp - 1);
    if (pivots_differ) {
#pragma omp task shared(arr)
        dual_pivot_quicksort_parallel(arr, lp + 1, rp - 1);
    }
#pragma omp task shared(arr)
    dual_pivot_quicksort_parallel(arr, rp + 1, high);
#pragma omp taskwait
}

static void dual_pivot_quicksort(int arr[], long n) {
#pragma omp parallel
    {
#pragma omp single
        {
            dual_pivot_quicksort_parallel(arr, 0, n - 1);
        }
    }
}
#else
static void dual_pivot_quicksort(int arr[], long n) {
    dual_pivot_quicksort_seq(arr, 0, n - 1);
}
#endif

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
#endif
    FILE *f = fopen(argv[1], "r");
    if (!f) {
        perror("fopen");
        return 1;
    }
    int seed;
    long n;
    if (fscanf(f, "%d", &seed) != 1 || fscanf(f, "%ld", &n) != 1) {
        fclose(f);
        fprintf(stderr, "Invalid input file\n");
        return 1;
    }
    fclose(f);

    if (n <= 0) {
        fprintf(stderr, "Invalid input size\n");
        return 1;
    }

    int *a = malloc(n * sizeof(int));
    if (!a) {
        perror("malloc");
        return 1;
    }
    srand((unsigned)seed);

    /* ---- Escolha da distribuicao: deixe apenas UMA linha ativa ---- */
    fill_uniform(a, n);
    /* fill_normal(a, n, 500.0, 150.0); */               /* mu=500, sigma=150 */
    /* fill_exponential(a, n, 1.0, 200.0); */            /* lambda=1, scale=200 */

    dual_pivot_quicksort(a, n);

    int is_sorted = 1;
#ifdef PARALLEL
#pragma omp parallel for reduction(&:is_sorted)
    for (long i = 0; i < n - 1; i++) {
        is_sorted &= (a[i] <= a[i + 1]);
    }
#else
    for (long i = 0; i < n - 1; i++) {
        if (a[i] > a[i + 1]) {
            is_sorted = 0;
            break;
        }
    }
#endif
    printf("Array esta ordenado: %s\n", is_sorted ? "SIM" : "NAO");
    free(a);
    return is_sorted ? 0 : 1;
}
