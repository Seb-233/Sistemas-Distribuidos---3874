/*#######################################################################################
 # Multiplicación de matrices (algoritmo clásico) con OpenMP
 # Uso: ./mmClasicaOpenMP N [hilos]
 #   N: tamaño de la matriz (NxN)
 #   hilos (opcional): número de hilos OpenMP (equivalente a OMP_NUM_THREADS)
 #
 # El programa imprime una línea con: N,threads,time_sec
 ######################################################################################*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

static double t0 = 0.0, t1 = 0.0;

static void InicioMuestra(void){ t0 = omp_get_wtime(); }
static void FinMuestra(void){ t1 = omp_get_wtime(); }

static void impMatrix(const double *A, int N){
    if (N > 10){ puts("(Matriz omitida, N>10)"); return; }
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++) printf("%7.2f ", A[i*(long)N + j]);
        putchar('\n');
    }
}

static void fill(double *A, int N){
    unsigned int seed = 1234567u;
    for (long i=0; i<(long)N*(long)N; i++){
        seed = 1103515245u*seed + 12345u;
        A[i] = (double)(seed % 1000) / 100.0; // 0.00..9.99 reproducible
    }
}

static void multiMatrix(const double *A, const double *B, double *C, int N){
    #pragma omp parallel for schedule(static)
    for (long i=0;i<(long)N*(long)N;i++) C[i]=0.0;

    // i-k-j para mejorar localidad de B; paralelizamos por i
    #pragma omp parallel for schedule(static)
    for (int i=0;i<N;i++){
        for (int k=0;k<N;k++){
            double aik = A[i*(long)N + k];
            const double *Bk = &B[k*(long)N];
            double *Ci = &C[i*(long)N];
            for (int j=0;j<N;j++){
                Ci[j] += aik * Bk[j];
            }
        }
    }
}

int main(int argc, char **argv){
    if (argc < 2){
        fprintf(stderr, "Uso: %s N [hilos]\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    if (N <= 0){ fprintf(stderr, "N debe ser > 0\n"); return 1; }

    if (argc >= 3){
        int thr = atoi(argv[2]);
        if (thr > 0) omp_set_num_threads(thr);
    }
    int threads = omp_get_max_threads();

    size_t n = (size_t)N*(size_t)N;
    double *A = (double*)malloc(sizeof(double)*n);
    double *B = (double*)malloc(sizeof(double)*n);
    double *C = (double*)malloc(sizeof(double)*n);
    if(!A || !B || !C){ fprintf(stderr, "Error de memoria\n"); return 1; }

    fill(A,N); fill(B,N);

    InicioMuestra();
    multiMatrix(A,B,C,N);
    FinMuestra();

    //impMatrix(C,N); // descomenta si N es pequeño

    double dt = t1 - t0;
    printf("%d,%d,%.6f\n", N, threads, dt);

    free(A); free(B); free(C);
    return 0;
}
