#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define WALL_TEMP 20.0
#define FIREPLACE_TEMP 100.0
#define FIREPLACE_START 3
#define FIREPLACE_END 7
#define BODY_TEMP 37.0
#define BODY_START 4
#define BODY_END 6
#define ROOM_SIZE 10

// Check if (i, j) lies inside the fixed 37 C body
__host__ __device__ bool is_inside_body(int i, int j, int n) {
    int body_start = (n * BODY_START) / ROOM_SIZE;
    int body_end = (n * BODY_END) / ROOM_SIZE;
    return (i >= body_start && i < body_end && j >= body_start && j < body_end);
}

// GPU kernel: skip outer walls and the 37 C body
__global__ void jacobi_step(double *h, double *g, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < n - 1 && j > 0 && j < n - 1 && !is_inside_body(i, j, n)) {
        g[i * n + j] = 0.25 * (h[(i - 1) * n + j] + h[(i + 1) * n + j] +
                               h[i * n + (j - 1)] + h[i * n + (j + 1)]);
    }
}

// Sequential CPU Jacobi reference
double* jacobi_cpu(double *h, double *g, int n, int iter_limit) {
    for (int iter = 0; iter < iter_limit; iter++) {
        for (int i = 1; i < n - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                if (!is_inside_body(i, j, n)) {
                    g[i * n + j] = 0.25 * (h[(i - 1) * n + j] + h[(i + 1) * n + j] +
                                           h[i * n + (j - 1)] + h[i * n + (j + 1)]);
                }
            }
        }
        double *temp = h;
        h = g;
        g = temp;
    }
    return h;
}

// Initialize room walls, hot body (37 C), and fireplace (100 C)
void initialize_room(double *h, int n) {
    int fireplace_start = (n * FIREPLACE_START) / ROOM_SIZE;
    int fireplace_end = (n * FIREPLACE_END) / ROOM_SIZE;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == 0 || i == n - 1 || j == 0 || j == n - 1) {
                h[i * n + j] = WALL_TEMP;
            } else if (is_inside_body(i, j, n)) {
                h[i * n + j] = BODY_TEMP;
            } else {
                h[i * n + j] = 0.0;
            }
        }
    }

    for (int j = fireplace_start; j < fireplace_end; j++) {
        h[(n - 1) * n + j] = FIREPLACE_TEMP;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        printf("Usage: %s <n> <iter_limit> <threads_per_block> <blocks_per_grid>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int iter_limit = atoi(argv[2]);
    int t = atoi(argv[3]);
    int b = atoi(argv[4]);

    if (b * t < n) {
        printf("Warning: Total threads (%d) < grid size (%d). Result will be incomplete!\n", b * t, n);
    }

    size_t bytes = n * n * sizeof(double);

    // Allocate and initialize host arrays
    double *cpu_h = (double *)malloc(bytes);
    double *cpu_g = (double *)malloc(bytes);
    double *gpu_h = (double *)malloc(bytes);
    initialize_room(cpu_h, n);
    initialize_room(cpu_g, n);
    initialize_room(gpu_h, n);

    // 1. Run and time CPU version
    struct timespec cpu_start, cpu_end;
    clock_gettime(CLOCK_MONOTONIC, &cpu_start);
    double *cpu_res = jacobi_cpu(cpu_h, cpu_g, n, iter_limit);
    clock_gettime(CLOCK_MONOTONIC, &cpu_end);
    double cpu_ms = (cpu_end.tv_sec - cpu_start.tv_sec) * 1000.0 +
                    (cpu_end.tv_nsec - cpu_start.tv_nsec) / 1e6;

    // 2. Allocate GPU memory and create CUDA events
    double *d_h, *d_g;
    cudaMalloc(&d_h, bytes);
    cudaMalloc(&d_g, bytes);

    cudaEvent_t start_h2d, stop_h2d, start_kern, stop_kern, start_d2h, stop_d2h;
    cudaEventCreate(&start_h2d); cudaEventCreate(&stop_h2d);
    cudaEventCreate(&start_kern); cudaEventCreate(&stop_kern);
    cudaEventCreate(&start_d2h); cudaEventCreate(&stop_d2h);

    // Measure Host-to-Device transfer
    cudaEventRecord(start_h2d);
    cudaMemcpy(d_h, gpu_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, gpu_h, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop_h2d);
    cudaEventSynchronize(stop_h2d);

    // Configure 2D grid and blocks
    dim3 threadsPerBlock(t, t);
    dim3 numBlocks(b, b);

    // Measure GPU kernel execution
    cudaEventRecord(start_kern);
    for (int iter = 0; iter < iter_limit; iter++) {
        jacobi_step<<<numBlocks, threadsPerBlock>>>(d_h, d_g, n);
        double *temp = d_h;
        d_h = d_g;
        d_g = temp;
    }
    cudaEventRecord(stop_kern);
    cudaEventSynchronize(stop_kern);

    // Measure Device-to-Host transfer
    cudaEventRecord(start_d2h);
    cudaMemcpy(gpu_h, d_h, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_d2h);
    cudaEventSynchronize(stop_d2h);

    // Calculate elapsed times and speedup
    float h2d_ms = 0, kern_ms = 0, d2h_ms = 0;
    cudaEventElapsedTime(&h2d_ms, start_h2d, stop_h2d);
    cudaEventElapsedTime(&kern_ms, start_kern, stop_kern);
    cudaEventElapsedTime(&d2h_ms, start_d2h, stop_d2h);
    float total_gpu_ms = h2d_ms + kern_ms + d2h_ms;

    // 3. Validate GPU result against CPU
    double max_diff = 0.0;
    for (int i = 0; i < n * n; i++) {
        double diff = fabs(cpu_res[i] - gpu_h[i]);
        if (diff > max_diff) max_diff = diff;
    }

    // Print metrics
    printf("Config: n=%d, iter=%d, t=%d, b=%d\n", n, iter_limit, t, b);
    printf("Validation: %s (max diff: %e)\n", (max_diff < 1e-6) ? "PASS" : "FAIL", max_diff);
    printf("CPU Time:        %.3f ms\n", cpu_ms);
    printf("GPU H2D Time:    %.3f ms\n", h2d_ms);
    printf("GPU Kernel Time: %.3f ms\n", kern_ms);
    printf("GPU D2H Time:    %.3f ms\n", d2h_ms);
    printf("GPU Total Time:  %.3f ms\n", total_gpu_ms);
    printf("Speedup (Total): %.2fx | Speedup (Kernel): %.2fx\n",
           cpu_ms / total_gpu_ms, cpu_ms / kern_ms);

    // Cleanup
    cudaEventDestroy(start_h2d); cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_kern); cudaEventDestroy(stop_kern);
    cudaEventDestroy(start_d2h); cudaEventDestroy(stop_d2h);
    cudaFree(d_h); cudaFree(d_g);
    free(cpu_h); free(cpu_g); free(gpu_h);

    return 0;
}