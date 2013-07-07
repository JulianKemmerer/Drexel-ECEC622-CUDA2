// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include <vector_dot_product_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void run_test(unsigned int);
float compute_on_device(float *, float *,int);
void check_for_error(char *);
extern "C" float compute_gold( float *, float *, unsigned int);

int main( int argc, char** argv) 
{
	if(argc != 2){
		printf("Usage: vector_dot_product <num elements> \n");
		exit(0);	
	}
	unsigned int num_elements = atoi(argv[1]);
	run_test(num_elements);
	return 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Perform vector dot product on the CPU and the GPU and compare results for correctness
////////////////////////////////////////////////////////////////////////////////
void run_test(unsigned int num_elements) 
{
	// Obtain the vector length
	unsigned int size = sizeof(float) * num_elements;

	// Allocate memory on the CPU for the input vectors A and B
	float *A = (float *)malloc(size);
	float *B = (float *)malloc(size);
	
	// Randomly generate input data. Initialize the input data to be floating point values between [-.5 , 5]
	printf("Generating random vectors with values between [-.5, .5]. \n");	
	srand(time(NULL));
	for(unsigned int i = 0; i < num_elements; i++){
		A[i] = (float)rand()/(float)RAND_MAX - 0.5;
		B[i] = (float)rand()/(float)RAND_MAX - 0.5;
	}
	
	printf("Generating dot product on the CPU. \n");
	float time;
	unsigned int cpu_timer;
	cutCreateTimer(&cpu_timer);
	cutStartTimer(cpu_timer);
	
	// Compute the reference solution on the CPU
	float reference = compute_gold(A, B, num_elements);
    
	cutStopTimer(cpu_timer);
	time = 1e-3 * cutGetTimerValue(cpu_timer);
	printf("CPU run time: %0.10f s\n", time);

	// Edit this function to compute the result vector on the GPU. The result should be placed in the gpu_result variable
	float gpu_result = compute_on_device(A, B, num_elements);

	printf("Result on CPU: %f, result on GPU (using %d threads per block): %f. \n", reference, BLOCK_SIZE, gpu_result);

	// cleanup memory
	free(A);
	free(B);
	
	return;
}

/* Edit this function to compute the dot product on the device using atomic intrinsics. */
float compute_on_device(float *A_on_host, float *B_on_host, int num_elements)
{
	float *A_on_device = NULL;
	float *B_on_device = NULL;
	float *C_on_device = NULL;

	/* alloc space on device for initial vectors, copy data */
	cudaMalloc( (void **)&A_on_device, num_elements * sizeof(float) );
	cudaMemcpy( A_on_device, A_on_host, num_elements * sizeof(float), cudaMemcpyHostToDevice);	

	cudaMalloc( (void **)&B_on_device, num_elements * sizeof(float) );
	cudaMemcpy( B_on_device, B_on_host, num_elements * sizeof(float), cudaMemcpyHostToDevice);

	/* alloc space for result, copy data */
	cudaMalloc( (void **)&C_on_device, GRID_SIZE * sizeof(float) ); // is vector instead of single val for testing purposes
	cudaMemset( C_on_device, 0.0f, GRID_SIZE * sizeof(float) );

	/* mutex for sync */
	int *mutex = NULL;
        cudaMalloc((void **)&mutex, sizeof(int));
        cudaMemset(mutex, 0, sizeof(int));

	/* Define grid parameters for GPU */
	dim3 thread_block(BLOCK_SIZE, 1, 1);
	dim3 grid(GRID_SIZE,1);

	/* start timer */
	float time;
	unsigned int gpu_timer;
	cutCreateTimer(&gpu_timer);
	cutStartTimer(gpu_timer);

	/* Launch kernel, sync ( for timing purposes ) */
	vector_dot_product_kernel <<< grid, thread_block >>> (A_on_device, B_on_device, C_on_device, num_elements,mutex);
	cudaThreadSynchronize();
	check_for_error("KERNEL FAILURE");

	/* end timer */
	cutStopTimer(gpu_timer);
	time = 1e-3 * cutGetTimerValue(gpu_timer);
	printf("GPU run time: %0.10f s\n", time);
	
	/* copy result back to host */
	float *C_host = (float *) malloc(GRID_SIZE*sizeof(float));
	float result = 0.0f;
	cudaMemcpy( &result, C_on_device, sizeof(float), cudaMemcpyDeviceToHost );
	
	/* Free mem on GPU */
	cudaFree(A_on_device);
	cudaFree(B_on_device);
	cudaFree(C_on_device);
	
	return result;
}
 
// This function checks for errors returned by the CUDA run time
void check_for_error(char *msg){
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err){
		printf("CUDA ERROR: %s (%s). \n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
} 
