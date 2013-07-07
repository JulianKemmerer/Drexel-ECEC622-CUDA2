#ifndef _VECTOR_DOT_PRODUCT_KERNEL_H_
#define _VECTOR_DOT_PRODUCT_KERNEL_H_

/* Edit this function to complete the functionality of dot product on the GPU using atomics. 
	You may add other kernel functions as you deem necessary. 
 */

#define BLOCK_SIZE 256
#define GRID_SIZE 240

/* prototypes */
// __device__ float atomicAddFloat( float *address, float val );
__device__ void lock(int *mutex);
__device__ void unlock(int *mutex);

__global__ void vector_dot_product_kernel( float *A, float *B, float *C, unsigned int numElements, int *mutex) {

	__shared__ float thread_sums[ BLOCK_SIZE ];

	/* thread ID and stride lengths (for coalescing memory) */
	unsigned int tID = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int stride_length = blockDim.x * gridDim.x;

	/* initialize local thread sum and starting location for thread*/
	float local_thread_sum = 0.0f;
	unsigned int i = tID;

	/* perform multiplication and add stride_length continuously until num_elements reached -->
		do not need to worry about thread edge cases due to numElements this way	 */
	while( i < numElements ) {

		/* multiply, increment by stride */
		local_thread_sum += A[i] * B[i];
		i += stride_length;
	}

	/* Put thread sum in shared mem */
	thread_sums[threadIdx.x] = local_thread_sum;
	__syncthreads();


	/* REDUCTION -- Reduce thread sums on a per-block basis (so result in one sum per block) */
	i = BLOCK_SIZE / 2;   // assumes block size is power of 2	
	while ( i != 0 ) {

		/* threads where i < 0 are threads on the second "half" which don't need to execute */
		if ( threadIdx.x < i ) {

			/* sum the calculating threads partial value with its second "half" counterpart */
			thread_sums[threadIdx.x] += thread_sums[ threadIdx.x + i ];
		}
		__syncthreads();

		/* reduces the threads by 2 each iteration */
		i = i / 2;
	}

	/* first thread in each block adds block-wide value to global mem location -> need atomic operation here */
	if (threadIdx.x == 0) {
		lock(mutex);
		C[0] += thread_sums[0] ;
		unlock(mutex);
	}
}


// Tried using this, and couldn't get it to work.  Using the mutex functions instead now.
/*  No atomicAdd with float support available , so need this (defined in CUDA guide)   */
/*__device__ float atomicAddFloat( float *address, float val ) {

	unsigned long long int* addr = (unsigned long long int*)address;
	unsigned long long int old = *addr;
	unsigned long long int assumed;

	do {
		assumed = old;
		old = atomicCAS(addr, assumed, __double_as_longlong( val + __longlong_as_double(assumed) ) );
	
	} while (assumed != old);

	return  (float)__longlong_as_double(old);

} */

/* Using CAS to acquire mutex. */
__device__ void lock(int *mutex){
                  while(atomicCAS(mutex, 0, 1) != 0);
}

/* Using exchange to release mutex. */
__device__ void unlock(int *mutex)
{
                  atomicExch(mutex, 0);
}



#endif // #ifndef _VECTOR_DOT_PRODUCT_KERNEL_H
