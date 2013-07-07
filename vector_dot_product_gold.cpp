#include <stdio.h>
#include <math.h>
#include <float.h>

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" float compute_gold( float *, float *, unsigned int);

float compute_gold( float* A, float* B, unsigned int num_elements){
  unsigned int i;
  double dot_product = 0.0f; 

  for( i = 0; i < num_elements; i++) 
			 dot_product += A[i] * B[i];

  return (float)dot_product;
}

