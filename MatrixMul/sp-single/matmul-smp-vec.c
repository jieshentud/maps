#include "jutil.h"

void matmulsmp(REAL * __restrict A, REAL * __restrict B, REAL * __restrict C, int iWA, int iWB, int iLwSize)
{
	int i, j, k;
	for(i = 0; i < iLwSize; i++)
		for(j = 0; j < iWB; j++){
			REAL sum = 0;
			for(k = 0; k < iWA; k++)
				sum += A[i * iWA + k] * B[k * iWB + j];
			C[i * iWB + j] = sum;
		}
}
