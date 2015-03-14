/*
 * kernel.h
 */

#ifndef KERNEL_H_
#define KERNEL_H_

#ifdef DP
#define REAL double
#else
#define REAL float
#endif

//local work size
#define BLOCK_SIZE 32

/*
#ifdef __cplusplus
extern "C"
{
#endif

//NDRange first worksize.dim0, then worksize.dim1
#pragma omp target device(opencl) copy_deps ndrange(2, iWB, iNumElements, iLwSize, iLwSize) file(kernel.cl) implements(matmulsmp)
#pragma omp task in(A[0;iNumElements*iWA], B[0;iWA*iWB]) out(C[0;iNumElements*iWB])
__kernel void matmulocl(__global REAL* A, __global REAL* B, __global REAL* C, int iNumElements, int iOffset, int iWA, int iWB, int iLwSize);

#ifdef __cplusplus
}
#endif
*/


#endif /* KERNEL_H_ */
