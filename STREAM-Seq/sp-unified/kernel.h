/*
 * kernel.h
 */

#ifndef KERNEL_H_
#define KERNEL_H_

#ifdef DP
#define REAL double
#define REAL4 double4
#else
#define REAL float
#define REAL4 float4
#endif


//local work size, define here if kernel.cl needs this
//#define BLOCK_SIZE 128


#ifdef __cplusplus
extern "C"
{
#endif

//NDRange first worksize.dim0, then worksize.dim1
#pragma omp target device(opencl) copy_deps ndrange(1, iNumElements, iLwSize) file(kernel.cl)
#pragma omp task in(a[0;iNumElements]) out(c[0;iNumElements])
__kernel void CopyOCL(__global REAL* a,
                      __global REAL* c,
                      int iNumElements,
                      int iLwSize);

#pragma omp target device(opencl) copy_deps ndrange(1, iNumElements, iLwSize) file(kernel.cl)
#pragma omp task in(c[0;iNumElements]) out(b[0;iNumElements])
__kernel void ScaleOCL(__global REAL* c,
                       __global REAL* b,
                       REAL scalar,
                       int iNumElements,
                       int iLwSize);

#pragma omp target device(opencl) copy_deps ndrange(1, iNumElements, iLwSize) file(kernel.cl)
#pragma omp task in(a[0;iNumElements], b[0;iNumElements]) out(c[0;iNumElements])
__kernel void AddOCL(__global REAL* a,
                     __global REAL* b,
                     __global REAL* c,
                     int iNumElements,
                     int iLwSize);

#pragma omp target device(opencl) copy_deps ndrange(1, iNumElements, iLwSize) file(kernel.cl)
#pragma omp task in(b[0;iNumElements], c[0;iNumElements]) out(a[0;iNumElements])
__kernel void TriadOCL(__global REAL* b,
                       __global REAL* c,
                       __global REAL* a,
                       REAL scalar,
                       int iNumElements,
                       int iLwSize);

#ifdef __cplusplus
}
#endif


#endif /* KERNEL_H_ */
