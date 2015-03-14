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

/*
#ifdef __cplusplus
extern "C"
{
#endif

//NDRange first worksize.dim0, then worksize.dim1
#pragma omp target device(opencl) copy_deps ndrange(2, col, iNumElements, iLwSize, iLwSize) file(kernel.cl)
#pragma omp task in(input_temp[0;(iNumElements+2)*col], power[0;(iNumElements+2)*col]) out(output_temp[0;iNumElements*col])
//argument iOffset is not used, it is used in the host
__kernel void HotspotOCL(__global REAL* input_temp,
                         __global REAL* output_temp,
                         __global REAL* power,
                         int row,
                         int col,
                         REAL Cap,
                         REAL Rx,
                         REAL Ry,
                         REAL Rz,
                         REAL step,
                         REAL amb_temp,
                         int iNumElements,
                         int iLwSize,
                  	  	 int iOffset);

#ifdef __cplusplus
}
#endif
*/

#endif /* KERNEL_H_ */
