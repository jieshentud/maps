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
#pragma omp target device(opencl) copy_deps ndrange(1, iNumElements, iLwSize) file(kernel.cl)
#pragma omp task in(input_positions[0;N*4], input_velocities[0;N*4]) out(output_positions[0;iNumElements*4], output_velocities[0;iNumElements*4])
__kernel void NBodyOCL(
		__global REAL * input_positions,   //each task see all N elements of input
		__global REAL * input_velocities,
		__global REAL * output_positions,  //each task see iNumElements of output, starting from iOffset*4
		__global REAL * output_velocities,
		const REAL dT,
		const REAL damping,
		const REAL softeningSquared,
		int N,
		int iOffset,
		int iNumElements,
		int iLwSize);

#ifdef __cplusplus
}
#endif
*/

#endif /* KERNEL_H_ */
