/*
 * nbody.c
 *
 *  Created on: Feb 4, 2015
 *  Jie Shen <j.shen@tudelft.nl>
 *  Delft University of Technology
 *
 */

#include "jutil.h"
#include <kernel.h>

void initialiseSystemRandomly(REAL* positions, REAL* velocities, int iNumElements);
void computeReferenceResult(REAL* input_positions, REAL* input_velocities, REAL* output_positions, REAL* output_velocities,
				const REAL dT, const REAL damping, const REAL softeningSquared, int iNumElements, int iNumIterations);
void verifyResults(REAL* positions, REAL* velocities,
		REAL* golden_positions, REAL* golden_velocities,  int iNumElements);
void printResults(REAL* positions, REAL* velocities,
		REAL* golden_positions, REAL* golden_velocities,  int iNumElements);

#pragma omp target device(smp) copy_deps
#pragma omp task in(input_positions[0;N*4], input_velocities[0;N*4]) out(output_positions[0;iTaskSize*4], output_velocities[0;iTaskSize*4])
void NBodySMP(REAL* __restrict input_positions, REAL* __restrict input_velocities, REAL* __restrict output_positions, REAL* __restrict output_velocities,
		const REAL dT, const REAL damping, const REAL softeningSquared, int N, int iTaskOffset, int iTaskSize, int iLwSize);

#ifdef __cplusplus
extern "C"
{
#endif

//NDRange first worksize.dim0, then worksize.dim1
#pragma omp target device(opencl) copy_deps ndrange(1, iNumElements, iLwSize) file(kernel.cl) implements(NBodySMP)
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

void RunKernelBoth(REAL* input_positions, REAL* input_velocities, REAL* output_positions, REAL* output_velocities,
		const REAL dT, const REAL damping, const REAL softeningSquared, int N, int iTaskSize, int iLwSize)
{
	int i;
	int iRemainder = N % iTaskSize;
	int iNumTasks = N / iTaskSize;
	int iTaskOffset;
#ifdef PRINT
	printf("Total NumTasks = %d, Remainder = %d\n", iNumTasks, iRemainder);
#endif
	for(i = 0; i < iNumTasks; i++){
		int iTaskOffset = i * iTaskSize;
		//call NBodySMP and NBodyOCL tasks
		NBodySMP(
				input_positions,   //each task sees all N elements of input
				input_velocities,
				&output_positions[iTaskOffset*4],  //each task sees iTaskSize of output, starting from iTaskOffset*4
				&output_velocities[iTaskOffset*4],
				dT,
				damping,
				softeningSquared,
				N,
				iTaskOffset,
				iTaskSize,
				iLwSize
				);

	}
}

int main(int argc, char** argv)
{
	int N, iTaskSize, iLwSize;
	int iPlatform;  //0, OCL; 1, SMP; 2, Both
	int iWarpSize = 32;

	int iNumIterations = 20;
#ifdef DP
	const REAL dT = 0.001;
	const REAL damping = 0.995;
	const REAL softeningSquared = 0.00125
#else
	const REAL dT = 0.001f;
	const REAL damping = 0.995f;
	const REAL softeningSquared = 0.00125f;
#endif

	if(argc != 5){
		printf("argc = %d\n", argc);
		printf("nbody <iPlatform - 2, Both> <N> <TaskSize> <LwSize> OR\n");
		printf("Error, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		exit(EXIT_FAILURE);
	}
	iPlatform = atoi(argv[1]);
	N = atoi(argv[2]);
	iTaskSize = atoi(argv[3]);
	iLwSize = atoi(argv[4]);
	if(iLwSize % iWarpSize != 0){
		int temp = RoundUp(iWarpSize, iLwSize);
		iLwSize = temp;
	}
	if(iTaskSize % iLwSize != 0){
		int temp = RoundUp(iLwSize, iTaskSize);
		iTaskSize = temp;
	}
	if(N % iTaskSize != 0){
		int temp = RoundUp(iTaskSize, N);
		N = temp;
	}

	if(iPlatform != 2){
		printf("Error, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		exit(EXIT_FAILURE);
	}
	printf("\nPlatform = %s, N = %d, TaskSize = %d, LwSize = %d\n\n", "Both", N, iTaskSize, iLwSize);

	printf("Size of 4 arrays = %f (KB) = %ld (Bytes)\n", (double)(N * sizeof(REAL) * 4 * 4) / 1024.0, N * sizeof(REAL) * 4 * 4);

	//set the host side memory buffers
	REAL* orig_positions = (REAL*)malloc(sizeof(REAL) * N * 4);
	REAL* orig_velocities = (REAL*)malloc(sizeof(REAL) * N * 4);
	initialiseSystemRandomly(orig_positions, orig_velocities, N);

	REAL* input_positions = (REAL*)malloc(sizeof(REAL) * N * 4);
	REAL* input_velocities = (REAL*)malloc(sizeof(REAL) * N * 4);
	REAL* output_positions = (REAL*)malloc(sizeof(REAL) * N * 4);
	REAL* output_velocities = (REAL*)malloc(sizeof(REAL) * N * 4);
	memcpy(input_positions, orig_positions, sizeof(REAL) * N * 4);
	memcpy(input_velocities, orig_velocities, sizeof(REAL) * N * 4);
	memset(output_positions, 0, sizeof(REAL) * N * 4);
	memset(output_velocities, 0, sizeof(REAL) * N * 4);

#ifdef PROFILING
	FILE* pFTime;
	pFTime = fopen ("timing","w");
	if(pFTime == NULL){
		printf("Error, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		exit(EXIT_FAILURE);
	}

	double dTBothStart = 0.0, dTBothEnd = 0.0, dTBoth = 0.0;
	int i, iStart = -10, iEnd = 10;
	for(i = iStart; i < iEnd; i++){
		if(i == 0)
			dTBothStart = GetTimeInMs();
#endif
		RunKernelBoth(input_positions, input_velocities, output_positions, output_velocities,
				dT, damping, softeningSquared, N, iTaskSize, iLwSize);
		#pragma omp taskwait
		if(iNumIterations != 1){
			REAL* temp_positions = input_positions;
			REAL* temp_velocities = input_velocities;
			input_positions = output_positions;
			input_velocities = output_velocities;
			output_positions = temp_positions;
			output_velocities = temp_velocities;
		}
#ifdef PROFILING
	}
	dTBothEnd = GetTimeInMs();
	dTBoth = (dTBothEnd - dTBothStart) / (double)iEnd;
	printf("Timing: %.6f / %d = %.6f ms\n", dTBothEnd - dTBothStart, iEnd, dTBoth);
	fprintf(pFTime, "Both: %.6f\n", dTBoth);
	fclose(pFTime);
#endif

	REAL* golden_input_positions = (REAL*)malloc(sizeof(REAL) * N * 4);
	REAL* golden_input_velocities = (REAL*)malloc(sizeof(REAL) * N * 4);
	REAL* golden_output_positions = (REAL*)malloc(sizeof(REAL) * N * 4);
	REAL* golden_output_velocities = (REAL*)malloc(sizeof(REAL) * N * 4);
	memcpy(golden_input_positions, orig_positions, sizeof(REAL) * N * 4);
	memcpy(golden_input_velocities, orig_velocities, sizeof(REAL) * N * 4);
	memset(golden_output_positions, 0, sizeof(REAL) * N * 4);
	memset(golden_output_velocities, 0, sizeof(REAL) * N * 4);

	computeReferenceResult(golden_input_positions, golden_input_velocities, golden_output_positions, golden_output_velocities,
			dT, damping, softeningSquared, N, iNumIterations);
	if(iNumIterations % 2  == 1){
		verifyResults(output_positions, output_velocities,
				golden_output_positions, golden_output_velocities, N);
		//printResults(output_positions, output_velocities,
		//		golden_output_positions, golden_output_velocities, N);
	}
	if(iNumIterations % 2  == 0){
		verifyResults(input_positions, input_velocities,
				golden_input_positions, golden_input_velocities, N);
		//printResults(input_positions, input_velocities,
		//		golden_input_positions, golden_input_velocities, N);
	}

	if(input_positions != NULL) free(input_positions);
	if(input_velocities != NULL) free(input_velocities);
	if(output_positions != NULL) free(output_positions);
	if(output_velocities != NULL) free(output_velocities);

	if(golden_input_positions != NULL) free(golden_input_positions);
	if(golden_input_velocities != NULL) free(golden_input_velocities);
	if(golden_output_positions != NULL) free(golden_output_positions);
	if(golden_output_velocities != NULL) free(golden_output_velocities);

	printf("DONE\n\n");

	return 0;
}
