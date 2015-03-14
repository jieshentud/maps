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

void RunKernelSMP(REAL* input_positions, REAL* input_velocities, REAL* output_positions, REAL* output_velocities,
						const REAL dT, const REAL damping, const REAL softeningSquared, int N, int iOffset, int iNumElements, int iLwSize)
{
	int i;
	int iRemainder = iNumElements % iLwSize;
	int iNumTasks = iNumElements / iLwSize + (iRemainder ? 1 : 0);
	int iTaskSize = iLwSize;  //the task size
	int iTaskOffset;          //the task offset in GLOBAL view
#ifdef PRINT
	printf("SMP NumTasks = %d, Remainder = %d\n", iNumTasks, iRemainder);
#endif
	for(i = 0; i < iNumTasks; i++){
		//call ompss task
		if(i == (iNumTasks-1) && iRemainder != 0){
			iTaskSize = iRemainder; //set the remaining task size if the remainder is not 0
		}
		iTaskOffset = iOffset + i * iTaskSize;
#ifdef PRINT
		printf("Task-%d, iTaksSize = %d\n", i, iTaskSize);
#endif
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
				iLwSize);
	}
}

void RunKernelOCL(REAL* input_positions, REAL* input_velocities, REAL* output_positions, REAL* output_velocities,
						const REAL dT, const REAL damping, const REAL softeningSquared, int N, int iOffset, int iNumElements, int iLwSize)
{
	NBodyOCL(
			input_positions,   //each task sees all N elements of input
			input_velocities,
			&output_positions[iOffset*4],  //each task sees iNumElements of output, starting from iOffset*4
			&output_velocities[iOffset*4],
			dT,
			damping,
			softeningSquared,
			N,
			iOffset,
			iNumElements,
			iLwSize);
}

int main(int argc, char** argv)
{
	int N;
	double fN;
	int iNumElementsSingle, iNumElementsOCL, iNumElementsSMP;
	int iOffsetSingle,iOffsetOCL, iOffsetSMP;
	int iLwSizeSingle, iLwSizeOCL, iLwSizeSMP;
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

	if(argc != 5 && argc != 6){
		printf("argc = %d\n", argc);
		printf("nbody <iPlatform - 0, OCL; 1, SMP> <N> <fN> <LwSize> OR\n");
		printf("nbody <iPlatform - 2, Both> <N> <fN> <LwSizeOCL> <LwSizeSMP>\n");
		printf("Error, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		exit(EXIT_FAILURE);
	}
	iPlatform = atoi(argv[1]);
	N = atoi(argv[2]);
	fN = atof(argv[3]);  //fN = R_GC / (1 + R_GC + R_GD)

#ifdef PROFILING
	FILE* pFTime;
	pFTime = fopen ("timing","w");
	if(pFTime == NULL){
		printf("Error, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		exit(EXIT_FAILURE);
	}
#endif

	if(iPlatform == 0 || iPlatform == 1){
		iLwSizeSingle = atoi(argv[4]);
		//when using OpenCL, round up LwSize to WarpSize
		if(iPlatform == 0 && iLwSizeSingle % iWarpSize != 0)
		{
			int temp = RoundUp(iWarpSize, iLwSizeSingle);
			iLwSizeSingle = temp;
		}
		printf("\nPlatform = %s, N = %d, fN = %f, LwSize = %d\n\n", ((iPlatform == 0) ? "OCL" : "SMP"), N, fN, iLwSizeSingle);

		iNumElementsSingle = (int)(round(N * fN));
		iOffsetSingle = 0;
		printf("iNumElementsSingle = %d, ", iNumElementsSingle);
		//when using OpenCL, round up NumElements to LwSize
		if(iPlatform == 0 && iNumElementsSingle % iLwSizeSingle != 0){
			int temp = RoundUp(iLwSizeSingle, iNumElementsSingle);
			iNumElementsSingle = temp;
		}
		printf("after rounding, iNumElementsSingle = %d\n", iNumElementsSingle);

		printf("Size of 4 arrays = %f (KB) = %ld (Bytes)\n", (double)(iNumElementsSingle * sizeof(REAL) * 4 * 4) / 1024.0, iNumElementsSingle * sizeof(REAL) * 4 * 4);
		printf("offset = %d, #options = %d\n\n", iOffsetSingle, iNumElementsSingle);

		//set the host side memory buffers
		REAL* orig_positions = (REAL*)malloc(sizeof(REAL) * iNumElementsSingle * 4);
		REAL* orig_velocities = (REAL*)malloc(sizeof(REAL) * iNumElementsSingle * 4);
		initialiseSystemRandomly(orig_positions, orig_velocities, iNumElementsSingle);

		REAL* input_positions = (REAL*)malloc(sizeof(REAL) * iNumElementsSingle * 4);
		REAL* input_velocities = (REAL*)malloc(sizeof(REAL) * iNumElementsSingle * 4);
		REAL* output_positions = (REAL*)malloc(sizeof(REAL) * iNumElementsSingle * 4);
		REAL* output_velocities = (REAL*)malloc(sizeof(REAL) * iNumElementsSingle * 4);
		memcpy(input_positions, orig_positions, sizeof(REAL) * iNumElementsSingle * 4);
		memcpy(input_velocities, orig_velocities, sizeof(REAL) * iNumElementsSingle * 4);
		memset(output_positions, 0, sizeof(REAL) * iNumElementsSingle * 4);
		memset(output_velocities, 0, sizeof(REAL) * iNumElementsSingle * 4);

		if(iPlatform == 0){
#ifdef PROFILING
		double dTOCLStart = 0.0, dTOCLEnd = 0.0, dTOCL = 0.0;
		int i, iStart = -10, iEnd = 10;   //iEnd - iStart = iNumIterations
		for(i = iStart; i < iEnd; i++){
			if(i == 0)
				dTOCLStart = GetTimeInMs();
#endif
					RunKernelOCL(input_positions, input_velocities, output_positions, output_velocities,
							dT, damping, softeningSquared, iNumElementsSingle, iOffsetSingle, iNumElementsSingle, iLwSizeSingle);
				//use taskwait because for the BOTH case, data needs sync after each iteration, and pointers need swapping
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
		dTOCLEnd = GetTimeInMs();
		dTOCL = (dTOCLEnd - dTOCLStart) / (double)iEnd;
		printf("Timing: %.6f / %d = %.6f ms\n", dTOCLEnd - dTOCLStart, iEnd, dTOCL);
		fprintf(pFTime, "OCL: %.6f\n", dTOCL);
		fclose(pFTime);
#endif
		}

		if(iPlatform == 1){
#ifdef PROFILING
		double dTSMPStart = 0.0, dTSMPEnd = 0.0, dTSMP = 0.0;
		int i, iStart = -10, iEnd = 10;
		for(i = iStart; i < iEnd; i++){
			if(i == 0)
				dTSMPStart = GetTimeInMs();
#endif
				RunKernelSMP(input_positions, input_velocities, output_positions, output_velocities,
						dT, damping, softeningSquared, iNumElementsSingle, iOffsetSingle, iNumElementsSingle, iLwSizeSingle);
				//use taskwait to sync all the smp tasks after an iteration, and then swapping the pointers
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
		dTSMPEnd = GetTimeInMs();
		dTSMP = (dTSMPEnd - dTSMPStart) / (double)iEnd;
		printf("Timing: %.6f / %d = %.6f ms\n", dTSMPEnd - dTSMPStart, iEnd, dTSMP);
		fprintf(pFTime, "SMP: %.6f\n", dTSMP);
		fclose(pFTime);
#endif
		}

		REAL* golden_input_positions = (REAL*)malloc(sizeof(REAL) * iNumElementsSingle * 4);
		REAL* golden_input_velocities = (REAL*)malloc(sizeof(REAL) * iNumElementsSingle * 4);
		REAL* golden_output_positions = (REAL*)malloc(sizeof(REAL) * iNumElementsSingle * 4);
		REAL* golden_output_velocities = (REAL*)malloc(sizeof(REAL) * iNumElementsSingle * 4);
		memcpy(golden_input_positions, orig_positions, sizeof(REAL) * iNumElementsSingle * 4);
		memcpy(golden_input_velocities, orig_velocities, sizeof(REAL) * iNumElementsSingle * 4);
		memset(golden_output_positions, 0, sizeof(REAL) * iNumElementsSingle * 4);
		memset(golden_output_velocities, 0, sizeof(REAL) * iNumElementsSingle * 4);

		computeReferenceResult(golden_input_positions, golden_input_velocities, golden_output_positions, golden_output_velocities,
				dT, damping, softeningSquared, iNumElementsSingle, iNumIterations);
		if(iNumIterations % 2  == 1){
			verifyResults(output_positions, output_velocities,
					golden_output_positions, golden_output_velocities, iNumElementsSingle);
			//printResults(output_positions, output_velocities,
			//		golden_output_positions, golden_output_velocities, iNumElementsSingle);
		}
		if(iNumIterations % 2  == 0){
			verifyResults(input_positions, input_velocities,
					golden_input_positions, golden_input_velocities, iNumElementsSingle);
			//printResults(input_positions, input_velocities,
			//		golden_input_positions, golden_input_velocities, iNumElementsSingle);
		}

		if(input_positions != NULL) free(input_positions);
		if(input_velocities != NULL) free(input_velocities);
		if(output_positions != NULL) free(output_positions);
		if(output_velocities != NULL) free(output_velocities);

		if(golden_input_positions != NULL) free(golden_input_positions);
		if(golden_input_velocities != NULL) free(golden_input_velocities);
		if(golden_output_positions != NULL) free(golden_output_positions);
		if(golden_output_velocities != NULL) free(golden_output_velocities);

	}
	if(iPlatform == 2){
		iLwSizeOCL = atoi(argv[4]);
		iLwSizeSMP = atoi(argv[5]);
		//when using OpenCL, round up LwSizeOCL to WarpSize
		if(iLwSizeOCL % iWarpSize != 0)
		{
			int temp = RoundUp(iWarpSize, iLwSizeOCL);
			iLwSizeOCL = temp;
		}
		printf("\nPlatform = %s, N = %d, fN = %f, LwSizeOCL = %d, LwSizeSMP = %d\n\n", "Both", N, fN, iLwSizeOCL, iLwSizeSMP);

		iNumElementsOCL = (int)(round(N * fN));
		iOffsetOCL = 0;
		printf("iNumElementsOCL = %d, ", iNumElementsOCL);
		if(iNumElementsOCL % iLwSizeOCL != 0){
			int temp = RoundUp(iLwSizeOCL, iNumElementsOCL);
			iNumElementsOCL = temp;
		}
		printf("after rounding, iNumElementsOCL = %d\n", iNumElementsOCL);

		iNumElementsSMP = N - iNumElementsOCL;
		iOffsetSMP = iNumElementsOCL;
		printf("iNumElementsSMP = %d\n", iNumElementsSMP);

		printf("Size of 4 arrays = %f (KB) = %ld (Bytes)\n", (double)(N * sizeof(REAL) * 4 * 4) / 1024.0, N * sizeof(REAL) * 4 * 4);
		printf("OCL: offset = %d, #options = %d, size of 4 arrays = %f (KB) = %ld (Bytes)\n", iOffsetOCL, iNumElementsOCL,
						(double)(iNumElementsOCL * sizeof(REAL) * 4 * 4) / 1024.0,
						iNumElementsOCL * sizeof(REAL) * 4 * 4);
		printf("SMP: offset = %d, #options = %d, size of 4 arrays = %f (KB) = %ld (Bytes)\n\n", iOffsetSMP, iNumElementsSMP,
						(double)(iNumElementsSMP * sizeof(REAL) * 4 * 4) / 1024.0,
						iNumElementsSMP * sizeof(REAL) * 4 * 4);

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

		if(iNumElementsOCL == 0){ //run only smp
			//RunKernelSMP(input_positions, input_velocities, output_positions, output_velocities,
			//	dT, damping, softeningSquared, N, iOffsetSMP, iNumElementsSMP, iLwSizeSMP);
			//#pragma omp taskwait
			printf("run only smp\n");
			fprintf(pFTime, "Both: only smp\n");
		}
		if(iNumElementsSMP == 0){ //run only ocl
			//RunKernelOCL(input_positions, input_velocities, output_positions, output_velocities,
			//	dT, damping, softeningSquared, N, iOffsetOCL, iNumElementsOCL, iLwSizeOCL);
			//#pragma omp taskwait
			printf("run only ocl\n");
			fprintf(pFTime, "Both: only ocl\n");
		}
		if(iNumElementsOCL != 0 && iNumElementsSMP != 0){ //run both
#ifdef PROFILING
		double dTBothStart = 0.0, dTBothEnd = 0.0, dTBoth = 0.0;
		int i, iStart = -10, iEnd = 10;
		for(i = iStart; i < iEnd; i++){
			if(i == 0)
				dTBothStart = GetTimeInMs();
#endif
			RunKernelOCL(input_positions, input_velocities, output_positions, output_velocities,
					dT, damping, softeningSquared, N, iOffsetOCL, iNumElementsOCL, iLwSizeOCL);
			RunKernelSMP(input_positions, input_velocities, output_positions, output_velocities,
					dT, damping, softeningSquared, N, iOffsetSMP, iNumElementsSMP, iLwSizeSMP);
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
		}

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
	}

	printf("DONE\n\n");

	return 0;
}
