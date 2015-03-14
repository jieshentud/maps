/*
 * matmul.c
 *
 *  Created on: Dec 16, 2014
 *  Ported to OmpSs by Jie Shen <j.shen@tudelft.nl>
 *  Delft University of Technology
 *
 */

#include "jutil.h"
#include <kernel.h>


void RunKernelOCL(REAL* pHostA, REAL* pHostB, REAL* pHostC, int iNumElements, int iOffset, int iWA, int iWB, int iLwSize)
{
	//call opencl kernel
	matmulocl(&(pHostA[iOffset * iWA]), pHostB, &(pHostC[iOffset * iWB]), iNumElements, iOffset, iWA, iWB, iLwSize);
}

#pragma omp target device(smp) copy_deps
#pragma omp task in(A[0;iLwSize*iWA], B[0;iWA*iWB]) out(C[0;iLwSize*iWB])
void matmulsmp(REAL *A, REAL *B, REAL *C, int iWA, int iWB, int iLwSize);


void RunKernelSMP(REAL* pHostA, REAL* pHostB, REAL* pHostC, int iNumElements, int iOffset, int iWA, int iWB, int iLwSize)
{
	int i;
	int iRemainder = iNumElements % iLwSize;
	int iNumTasks = iNumElements / iLwSize + (iRemainder ? 1 : 0);
	int iTaskSize = iLwSize;
#ifdef PRINT
	printf("SMP NumTasks = %d, Remainder = %d\n", iNumTasks, iRemainder);
#endif
	for(i = 0; i < iNumTasks; i++){
		//call ompss task
		if(i == (iNumTasks-1) && iRemainder != 0){
			iTaskSize = iRemainder; //set the remaining task size if the remainder is not 0
		}
#ifdef PRINT
		printf("Task-%d, iTaksSize = %d\n", i, iTaskSize);
#endif
		matmulsmp(&(pHostA[iOffset * iWA + i * iLwSize * iWA]), pHostB, &(pHostC[iOffset * iWB + i * iLwSize * iWB]), iWA, iWB, iTaskSize);
	}
}

int main(int argc, char** argv)
{
	int N;
	double fN;
	int iNumElementsSingle, iNumElementsOCL, iNumElementsSMP;
	int iOffsetSingle,iOffsetOCL, iOffsetSMP;
	int iLwSizeSingle, iLwSizeOCL, iLwSizeSMP;
	int iHA, iWA, iWB;
	int iPlatform;  //0, OCL; 1, SMP; 2, Both
	int iWarpSize = 32;

	if(argc != 5 && argc != 6){
		printf("argc = %d\n", argc);
		printf("matmul <iPlatform - 0, OCL; 1, SMP> <N> <fN> <LwSize> OR\n");
		printf("matmul <iPlatform - 2, Both> <N> <fN> <LwSizeOCL> <LwSizeSMP>\n");
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

		iHA = iNumElementsSingle;
		iWA = N;
		iWB = N;
		printf("\nMatrix dims: A(%i x %i), B(%i x %i), C(%i x %i)\n", iHA, iWA, iWA, iWB, iHA, iWB);
		printf("Size of 3 arrays = %f (KB) = %ld (Bytes)\n", (double)(iHA * iWA + iWA * iWB + iHA * iWB) * sizeof(REAL) / 1024.0, (iHA * iWA + iWA * iWB + iHA * iWB) * sizeof(REAL));
		printf("offset = %i, # elements in iHA = %i\n\n", iOffsetSingle, iNumElementsSingle);

		REAL* pHostA = (REAL*)malloc(sizeof(REAL) * iHA * iWA);
		REAL* pHostB = (REAL*)malloc(sizeof(REAL) * iWA * iWB);
		REAL* pHostC = (REAL*)malloc(sizeof(REAL) * iHA * iWB);
		InitMatrix(pHostA, iHA, iWA);
		InitMatrix(pHostB, iWA, iWB);
		memset(pHostC, 0, sizeof(REAL) * iHA * iWB);

		if(iPlatform == 0){
#ifdef PROFILING
		double dTOCLStart = 0.0, dTOCLEnd = 0.0, dTOCL = 0.0;
		int i, iStart = -10, iEnd = 10;
		for(i = iStart; i < iEnd; i++){
			if(i == 0)
				dTOCLStart = GetTimeInMs();
#endif
			RunKernelOCL(pHostA, pHostB, pHostC, iNumElementsSingle, iOffsetSingle, iWA, iWB, iLwSizeSingle);
			#pragma omp taskwait
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
			RunKernelSMP(pHostA, pHostB, pHostC, iNumElementsSingle, iOffsetSingle, iWA, iWB, iLwSizeSingle);
			#pragma omp taskwait
#ifdef PROFILING
		}
		dTSMPEnd = GetTimeInMs();
		dTSMP = (dTSMPEnd - dTSMPStart) / (double)iEnd;
		printf("Timing: %.6f / %d = %.6f ms\n", dTSMPEnd - dTSMPStart, iEnd, dTSMP);
		fprintf(pFTime, "SMP: %.6f\n", dTSMP);
		fclose(pFTime);
#endif
		}

        if(pHostA != NULL) free(pHostA);
        if(pHostB != NULL) free(pHostB);
        if(pHostC != NULL) free(pHostC);
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

		iHA = N;
		iWA = N;
		iWB = N;
		printf("\nMatrix dims: A(%i x %i), B(%i x %i), C(%i x %i)\n", iHA, iWA, iWA, iWB, iHA, iWB);
		printf("Size of 3 arrays = %f (KB) = %ld (Bytes)\n", (double)(iHA * iWA + iWA * iWB + iHA * iWB) * sizeof(REAL) / 1024.0, (iHA * iWA + iWA * iWB + iHA * iWB) * sizeof(REAL));
		printf("OCL: offset = %i, # elements in iHA = %i, size of 3 arrays = %f (KB) = %ld (Bytes)\n", iOffsetOCL, iNumElementsOCL, (double)(iNumElementsOCL * iWA + iWA * iWB + iNumElementsOCL * iWB) * sizeof(REAL) / 1024.0, (iNumElementsOCL * iWA + iWA * iWB + iNumElementsOCL * iWB) * sizeof(REAL));
		printf("SMP: offset = %i, # elements in iHA = %i, size of 3 arrays = %f (KB) = %ld (Bytes)\n\n", iOffsetSMP, iNumElementsSMP, (double)(iNumElementsSMP * iWA + iWA * iWB + iNumElementsSMP * iWB) * sizeof(REAL) / 1024.0, (iNumElementsSMP * iWA + iWA * iWB + iNumElementsSMP * iWB) * sizeof(REAL));

		REAL* pHostA = (REAL*)malloc(sizeof(REAL) * iHA * iWA);
		REAL* pHostB = (REAL*)malloc(sizeof(REAL) * iWA * iWB);
		REAL* pHostC = (REAL*)malloc(sizeof(REAL) * iHA * iWB);
		InitMatrix(pHostA, iHA, iWA);
		InitMatrix(pHostB, iWA, iWB);
		memset(pHostC, 0, sizeof(REAL) * iHA * iWB);

		if(iNumElementsOCL == 0){ //run only smp
			//RunKernelSMP(pHostA, pHostB, pHostC, iNumElementsSMP, iOffsetSMP, iWA, iWB, iLwSizeSMP);
			//#pragma omp taskwait
			printf("run only smp\n");
			fprintf(pFTime, "Both: only smp\n");
		}
		if(iNumElementsSMP == 0){ //run only ocl
			//RunKernelOCL(pHostA, pHostB, pHostC, iNumElementsOCL, iOffsetOCL, iWA, iWB, iLwSizeOCL);
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
			RunKernelOCL(pHostA, pHostB, pHostC, iNumElementsOCL, iOffsetOCL, iWA, iWB, iLwSizeOCL);
			RunKernelSMP(pHostA, pHostB, pHostC, iNumElementsSMP, iOffsetSMP, iWA, iWB, iLwSizeSMP);
			#pragma omp taskwait
#ifdef PROFILING
		}
		dTBothEnd = GetTimeInMs();
		dTBoth = (dTBothEnd - dTBothStart) / (double)iEnd;
		printf("Timing: %.6f / %d = %.6f ms\n", dTBothEnd - dTBothStart, iEnd, dTBoth);
		fprintf(pFTime, "Both: %.6f\n", dTBoth);
		fclose(pFTime);
#endif
		}
        
        if(pHostA != NULL) free(pHostA);
        if(pHostB != NULL) free(pHostB);
        if(pHostC != NULL) free(pHostC);
	}

	printf("DONE\n\n");

	return 0;

}
