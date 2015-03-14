/*
 * blackscholes.c
 *
 *  Created on: Jan 28, 2015
 *  Ported to OmpSs by Jie Shen <j.shen@tudelft.nl>
 *  Delft University of Technology
 *
 */

#include "jutil.h"
#include <kernel.h>


void RunKernelOCL(REAL* pCall, REAL* pPut, REAL* pS, REAL * pX, REAL * pT, REAL R, REAL V, int iNumElements, int iOffset, int iLwSize)
{
	//call opencl kernel
	BlackScholesOCL(
			&(pCall[iOffset]),
			&(pPut[iOffset]),
			&(pS[iOffset]),
			&(pX[iOffset]),
			&(pT[iOffset]),
			R,
			V,
			iNumElements,
			iLwSize);
}

#pragma omp target device(smp) copy_deps
#pragma omp task in(pS[0;iLwSize], pX[0;iLwSize], pT[0;iLwSize]) out(pCall[0;iLwSize], pPut[0;iLwSize])
extern void BlackScholesSMP(REAL* restrict pCall, REAL* restrict pPut, REAL* restrict pS, REAL * restrict pX, REAL * restrict pT, REAL R, REAL V, int iLwSize);

void RunKernelSMP(REAL* pCall, REAL* pPut, REAL* pS, REAL * pX, REAL * pT, REAL R, REAL V, int iNumElements, int iOffset, int iLwSize)
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
		BlackScholesSMP(
				&(pCall[iOffset + i * iLwSize]),
				&(pPut[iOffset + i * iLwSize]),
				&(pS[iOffset + i * iLwSize]),
				&(pX[iOffset + i * iLwSize]),
				&(pT[iOffset + i * iLwSize]),
				R,
				V,
				iTaskSize);
	}
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

#ifdef DP
	const REAL                   R = 0.02;
	const REAL                   V = 0.30;
#else
	const REAL                   R = 0.02f;
	const REAL                   V = 0.30f;
#endif

	if(argc != 5 && argc != 6){
		printf("argc = %d\n", argc);
		printf("blackscholes <iPlatform - 0, OCL; 1, SMP> <N> <fN> <LwSize> OR\n");
		printf("blackscholes <iPlatform - 2, Both> <N> <fN> <LwSizeOCL> <LwSizeSMP>\n");
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

		printf("Size of 5 arrays = %f (KB) = %ld (Bytes)\n", (double)(iNumElementsSingle * sizeof(REAL) * 5) / 1024.0, iNumElementsSingle * sizeof(REAL) * 5);
		printf("offset = %d, #options = %d\n\n", iOffsetSingle, iNumElementsSingle);

		//set the host side memory buffers
		REAL* pCall = (REAL*)malloc(sizeof(REAL) * iNumElementsSingle);
		REAL* pPut = (REAL*)malloc(sizeof(REAL) * iNumElementsSingle);
		REAL* pS = (REAL*)malloc(sizeof(REAL) * iNumElementsSingle);
		REAL* pX = (REAL*)malloc(sizeof(REAL) * iNumElementsSingle);
		REAL* pT = (REAL*)malloc(sizeof(REAL) * iNumElementsSingle);

		InitArrayInRange(pS, iNumElementsSingle, 5.0f, 30.0f);
		InitArrayInRange(pX, iNumElementsSingle, 1.0f, 100.0f);
		InitArrayInRange(pT, iNumElementsSingle, 0.25f, 10.0f);
		memset(pCall, 0, sizeof(REAL) * iNumElementsSingle);
		memset(pPut, 0, sizeof(REAL) * iNumElementsSingle);

		//PrintArray(pS, iNumElementsSingle);

		if(iPlatform == 0){
#ifdef PROFILING
		double dTOCLStart = 0.0, dTOCLEnd = 0.0, dTOCL = 0.0;
		int i, iStart = -10, iEnd = 10;
		for(i = iStart; i < iEnd; i++){
			if(i == 0)
				dTOCLStart = GetTimeInMs();
#endif
			//double t1 = GetTimeInMs();
			RunKernelOCL(pCall, pPut, pS, pX, pT, R, V, iNumElementsSingle, iOffsetSingle, iLwSizeSingle);
			#pragma omp taskwait
			//double t2 = GetTimeInMs();
			//printf("i = %d, time = %.6f\n", i, t2 - t1);
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
			RunKernelSMP(pCall, pPut, pS, pX, pT, R, V, iNumElementsSingle, iOffsetSingle, iLwSizeSingle);
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

		REAL* pCallGolden = (REAL*)malloc(sizeof(REAL) * iNumElementsSingle);
		REAL* pPutGolden  = (REAL*)malloc(sizeof(REAL) * iNumElementsSingle);
		memset(pCallGolden, 0, sizeof(REAL) * iNumElementsSingle);
		memset(pPutGolden, 0, sizeof(REAL) * iNumElementsSingle);
		ComputeHost(pCallGolden, pPutGolden, pS, pX, pT, R, V, iNumElementsSingle);
		CheckResult(pCall, pPut, pCallGolden, pPutGolden, iNumElementsSingle);

		//PrintMatrix(pCall, iNumElementsSingle);
		//PrintMatrix(pCallGolden, iNumElementsSingle);
        
        if(pCall != NULL) free(pCall);
        if(pPut != NULL) free(pPut);
        if(pS != NULL) free(pS);
        if(pX != NULL) free(pX);
        if(pT != NULL) free(pT);
        if(pCallGolden != NULL) free(pCallGolden);
        if(pPutGolden != NULL) free(pPutGolden);
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

		printf("Size of 5 arrays = %f (KB) = %ld (Bytes)\n", (double)(N * sizeof(REAL) * 5) / 1024.0, N * sizeof(REAL) * 5);
		printf("OCL: offset = %d, #options = %d, size of 5 arrays = %f (KB) = %ld (Bytes)\n", iOffsetOCL, iNumElementsOCL,
						(double)(iNumElementsOCL * sizeof(REAL) * 5) / 1024.0,
						iNumElementsOCL * sizeof(REAL) * 5);
		printf("SMP: offset = %d, #options = %d, size of 5 arrays = %f (KB) = %ld (Bytes)\n\n", iOffsetSMP, iNumElementsSMP,
								(double)(iNumElementsSMP * sizeof(REAL) * 5) / 1024.0,
								iNumElementsSMP * sizeof(REAL) * 5);

		//set the host side memory buffers
		REAL* pCall = (REAL*)malloc(sizeof(REAL) * N);
		REAL* pPut = (REAL*)malloc(sizeof(REAL) * N);
		REAL* pS = (REAL*)malloc(sizeof(REAL) * N);
		REAL* pX = (REAL*)malloc(sizeof(REAL) * N);
		REAL* pT = (REAL*)malloc(sizeof(REAL) * N);

		InitArrayInRange(pS, N, 5.0f, 30.0f);
		InitArrayInRange(pX, N, 1.0f, 100.0f);
		InitArrayInRange(pT, N, 0.25f, 10.0f);
		memset(pCall, 0, sizeof(REAL) * N);
		memset(pPut, 0, sizeof(REAL) * N);

		//PrintArray(pS, iNumElementsSingle);

		if(iNumElementsOCL == 0){ //run only smp
			//RunKernelSMP(pCall, pPut, pS, pX, pT, R, V, iNumElementsSMP, iOffsetSMP, iLwSizeSMP);
			//#pragma omp taskwait
			printf("run only smp\n");
			fprintf(pFTime, "Both: only smp\n");
		}
		if(iNumElementsSMP == 0){ //run only ocl
			//RunKernelOCL(pCall, pPut, pS, pX, pT, R, V, iNumElementsOCL, iOffsetOCL, iLwSizeOCL);
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
			RunKernelOCL(pCall, pPut, pS, pX, pT, R, V, iNumElementsOCL, iOffsetOCL, iLwSizeOCL);
			RunKernelSMP(pCall, pPut, pS, pX, pT, R, V, iNumElementsSMP, iOffsetSMP, iLwSizeSMP);
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

		REAL* pCallGolden = (REAL*)malloc(sizeof(REAL) * N);
		REAL* pPutGolden  = (REAL*)malloc(sizeof(REAL) * N);
		memset(pCallGolden, 0, sizeof(REAL) * N);
		memset(pPutGolden, 0, sizeof(REAL) * N);
		ComputeHost(pCallGolden, pPutGolden, pS, pX, pT, R, V, N);
		CheckResult(pCall, pPut, pCallGolden, pPutGolden, N);

		//PrintMatrix(pCall, N);
		//PrintMatrix(pCallGolden, N);

        if(pCall != NULL) free(pCall);
        if(pPut != NULL) free(pPut);
        if(pS != NULL) free(pS);
        if(pX != NULL) free(pX);
        if(pT != NULL) free(pT);
        if(pCallGolden != NULL) free(pCallGolden);
        if(pPutGolden != NULL) free(pPutGolden);
	}

	printf("DONE\n\n");

	return 0;

}
