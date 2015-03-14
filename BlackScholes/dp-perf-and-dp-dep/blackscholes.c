/*
 * jutil.h
 *
 *  Created on: Jan 28, 2015
 *  Ported to OmpSs by Jie Shen <j.shen@tudelft.nl>
 *  Delft University of Technology
 *
 */

#include "jutil.h"
#include <kernel.h>



#pragma omp target device(smp) copy_deps
#pragma omp task in(pS[0;iNumElements], pX[0;iNumElements], pT[0;iNumElements]) out(pCall[0;iNumElements], pPut[0;iNumElements])
extern void BlackScholesSMP(REAL* pCall, REAL* pPut, REAL* pS, REAL * pX, REAL * pT, REAL R, REAL V, int iNumElements, int iLwSize);

#ifdef __cplusplus
extern "C"
{
#endif

//NDRange first worksize.dim0, then worksize.dim1
#pragma omp target device(opencl) copy_deps ndrange(1, iNumElements, iLwSize) file(kernel.cl) implements(BlackScholesSMP)
#pragma omp task in(pS[0;iNumElements], pX[0;iNumElements], pT[0;iNumElements]) out(pCall[0;iNumElements], pPut[0;iNumElements])
__kernel void BlackScholesOCL(
    __global REAL *pCall, //Call option price
    __global REAL *pPut,  //Put option price
    __global REAL *pS,    //Current stock price
    __global REAL *pX,    //Option strike price
    __global REAL *pT,    //Option years
    REAL R,               //Riskless rate of return
    REAL V,               //Stock volatility
    int iNumElements,
    int iLwSize
    );

#ifdef __cplusplus
}
#endif

void RunKernelBoth(REAL* pCall, REAL* pPut, REAL* pS, REAL* pX, REAL* pT, REAL R, REAL V, int N, int iTaskSize, int iLwSize)
{
	int i;
	int iRemainder = N % iTaskSize; //iRemainder should be 0
	int iNumTasks = N / iTaskSize;
	int iOffset;
#ifdef PRINT
	printf("Total NumTasks = %d, Remainder = %d\n", iNumTasks, iRemainder);
#endif
	for(i = 0; i < iNumTasks; i++){
		int iOffset = i * iTaskSize;
		//call BlackScholesSMP and BlackScholesOCL tasks
		BlackScholesSMP(
				&(pCall[iOffset]),
				&(pPut[iOffset]),
				&(pS[iOffset]),
				&(pX[iOffset]),
				&(pT[iOffset]),
				R,
				V,
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

#ifdef DP
	const REAL                   R = 0.02;
	const REAL                   V = 0.30;
#else
	const REAL                   R = 0.02f;
	const REAL                   V = 0.30f;
#endif

	if(argc != 5){
		printf("argc = %d\n", argc);
		printf("blackscholes <iPlatform - 2, Both> <N> <TaskSize> <LwSize> OR\n");
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

	printf("Size of 5 arrays = %f (KB) = %ld (Bytes)\n", (double)(N * sizeof(REAL) * 5) / 1024.0, N * sizeof(REAL) * 5);

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
		//double t1 = GetTimeInMs();
		RunKernelBoth(pCall, pPut, pS, pX, pT, R, V, N, iTaskSize, iLwSize);
		#pragma omp taskwait
		//double t2 = GetTimeInMs();
		//printf("i = %d, time = %.6f\n", i, t2 - t1);
#ifdef PROFILING
	}
	dTBothEnd = GetTimeInMs();
	dTBoth = (dTBothEnd - dTBothStart) / (double)iEnd;
	printf("Timing: %.6f / %d = %.6f ms\n", dTBothEnd - dTBothStart, iEnd, dTBoth);
	fprintf(pFTime, "Both: %.6f\n", dTBoth);
	fclose(pFTime);
#endif

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

	printf("DONE\n\n");

	return 0;

}
