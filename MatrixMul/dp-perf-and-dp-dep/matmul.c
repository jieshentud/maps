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


#pragma omp target device(smp) copy_deps
#pragma omp task in(A[0;iNumElements*iWA], B[0;iWA*iWB]) out(C[0;iNumElements*iWB])
void matmulsmp(REAL *A, REAL *B, REAL *C, int iNumElements, int iOffset, int iWA, int iWB, int iLwSize);

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

void RunKernelBoth(REAL* pHostA, REAL* pHostB, REAL* pHostC, int iHA, int iWA, int iWB, int iTaskSize, int iLwSize)
{
	int i;
	int iRemainder = iHA % iTaskSize;
	int iNumTasks = iHA / iTaskSize;
	int iOffset;
#ifdef PRINT
	printf("Total NumTasks = %d, Remainder = %d\n", iNumTasks, iRemainder);
#endif
	for(i = 0; i < iNumTasks; i++){
		int iOffset = i * iTaskSize;
		//call matmulsmp and matmulocl tasks
		matmulsmp(&(pHostA[iOffset * iWA]), pHostB, &(pHostC[iOffset * iWA]), iTaskSize, iOffset, iWA, iWB, iLwSize);
	}
}

int main(int argc, char** argv)
{
	int iPlatform;  //0, OCL; 1, SMP; 2, Both
	int N, iTaskSize, iLwSize;
	int iHA, iWA, iWB;

	int iWarpSize = 32;

	if(argc != 5){
		printf("argc = %d\n", argc);
		printf("matmul <iPlatform - 2, Both> <N> <TaskSize> <LwSize> OR\n");
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

	iHA = N;
	iWA = N;
	iWB = N;

	printf("\nMatrix dims: A(%i x %i), B(%i x %i), C(%i x %i)\n", iHA, iWA, iWA, iWB, iHA, iWB);
	printf("Size of 3 arrays = %f (KB) = %ld (Bytes)\n", (double)(iHA * iWA + iWA * iWB + iHA * iWB) * sizeof(REAL) / 1024.0, (iHA * iWA + iWA * iWB + iHA * iWB) * sizeof(REAL));

	REAL* pHostA = (REAL*)malloc(sizeof(REAL) * iHA * iWA);
	REAL* pHostB = (REAL*)malloc(sizeof(REAL) * iWA * iWB);
	REAL* pHostC = (REAL*)malloc(sizeof(REAL) * iHA * iWB);
	InitMatrix(pHostA, iHA, iWA);
	InitMatrix(pHostB, iWA, iWB);
	memset(pHostC, 0, sizeof(REAL) * iHA * iWB);

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
	RunKernelBoth(pHostA, pHostB, pHostC, iHA, iWA, iWB, iTaskSize, iLwSize);
	#pragma omp taskwait
	//double t2 = GetTimeInMs();
	//printf("i = %d, time = %.6f ms\n", i, t2-t1);

#ifdef PROFILING
	}
	dTBothEnd = GetTimeInMs();
	dTBoth = (dTBothEnd - dTBothStart) / (double)iEnd;
	printf("Timing: %.6f / %d = %.6f ms\n", dTBothEnd - dTBothStart, iEnd, dTBoth);
	fprintf(pFTime, "Both: %.6f\n", dTBoth);
	fclose(pFTime);
#endif

	if(pHostA != NULL) free(pHostA);
	if(pHostB != NULL) free(pHostB);
	if(pHostC != NULL) free(pHostC);

	printf("DONE\n\n");

	return 0;

}
