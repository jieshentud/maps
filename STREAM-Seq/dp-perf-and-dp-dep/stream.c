/*
 * stream.c
 *
 *	Created on: Feb 11, 2015
 *  Downloaded from stream benchmark, ported to OmpSs by Jie Shen <j.shen@tudelft.nl>
 *  Delft University of Technology
 *
 */

#include "jutil.h"
#include <kernel.h>

void checkSTREAMresults (REAL* a, REAL* b, REAL* c, int iNumElements, int iNumIterations);

#pragma omp target device(smp) copy_deps
#pragma omp task in(a[0;iNumElements]) out(c[0;iNumElements])
void CopySMP(REAL* __restrict a, REAL* __restrict c, int iNumElements, int iLwSize);

#pragma omp target device(smp) copy_deps
#pragma omp task in(c[0;iNumElements]) out(b[0;iNumElements])
void ScaleSMP(REAL* __restrict c, REAL* __restrict b, REAL scalar, int iNumElements, int iLwSize);

#pragma omp target device(smp) copy_deps
#pragma omp task in(a[0;iNumElements], b[0;iNumElements]) out(c[0;iNumElements])
void AddSMP(REAL* __restrict a, REAL* __restrict b, REAL* __restrict c, int iNumElements, int iLwSize);

#pragma omp target device(smp) copy_deps
#pragma omp task in(b[0;iNumElements], c[0;iNumElements]) out(a[0;iNumElements])
void TriadSMP(REAL* __restrict b, REAL* __restrict c, REAL* __restrict a, REAL scalar, int iNumElements, int iLwSize);

#ifdef __cplusplus
extern "C"
{
#endif

//NDRange first worksize.dim0, then worksize.dim1
#pragma omp target device(opencl) copy_deps ndrange(1, iNumElements, iLwSize) file(kernel.cl) implements(CopySMP)
#pragma omp task in(a[0;iNumElements]) out(c[0;iNumElements])
__kernel void CopyOCL(__global REAL* a,
                      __global REAL* c,
                      int iNumElements,
                      int iLwSize);

#pragma omp target device(opencl) copy_deps ndrange(1, iNumElements, iLwSize) file(kernel.cl) implements(ScaleSMP)
#pragma omp task in(c[0;iNumElements]) out(b[0;iNumElements])
__kernel void ScaleOCL(__global REAL* c,
                       __global REAL* b,
                       REAL scalar,
                       int iNumElements,
                       int iLwSize);

#pragma omp target device(opencl) copy_deps ndrange(1, iNumElements, iLwSize) file(kernel.cl) implements(AddSMP)
#pragma omp task in(a[0;iNumElements], b[0;iNumElements]) out(c[0;iNumElements])
__kernel void AddOCL(__global REAL* a,
                     __global REAL* b,
                     __global REAL* c,
                     int iNumElements,
                     int iLwSize);

#pragma omp target device(opencl) copy_deps ndrange(1, iNumElements, iLwSize) file(kernel.cl) implements(TriadSMP)
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

void RunKernelBoth(REAL* a, REAL* b, REAL* c, REAL scalar, int N, int iTaskSize, int iLwSize)
{
	int i;
	int iRemainder = N % iTaskSize;
	int iNumTasks = N / iTaskSize;
	int iTaskOffset;  //the task offset in GLOBAL view
#ifdef PRINT
	printf("Total NumTasks = %d, Remainder = %d\n", iNumTasks, iRemainder);
#endif
	//enable the four taskwaits to mimic the application that needs synchronization
	for(i = 0; i < iNumTasks; i++){
		int iTaskOffset = i * iTaskSize;
		//call SMP and OCL tasks
		CopySMP(&a[iTaskOffset], &c[iTaskOffset], iTaskSize, iLwSize);
	}
	//#pragma omp taskwait
	for(i = 0; i < iNumTasks; i++){
		int iTaskOffset = i * iTaskSize;
		//call SMP and OCL tasks
		ScaleSMP(&c[iTaskOffset], &b[iTaskOffset], scalar, iTaskSize, iLwSize);
	}
	//#pragma omp taskwait
	for(i = 0; i < iNumTasks; i++){
		int iTaskOffset = i * iTaskSize;
		//call SMP and OCL tasks
		AddSMP(&a[iTaskOffset], &b[iTaskOffset], &c[iTaskOffset], iTaskSize, iLwSize);
	}
	//#pragma omp taskwait
	for(i = 0; i < iNumTasks; i++){
		int iTaskOffset = i * iTaskSize;
		//call SMP and OCL tasks
		TriadSMP(&b[iTaskOffset], &c[iTaskOffset], &a[iTaskOffset], scalar, iTaskSize, iLwSize);
	}
	//#pragma omp taskwait

}

int main(int argc, char** argv)
{
	int N, iTaskSize, iLwSize;
	int iPlatform;  //0, OCL; 1, SMP; 2, Both
	int iWarpSize = 32;

	int iNumIterations = 20;

	if(argc != 5){
		printf("argc = %d\n", argc);
		printf("stream <iPlatform - 2, Both> <N> <TaskSize> <LwSize> OR\n");
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

    ssize_t		j;
    REAL		scalar;

    printf("Size of 3 arrays = %f (KB) = %ld (Bytes)\n", (double)(N * sizeof(REAL) * 3) / 1024.0, N * sizeof(REAL) * 3);

	//make the memory align to 128 bytes
	int min_align = 1024;
	REAL* a = (REAL*)memalign(min_align/8, sizeof(REAL) * N);
	REAL* b = (REAL*)memalign(min_align/8, sizeof(REAL) * N);
	REAL* c = (REAL*)memalign(min_align/8, sizeof(REAL) * N);

	for (j=0; j<N; j++){
		a[j] = 1.0;
		b[j] = 2.0;
		c[j] = 0.0;
		a[j] = 2.0E0 * a[j];
	}
	scalar = 3.0;

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
		//use taskwait to get correct timing for one iteration
		RunKernelBoth(a, b, c, scalar, N, iTaskSize, iLwSize);
		#pragma omp taskwait
#ifdef PROFILING
	}
	dTBothEnd = GetTimeInMs();
	dTBoth = (dTBothEnd - dTBothStart) / (double)iEnd;
	printf("Timing: %.6f / %d = %.6f ms\n", dTBothEnd - dTBothStart, iEnd, dTBoth);
	fprintf(pFTime, "Both: %.6f\n", dTBoth);
	fclose(pFTime);
#endif

	//PrintArray(a, iNumElementsSingle);
	//PrintArray(b, iNumElementsSingle);
	//PrintArray(c, iNumElementsSingle);
	checkSTREAMresults (a, b, c, N, iNumIterations);

	if(a != NULL) free(a);
	if(b != NULL) free(b);
	if(c != NULL) free(c);

	printf("DONE\n\n");

	return 0;

}
