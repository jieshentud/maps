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

void RunKernelSMP(REAL* a, REAL* b, REAL* c, REAL scalar, int iNumElements, int iLwSize, int iOffset)
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
		CopySMP(&a[iTaskOffset], &c[iTaskOffset], iTaskSize, iLwSize);
	}
//enable the four taskwaits to mimic the application that needs synchronization
//#pragma omp taskwait
	for(i = 0; i < iNumTasks; i++){
		//call ompss task
		if(i == (iNumTasks-1) && iRemainder != 0){
			iTaskSize = iRemainder; //set the remaining task size if the remainder is not 0
		}
		iTaskOffset = iOffset + i * iTaskSize;
		ScaleSMP(&c[iTaskOffset], &b[iTaskOffset], scalar, iTaskSize, iLwSize);
	}
//#pragma omp taskwait
	for(i = 0; i < iNumTasks; i++){
		//call ompss task
		if(i == (iNumTasks-1) && iRemainder != 0){
			iTaskSize = iRemainder; //set the remaining task size if the remainder is not 0
		}
		iTaskOffset = iOffset + i * iTaskSize;
		AddSMP(&a[iTaskOffset], &b[iTaskOffset], &c[iTaskOffset], iTaskSize, iLwSize);
	}
//#pragma omp taskwait
	for(i = 0; i < iNumTasks; i++){
		//call ompss task
		if(i == (iNumTasks-1) && iRemainder != 0){
			iTaskSize = iRemainder; //set the remaining task size if the remainder is not 0
		}
		iTaskOffset = iOffset + i * iTaskSize;
		TriadSMP(&b[iTaskOffset], &c[iTaskOffset], &a[iTaskOffset], scalar, iTaskSize, iLwSize);
	}
//#pragma omp taskwait
}

void RunKernelOCL(REAL* a, REAL* b, REAL* c, REAL scalar, int iNumElements, int iLwSize, int iOffset)
{
	CopyOCL(&a[iOffset], &c[iOffset], iNumElements, iLwSize);
//enable the four taskwaits to mimic the application that needs synchronization
//#pragma omp taskwait
	ScaleOCL(&c[iOffset], &b[iOffset], scalar, iNumElements, iLwSize);
//#pragma omp taskwait
	AddOCL(&a[iOffset], &b[iOffset], &c[iOffset], iNumElements, iLwSize);
//#pragma omp taskwait
	TriadOCL(&b[iOffset], &c[iOffset], &a[iOffset], scalar, iNumElements, iLwSize);
//#pragma omp taskwait
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

	if(argc != 5 && argc != 6){
		printf("argc = %d\n", argc);
		printf("stream <iPlatform - 0, OCL; 1, SMP> <N> <fN> <LwSize> OR\n");
		printf("stream <iPlatform - 2, Both> <N> <fN> <LwSizeOCL> <LwSizeSMP>\n");
		printf("Error, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		exit(EXIT_FAILURE);
	}
	iPlatform = atoi(argv[1]);
	N = atoi(argv[2]);
	fN = atof(argv[3]);  //fN = R_GC / (1 + R_GC + R_GD)

    ssize_t		j;
    REAL		scalar;

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

		printf("Size of 3 arrays = %f (KB) = %ld (Bytes)\n", (double)(iNumElementsSingle * sizeof(REAL) * 3) / 1024.0, iNumElementsSingle * sizeof(REAL) * 3);
		printf("offset = %d, #options = %d\n\n", iOffsetSingle, iNumElementsSingle);

		//make the memory align to 128 bytes
		int min_align = 1024;
		REAL* a = (REAL*)memalign(min_align/8, sizeof(REAL) * iNumElementsSingle);
		REAL* b = (REAL*)memalign(min_align/8, sizeof(REAL) * iNumElementsSingle);
		REAL* c = (REAL*)memalign(min_align/8, sizeof(REAL) * iNumElementsSingle);

		for (j=0; j<iNumElementsSingle; j++){
			a[j] = 1.0;
			b[j] = 2.0;
			c[j] = 0.0;
			a[j] = 2.0E0 * a[j];
		}
		scalar = 3.0;

		if(iPlatform == 0){
#ifdef PROFILING
		double dTOCLStart = 0.0, dTOCLEnd = 0.0, dTOCL = 0.0;
		int i, iStart = -10, iEnd = 10;   //iEnd - iStart = iNumIterations
		for(i = iStart; i < iEnd; i++){
			if(i == 0){
				#pragma omp taskwait
				dTOCLStart = GetTimeInMs();
			}
#endif
			RunKernelOCL(a, b, c, scalar, iNumElementsSingle, iLwSizeSingle, iOffsetSingle);
#ifdef PROFILING
		}
		//get timing after the loop
		#pragma omp taskwait
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
			if(i == 0){
				#pragma omp taskwait
				dTSMPStart = GetTimeInMs();
			}
#endif
			RunKernelSMP(a, b, c, scalar, iNumElementsSingle, iLwSizeSingle, iOffsetSingle);
#ifdef PROFILING
		}
		//get timing after the loop
		#pragma omp taskwait
		dTSMPEnd = GetTimeInMs();
		dTSMP = (dTSMPEnd - dTSMPStart) / (double)iEnd;
		printf("Timing: %.6f / %d = %.6f ms\n", dTSMPEnd - dTSMPStart, iEnd, dTSMP);
		fprintf(pFTime, "SMP: %.6f\n", dTSMP);
		fclose(pFTime);
#endif
		}

		//PrintArray(a, iNumElementsSingle);
		//PrintArray(b, iNumElementsSingle);
		//PrintArray(c, iNumElementsSingle);
		checkSTREAMresults (a, b, c, iNumElementsSingle, iNumIterations);

		if(a != NULL) free(a);
		if(b != NULL) free(b);
		if(c != NULL) free(c);
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

		printf("Size of 3 arrays = %f (KB) = %ld (Bytes)\n", (double)(N * sizeof(REAL) * 3) / 1024.0, N * sizeof(REAL) * 3);
		printf("OCL: offset = %i, # elements = %i\n", iOffsetOCL, iNumElementsOCL);
		printf("SMP: offset = %i, # elements = %i\n\n", iOffsetSMP, iNumElementsSMP);

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

		if(iNumElementsOCL == 0){ //run only smp
			//RunKernelSMP(a, b, c, scalar, iNumElementsSMP, iLwSizeSMP, iOffsetSMP);
			//#pragma omp taskwait
			printf("run only smp\n");
			fprintf(pFTime, "Both: only smp\n");
		}
		if(iNumElementsSMP == 0){ //run only ocl
			//RunKernelOCL(a, b, c, scalar, iNumElementsOCL, iLwSizeOCL, iOffsetOCL);
			//#pragma omp taskwait
			printf("run only ocl\n");
			fprintf(pFTime, "Both: only ocl\n");
		}
		if(iNumElementsOCL != 0 && iNumElementsSMP != 0){ //run both
#ifdef PROFILING
		double dTBothStart = 0.0, dTBothEnd = 0.0, dTBoth = 0.0;
		int i, iStart = -10, iEnd = 10;
		for(i = iStart; i < iEnd; i++){
			if(i == 0){
				#pragma omp taskwait
				dTBothStart = GetTimeInMs();
			}
#endif
			RunKernelOCL(a, b, c, scalar, iNumElementsOCL, iLwSizeOCL, iOffsetOCL);
			RunKernelSMP(a, b, c, scalar, iNumElementsSMP, iLwSizeSMP, iOffsetSMP);
#ifdef PROFILING
		}
		//get timing after the loop
		#pragma omp taskwait
		dTBothEnd = GetTimeInMs();
		dTBoth = (dTBothEnd - dTBothStart) / (double)iEnd;
		printf("Timing: %.6f / %d = %.6f ms\n", dTBothEnd - dTBothStart, iEnd, dTBoth);
		fprintf(pFTime, "Both: %.6f\n", dTBoth);
		fclose(pFTime);
#endif
		}

		//PrintArray(a, iNumElementsSingle);
		//PrintArray(b, iNumElementsSingle);
		//PrintArray(c, iNumElementsSingle);
		checkSTREAMresults (a, b, c, N, iNumIterations);

		if(a != NULL) free(a);
		if(b != NULL) free(b);
		if(c != NULL) free(c);
	}

	printf("DONE\n\n");

	return 0;

}
