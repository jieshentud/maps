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

void RunKernelCopySMP(REAL* a, REAL* b, REAL* c, REAL scalar, int iNumElements, int iLwSize, int iOffset)
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
}

void RunKernelScaleSMP(REAL* a, REAL* b, REAL* c, REAL scalar, int iNumElements, int iLwSize, int iOffset)
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
#ifdef PRINT
		printf("Task-%d, iTaksSize = %d\n", i, iTaskSize);
#endif
		iTaskOffset = iOffset + i * iTaskSize;
		ScaleSMP(&c[iTaskOffset], &b[iTaskOffset], scalar, iTaskSize, iLwSize);
	}
}

void RunKernelAddSMP(REAL* a, REAL* b, REAL* c, REAL scalar, int iNumElements, int iLwSize, int iOffset)
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
#ifdef PRINT
		printf("Task-%d, iTaksSize = %d\n", i, iTaskSize);
#endif
		iTaskOffset = iOffset + i * iTaskSize;
		AddSMP(&a[iTaskOffset], &b[iTaskOffset], &c[iTaskOffset], iTaskSize, iLwSize);
	}
}

void RunKernelTriadSMP(REAL* a, REAL* b, REAL* c, REAL scalar, int iNumElements, int iLwSize, int iOffset)
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
#ifdef PRINT
		printf("Task-%d, iTaksSize = %d\n", i, iTaskSize);
#endif
		iTaskOffset = iOffset + i * iTaskSize;
		TriadSMP(&b[iTaskOffset], &c[iTaskOffset], &a[iTaskOffset], scalar, iTaskSize, iLwSize);
	}
}

void RunKernelCopyOCL(REAL* a, REAL* b, REAL* c, REAL scalar, int iNumElements, int iLwSize, int iOffset)
{
	CopyOCL(&a[iOffset], &c[iOffset], iNumElements, iLwSize);
}

void RunKernelScaleOCL(REAL* a, REAL* b, REAL* c, REAL scalar, int iNumElements, int iLwSize, int iOffset)
{
	ScaleOCL(&c[iOffset], &b[iOffset], scalar, iNumElements, iLwSize);
}

void RunKernelAddOCL(REAL* a, REAL* b, REAL* c, REAL scalar, int iNumElements, int iLwSize, int iOffset)
{
	AddOCL(&a[iOffset], &b[iOffset], &c[iOffset], iNumElements, iLwSize);
}

void RunKernelTriadOCL(REAL* a, REAL* b, REAL* c, REAL scalar, int iNumElements, int iLwSize, int iOffset)
{
	TriadOCL(&b[iOffset], &c[iOffset], &a[iOffset], scalar, iNumElements, iLwSize);
}


int main(int argc, char** argv)
{
	int N;
	double fN;
	double fN1, fN2, fN3, fN4;
	int iNumElementsSingle;
	int iNumElementsOCL1, iNumElementsSMP1, iNumElementsOCL2, iNumElementsSMP2, iNumElementsOCL3, iNumElementsSMP3, iNumElementsOCL4, iNumElementsSMP4;
	int iOffsetSingle;
	int iOffsetOCL1, iOffsetSMP1, iOffsetOCL2, iOffsetSMP2, iOffsetOCL3, iOffsetSMP3, iOffsetOCL4, iOffsetSMP4;
	int iLwSizeSingle, iLwSizeOCL, iLwSizeSMP;
	int iPlatform;  //0, OCL; 1, SMP; 2, Both
	int iWarpSize = 1;

	int iNumIterations = 20;

	if(argc != 5 && argc != 9){
		printf("argc = %d\n", argc);
		printf("stream <iPlatform - 0, OCL; 1, SMP> <N> <fN> <LwSize> OR\n");
		printf("stream <iPlatform - 2, Both> <N> <fN1> <fN2> <fN3> <fN4> <LwSizeOCL> <LwSizeSMP>\n");
		printf("Error, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		exit(EXIT_FAILURE);
	}
	iPlatform = atoi(argv[1]);
	N = atoi(argv[2]);

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
		fN = atof(argv[3]);  //fN = R_GC / (1 + R_GC + R_GD)

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
				dTOCLStart = GetTimeInMs();
			}
#endif
			//use taskwait to get correct timing for each kernel
			RunKernelCopyOCL(a, b, c, scalar, iNumElementsSingle, iLwSizeSingle, iOffsetSingle);
			#pragma omp taskwait
			RunKernelScaleOCL(a, b, c, scalar, iNumElementsSingle, iLwSizeSingle, iOffsetSingle);
			#pragma omp taskwait
			RunKernelAddOCL(a, b, c, scalar, iNumElementsSingle, iLwSizeSingle, iOffsetSingle);
			#pragma omp taskwait
			RunKernelTriadOCL(a, b, c, scalar, iNumElementsSingle, iLwSizeSingle, iOffsetSingle);
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
			//use taskwait to get correct timing for each kernel
			RunKernelCopySMP(a, b, c, scalar, iNumElementsSingle, iLwSizeSingle, iOffsetSingle);
			#pragma omp taskwait
			RunKernelScaleSMP(a, b, c, scalar, iNumElementsSingle, iLwSizeSingle, iOffsetSingle);
			#pragma omp taskwait
			RunKernelAddSMP(a, b, c, scalar, iNumElementsSingle, iLwSizeSingle, iOffsetSingle);
			#pragma omp taskwait
			RunKernelTriadSMP(a, b, c, scalar, iNumElementsSingle, iLwSizeSingle, iOffsetSingle);
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

		//PrintArray(a, iNumElementsSingle);
		//PrintArray(b, iNumElementsSingle);
		//PrintArray(c, iNumElementsSingle);
		checkSTREAMresults (a, b, c, iNumElementsSingle, iNumIterations);

		if(a != NULL) free(a);
		if(b != NULL) free(b);
		if(c != NULL) free(c);
	}

	if(iPlatform == 2){
		fN1 = atof(argv[3]);
		fN2 = atof(argv[4]);
		fN3 = atof(argv[5]);
		fN4 = atof(argv[6]);

		iLwSizeOCL = atoi(argv[7]);
		iLwSizeSMP = atoi(argv[8]);
		//when using OpenCL, round up LwSizeOCL to WarpSize
		if(iLwSizeOCL % iWarpSize != 0)
		{
			int temp = RoundUp(iWarpSize, iLwSizeOCL);
			iLwSizeOCL = temp;
		}
		printf("\nPlatform = %s, N = %d, fN = %f %f %f %f, LwSizeOCL = %d, LwSizeSMP = %d\n\n", "Both", N, fN1, fN2, fN3, fN4, iLwSizeOCL, iLwSizeSMP);

		GetNumElements(N, fN1, &iNumElementsOCL1, &iOffsetOCL1, iLwSizeOCL, &iNumElementsSMP1, &iOffsetSMP1, iLwSizeSMP);
		GetNumElements(N, fN2, &iNumElementsOCL2, &iOffsetOCL2, iLwSizeOCL, &iNumElementsSMP2, &iOffsetSMP2, iLwSizeSMP);
		GetNumElements(N, fN3, &iNumElementsOCL3, &iOffsetOCL3, iLwSizeOCL, &iNumElementsSMP3, &iOffsetSMP3, iLwSizeSMP);
		GetNumElements(N, fN4, &iNumElementsOCL4, &iOffsetOCL4, iLwSizeOCL, &iNumElementsSMP4, &iOffsetSMP4, iLwSizeSMP);

		printf("Size of 3 arrays = %f (KB) = %ld (Bytes)\n", (double)(N * sizeof(REAL) * 3) / 1024.0, N * sizeof(REAL) * 3);
		//printf("OCL: offset = %i, # elements = %i\n", iOffsetOCL1, iNumElementsOCL1);
		//printf("SMP: offset = %i, # elements = %i\n\n", iOffsetSMP1, iNumElementsSMP1);

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
		double dTBothStart = 0.0, dTBothEnd = 0.0, dTBoth = 0.0;
		int i, iStart = -10, iEnd = 10;
		for(i = iStart; i < iEnd; i++){
			if(i == 0)
				dTBothStart = GetTimeInMs();
#endif
		//SP-Varied requires taskwait after each kernel to assemble the output of one kernel for the correct input of the next kernel
		//copy
		if(iNumElementsOCL1 == 0){ //run only smp
			//RunKernelCopySMP(a, b, c, scalar, iNumElementsSMP1, iLwSizeSMP, iOffsetSMP1);
			printf("run only smp\n");
			fprintf(pFTime, "Both: only smp\n");
		}
		if(iNumElementsSMP1 == 0){ //run only ocl
			//RunKernelCopyOCL(a, b, c, scalar, iNumElementsOCL1, iLwSizeOCL, iOffsetOCL1);
			printf("run only ocl\n");
			fprintf(pFTime, "Both: only ocl\n");
		}
		if(iNumElementsOCL1 != 0 && iNumElementsSMP1 != 0){ //run both
			RunKernelCopyOCL(a, b, c, scalar, iNumElementsOCL1, iLwSizeOCL, iOffsetOCL1);
			RunKernelCopySMP(a, b, c, scalar, iNumElementsSMP1, iLwSizeSMP, iOffsetSMP1);
		}
		#pragma omp taskwait
		//scale
		if(iNumElementsOCL2 == 0){ //run only smp
			//RunKernelScaleSMP(a, b, c, scalar, iNumElementsSMP2, iLwSizeSMP, iOffsetSMP2);
			printf("run only smp\n");
			fprintf(pFTime, "Both: only smp\n");
		}
		if(iNumElementsSMP2 == 0){ //run only ocl
			//RunKernelScaleOCL(a, b, c, scalar, iNumElementsOCL2, iLwSizeOCL, iOffsetOCL2);
			printf("run only ocl\n");
			fprintf(pFTime, "Both: only ocl\n");
		}
		if(iNumElementsOCL2 != 0 && iNumElementsSMP2 != 0){ //run both
			RunKernelScaleOCL(a, b, c, scalar, iNumElementsOCL2, iLwSizeOCL, iOffsetOCL2);
			RunKernelScaleSMP(a, b, c, scalar, iNumElementsSMP2, iLwSizeSMP, iOffsetSMP2);
		}
		#pragma omp taskwait
		//add
		if(iNumElementsOCL3 == 0){ //run only smp
			//RunKernelAddSMP(a, b, c, scalar, iNumElementsSMP3, iLwSizeSMP, iOffsetSMP3);
			printf("run only smp\n");
			fprintf(pFTime, "Both: only smp\n");
		}
		if(iNumElementsSMP3 == 0){ //run only ocl
			//RunKernelAddOCL(a, b, c, scalar, iNumElementsOCL3, iLwSizeOCL, iOffsetOCL3);
			printf("run only ocl\n");
			fprintf(pFTime, "Both: only ocl\n");
		}
		if(iNumElementsOCL3 != 0 && iNumElementsSMP3 != 0){ //run both
			RunKernelAddOCL(a, b, c, scalar, iNumElementsOCL3, iLwSizeOCL, iOffsetOCL3);
			RunKernelAddSMP(a, b, c, scalar, iNumElementsSMP3, iLwSizeSMP, iOffsetSMP3);
		}
		#pragma omp taskwait
		//triad
		if(iNumElementsOCL4 == 0){ //run only smp
			//RunKernelTriadSMP(a, b, c, scalar, iNumElementsSMP4, iLwSizeSMP, iOffsetSMP4);
			printf("run only smp\n");
			fprintf(pFTime, "Both: only smp\n");
		}
		if(iNumElementsSMP4 == 0){ //run only ocl
			//RunKernelTriadOCL(a, b, c, scalar, iNumElementsOCL4, iLwSizeOCL, iOffsetOCL4);
			printf("run only ocl\n");
			fprintf(pFTime, "Both: only ocl\n");
		}
		if(iNumElementsOCL4 != 0 && iNumElementsSMP4 != 0){ //run both
			RunKernelTriadOCL(a, b, c, scalar, iNumElementsOCL4, iLwSizeOCL, iOffsetOCL4);
			RunKernelTriadSMP(a, b, c, scalar, iNumElementsSMP4, iLwSizeSMP, iOffsetSMP4);
		}
		#pragma omp taskwait
#ifdef PROFILING
		}
		dTBothEnd = GetTimeInMs();
		dTBoth = (dTBothEnd - dTBothStart) / (double)iEnd;
		printf("Timing: %.6f / %d = %.6f ms\n", dTBothEnd - dTBothStart, iEnd, dTBoth);
		fprintf(pFTime, "Both: %.6f\n", dTBoth);
		fclose(pFTime);
#endif

		//PrintArray(a, N);
		//PrintArray(b, N);
		//PrintArray(c, N);
		checkSTREAMresults (a, b, c, N, iNumIterations);

		if(a != NULL) free(a);
		if(b != NULL) free(b);
		if(c != NULL) free(c);
	}

	printf("DONE\n\n");

	return 0;

}
