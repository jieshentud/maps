/*
 * hotspot.c
 *
 *  Created on: Feb 8, 2015
 *  Ported to OmpSs by Jie Shen <j.shen@tudelft.nl>
 *  Delft University of Technology
 *
 */

#include "jutil.h"
#include <kernel.h>


/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

void ComputeHost(REAL* input_temp, REAL* output_temp, REAL* power, int row, int col,
		REAL Cap, REAL Rx, REAL Ry, REAL Rz, REAL step, REAL amb_temp, int iNumIterations);

#pragma omp target device(smp) copy_deps
#pragma omp task in(input_temp[0;(iTaskSize+2)*col], power[0;(iTaskSize+2)*col]) out(output_temp[0;iTaskSize*col])
void HotspotSMP(REAL* __restrict input_temp, REAL* __restrict output_temp, REAL* __restrict power, int row, int col,
		REAL Cap, REAL Rx, REAL Ry, REAL Rz, REAL step, REAL amb_temp, int iTaskSize, int iLwSize, int iTaskOffset);

void RunKernelSMP(REAL* input_temp, REAL* output_temp, REAL* power, int row, int col,
		REAL Cap, REAL Rx, REAL Ry, REAL Rz, REAL step, REAL amb_temp, int iNumElements, int iLwSize, int iOffset)
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
		HotspotSMP(
				&(input_temp[(iTaskOffset-1) * col]),  //input starts from the padding row
				&(output_temp[iTaskOffset * col]),    //output starts from the valid row
				&(power[(iTaskOffset-1) * col]),       //input starts from the padding row
				row,
				col,
				Cap,
				Rx,
				Ry,
				Rz,
				step,
				amb_temp,
				iTaskSize,
				iLwSize,
				iTaskOffset);
	}
}


void RunKernelOCL(REAL* input_temp, REAL* output_temp, REAL* power, int row, int col,
		REAL Cap, REAL Rx, REAL Ry, REAL Rz, REAL step, REAL amb_temp, int iNumElements, int iLwSize, int iOffset)
{
	HotspotOCL(&(input_temp[(iOffset-1) * col]),  //input starts from the padding row
	           &(output_temp[iOffset * col]),     //output starts from the valid row
	           &(power[(iOffset-1) * col]),       //input starts from the padding row
	           row,
	           col,
	           Cap,
	           Rx,
	           Ry,
	           Rz,
	           step,
	           amb_temp,
	           iNumElements,
	           iLwSize,
	           iOffset);
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

	int grid_rows, grid_cols;
	int iNumIterations = 20;

	// chip parameter
	const static REAL t_chip = 0.0005;
	const static REAL chip_height = 0.016;
	const static REAL chip_width = 0.016;
	// ambient temperature, assuming no package at all
	const static REAL amb_temp = 80.0;

	if(argc != 5 && argc != 6){
		printf("argc = %d\n", argc);
		printf("hotspot <iPlatform - 0, OCL; 1, SMP> <N> <fN> <LwSize> OR\n");
		printf("hotspot <iPlatform - 2, Both> <N> <fN> <LwSizeOCL> <LwSizeSMP>\n");
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

		//add padding rows to the top and bottom borders
		grid_rows = iNumElementsSingle + 2;
		grid_cols = N;
		int iOffsetSingleAfterPadding = iOffsetSingle + 1;
		printf("Size of 3 arrays = %f (KB) = %ld (Bytes)\n", (double)(grid_rows * grid_cols * sizeof(REAL) * 3) / 1024.0, (grid_rows * grid_cols * sizeof(REAL) * 3));
		printf("offset = %i, offset after padding = %d, # elements in grid rows = %i\n\n", iOffsetSingle, iOffsetSingleAfterPadding, iNumElementsSingle);

		REAL grid_height = chip_height / iNumElementsSingle;
		REAL grid_width = chip_width / grid_cols;

		REAL Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
		REAL Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
		REAL Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
		REAL Rz = t_chip / (K_SI * grid_height * grid_width);

		REAL max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
		REAL step = PRECISION / max_slope;

		//make the memory align to 128 bytes
		int min_align = 1024;
		REAL* orig_temp = (REAL*)memalign(min_align/8, sizeof(REAL) * iNumElementsSingle * grid_cols);
		REAL* input_temp = (REAL*)memalign(min_align/8, sizeof(REAL) * grid_rows * grid_cols);
		REAL* output_temp = (REAL*)memalign(min_align/8, sizeof(REAL) * grid_rows * grid_cols);
		REAL* power = (REAL*)memalign(min_align/8, sizeof(REAL) * grid_rows * grid_cols);

		InitMatrix(orig_temp, iNumElementsSingle, grid_cols);
		InitMatrix(power, grid_rows, grid_cols);
		//pad one row of 0.0 before the top row
		memset(&input_temp[0*grid_cols], 0, sizeof(REAL) * grid_cols);
		//copy orig_temp to input_temp
		memcpy(&input_temp[1*grid_cols], orig_temp, sizeof(REAL) * iNumElementsSingle * grid_cols);
		//pad one row of 0.0 after the bottom row
		memset(&input_temp[(iNumElementsSingle+1)*grid_cols], 0, sizeof(REAL) * grid_cols);
		memset(output_temp, 0, sizeof(REAL) * grid_rows * grid_cols);

		//PrintMatrix(input_temp, grid_rows, grid_cols);
		//PrintMatrix(power, grid_rows, grid_cols);

		if(iPlatform == 0){
#ifdef PROFILING
		double dTOCLStart = 0.0, dTOCLEnd = 0.0, dTOCL = 0.0;
		int i, iStart = -10, iEnd = 10;   //iEnd - iStart = iNumIterations
		for(i = iStart; i < iEnd; i++){
			if(i == 0)
				dTOCLStart = GetTimeInMs();
#endif
				RunKernelOCL(input_temp, output_temp, power, grid_rows, grid_cols,
					Cap, Rx, Ry, Rz, step, amb_temp, iNumElementsSingle, iLwSizeSingle, iOffsetSingleAfterPadding);
				//use taskwait because for the BOTH case, data needs sync after each iteration, and pointers need swapping
				#pragma omp taskwait
				if(iNumIterations != 1){
					REAL* tmp_temp = input_temp;
					input_temp = output_temp;
					output_temp = tmp_temp;
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
				RunKernelSMP(input_temp, output_temp, power, grid_rows, grid_cols,
						Cap, Rx, Ry, Rz, step, amb_temp, iNumElementsSingle, iLwSizeSingle, iOffsetSingleAfterPadding);
				//use taskwait to sync all the smp tasks after an iteration, and then swapping the pointers
				#pragma omp taskwait
				if(iNumIterations != 1){
					REAL* tmp_temp = input_temp;
					input_temp = output_temp;
					output_temp = tmp_temp;
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

		REAL* golden_input_temp = (REAL*)malloc(sizeof(REAL) * grid_rows * grid_cols);
		REAL* golden_output_temp = (REAL*)malloc(sizeof(REAL) * grid_rows * grid_cols);
		memset(&golden_input_temp[0*grid_cols], 0, sizeof(REAL) * grid_cols);
		memcpy(&golden_input_temp[1*grid_cols], orig_temp, sizeof(REAL) * iNumElementsSingle * grid_cols);
		memset(&golden_input_temp[(iNumElementsSingle+1)*grid_cols], 0, sizeof(REAL) * grid_cols);
		memset(golden_output_temp, 0, sizeof(REAL) * grid_rows * grid_cols);

		ComputeHost(golden_input_temp, golden_output_temp, power, grid_rows, grid_cols,
				Cap, Rx, Ry, Rz, step, amb_temp, iNumIterations);
		if(iNumIterations % 2  == 1){
			CompareMatrix(output_temp, golden_output_temp, grid_rows, grid_cols);
			//PrintMatrix(output_temp, grid_rows, grid_cols);
			//PrintMatrix(golden_output_temp, grid_rows, grid_cols);
		}
		if(iNumIterations % 2  == 0){
			CompareMatrix(input_temp, golden_input_temp, grid_rows, grid_cols);
			//PrintMatrix(input_temp, grid_rows, grid_cols);
			//PrintMatrix(golden_input_temp, grid_rows, grid_cols);
		}

		if(golden_input_temp != NULL) free(golden_input_temp);
		if(golden_output_temp != NULL) free(golden_output_temp);

		if(orig_temp != NULL) free(orig_temp);
		if(input_temp != NULL) free(input_temp);
		if(output_temp != NULL) free(output_temp);
		if(power != NULL) free(power);
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

		//add padding rows to the top and bottom borders
		grid_rows = N + 2;
		grid_cols = N;
		int iOffsetOCLAfterPadding = iOffsetOCL + 1;
		int iOffsetSMPAfterPadding = iOffsetSMP + 1;
		printf("Size of 3 arrays = %f (KB) = %ld (Bytes)\n", (double)(grid_rows * grid_cols * sizeof(REAL) * 3) / 1024.0, (grid_rows * grid_cols * sizeof(REAL) * 3));
		printf("OCL: offset = %i, offset after padding = %d, # elements in grid rows = %i\n", iOffsetOCL, iOffsetOCLAfterPadding, iNumElementsOCL);
		printf("SMP: offset = %i, offset after padding = %d, # elements in grid rows = %i\n\n", iOffsetSMP, iOffsetSMPAfterPadding, iNumElementsSMP);

		REAL grid_height = chip_height / N;
		REAL grid_width = chip_width / grid_cols;

		REAL Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
		REAL Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
		REAL Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
		REAL Rz = t_chip / (K_SI * grid_height * grid_width);

		REAL max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
		REAL step = PRECISION / max_slope;

		//make the memory align to 128 bytes
		int min_align = 1024;
		REAL* orig_temp = (REAL*)memalign(min_align/8, sizeof(REAL) * N * grid_cols);
		REAL* input_temp = (REAL*)memalign(min_align/8, sizeof(REAL) * grid_rows * grid_cols);
		REAL* output_temp = (REAL*)memalign(min_align/8, sizeof(REAL) * grid_rows * grid_cols);
		REAL* power = (REAL*)memalign(min_align/8, sizeof(REAL) * grid_rows * grid_cols);

		InitMatrix(orig_temp, N, grid_cols);
		InitMatrix(power, grid_rows, grid_cols);
		//pad one row of 0.0 before the top row
		memset(&input_temp[0*grid_cols], 0, sizeof(REAL) * grid_cols);
		//copy orig_temp to input_temp
		memcpy(&input_temp[1*grid_cols], orig_temp, sizeof(REAL) * N * grid_cols);
		//pad one row of 0.0 after the bottom row
		memset(&input_temp[(N+1)*grid_cols], 0, sizeof(REAL) * grid_cols);
		memset(output_temp, 0, sizeof(REAL) * grid_rows * grid_cols);

		if(iNumElementsOCL == 0){ //run only smp
			//RunKernelSMP(input_temp, output_temp, power, grid_rows, grid_cols,
			//	Cap, Rx, Ry, Rz, step, amb_temp, N, iLwSizeSMP, iOffsetSMPAfterPadding);
			//#pragma omp taskwait
			printf("run only smp\n");
			fprintf(pFTime, "Both: only smp\n");
		}
		if(iNumElementsSMP == 0){ //run only ocl
			//RunKernelOCL(input_temp, output_temp, power, grid_rows, grid_cols,
			//	Cap, Rx, Ry, Rz, step, amb_temp, N, iLwSizeOCL, iOffsetOCLAfterPadding);
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
			RunKernelOCL(input_temp, output_temp, power, grid_rows, grid_cols,
					Cap, Rx, Ry, Rz, step, amb_temp, iNumElementsOCL, iLwSizeOCL, iOffsetOCLAfterPadding);
			RunKernelSMP(input_temp, output_temp, power, grid_rows, grid_cols,
					Cap, Rx, Ry, Rz, step, amb_temp, iNumElementsSMP, iLwSizeSMP, iOffsetSMPAfterPadding);
			#pragma omp taskwait
			if(iNumIterations != 1){
				REAL* tmp_temp = input_temp;
				input_temp = output_temp;
				output_temp = tmp_temp;
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

		REAL* golden_input_temp = (REAL*)malloc(sizeof(REAL) * grid_rows * grid_cols);
		REAL* golden_output_temp = (REAL*)malloc(sizeof(REAL) * grid_rows * grid_cols);
		memset(&golden_input_temp[0*grid_cols], 0, sizeof(REAL) * grid_cols);
		memcpy(&golden_input_temp[1*grid_cols], orig_temp, sizeof(REAL) * N * grid_cols);
		memset(&golden_input_temp[(N+1)*grid_cols], 0, sizeof(REAL) * grid_cols);
		memset(golden_output_temp, 0, sizeof(REAL) * grid_rows * grid_cols);

		ComputeHost(golden_input_temp, golden_output_temp, power, grid_rows, grid_cols,
				Cap, Rx, Ry, Rz, step, amb_temp, iNumIterations);
		if(iNumIterations % 2  == 1){
			CompareMatrix(output_temp, golden_output_temp, grid_rows, grid_cols);
			//PrintMatrix(output_temp, grid_rows, grid_cols);
			//PrintMatrix(golden_output_temp, grid_rows, grid_cols);
		}
		if(iNumIterations % 2  == 0){
			CompareMatrix(input_temp, golden_input_temp, grid_rows, grid_cols);
			//PrintMatrix(input_temp, grid_rows, grid_cols);
			//PrintMatrix(golden_input_temp, grid_rows, grid_cols);
		}

		if(golden_input_temp != NULL) free(golden_input_temp);
		if(golden_output_temp != NULL) free(golden_output_temp);

		if(orig_temp != NULL) free(orig_temp);
		if(input_temp != NULL) free(input_temp);
		if(output_temp != NULL) free(output_temp);
		if(power != NULL) free(power);
	}

	printf("DONE\n\n");

	return 0;

}

