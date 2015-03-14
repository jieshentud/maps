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

#ifdef __cplusplus
extern "C"
{
#endif

//NDRange first worksize.dim0, then worksize.dim1
#pragma omp target device(opencl) copy_deps ndrange(2, col, iNumElements, iLwSize, iLwSize) file(kernel.cl) implements(HotspotSMP)
#pragma omp task in(input_temp[0;(iNumElements+2)*col], power[0;(iNumElements+2)*col]) out(output_temp[0;iNumElements*col])
__kernel void HotspotOCL(__global REAL* input_temp,
                         __global REAL* output_temp,
                         __global REAL* power,
                         int row,
                         int col,
                         REAL Cap,
                         REAL Rx,
                         REAL Ry,
                         REAL Rz,
                         REAL step,
                         REAL amb_temp,
                         int iNumElements,
                         int iLwSize,
                  	  	 int iOffset);

#ifdef __cplusplus
}
#endif

void RunKernelBoth(REAL* input_temp, REAL* output_temp, REAL* power, int row, int col,
		REAL Cap, REAL Rx, REAL Ry, REAL Rz, REAL step, REAL amb_temp, int N, int iTaskSize, int iLwSize)
{
	int i;
	int iRemainder = N % iTaskSize;
	int iNumTasks = N / iTaskSize;
	int iTaskOffset;  //the task offset in GLOBAL view
#ifdef PRINT
	printf("Total NumTasks = %d, Remainder = %d\n", iNumTasks, iRemainder);
#endif
	for(i = 0; i < iNumTasks; i++){
		int iTaskOffset = i * iTaskSize + 1;  //add padding to the top and bottom borders, valid row starts from 1
		//call HotspotSMP and HotspotOCL tasks
		HotspotSMP(&(input_temp[(iTaskOffset-1) * col]),  //input starts from the padding row
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
				   iTaskOffset
				);
	}

}

int main(int argc, char** argv)
{
	int N, iTaskSize, iLwSize;
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

	if(argc != 5){
		printf("argc = %d\n", argc);
		printf("hotspot <iPlatform - 2, Both> <N> <TaskSize> <LwSize> OR\n");
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

	grid_rows = N + 2;
	grid_cols = N;

	printf("Size of 3 arrays = %f (KB) = %ld (Bytes)\n", (double)(grid_rows * grid_cols * sizeof(REAL) * 3) / 1024.0, grid_rows * grid_cols * sizeof(REAL) * 3);

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
		RunKernelBoth(input_temp, output_temp, power, grid_rows, grid_cols,
				Cap, Rx, Ry, Rz, step, amb_temp, N, iTaskSize, iLwSize);
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

	printf("DONE\n\n");

	return 0;
}
