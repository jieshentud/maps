/*
 * jutil.c
 *
 *  Created on: Dec 17, 2014
 *      Author: Jie Shen
 *  Revised on: Feb 4, 2015
 *
 */

#include "jutil.h"

double GetTimeInS(void)  //in second
{
        struct timeval current_time;
        gettimeofday(&current_time, NULL);

        return (double)(current_time.tv_sec) + (double)(current_time.tv_usec * 1.0e-6);
}

double GetTimeInMs(void)  //in millisecond
{
        struct timeval current_time;
        gettimeofday(&current_time, NULL);

        return (double)(current_time.tv_sec * 1.0e3) + (double)(current_time.tv_usec * 1.0e-3);
}

int RoundUp(int iDivisor, int iDividend)
{
	return (iDividend % iDivisor == 0) ? iDividend : (iDividend + (iDivisor - iDividend % iDivisor));
}

void InitArray(REAL* pArray, int iNumElements)
{
	srand(time(NULL));
	//srand(999);
	REAL rScale;
#ifdef DP
	rScale = 1.0 / RAND_MAX;
#else
	rScale = 1.0f / (float)RAND_MAX;
#endif
	int i;
	for(i = 0; i < iNumElements; i++){
		pArray[i] = rScale * rand();
	}
}

void InitArrayInRange(REAL* pArray, int iNumElements, REAL rLow, REAL rHigh)
{
	srand(time(NULL));
	//srand(999);
	REAL rScale;
#ifdef DP
	rScale = 1.0 / RAND_MAX;
#else
	rScale = 1.0f / (float)RAND_MAX;
#endif
	int i;
	for(i = 0; i < iNumElements; i++){
		REAL rVar = rScale * rand();
		pArray[i] = rLow + rVar * (rHigh - rLow);
	}
}

void InitMatrix(REAL* pMatrix, int iRows, int iCols)
{
	srand(time(NULL));
	//srand(999);
	REAL rScale;
#ifdef DP
	rScale = 1.0 / RAND_MAX;
#else
	rScale = 1.0f / (float)RAND_MAX;
#endif
	int i, j;
	for(i = 0; i < iRows; i++)
		for(j = 0; j < iCols; j++)
			pMatrix[i * iCols + j] = rScale * rand();
}

void PrintArray(REAL* pArray, int iNumElements)
{
	int i;
	for(i = 0; i < iNumElements; i++){
		printf("%f\t", pArray[i]);
	}
	printf("\n");
}

void PrintMatrix(REAL* pMatrix, int iRows, int iCols)
{
	int i, j;
	for(i = 0; i < iRows; i++){
			for(j = 0; j < iCols; j++)
				printf("%f\t", pMatrix[i * iCols + j]);
			printf("\n");
	}
	printf("\n");
}

void CompareArray(REAL* pResult, REAL* pReference, int iNumElements)
{
	int i, count = 0;
	printf("\nCHECK...\n");
	for(i = 0; i < iNumElements; i++)
		if(fabs(pResult[i] - pReference[i]) > REAL_EPSILON){
			count++;
			printf("\nDiff = %.12f, EPSILON = %.12f\t\t", fabs(pResult[i] - pReference[i]), REAL_EPSILON);
			printf("Result[%d] = %.6f\t, Reference[%d] = %.6f\n", i, pResult[i], i, pReference[i]);
		}
	if(count == 0)
		printf("\nRESULT PASS\n\n");
	else printf("\nRESULT FAIL in %d/%d\n\n", count, iNumElements);
}

void CompareMatrix(REAL* pResult, REAL* pReference, int iRows, int iCols)
{
	int i, j, count = 0;
	printf("\nCHECK...\n");
	for(i = 0; i < iRows; i++)
		for(j = 0; j < iCols; j++)
			if(fabs(pResult[i * iCols + j] - pReference[i * iCols + j]) > REAL_EPSILON){
				count++;
				//printf("Diff = %.12f, EPSILON = %.12f\t\t", fabs(pResult[i * iCols + j] - pReference[i * iCols + j]), REAL_EPSILON);
				//printf("Result[%d][%d] = %.6f\t, Reference[%d][%d] = %.6f\n", i, j, pResult[i * iCols + j], i, j, pReference[i * iCols + j]);
			}
	if(count == 0)
		printf("\nRESULT PASS\n\n");
	else printf("\nRESULT FAIL in %d/%d\n\n", count, iRows * iCols);
}

/*void CompareMatrix(REAL* pResult, REAL* pReference, int iRows, int iCols)
{
	int i, j, count = 0;
	for(i = 0; i < iRows; i++)
		for(j = 0; j < iCols; j++)
			if(fabs(pResult[i * iCols + j] - pReference[i * iCols + j]) > REAL_EPSILON){
				REAL RelativeErr;
				if(pResult[i * iCols + j] > pReference[i * iCols + j])
					RelativeErr = fabs(pResult[i * iCols + j] - pReference[i * iCols + j]) / pResult[i * iCols + j];
				else RelativeErr = fabs(pResult[i * iCols + j] - pReference[i * iCols + j]) / pReference[i * iCols + j];
				if(RelativeErr > REAL_EPSILON){
					count++;
					printf("Result[%d][%d] = %.6f\t Reference[%d][%d] = %.6f\n", i, j, pResult[i * iCols + j], i, j, pReference[i * iCols + j]);
				}
			}
	if(count == 0)
		printf("\nRESULT PASS\n\n");
	else printf("\nRESULT FAIL\n\n");
}
*/

void ConvertMatrixLinear2Block(REAL* pMatrix, REAL** pMatrixBlock, int iRows, int iCols, int iDimCols, int iLwSize)
{
	int i, j;
	for(i = 0; i < iRows; i++)
		for(j = 0; j < iCols; j++)
			pMatrixBlock[i/iLwSize * iDimCols + j/iLwSize][(i%iLwSize)*iLwSize+(j%iLwSize)] = pMatrix[i*iCols +j];
}

void ConvertMatrixBlock2Linear(REAL* pMatrix, REAL** pMatrixBlock, int iRows, int iCols, int iDimCols, int iLwSize)
{
	int i, j;
	for(i = 0; i < iRows; i++)
		for(j = 0; j < iCols; j++)
			pMatrix[i*iCols +j] = pMatrixBlock[i/iLwSize * iDimCols + j/iLwSize][(i%iLwSize)*iLwSize+(j%iLwSize)];
}

void PrintMatrixInBlock(REAL** pMatrixBlock, int iDimRows, int iDimCols, int iLwSize)
{
	int i, j, k, l;
	for(i = 0; i < iDimRows; i++){
		for(j = 0; j < iDimCols; j++){
			for(k = 0; k < iLwSize; k++){
				for(l = 0; l < iLwSize; l++)
					printf("%f\t", pMatrixBlock[i * iDimCols + j][k * iLwSize + l]);
				printf("\n");
			}
			printf("\n");
		}
		printf("\n");
	}
}




