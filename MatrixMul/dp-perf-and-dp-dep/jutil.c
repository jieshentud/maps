/*
 * jutil.c
 *
 *  Created on: Dec 17, 2014
 *      Author: Jie Shen
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
	//srand(time(NULL));
	srand(999);
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

void InitMatrix(REAL* pMatrix, int iRows, int iCols)
{
	//srand(time(NULL));
	srand(999);
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

void ComputeHost(REAL* pA, REAL* pB, REAL* pC, int iHA, int iWA, int iWB)
{
	int i, j, k;
	for(i = 0; i < iHA; i++)
		for(j = 0; j < iWB; j++){
			REAL sum = 0;
			for(k = 0; k < iWA; k++)
				sum += pA[i * iWA + k] * pB[k * iWB + j];
			pC[i * iWB + j] = sum;
		}
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

