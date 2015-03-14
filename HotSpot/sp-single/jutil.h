/*
 * jutil.h
 *
 *  Created on: Dec 17, 2014
 *      Author: Jie Shen
 *  Revised on: Feb 8, 2015
 *
 */

#ifndef JUTIL_H_
#define JUTIL_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <unistd.h>
#include <malloc.h>

#ifdef DP
#define REAL double
#define REAL_EPSILON DBL_EPSILON
#else
#define REAL float
#define REAL_EPSILON 1e-6
//#define REAL_EPSILON FLT_EPSILON
#endif

#define PROFILING
//#define PRINT

double GetTimeInS(void);
double GetTimeInMs(void);
int RoundUp(int iDivisor, int iDividend);
void InitArray(REAL* pArray, int iNumElements);
void InitArrayInRange(REAL* pArray, int iNumElements, REAL rLow, REAL rHigh);
void InitMatrix(REAL* pMatrix, int iRows, int iCols);
void PrintArray(REAL* pArray, int iNumElements);
void PrintMatrix(REAL* pMatrix, int iRows, int iCols);
void CompareArray(REAL* pResult, REAL* pReference, int iNumElements);
void CompareMatrix(REAL* pResult, REAL* pReference, int iRows, int iCols);
void ConvertMatrixLinear2Block(REAL* pMatrix, REAL** pMatrixBlock, int iRows, int iCols, int iDimCols, int iLwSize);
void ConvertMatrixBlock2Linear(REAL* pMatrix, REAL** pMatrixBlock, int iRows, int iCols, int iDimCols, int iLwSize);
void PrintMatrixInBlock(REAL** pMatrixBlock, int iDimRows, int iDimCols, int iLwSize);

#endif /* JUTIL_H_ */
