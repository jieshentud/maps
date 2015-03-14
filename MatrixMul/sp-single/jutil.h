/*
 * jutil.h
 *
 *  Created on: Dec 17, 2014
 *      Author: Jie Shen
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
void InitMatrix(REAL* pMatrix, int iRows, int iCols);
void PrintArray(REAL* pArray, int iNumElements);
void PrintMatrix(REAL* pMatrix, int iRows, int iCols);
void ComputeHost(REAL* pA, REAL* pB, REAL* pC, int iHA, int iWA, int iWB);
void CompareArray(REAL* pResult, REAL* pReference, int iNumElements);
void CompareMatrix(REAL* pResult, REAL* pReference, int iRows, int iCols);

#endif /* JUTIL_H_ */
