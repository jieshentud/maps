/*
 * jutil.c
 *
 *  Created on: Dec 17, 2014
 *      Author: Jie Shen
 *  Revised on: Jan 28, 2015
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

///////////////////////////////////////////////////////////////////////////////
// Rational approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
#ifdef DP
static REAL CND(REAL d){
    const REAL       A1 = 0.31938153;
    const REAL       A2 = -0.356563782;
    const REAL       A3 = 1.781477937;
    const REAL       A4 = -1.821255978;
    const REAL       A5 = 1.330274429;
    const REAL RSQRT2PI = 0.39894228040143267793994605993438;

    REAL
        K = 1.0 / (1.0 + 0.2316419 * fabs(d));

    REAL
        cnd = RSQRT2PI * exp(- 0.5f * d * d) *
        (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if(d > 0)
        cnd = 1.0 - cnd;

    return cnd;
}
#else
static REAL CND(REAL d){
    const REAL       A1 = 0.31938153f;
    const REAL       A2 = -0.356563782f;
    const REAL       A3 = 1.781477937f;
    const REAL       A4 = -1.821255978f;
    const REAL       A5 = 1.330274429f;
    const REAL RSQRT2PI = 0.39894228040143267793994605993438f;

    REAL
        K = 1.0f / (1.0f + 0.2316419f * fabs(d));

    REAL
        cnd = RSQRT2PI * exp(- 0.5f * d * d) *
        (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if(d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}
#endif

///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
static void BlackScholesBodyCPU(
    REAL* call, //Call option price
    REAL* put,  //Put option price
    REAL Sf,    //Current stock price
    REAL Xf,    //Option strike price
    REAL Tf,    //Option years
    REAL Rf,    //Riskless rate of return
    REAL Vf     //Stock volatility
){
	REAL S = Sf, X = Xf, T = Tf, R = Rf, V = Vf;

	REAL sqrtT = sqrt(T);
	REAL    d1 = (log(S / X) + (R + 0.5f * V * V) * T) / (V * sqrtT);
	REAL    d2 = d1 - V * sqrtT;
	REAL CNDD1 = CND(d1);
	REAL CNDD2 = CND(d2);

    //Calculate Call and Put simultaneously
	REAL expRT = exp(- R * T);
    *call = (S * CNDD1 - X * expRT * CNDD2);
#ifdef DP
    *put  = (X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1));
#else
    *put  = (X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1));
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options
////////////////////////////////////////////////////////////////////////////////
void ComputeHost(
    REAL *pCallGolden, //Call option price
    REAL *pPutGolden,  //Put option price
    REAL *pS,    //Current stock price
    REAL *pX,    //Option strike price
    REAL *pT,    //Option years
    REAL R,       //Riskless rate of return
    REAL V,       //Stock volatility
    int iNumElements
){
	int i;
    for(i = 0; i < iNumElements; i++)
        BlackScholesBodyCPU(
        	&(pCallGolden[i]),
        	&(pPutGolden[i]),
            pS[i],
            pX[i],
            pT[i],
            R,
            V
        );
}

void CheckResult(REAL* pCall, REAL* pPut, REAL* pCallGolden, REAL* pPutGolden, int iNumElements)
{
	double deltaCall = 0.0, deltaPut = 0.0, sumCall = 0.0, sumPut = 0.0;
	double L1call, L1put;
	int i;
	for(i = 0; i < iNumElements; i++)
	{
		sumCall += fabs(pCallGolden[i]);
		sumPut  += fabs(pPutGolden[i]);
		deltaCall += fabs(pCallGolden[i] - pCall[i]);
		deltaPut  += fabs(pPutGolden[i] - pPut[i]);
	}
	L1call = deltaCall / sumCall;
	L1put = deltaPut / sumPut;
	printf("Relative L1 (call, put) = (%.3e, %.3e)\n", L1call, L1put);
	if((L1call < 1E-6) && (L1put < 1E-6))
		printf("\nRESULT PASS\n\n");
}





