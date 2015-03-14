#include "jutil.h"

void CopySMP(REAL* __restrict a, REAL* __restrict c, int iNumElements, int iLwSize)
{
	int j;
	for(j = 0; j < iNumElements; j++)
		c[j] = a[j];
}

void ScaleSMP(REAL* __restrict c, REAL* __restrict b, REAL scalar, int iNumElements, int iLwSize)
{
	int j;
	for(j = 0; j < iNumElements; j++)
		b[j] = scalar * c[j];
}

void AddSMP(REAL* __restrict a, REAL* __restrict b, REAL* __restrict c, int iNumElements, int iLwSize)
{
	int j;
	for(j = 0; j < iNumElements; j++)
		c[j] = a[j] + b[j];
}

void TriadSMP(REAL* __restrict b, REAL* __restrict c, REAL* __restrict a, REAL scalar, int iNumElements, int iLwSize)
{
	int j;
	for(j = 0; j < iNumElements; j++)
		a[j] = b[j]+scalar*c[j];
}
