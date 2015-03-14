#include "jutil.h"

///////////////////////////////////////////////////////////////////////////////
// Rational approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
#ifdef DP
static inline REAL CND(REAL d){
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
static inline REAL CND(REAL d){
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
static inline void BlackScholesBodyCPU(
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

void BlackScholesSMP(REAL* __restrict pCall, REAL* __restrict pPut, REAL* __restrict pS, REAL * __restrict pX, REAL * __restrict pT, REAL R, REAL V, int iNumElements, int iLwSize)
{
	int i;
	for(i = 0; i < iNumElements; i++)
		BlackScholesBodyCPU(
				&(pCall[i]),
				&(pPut[i]),
				pS[i],
				pX[i],
				pT[i],
				R,
				V
	            );
}
