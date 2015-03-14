/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * kernel.cl
 */

#include <kernel.h>

#if(0)
    #define EXP(a) native_exp(a)
    #define LOG(a) native_log(a)
    #define SQRT(a) native_sqrt(a)
#else
    #define EXP(a) exp(a)
    #define LOG(a) log(a)
    #define SQRT(a) sqrt(a)
#endif


///////////////////////////////////////////////////////////////////////////////
// Predefine functions to avoid bug in OpenCL compiler on Mac OSX 10.7 systems
///////////////////////////////////////////////////////////////////////////////
REAL CND(REAL d);
void BlackScholesBody(__global REAL *call, __global REAL *put,  REAL S,
					  REAL X, REAL T, REAL R, REAL V);

///////////////////////////////////////////////////////////////////////////////
// Rational approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
#ifdef DP
REAL CND(REAL d){
    const REAL       A1 = 0.31938153;
    const REAL       A2 = -0.356563782;
    const REAL       A3 = 1.781477937;
    const REAL       A4 = -1.821255978;
    const REAL       A5 = 1.330274429;
    const REAL RSQRT2PI = 0.39894228040143267793994605993438;

    REAL
        K = 1.0 / (1.0 + 0.2316419 * fabs(d));

    REAL
        cnd = RSQRT2PI * EXP(- 0.5 * d * d) * 
        (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if(d > 0)
        cnd = 1.0 - cnd;

    return cnd;
}
#else
REAL CND(REAL d){
    const REAL       A1 = 0.31938153f;
    const REAL       A2 = -0.356563782f;
    const REAL       A3 = 1.781477937f;
    const REAL       A4 = -1.821255978f;
    const REAL       A5 = 1.330274429f;
    const REAL RSQRT2PI = 0.39894228040143267793994605993438f;
    
    REAL
    K = 1.0f / (1.0f + 0.2316419f * fabs(d));
    
    REAL
    cnd = RSQRT2PI * EXP(- 0.5f * d * d) * 
    (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
    
    if(d > 0)
        cnd = 1.0f - cnd;
    
    return cnd;
}

#endif


///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
void BlackScholesBody(
    __global REAL *call, //Call option price
    __global REAL *put,  //Put option price
    REAL S,              //Current stock price
    REAL X,              //Option strike price
    REAL T,              //Option years
    REAL R,              //Riskless rate of return
    REAL V               //Stock volatility
){
    REAL sqrtT = SQRT(T);
    REAL    d1 = (LOG(S / X) + (R + 0.5f * V * V) * T) / (V * sqrtT);
    REAL    d2 = d1 - V * sqrtT;
    REAL CNDD1 = CND(d1);
    REAL CNDD2 = CND(d2);

    //Calculate Call and Put simultaneously
    REAL expRT = EXP(- R * T);
    *call = (S * CNDD1 - X * expRT * CNDD2);
#ifdef DP
    *put  = (X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1));
#else
    *put  = (X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1));
#endif
}



__kernel void BlackScholesOCL(
    __global REAL *pCall, //Call option price
    __global REAL *pPut,  //Put option price
    __global REAL *pS,    //Current stock price
    __global REAL *pX,    //Option strike price
    __global REAL *pT,    //Option years
    REAL R,                //Riskless rate of return
    REAL V,                //Stock volatility
    int iNumElements,
    int iLwSize                    
){
    for(int opt = get_global_id(0); opt < iNumElements; opt += get_global_size(0))
        BlackScholesBody(
            &pCall[opt],
            &pPut[opt],
            pS[opt],
            pX[opt],
            pT[opt],
            R,
            V
        );
}
