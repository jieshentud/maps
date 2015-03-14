/*
 * kernel.h
 */

#ifndef KERNEL_H_
#define KERNEL_H_

#ifdef DP
#define REAL double
#else
#define REAL float
#endif

//local work size, define here if kernel.cl needs this
//#define BLOCK_SIZE 32

/*
#ifdef __cplusplus
extern "C"
{
#endif

//NDRange first worksize.dim0, then worksize.dim1
#pragma omp target device(opencl) copy_deps ndrange(1, iNumElements, iLwSize) file(kernel.cl)
#pragma omp task in(pS[0;iNumElements], pX[0;iNumElements], pT[0;iNumElements]) out(pCall[0;iNumElements], pPut[0;iNumElements])
__kernel void BlackScholesOCL(
    __global REAL *pCall, //Call option price
    __global REAL *pPut,  //Put option price
    __global REAL *pS,    //Current stock price
    __global REAL *pX,    //Option strike price
    __global REAL *pT,    //Option years
    REAL R,               //Riskless rate of return
    REAL V,               //Stock volatility
    int iNumElements,
    int iLwSize
    );

#ifdef __cplusplus
}
#endif
*/



#endif /* KERNEL_H_ */
