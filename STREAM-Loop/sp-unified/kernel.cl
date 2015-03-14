/*
 * kernel.cl
 */

#include <kernel.h>


__kernel void CopyOCL(__global REAL* a,
                      __global REAL* c,
                      int iNumElements,
                      int iLwSize)
{
    int j = get_global_id(0);
    c[j] = a[j];
}

__kernel void ScaleOCL(__global REAL* c,
                       __global REAL* b,
                       REAL scalar,
                       int iNumElements,
                       int iLwSize)
{
    int j = get_global_id(0);
    b[j] = scalar * c[j];
}

__kernel void AddOCL(__global REAL* a,
                     __global REAL* b,
                     __global REAL* c,
                     int iNumElements,
                     int iLwSize)
{
    int j = get_global_id(0);
    c[j] = a[j] + b[j];
}

__kernel void TriadOCL(__global REAL* b,
                       __global REAL* c,
                       __global REAL* a,
                       REAL scalar,
                       int iNumElements,
                       int iLwSize)
{
    int j = get_global_id(0);
    a[j] = b[j]+scalar*c[j];
}

