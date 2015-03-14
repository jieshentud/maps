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

/* Matrix multiplication: C = A * B.
 * Device code.
 */

/*
 * kernel.cl
 */

#include <kernel.h>

#define AS(i, j) As[j + i * BLOCK_SIZE]
#define BS(i, j) Bs[j + i * BLOCK_SIZE]

__kernel void matmulocl(__global REAL* A,__global REAL* B, __global REAL* C, int iNumElements, int iOffset, int iWA, int iWB, int iLwSize)
{
    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);
    
    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    
    // Index of the first sub-matrix of A processed by the block
    int aBegin = iWA * BLOCK_SIZE * by;
    
    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + iWA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;
    
    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;
    
    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * iWB;

    // Shared memory for the sub-matrix of A  
    __local REAL As[BLOCK_SIZE * BLOCK_SIZE];
    // Shared memory for the sub-matrix of B  
    __local REAL Bs[BLOCK_SIZE * BLOCK_SIZE];
    
    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    REAL Csub = 0.0;
    
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep) {
        
        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        AS(ty, tx) = A[a + iWA * ty + tx];
        BS(ty, tx) = B[b + iWB * ty + tx];
        
        // Synchronize to make sure the matrices are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix        
        //#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += AS(ty, k) * BS(k, tx);

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (get_global_id(1) < iNumElements)
        // Write the block sub-matrix to device memory;
        // each thread writes one element
        C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = Csub;
}
