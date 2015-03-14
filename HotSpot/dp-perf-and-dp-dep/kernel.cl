/*
 * kernel.cl
 */

#include <kernel.h>

#define max(a, b) (a > b ? a : b)
#define min(a, b) (a <= b ? a : b)

__kernel void HotspotOCL(__global REAL* input_temp,
                         __global REAL* output_temp,
                         __global REAL* power,
                         int row,
                         int col,
                         REAL Cap,
                         REAL Rx,
                         REAL Ry,
                         REAL Rz,
                         REAL step,
                         REAL amb_temp,
                         int iNumElements,
                         int iLwSize,
                  	  	 int iOffset)
{
    int c = get_global_id(0);
    int r = get_global_id(1) + 1; //input sees iNumElements+2 rows, valid row starts from 1
    
    if((c >= 0) && (c <= col-1) && (r >= 1) && (r <= iNumElements)){
     
        int top = r-1; 
        int bottom = r+1;
        int left = max(c-1, 0);        //process the left border and the right border as they are not padded 
        int right = min(c+1, col-1);
    
        REAL delta;
        delta = (step / Cap) * (power[r * col + c] + 
                        (input_temp[top * col + c] + input_temp[bottom * col + c] - 2.0 * input_temp[r * col + c]) / Ry + 
                        (input_temp[r * col + right] + input_temp[r * col + left] - 2.0 * input_temp[r * col + c]) / Rx + 
                        (amb_temp - input_temp[r * col + c]) / Rz);
        
        
        //output sees iNumElements rows, valid row starts from 0
        int output_r = r - 1;
        output_temp[output_r * col + c] = input_temp[r * col + c] + delta;
    }
}
