#include "jutil.h"

#define max(a, b) (a > b ? a : b)
#define min(a, b) (a <= b ? a : b)

void HotspotSMP(REAL* __restrict input_temp, REAL* __restrict output_temp, REAL* __restrict power, int row, int col,
		REAL Cap, REAL Rx, REAL Ry, REAL Rz, REAL step, REAL amb_temp, int iTaskSize, int iLwSize, int iTaskOffset)
{
	int i, j;
	//limit the computation within the valid range, excluding the padded rows
	for(i = 1; i <= iTaskSize; i++)
		for(j = 0; j <= col-1; j++)
		{
			int top = i-1;
			int bottom = i+1;
			int left = max(j-1, 0);       //process the left border and the right border as they are not padded
			int right = min(j+1, col-1);

			REAL delta;
			delta = (step / Cap) * (power[i * col + j] +
                    (input_temp[top * col + j] + input_temp[bottom * col + j] - 2.0 * input_temp[i * col + j]) / Ry +
                    (input_temp[i * col + right] + input_temp[i * col + left] - 2.0 * input_temp[i * col + j]) / Rx +
                    (amb_temp - input_temp[i * col + j]) / Rz);

			int output_i = i - 1;
			output_temp[output_i * col + j] = input_temp[i * col + j] + delta;
		}
}
