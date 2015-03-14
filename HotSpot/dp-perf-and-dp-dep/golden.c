#include "jutil.h"

#define STR_SIZE 256

#define max(a, b) (a > b ? a : b)
#define min(a, b) (a <= b ? a : b)

void readinput(REAL *vect, int grid_rows, int grid_cols, char *file)
{

  	int i,j;
	FILE *fp;
	char str[STR_SIZE];
	float val;

	if( (fp  = fopen(file, "r" )) ==0 ){
		printf("The file was not opened");
		exit(EXIT_FAILURE);
	}


	for (i=0; i <= grid_rows-1; i++)
	 for (j=0; j <= grid_cols-1; j++)
	 {
		if (fgets(str, STR_SIZE, fp) == NULL){
			printf("Error reading file\n");
			exit(EXIT_FAILURE);
		}
		if (feof(fp)){
			printf("not enough lines in file");
			exit(EXIT_FAILURE);
		}
		//if ((sscanf(str, "%d%f", &index, &val) != 2) || (index != ((i-1)*(grid_cols-2)+j-1)))
		if ((sscanf(str, "%f", &val) != 1)){
			printf("invalid file format");
			exit(EXIT_FAILURE);
		}
		vect[i*grid_cols+j] = val;
	}

	fclose(fp);

}

void ComputeHost(REAL* input_temp, REAL* output_temp, REAL* power, int row, int col,
		REAL Cap, REAL Rx, REAL Ry, REAL Rz, REAL step, REAL amb_temp, int iNumIterations)
{

	int k;
	for(k = 0; k < iNumIterations; k++){

		int i, j;
		//limit the computation within the valid range, excluding the padded rows
		for(i = 1; i <= row-2; i++){
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

				output_temp[i * col + j] = input_temp[i * col + j] + delta;
			}
		}

		if(iNumIterations != 1){
			REAL* tmp_temp = input_temp;
			input_temp = output_temp;
			output_temp = tmp_temp;
		}

	}

}


