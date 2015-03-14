#include "jutil.h"

void initialiseSystemRandomly(REAL* positions, REAL* velocities, int iNumElements)
{
	int i;
	for ( i = 0; i < iNumElements; i++ )
	{
#ifdef DP
		positions[4 * i] = rand() / RAND_MAX * 2 - 1;
		positions[(4 * i) + 1] = rand() / RAND_MAX * 2 - 1;
		positions[(4 * i) + 2] = rand() / RAND_MAX * 2 - 1;
		positions[(4 * i) + 3] = 1.0; // mass

		velocities[4 * i] = rand() / RAND_MAX * 2 - 1;
		velocities[(4 * i)+1] = rand() / RAND_MAX * 2 - 1;
		velocities[(4 * i)+2] = rand() / RAND_MAX * 2 - 1;
		velocities[(4 * i)+3] = 1.0; // inverse mass
#else
		positions[4 * i] = rand() / (float) RAND_MAX * 2 - 1;
		positions[(4 * i) + 1] = rand() / (float) RAND_MAX * 2 - 1;
		positions[(4 * i) + 2] = rand() / (float) RAND_MAX * 2 - 1;
		positions[(4 * i) + 3] = 1.0f; // mass

		velocities[4 * i] = rand() / (float) RAND_MAX * 2 - 1;
		velocities[(4 * i)+1] = rand() / (float) RAND_MAX * 2 - 1;
		velocities[(4 * i)+2] = rand() / (float) RAND_MAX * 2 - 1;
		velocities[(4 * i)+3] = 1.0f; // inverse mass
#endif
	}
}

void computeReferenceResult(REAL* input_positions, REAL* input_velocities, REAL* output_positions, REAL* output_velocities,
				const REAL dT, const REAL damping, const REAL softeningSquared, int iNumElements, int iNumIterations)
{
	int k;
	for(k = 0; k < iNumIterations; k++){

		int i;
		for(i = 0; i < iNumElements; i++){

			REAL pos[4], vel[4];
			pos[0] = input_positions[i*4+0];
			pos[1] = input_positions[i*4+1];
			pos[2] = input_positions[i*4+2];
			pos[3] = input_positions[i*4+3];
			vel[0] = input_velocities[i*4+0];
			vel[1] = input_velocities[i*4+1];
			vel[2] = input_velocities[i*4+2];
			vel[3] = input_velocities[i*4+3];

			REAL force[3];
			force[0] = 0.0;
			force[1] = 0.0;
			force[2] = 0.0;

			int j;
			for(j = 0; j < iNumElements; j++){

				REAL r[3];
		        r[0] = pos[0] - input_positions[j*4+0];
		        r[1] = pos[1] - input_positions[j*4+1];
		        r[2] = pos[2] - input_positions[j*4+2];

		        REAL distSqr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
		        distSqr += softeningSquared;
#ifdef DP
		        REAL invDist = (REAL)1.0 / (REAL)sqrt(distSqr);
#else
		        REAL invDist = (REAL)1.0f / (REAL)sqrt(distSqr);
#endif
		        REAL invDistCube =  invDist * invDist * invDist;
		        REAL s = pos[3] * invDistCube;

		        force[0] += r[0] * s;
		        force[1] += r[1] * s;
		        force[2] += r[2] * s;
			}

			//REAL invMass = input_velocities[i*4+3];
			REAL invMass = vel[3];

			// acceleration = force / mass;
			// new velocity = old velocity + acceleration * deltaTime
			vel[0] += (force[0] * invMass) * dT;
			vel[1] += (force[1] * invMass) * dT;
			vel[2] += (force[2] * invMass) * dT;

			vel[0] *= damping;
			vel[1] *= damping;
			vel[2] *= damping;

			// new position = old position + velocity * deltaTime
			pos[0] += vel[0] * dT;
			pos[1] += vel[1] * dT;
			pos[2] += vel[2] * dT;

			output_positions[i*4+0] = pos[0];
			output_positions[i*4+1] = pos[1];
			output_positions[i*4+2] = pos[2];
			output_positions[i*4+3] = pos[3];  //the last element stays the same for all the iterations

			output_velocities[i*4+0] = vel[0];
			output_velocities[i*4+1] = vel[1];
			output_velocities[i*4+2] = vel[2];
			output_velocities[i*4+3] = vel[3]; //the last element stays the same for all the iterations
		}

		if(iNumIterations != 1){
			REAL* temp_positions = input_positions;
			REAL* temp_velocities = input_velocities;
			input_positions = output_positions;
			input_velocities = output_velocities;
			output_positions = temp_positions;
			output_velocities = temp_velocities;
		}
	}
}

inline int compareParticleVector(const REAL* benchmarkParticle, const REAL* referenceParticle )
{
	REAL thresholdX = fabs( referenceParticle[0] * 0.05 ) + 0.01;
	REAL thresholdY = fabs( referenceParticle[1] * 0.05 ) + 0.01;
	REAL thresholdZ = fabs( referenceParticle[2] * 0.05 ) + 0.01;

	REAL deltaX = fabs(benchmarkParticle[0] - referenceParticle[0]);
	REAL deltaY = fabs(benchmarkParticle[1] - referenceParticle[1]);
	REAL deltaZ = fabs(benchmarkParticle[2] - referenceParticle[2]);

    if ( deltaX > thresholdX )
    {
        return 0;
    }

    if ( deltaY > thresholdY )
    {
        return 0;
    }

    if ( deltaZ > thresholdZ )
    {
        return 0;
    }

    return 1;
}

inline void printParticleVector( const char* string, const REAL* particle  )
{
    printf( "%-32s %f, %f, %f, %f\n", string, particle[0], particle[1], particle[2], particle[3] );
}

void verifyResults(REAL* positions, REAL* velocities,
		REAL* golden_positions, REAL* golden_velocities,  int iNumElements)
{
	int i;
	int count = 0;
	for(i = 0; i < iNumElements; i++){
		if(compareParticleVector( &positions[i*4], &golden_positions[i*4] ) == 0){
			printf( "particle position failed at %d\n", (int)i );
			printParticleVector( "benchmark output", &positions[i*4]);
			printParticleVector( "golden output", &golden_positions[i*4]);
			count++;
			printf("count = %d\n", count);
			printf("\n");
		}
	}
	printf("positions count = %d\n", count);
	for(i = 0; i < iNumElements; i++){
		if(compareParticleVector( &velocities[i*4], &golden_velocities[i*4] ) == 0){
			printf( "particle velocity failed at %d\n", (int)i );
			printParticleVector( "benchmark output", &velocities[i*4]);
			printParticleVector( "golden output", &golden_velocities[i*4]);
			count++;
			printf("count = %d\n", count);
			printf("\n");
		}
	}
	printf("positions+velocities count = %d\n", count);
	if(count == 0)
		printf("PASS\n");
	else
		printf("FAIL\n");
}


void printResults(REAL* positions, REAL* velocities,
		REAL* golden_positions, REAL* golden_velocities,  int iNumElements)
{
	printf("------------Positions-------------\n");
    int i;
    for ( i = 0; i < iNumElements; i++ )
    {
			printf( "particle position at %d\n", (int)i );
            printParticleVector( "benchmark output", &positions[i*4] );
            printParticleVector( "golden output", &golden_positions[i*4] );
            printf("\n");
    }

    printf("------------Velocities-------------\n");
    for ( i = 0; i < iNumElements; i++ )
    {
			printf( "particle velocity at %d\n", (int)i );
            printParticleVector( "benchmark output", &velocities[i*4] );
            printParticleVector( "golden output", &golden_velocities[i*4] );
            printf( "\n");
    }

    printf("------------End-------------\n");
}






