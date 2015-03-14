#include "jutil.h"

//argument iLwSize is not used
void NBodySMP(REAL* __restrict input_positions, REAL* __restrict input_velocities, REAL* __restrict output_positions, REAL* __restrict output_velocities,
		const REAL dT, const REAL damping, const REAL softeningSquared, int N, int iTaskOffset, int iTaskSize, int iLwSize)
{
	int i;
	for(i = 0; i < iTaskSize; i++){
		//input is the global input, output is a local part of the global output
		REAL pos[4], vel[4];
		pos[0] = input_positions[iTaskOffset*4 + i*4+0];
		pos[1] = input_positions[iTaskOffset*4 + i*4+1];
		pos[2] = input_positions[iTaskOffset*4 + i*4+2];
		pos[3] = input_positions[iTaskOffset*4 + i*4+3];
		vel[0] = input_velocities[iTaskOffset*4 + i*4+0];
		vel[1] = input_velocities[iTaskOffset*4 + i*4+1];
		vel[2] = input_velocities[iTaskOffset*4 + i*4+2];
		vel[3] = input_velocities[iTaskOffset*4 + i*4+3];

		REAL force[3];
		force[0] = 0.0;
		force[1] = 0.0;
		force[2] = 0.0;

		int j;
		for(j = 0; j < N; j++){

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
}
