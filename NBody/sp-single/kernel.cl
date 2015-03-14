/*
 * kernel.cl
 */
 
#include <kernel.h>

//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void NBodyOCL(
                              __global REAL * input_positions,
                              __global REAL * input_velocities,
                              __global REAL * output_positions,
                              __global REAL * output_velocities,
                              const REAL dT,
                              const REAL damping,
                              const REAL softeningSquared,
                              int N,
                              int iOffset,
                              int iNumElements,
                              int iLwSize)
{
    
    int i = get_global_id(0);
    
    if(i < iNumElements){
        //input is the global input, output is a local part of the global output
        REAL4 pos = vload4(0, &input_positions[iOffset*4 + i*4]);   
        REAL4 vel = vload4(0, &input_velocities[iOffset*4 + i*4]);
        
        REAL4 force = (REAL4)(0.0, 0.0, 0.0, 0.0);
        
        int j;
        for(j = 0; j < N; j++){
            
            REAL4 posJ = vload4(0, &input_positions[j*4]);
            REAL4 r = pos - posJ;

            REAL distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
            distSqr += softeningSquared;
#ifdef DP
            REAL invDist = (REAL)1.0 / (REAL)sqrt(distSqr);
#else
            REAL invDist = (REAL)1.0f / (REAL)sqrt(distSqr);
#endif
            REAL invDistCube =  invDist * invDist * invDist;
            REAL s = pos.w * invDistCube;
            
            force += r * s;
        }
        
        //REAL invMass = input_velocities[i*4+3];
        REAL invMass = vel.w;
        
        // acceleration = force / mass;
        // new velocity = old velocity + acceleration * deltaTime      
        vel.x += (force.x * invMass) * dT;
        vel.y += (force.y * invMass) * dT;
        vel.z += (force.z * invMass) * dT;
        
        vel.x *= damping;
        vel.y *= damping;
        vel.z *= damping;
        
        // new position = old position + velocity * deltaTime
        pos.x += vel.x * dT;
        pos.y += vel.y * dT;
        pos.z += vel.z * dT;
        
        vstore4(pos, 0, &output_positions[i*4]);
        vstore4(vel, 0, &output_velocities[i*4]);
    }
}