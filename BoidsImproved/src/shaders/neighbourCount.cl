/*CONSTANTS
* float: CONFINERAD = confinementRadius
* float: COHESIONRAD = cohesionRadius
* float: AVOIDRAD = avoidanceRadius
* float: CONFORMRAD = conformanceRadius
* int: DIM = dimension
*/

//#pragma OPENCL EXTENSION cl_amd_printf : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

int spatialCellsIndex(const float3* pos){
	int3 offPos = getCellPos(pos);

	return offPos.x + offPos.y * DIM + offPos.z * DIM * DIM;
}

kernel void neighbourCount(
	__global Boid* boids,
	__global int* spatialCellsCount
) {
	int i = get_global_id(0);
	Boid b = boids[i];
	int index = spatialCellsIndex(&b.pos);
	atomic_inc(spatialCellsCount + index);
}