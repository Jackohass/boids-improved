/*CONSTANTS
* float: CONFINERAD = confinementRadius
* float: COHESIONRAD = cohesionRadius
* float: AVOIDRAD = avoidanceRadius
* float: CONFORMRAD = conformanceRadius
* int: DIM = dimension
*/

//#pragma OPENCL EXTENSION cl_amd_printf : enable
//#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

kernel void neighbourInsert(
	__global Boid* boids,
	__global int* spatialCells,
	__global int* spatialCellsOffset,
	__global int* spatialCellsCount
) {
	int i = get_global_id(0);
	Boid b = boids[i];
	int index = spatialCellsIndex(&b.pos);
	int neighPos = atomic_dec(spatialCellsCount + index);
	spatialCells[spatialCellsOffset[index] + neighPos - 1] = i;
}