/*CONSTANTS
* float: CONFINERAD = confinementRadius
* float: COHESIONRAD = cohesionRadius
* float: AVOIDRAD = avoidanceRadius
* float: CONFORMRAD = conformanceRadius
* int: DIM = dimension
*/

//#pragma OPENCL EXTENSION cl_amd_printf : enable

kernel void neighbourExclusiveScan(
	__global int* spatialCellsOffset,
	__global int* spatialCellsCount
) {
	int i = get_global_id(0);
	//spatialCellsOffset[i] = work_group_scan_exclusive_add(spatialCellsCount[i]);
}