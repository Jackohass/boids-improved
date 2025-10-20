/*CONSTANTS
* float: CONFINERAD = confinementRadius
* float: COHESIONRAD = cohesionRadius
* float: AVOIDRAD = avoidanceRadius
* float: CONFORMRAD = conformanceRadius
* int: DIM = dimension
*/

//#pragma OPENCL EXTENSION cl_amd_printf : enable

#pragma __opencl_c_work_group_collective_functions

typedef struct Boid
{
	float3 pos;
	float3 vel;
} Boid;

__constant const float normalizer = 1.0f / 1000.f;

//Avoidance
__constant const float avoidanceStrength = 0.4f;

//Cohesion
__constant const float coheshionStrength = 0.1f;
/*const float coheshionEpsilon = 1.0f / 10.0f;
const float coheshionEpsInvSqr = 1.0f / glm::pow(coheshionEpsilon, 2.0f);*/

//Conformance
__constant const float conformanceStrength = 0.2f;

__constant const float3 boidForward = (float3)(0.0f, 0.591751695f, 0.0f);

float3 confinement(float3 currPos) {
	const float confinementStrength = 0.1f;
	float3 confinementV = (float3)(0.0f);
	const float radius = CONFINERAD;

	// confinementV.x = (currPos.x < radius*(-1.0f)) * 1.0f; 
	// confinementV.x = (currPos.x > radius) * -1.0f;

	// confinementV.y = (currPos.y < radius*(-1.0f)) * 1.0f; 
	// confinementV.y = (currPos.y > radius) * -1.0f; 

	// confinementV.z = (currPos.z < radius*(-1.0f)) * 1.0f; 
	// confinementV.z = (currPos.z > radius) * -1.0f; 

	if(currPos.x < -radius) confinementV.x = 1.0f;
	else if(currPos.x > radius) confinementV.x = -1.0f;

	if(currPos.y < -radius) confinementV.y = 1.0f;
	else if(currPos.y > radius) confinementV.y = -1.0f;

	if(currPos.z < -radius) confinementV.z = 1.0f;
	else if(currPos.z > radius) confinementV.z = -1.0f;

	if(confinementV.x != 0.0f || confinementV.y != 0.0f || confinementV.z != 0.0f) return normalize(confinementV) * confinementStrength;
	else return 0;
}

float3 drag(float3 currVel) {
	const float drag = 0.1f;
	return -currVel * drag;
}

float3 clamp(float3* original, float3 increment, const float dt) {
	const float speedLimitUpper = 1.0f * normalizer;
	const float speedLimitLower = 0.1f * normalizer;

	float3 newBoidVel = *original + increment * normalizer * dt;
	float3 newVel = 0.5f * increment * dt * normalizer + newBoidVel;

	float newVelLen = length(newVel);
	float newBoidVelLen = length(newBoidVel);

	newVel = normalize(newVel) * clamp(length(newVel), speedLimitLower, speedLimitUpper);
	newBoidVel = normalize(newBoidVel) * clamp(length(newBoidVel), speedLimitLower, speedLimitUpper);

	*original = newBoidVel;
	return newVel;
}

float orientedAngle(float3 x, float3 y, float3 ref) {
	float Angle = (float)(acos(clamp(dot(x, y), (float)(-1), (float)(1))));
	return mix(Angle, -Angle, dot(ref, cross(x, y)) < (float)(0));
}

float16 rotate(float16* m, float angle, float3* v) {
	float a = angle;
	float c = cos(a);
	float s = sin(a);

	float3 axis = normalize(*v);
	float3 temp = ((float)(1) - c) * axis;

	float16 Rotate;
	Rotate.s0 = c + temp.x * axis.x;
	Rotate.s1 = temp.x * axis.y + s * axis.z;
	Rotate.s3 = temp.x * axis.z - s * axis.y;

	Rotate.s4 = temp.y * axis.x - s * axis.z;
	Rotate.s5 = c + temp.y * axis.y;
	Rotate.s6 = temp.y * axis.z + s * axis.x;

	Rotate.s7 = temp.z * axis.x + s * axis.y;
	Rotate.s8 = temp.z * axis.y - s * axis.x;
	Rotate.s9 = c + temp.z * axis.z;

	float16 Result;
	Result.s0123 = (*m).s0123 * Rotate.s0 + (*m).s4567 * Rotate.s1 + (*m).s89ab * Rotate.s3;
	Result.s4567 = (*m).s0123 * Rotate.s4 + (*m).s4567 * Rotate.s5 + (*m).s89ab * Rotate.s6;
	Result.s89ab = (*m).s0123 * Rotate.s7 + (*m).s4567 * Rotate.s8 + (*m).s89ab * Rotate.s9;
	Result.scdef = (*m).scdef;
	return Result;
}

int3 getCellPos(const float3* pos){
	//Calculate how much the coordinates should be scaled to transformed into cells
	const float scale = DIM / (CONFINERAD * 2);

	//Calculate the per axis cell indexes
	float3 tempPos = (((*pos) + (float3)(CONFINERAD)) * (scale));
	int3 offPos = (int3)((int)tempPos.x, (int)tempPos.y, (int)tempPos.z);

	//Check if any pos was the maximum of said dimension or outside, and then
	//treat it as being in the last cell of that dimension
	offPos.x = clamp(offPos.x, 0, DIM - 1);
	offPos.y = clamp(offPos.y, 0, DIM - 1);
	offPos.z = clamp(offPos.z, 0, DIM - 1);

	return offPos;
}

kernel void simulateBoid(
	__global Boid* boids,
	__global int* spatialCells,
	__global int* spatialCellsOffset,
	__global float16* boidModels,
	const float dt,
	const int maxNumNeighs
) {
	int i = get_global_id(0);
	Boid b = boids[i];

	float3 currPos = b.pos;
	float3 currVel = b.vel;

	float3 a = (float3)(0.0f);

	//Avoidance
	float3 avoidanceRes = (float3)(0.0f);;

	//Cohesion
	float3 coheshionCenter = (float3)(0.0f);
	int coheshionNumNear = 0;

	//Conformance
	float3 conformanceVelocity = (float3)(0.0f);
	int conformanceNumNear = 0;

	int3 cellPos = getCellPos(&currPos);
	const float scale = DIM / (CONFINERAD * 2);
	float3 tempPos = ((currPos + (float3)(CONFINERAD)) * scale);
	int3 offPos = (int3)((int)tempPos.x, (int)tempPos.y, (int)tempPos.z);
	int currCellIndex = offPos.x + offPos.y * DIM + offPos.z * DIM * DIM;

	for (int z = -1; z <= 1; z++) {
		for (int y = -1; y <= 1; y++) {
			for (int x = -1; x <= 1; x++) {

				if ((cellPos.z + z > DIM - 1) ||
					(cellPos.z + z < 0) ||
					(cellPos.y + y > DIM - 1) ||
					(cellPos.y + y < 0) ||
					(cellPos.x + x > DIM - 1) ||
					(cellPos.x + x < 0)
				) continue;

				//The cell is valid, get its index
				
				int cellIndex = currCellIndex + z * DIM * DIM + y * DIM + x;

				int offsetLo = spatialCellsOffset[cellIndex];
				int offsetHi = spatialCellsOffset[
					(cellIndex + 1 < DIM * DIM * DIM ? cellIndex + 1 : maxNumNeighs)
					];

				for (int neighIndex = offsetLo; neighIndex < offsetHi; neighIndex++) {
					Boid neighbour = boids[spatialCells[neighIndex]];
					float3 neighPos = neighbour.pos;
					float3 neighVel = neighbour.vel;

					float d = distance(currPos, neighPos);
					if (d < AVOIDRAD) {
						//Avoidance
						avoidanceRes -= (neighPos - currPos) * ((AVOIDRAD - d) / d);

						//Cohesion
						coheshionCenter += neighPos;
						coheshionNumNear++;

						//Conformance
						conformanceVelocity += neighVel;
						conformanceNumNear++;
					}
					else if (d < COHESIONRAD) {
						//Cohesion
						coheshionCenter += neighPos;
						coheshionNumNear++;

						//Conformance
						conformanceVelocity += neighVel;
						conformanceNumNear++;
					}
					else if (d < CONFORMRAD) {
						//Conformance
						conformanceVelocity += neighVel;
						conformanceNumNear++;
					}
				}
			}
		}
	}

	/*for (int neighIndex = 0; neighIndex < 22222; neighIndex++) {
		Boid neighbour = boids[neighIndex];
		float3 neighPos = neighbour.pos;
		float3 neighVel = neighbour.vel;

		float d = distance(currPos, neighPos);
		if (d < AVOIDRAD) {
			//Avoidance
			avoidanceRes -= (neighPos - currPos) * ((AVOIDRAD - d) / d);

			//Cohesion
			coheshionCenter += neighPos;
			coheshionNumNear++;

			//Conformance
			conformanceVelocity += neighVel;
			conformanceNumNear++;
		}
		else if (d < COHESIONRAD) {
			//Cohesion
			coheshionCenter += neighPos;
			coheshionNumNear++;

			//Conformance
			conformanceVelocity += neighVel;
			conformanceNumNear++;
		}
		else if (d < CONFORMRAD) {
			//Conformance
			conformanceVelocity += neighVel;
			conformanceNumNear++;
		}
	}*/

	//Avoidance
	a += avoidanceRes * avoidanceStrength;

	//Cohesion
	coheshionCenter = (coheshionNumNear > 0) ? coheshionCenter / (float)coheshionNumNear - currPos : coheshionCenter;
	a += coheshionCenter * coheshionStrength;

	//Conformance
	conformanceVelocity = (conformanceNumNear > 0) ? conformanceVelocity / (float)conformanceNumNear : currVel;
	a += (conformanceVelocity - currVel) * conformanceStrength;

	//Confinement
	a += confinement(currPos);

	//Drag
	a += drag(currVel);

	//Clamp
	currPos += clamp(&currVel, a, dt) * dt;

	boids[i].pos = currPos;
	boids[i].vel = currVel;
	
	float3 ref = normalize(cross(boidForward, currVel));
	
	float angle = orientedAngle(normalize(boidForward), normalize(currVel), ref);

	float16 model = (float16)(
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		currPos.x, currPos.y, currPos.z, 1.0f
	);

	boidModels[i] = rotate(&model, angle, &ref);
	// printf(
	// "%f, %f, %f, %f\n"
	// "%f, %f, %f, %f\n"
	// "%f, %f, %f, %f\n"
	// "%f, %f, %f, %f\n\n\n",
	// boidModels[i].s0, boidModels[i].s1, boidModels[i].s2, boidModels[i].s3, 
	// boidModels[i].s4, boidModels[i].s5, boidModels[i].s6, boidModels[i].s7, 
	// boidModels[i].s8, boidModels[i].s9, boidModels[i].sa, boidModels[i].sb,
	// boidModels[i].sc, boidModels[i].sd, boidModels[i].se , boidModels[i].sf
	// );
}