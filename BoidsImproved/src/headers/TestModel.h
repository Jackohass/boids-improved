#ifndef TEST_MODEL_CORNEL_BOX_H
#define TEST_MODEL_CORNEL_BOX_H

// Defines a simple test model: The Cornel Box

#include <glm.hpp>
#include <gtx/vector_angle.hpp>
#include <gtc/random.hpp>
#include <gtc/quaternion.hpp>
#include <vector>
#include <iostream>

using glm::vec3;
using glm::sphericalRand;
using glm::linearRand;



//Boid Simulation constants, precision of 8 for the GPU (just an arbitrary number)
constexpr float confinementRadius = 4.0f;

// WARNING: for GPU version, each radius must increase size. Ie avoidamce < cohesion < conformance
constexpr float avoidanceRadius = 0.1f;
constexpr float cohesionRadius = 0.15f;
constexpr float conformanceRadius = 0.25f;
constexpr float largestRadius = cohesionRadius > avoidanceRadius ? 
	(cohesionRadius > conformanceRadius ? cohesionRadius : conformanceRadius) : 
	(avoidanceRadius > conformanceRadius ? avoidanceRadius : conformanceRadius);

constexpr int dimension = 2 * confinementRadius / largestRadius;

constexpr int numBoids = 42222;

// Used to describe a triangular surface:
class Triangle
{
public:
	std::vector<glm::vec3> v;
	glm::vec3 normal;

	Triangle( glm::vec3 v0, glm::vec3 v1, glm::vec3 v2)
	{
		v.push_back(v0);
		v.push_back(v1);
		v.push_back(v2);
		ComputeNormal();
	}

	void ComputeNormal()
	{
		glm::vec3 e1 = v[1]-v[0];
		glm::vec3 e2 = v[2]-v[0];
		normal = glm::normalize( glm::cross( e2, e1 ) );
	}
};

struct Object {
	std::vector<Triangle> triangles;
	glm::vec3 colour;
	int offset = 0;
};

struct Boid {
	glm::vec3 pos;
	glm::vec3 rot;
	glm::vec3 vel;
	Object* mesh;

	const vec3 boidForward = vec3(0.5, 1, 1.0f / (2.0f * glm::sqrt(3.0f)))
		- vec3(1.0f / 2.0f, glm::sqrt(2.0f / 3.0f) / 2.0f, 1.0f / (2.0f * glm::sqrt(3.0f)));

	Boid(const glm::vec3& p, const glm::vec3& r, const glm::vec3& v, Object* obj) {
		pos = p;
		rot = r;
		vel = v;
		mesh = obj;
	}

	void rotate(glm::vec3 r) {
		rot += r;
	}

	void move(const glm::vec3& p) {
		pos += p;
	}

	glm::mat4 getModel() {
		//Calculate the angle the boid needs to rotate to point forward
		//glm::vec3 forward(0, 1, 0);
		
		vec3 ref = glm::normalize(glm::cross(boidForward, vel));
		float angle = glm::orientedAngle(glm::normalize(boidForward), glm::normalize(vel), ref);

		glm::mat4 model = glm::translate(glm::mat4(1.0f), pos);
		//model *= glm::eulerAngleYXZ(rot[1], rot[0], rot[2]);

		model = glm::rotate(model, angle, ref);
		return model;
	}
};

void LoadLevel(std::vector<Object>& objects, std::vector<Boid>& boids)
{
	using glm::vec3;

	// Defines colors:
	vec3 red(0.75f, 0.15f, 0.15f);
	vec3 yellow(0.75f, 0.75f, 0.15f);
	vec3 green(0.15f, 0.75f, 0.15f);
	vec3 cyan(0.15f, 0.75f, 0.75f);
	vec3 blue(0.15f, 0.15f, 0.75f);
	vec3 purple(0.75f, 0.15f, 0.75f);
	vec3 white(0.75f, 0.75f, 0.75f);

	objects.clear();
	objects.resize(1);

	boids.clear();
	boids.reserve(numBoids);

	// ---------------------------------------------------------------------------
	// Room
	float L = 40;			// Length of Cornell Box side.

	
	vec3 M(1.0f / 2.0f, glm::sqrt(2.0f / 3.0f) / 2.0f, 1.0f / (2.0f * glm::sqrt(3.0f)));
	vec3 A = vec3(0, 0, 0) - M;
	vec3 B = vec3(1, 0, 0) - M;
	vec3 C = vec3(0.5, 0, glm::sqrt(3.0f) / 2.0f) - M;
	//vec3 D = vec3(0.5, glm::sqrt(2.0f / 3.0f), 1.0f / (2.0f * glm::sqrt(3.0f))) - M;
	vec3 D = vec3(0.5, 1, 1.0f / (2.0f * glm::sqrt(3.0f))) - M;
	
	objects[0].colour = red;
	objects[0].offset = 0;
	objects[0].triangles.push_back(Triangle(A, B, C));
	objects[0].triangles.push_back(Triangle(A, C, D));
	objects[0].triangles.push_back(Triangle(B, C, D));
	objects[0].triangles.push_back(Triangle(A, B, D));

	// ----------------------------------------------
	// Scale to the volume [-1,1]^3

	for (size_t j = 0; j < objects.size(); ++j)
	{
		std::vector<Triangle>& triangles = objects[j].triangles;
		for (int i = 0; i < triangles.size(); i++)
		{

			triangles[i].v[0] *= 2 / L;
			triangles[i].v[1] *= 2 / L;
			triangles[i].v[2] *= 2 / L;

			triangles[i].ComputeNormal();
		}

	}

	for(int i = 0; i < numBoids; i++){
		vec3 pos = sphericalRand(linearRand(0.0f, confinementRadius));
		vec3 vel = sphericalRand(0.001f);

		boids.push_back(Boid(
			pos, 
			vec3(0, 0, 0), 
			vel,
			&objects[0])
		);

	}
}

#endif
