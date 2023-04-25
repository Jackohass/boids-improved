#include <iostream>
#include<sstream>
#include<fstream>
#include<string>
#include <glm/glm.hpp>
#include <SDL.h>
#include "SDLauxiliary.h"
#include <glm/gtx/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/euler_angles.hpp>
#include "TestModel.h"
#include "glad.h"

struct Pixel{
	int x;
	int y;
	float zinv;
	glm::vec3 pos3d; //Pixel Illumination
	//glm::vec3 illumination; //Vertex Illumination
};

struct Vertex{
	glm::vec3 position;
	//glm::vec3 normal; //Vertex Illumination
	//glm::vec3 reflectance; //Vertex Illumination
};

using namespace std;
using glm::vec3;
using glm::vec4;
using glm::ivec2;
using glm::mat3;
using glm::mat4;

// ----------------------------------------------------------------------------
// GLOBAL VARIABLES

const int SCREEN_WIDTH = 1000;
const int SCREEN_HEIGHT = 1000;
SDL_Surface* screen;
int t;
int dSimT;
vector<Object> objects;
vector<Boid> boids;
vec4 cameraPos(0, 0, -3.001, 1);
mat4 cameraMatrix(1);
mat4 projectionMatrix = glm::perspective(53.0f,
			(float) SCREEN_WIDTH / (float) SCREEN_HEIGHT,
			0.1f, 100.0f);
const float focalLength = SCREEN_WIDTH;
mat4 R(1);
float yaw = 0;
vec3 currentColor(1, 1, 1);
vec3 currentNormal; //Per Pixel Illumination
vec3 currentReflectance; //Per Pixel Illunination
float depthBuffer[SCREEN_HEIGHT][SCREEN_WIDTH];
vec3 lightPos = glm::rotateX(vec3(0, -0.5, -0.7), 180.0f);
vec3 lightPower = 14.1f * vec3(1, 1, 1);
vec3 indirectLightPowerPerArea = 0.5f * vec3(1, 1, 1);
unsigned int shader;
unsigned int VAO;

// ----------------------------------------------------------------------------
// FUNCTIONS

void ComputePolygonRows(const vector<Pixel>& vertexPixels, vector<Pixel>& leftPixels,
	vector<Pixel>& rightPixels);
void Update();
void Draw();
void Interpolate(ivec2 a, ivec2 b, vector<ivec2>& result);
void Interpolate(Pixel a, Pixel b, vector<Pixel>& result);
void VertexShader(const Vertex& v, Pixel& p);
void PixelShader(const Pixel& p);




void updateShaders(mat4 model, vec3 objectColor)
{
	glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, &model[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(shader, "camera"), 1, GL_FALSE, &cameraMatrix[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, &projectionMatrix[0][0]);

	glUniform3fv(glGetUniformLocation(shader, "objectColor"), 1, &objectColor[0]);
	glUniform3fv(glGetUniformLocation(shader, "lightPos"), 1, &lightPos[0]);
	glUniform3fv(glGetUniformLocation(shader, "lightPower"), 1, &lightPower[0]);
	glUniform3fv(glGetUniformLocation(shader, "indirectLightPowerPerArea"), 1, &indirectLightPowerPerArea[0]);
}

void checkShaderErrors(int checkShader) {
	int success;
	glGetShaderiv(checkShader, GL_COMPILE_STATUS, &success);
	if (success == GL_FALSE) {
		const int maxLength = 1024;
		//glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);
		GLchar errorLog[maxLength];
		glGetShaderInfoLog(shader, maxLength, NULL, errorLog);
		std::cout << "ERROR::SHADER_COMPILATION_ERROR: \n";
		for (int i = 0; i < maxLength; i++)
		{
			cout << errorLog[i];
		}
		std::cout << "\n -- --------------------------------------------------- -- " << std::endl;
	}
	else {
		std::cout << "Compilation was successful" << std::endl;
	}
}

void setupShaders()
{
	string vertexString;
	string pixelString;
	ifstream vertexFile;
	ifstream pixelFile;
	std::stringstream vShaderStream, fShaderStream;
	vertexFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	pixelFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

	try {
		vertexFile.open("shader/vertexShader.vert");
		pixelFile.open("shader/pixelShader.frag");
		vShaderStream << vertexFile.rdbuf();
		fShaderStream << pixelFile.rdbuf();
		vertexString = vShaderStream.str();
		pixelString = fShaderStream.str();
		vertexFile.close();
		pixelFile.close();
	} catch (std::ifstream::failure& e){
		std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ: " << e.what() << std::endl;
	}
	//std::cout << "vertex shader: " << vertexString << std::endl;
	//std::cout << "pixel shader: " << pixelString << std::endl;
	
	

	const char* codeVertex = vertexString.c_str();
	const char* codePixel = pixelString.c_str();
	

	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &codeVertex, NULL);
	glCompileShader(vertexShader);
	checkShaderErrors(vertexShader);

	unsigned int pixelShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(pixelShader, 1, &codePixel, NULL);
	glCompileShader(pixelShader);
	checkShaderErrors(pixelShader);

	shader = glCreateProgram();
	glAttachShader(shader, vertexShader);
	glAttachShader(shader, pixelShader);

	glLinkProgram(shader);
	glUseProgram(shader);
	glDeleteShader(vertexShader);
	glDeleteShader(pixelShader);
}

int main(int argc, char* argv[])
{
	LoadLevel(objects, boids);
	screen = InitializeSDL(SCREEN_WIDTH, SCREEN_HEIGHT);

	if (!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}
	//Add depth buffer testing
	glEnable(GL_DEPTH_TEST);

	setupShaders();

	//Prepare the triangle buffer
	const int numVerts = 30 /*HARDCODE BE CAREFUL, SPICY*/ * 3 * 3 * 2;
	float vertices[numVerts];
	int curObjOffset = 0;
	for (int objIndex = 0; objIndex < objects.size(); objIndex++)
	{
		vector<Triangle>& triangles = objects[objIndex].triangles;
		for (int i = 0; i < triangles.size(); i++) {
			int curTriOffset = (curObjOffset + i) * 6 * 3;

			for (int j = 0; j < 3; j++) {
				int curVertOffset = curTriOffset + j * 6;

				for (int k = 0; k < 3; k++) {
					vertices[curVertOffset + k] = triangles[i].v[j][k];
				}
				for (int k = 0; k < 3; k++) {
					vertices[curVertOffset + 3 + k] = triangles[i].normal[k];
				}
			}

			//cout << triangles[i].normal[0] << " " << triangles[i].normal[1] << " " << triangles[i].normal[2] << endl;
		}
		curObjOffset += triangles.size();
	}
	

	unsigned int VBO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindVertexArray(VAO);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// normal attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	
	t = SDL_GetTicks();	// Set start value for timer.

	/*vector<ivec2> vertexPixels(3);
	vertexPixels[0] = ivec2(10, 5);
	vertexPixels[1] = ivec2(5, 10);
	vertexPixels[2] = ivec2(15, 15);
	vector<ivec2> leftPixels;
	vector<ivec2> rightPixels;
	ComputePolygonRows(vertexPixels, leftPixels, rightPixels);
	for (int row = 0; row < leftPixels.size(); ++row)
	{
		cout << "Start: ("
			<< leftPixels[row].x << ","
			<< leftPixels[row].y << "). "
			<< "End: ("
			<< rightPixels[row].x << ","
			<< rightPixels[row].y << "). " << endl;
	}*/

	const int precision = 10;
	int avgNum[precision];
	int i = 0;
	while( NoQuitMessageSDL() )
	{
		Update();
		Draw();

		avgNum[i] = dSimT;
		i = (i + 1) % precision;
	}

	int sum = 0;
	for (int j = 0; j < precision; j++)
	{
		sum += avgNum[j];
	}

	printf("Average simulate time: %d", sum/precision);

	SDL_SaveBMP( screen, "screenshot.bmp" );
	return 0;
}

void handleInput(float dt){
	Uint8* keystate = SDL_GetKeyState(0);

	vec4 forward(R[2][0], R[2][1], R[2][2], 0);
	vec4 right(R[0][0], R[0][1], R[0][2], 0);
	vec4 down(R[1][0], R[1][1], R[1][2], 0);

	float speed = 0.001f * dt; 
	float rotSpeed = 0.05f * dt;

	if (keystate[SDLK_UP])
	{
		// Move camera forward
		cameraPos += speed * vec4(0, 0, 1, 0);
	}
	if (keystate[SDLK_DOWN])
	{
		// Move camera backward
		cameraPos -= speed * vec4(0, 0, 1, 0);;
	}
	if (keystate[SDLK_LEFT])
	{
		// Move camera to the left
		//cameraPos[0] -= 0.1;
		yaw += rotSpeed;
		float rad = glm::radians(yaw);
		R = mat4(glm::cos(rad), 0, glm::sin(rad), 0,
			0, 1, 0, 0,
			-glm::sin(rad), 0, glm::cos(rad), 0,
			0, 0, 0, 1);
	}
	if (keystate[SDLK_RIGHT])
	{
		// Move camera to the right
		//cameraPos[0] += 0.1;
		yaw -= rotSpeed;
		float rad = glm::radians(yaw);
		R = mat4(glm::cos(rad), 0, glm::sin(rad), 0,
			0, 1, 0, 0, 
			-glm::sin(rad), 0, glm::cos(rad), 0,
			0, 0, 0, 1);
	}
	//Move forwad, back, right, left, down and up respectively.
	vec3 f(forward);
	vec3 r(right);
	vec3 d(down);

	if (keystate[SDLK_w]) lightPos -= speed * f;
	if (keystate[SDLK_s]) lightPos += speed * f;
	if (keystate[SDLK_d]) lightPos += speed * r;
	if (keystate[SDLK_a]) lightPos -= speed * r;
	if (keystate[SDLK_q]) lightPos -= speed * d;
	if (keystate[SDLK_e]) lightPos += speed * d;

	if( keystate[SDLK_RSHIFT] )
		;

	if( keystate[SDLK_RCTRL] )
		;

	mat4 trans(vec4(1, 0, 0, 0), vec4(0, 1, 0, 0), vec4(0, 0, 1, 0), cameraPos);
	cameraMatrix = trans * R /* * glm::rotate(mat4(1), 180.0f, vec3(1, 0, 0))*/;
}

vec3 cohesion(Boid &current){
	/*
		Find the center of mass amongst nearby boids, with nearby beind defined
		as those boids within a sphere centered on this boid with a given radius.
		Weight the boids effect on the center of mass by their inverse distance²

		Lastly calculate the vector required to move to said center
	*/
	const float radius = 0.15f;
	const float strength = 0.1f;
	const float epsilon = 1.0f / 10.0f;
	const float epsInvSqr = 1.0f / glm::pow(epsilon, 2.0f);

	vec3 center(0, 0, 0);
	int numNear = 0;
	for(Boid& b : boids){
		if (&b == &current) continue;

		float d = glm::distance(current.pos, b.pos);
		if(d < radius){
			//TODO: This should probably be weighted by the inv square dist
			//float w = (d > epsilon) ? (1.0f / glm::pow(d, 2.0f)) / epsInvSqr : 1.0f;

			center += b.pos /* ((radius - d) / d)*/;
			numNear++;
		}
	}

	//Find the averaged center, or self if no neighbours
	center = (numNear > 0) ? center / (float) numNear - current.pos : center;

	return center * strength;
}

vec3 avoidance(Boid& current){
	const float radius = 0.1f;
	const float strength = 0.4f;

	vec3 res(0, 0, 0);

	for(Boid& b : boids){
		if (&b == &current) continue;

		float dist = glm::distance(current.pos, b.pos);

		if(dist < radius){
			res -= (b.pos - current.pos) * ((radius - dist) /  dist);
		}
	}

	return res * strength;
}

vec3 conformance(Boid& current){
	const float radius = 0.25f;
	const float strength = 0.2f;

	vec3 velocity(0, 0, 0);
	int numNear = 0;
	for(Boid& b : boids){
		if (&b == &current) continue;

		if(glm::distance(current.pos, b.pos) < radius){
			//TODO: This should probably be weighted by the inv square dist
			velocity += b.vel;
			numNear++;
		}
	}

	velocity = (numNear > 0) ? velocity / (float) numNear : current.vel;

	return (velocity - current.vel) * strength; 
}

vec3 confinement(Boid& current){
	const float strength = 0.1f;

	vec3 v(0, 0, 0);

	/*if(glm::length(current.pos) > confinementRadius){
		v = glm::normalize(-current.pos);
	}*/


	for(int i =  0; i < 3; i++){
		if(current.pos[i] < -confinementRadius) v[i] = 1;
		else if(current.pos[i] > confinementRadius) v[i] = -1;
	}

	if(v != vec3(0, 0, 0)) v = glm::normalize(v);
	
	return v * strength;
}

vec3 clamp(vec3& original, vec3& increment, const float normalizer, const float dt){
	const float speedLimitUpper = 1.0f * normalizer;
	const float speedLimitLower = 0.1f * normalizer;

	vec3 newBoidVel = original + increment * normalizer * dt;
	vec3 newVel = 0.5f * increment * dt * normalizer + newBoidVel;
	if(glm::length(newVel) > speedLimitUpper){
		newVel = glm::normalize(newVel) * speedLimitUpper;
	}
	else if (glm::length(newVel) < speedLimitLower) {
		newVel = glm::normalize(newVel) * speedLimitLower;
	}

	if (glm::length(newBoidVel) > speedLimitUpper) {
		newBoidVel = glm::normalize(newBoidVel) * speedLimitUpper;
	}
	else if (glm::length(newBoidVel) < speedLimitLower) {
		newBoidVel = glm::normalize(newBoidVel) * speedLimitLower;
	}

	original = newBoidVel;
	return newVel;
}

vec3 drag(Boid& current) {
	const float drag = 0.1;
	return -current.vel * drag;
}

void simulateBoid(float dt){
	/*
		For each boid, calulate the vector for each of the three rules
			Simulate a fourth bounds rule to keep the boids in line
		Then sum these vectors (without prioritising them, which differs from the paper)
		Add this sum to the velocity of the boid
		Update the position by the given velocity	

		Sources: https://vergenet.net/~conrad/boids/pseudocode.html, https://dl.acm.org/doi/10.1145/37402.37406 
	*/

	const float normalizer = 1.0f/1000.f;

	int i = 0;
	for(Boid& b : boids){
		vec3 a = cohesion(b);
		a += avoidance(b); 
		a += conformance(b);
		a += confinement(b);
		a += drag(b);

		//b.vel += v;
		//cout << i << ": " << "(" << b.vel[0] << ", " << b.vel[1] << ", " << b.vel[2] << ")" << endl;
		b.move(clamp(b.vel, a, normalizer, dt) * dt);
		//v = v0 + a * dt
		//p = 0.5 * dt * dt * a + dt * v = 0.5 * dt * dt + dt * v0 + a * dt * dt = 1.5*a*dt^2 + v0*dt
	}
}

void Update()
{
	// Compute frame time:
	int t2 = SDL_GetTicks();
	float dt = float(t2-t);
	t = t2;
	// cout << "Render time: " << dt << " ms." << endl;

	handleInput(dt);

	int simT = SDL_GetTicks();
	simulateBoid(dt);
	int simT2 = SDL_GetTicks();
	dSimT = simT2 - simT;

	cout << "Prev Tot time: " << dt << " ms." << " Simulation time: " << dSimT << " ms." << endl;
}

void Draw(){

	glClearColor(0.0f, 0.0f, 0.0f, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//Update our shaders based on the camera positions and stuff
	glUseProgram(shader); //TODO: Do once

	//Render
	glBindVertexArray(VAO);
	/*int currObjOffset = 0;
	for (int i = 0; i < objects.size(); i++)
	{
		int numtri = objects[i].triangles.size() * 3;
		updateShaders(mat4(1) * glm::rotate(mat4(1), 180.0f, vec3(1, 0, 0)), objects[i].colour); //TODO: By copy or by reference?
		glDrawArrays(GL_TRIANGLES, currObjOffset, numtri);
		currObjOffset += numtri;
	}*/
	for (int i = 0; i < boids.size(); i++)
	{
		int numtri = boids[i].mesh->triangles.size() * 3;
		updateShaders(boids[i].getModel(), boids[i].mesh->colour); //TODO: By copy or by reference?
		glDrawArrays(GL_TRIANGLES, boids[i].mesh->offset, numtri);
	}
	

	SDL_GL_SwapBuffers();
}
