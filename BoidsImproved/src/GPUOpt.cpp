#include<iostream>
#include<sstream>
#include<fstream>
#include<string>
#include<glm.hpp>
#include<omp.h>
#include"SDL.h"
#include"headers/SDLauxiliary.h"
#include<gtx/constants.hpp>
#include<gtc/matrix_transform.hpp>
#include<gtx/rotate_vector.hpp>
#include<gtx/euler_angles.hpp>
#include"headers/TestModel.h"
#include"headers/glad.h"

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

struct Pos
{
	float x, y, z, w; // Positions. Variable w is not used, but is set for byte offset reasons.
};
struct Vel
{
	float vx, vy, vz, vw; // Velocities. Variable vw is not used, but is set for byte offset reasons.
};
struct CellIndex
{
	int start, end;
};


using namespace std;
using glm::vec3;
using glm::vec4;
using glm::ivec2;
using glm::ivec3;
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
vec3 lightPos = glm::rotateX(vec3(0, -4.0, -0.7), 180.0f);
vec3 lightPower = 280.1f * vec3(1, 1, 1);
vec3 indirectLightPowerPerArea = 0.5f * vec3(1, 1, 1);

unsigned int shader;
GLuint getBoidHashProgram;

unsigned int VAO;
GLuint posBuffer;
GLuint velBuffer;
GLuint hashBuffer;
GLuint idsBuffer;

omp_lock_t writelock[dimension * dimension * dimension];

//Spatial Partitioning
vector<Boid *> spatialCells[dimension * dimension * dimension];
vector<vector<Boid *> *> neighbours[dimension * dimension * dimension];

//GPU optimization necessary variables
Pos* boidPos;
Vel* boidVel;
int boidHashes[numBoids];
CellIndex cellIndexes[numBoids];


// ----------------------------------------------------------------------------
// FUNCTIONS
void Update();
void Draw();
void calculateCellNeighbours();

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
	string getBoidHashString;
	ifstream vertexFile;
	ifstream pixelFile;
	ifstream getBoidHashFile;
	std::stringstream vShaderStream, fShaderStream, getBoidHashStream;
	vertexFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	pixelFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	getBoidHashFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

	try {
		vertexFile.open("src/shaders/vertexShader.vert");
		pixelFile.open("src/shaders/pixelShader.frag");
		getBoidHashFile.open("src/shaders/getBoidHash.comp");
		vShaderStream << vertexFile.rdbuf();
		fShaderStream << pixelFile.rdbuf();
		getBoidHashStream << getBoidHashFile.rdbuf();
		vertexString = vShaderStream.str();
		pixelString = fShaderStream.str();
		getBoidHashString = getBoidHashStream.str();
		vertexFile.close();
		pixelFile.close();
		getBoidHashFile.close();
	} catch (std::ifstream::failure& e){
		std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ: " << e.what() << std::endl;
	}
	std::cout << "vertex shader: " << vertexString << std::endl;
	std::cout << "pixel shader: " << pixelString << std::endl;
	std::cout << "getBoidHash shader: " << getBoidHashString << std::endl;

	const char* codeVertex = vertexString.c_str();
	const char* codePixel = pixelString.c_str();
	const char* codeGetBoidHash = getBoidHashString.c_str();

	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &codeVertex, NULL);
	glCompileShader(vertexShader);
	checkShaderErrors(vertexShader);

	unsigned int pixelShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(pixelShader, 1, &codePixel, NULL);
	glCompileShader(pixelShader);
	checkShaderErrors(pixelShader);

	unsigned int getBoidHashShader = glCreateShader(GL_COMPUTE_SHADER);
	glShaderSource(getBoidHashShader, 1, &codeGetBoidHash, NULL);
	glCompileShader(getBoidHashShader);
	checkShaderErrors(getBoidHashShader);

	//Create render program
	shader = glCreateProgram();
	glAttachShader(shader, vertexShader);
	glAttachShader(shader, pixelShader);
	glLinkProgram(shader);
	glUseProgram(shader);
	glDeleteShader(vertexShader);
	glDeleteShader(pixelShader);

	//Create getBoidHash program
	getBoidHashProgram = glCreateProgram();
	glAttachShader(getBoidHashProgram, getBoidHashShader);
	glLinkProgram(getBoidHashProgram);
	glDeleteShader(getBoidHashProgram);
}

int main(int argc, char* argv[])
{
	for (int i = 0; i < dimension * dimension * dimension; i++)
	{
		omp_init_lock(&writelock[i]);
	}
	

	LoadLevel(objects, boids);

	screen = InitializeSDL(SCREEN_WIDTH, SCREEN_HEIGHT);

	if (!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}
	//Add depth buffer testing
	glEnable(GL_DEPTH_TEST);

	const GLubyte* renderer = glGetString(GL_RENDERER);
	const GLubyte* vendor = glGetString(GL_VENDOR);
	const GLubyte* version = glGetString(GL_VERSION);
	const GLubyte* glslVersion =
		glGetString(GL_SHADING_LANGUAGE_VERSION);

	GLint major, minor;
	glGetIntegerv(GL_MAJOR_VERSION, &major);
	glGetIntegerv(GL_MINOR_VERSION, &minor);

	printf("GL Vendor            : %s\n", vendor);
	printf("GL Renderer          : %s\n", renderer);
	printf("GL Version (string)  : %s\n", version);
	printf("GL Version (integer) : %d.%d\n", major, minor);
	printf("GLSL Version         : %s\n", glslVersion);

	setupShaders();

	//Prepare the triangle buffer
	const int numVerts = 30 /*HARDCODED, BE CAREFUL*/ * 3 * 3 * 2;
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

	//Setup Shader Storage Buffer Objects
	GLint bufMask = GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT; // the invalidate makes a big difference when re-writing

	glGenBuffers(1, &posBuffer);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, posBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numBoids * sizeof(struct Pos), NULL, GL_STATIC_DRAW);
	boidPos = (struct Pos*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, numBoids * sizeof(struct Pos), bufMask);
	for (int i = 0; i < numBoids; i++)
	{
		boidPos[i].x = boids[i].pos.x;
		boidPos[i].y = boids[i].pos.y;
		boidPos[i].z = boids[i].pos.z;
		boidPos[i].w = 1.;
	}
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

	glGenBuffers(1, &velBuffer);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, velBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numBoids * sizeof(struct Vel), NULL, GL_STATIC_DRAW);
	boidVel = (struct Vel*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, numBoids * sizeof(struct Vel), bufMask);
	for (int i = 0; i < numBoids; i++)
	{
		boidVel[i].vx = boids[i].vel.x;
		boidVel[i].vy = boids[i].vel.y;
		boidVel[i].vz = boids[i].vel.z;
		boidVel[i].vw = 0.;
	}
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

	glGenBuffers(1, &hashBuffer);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, hashBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(boidHashes), NULL, GL_DYNAMIC_COPY);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	glGenBuffers(1, &idsBuffer);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, idsBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(cellIndexes), NULL, GL_DYNAMIC_COPY);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	// --------
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, posBuffer);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, hashBuffer);

	glUseProgram(getBoidHashProgram);
	glUniform1i(glGetUniformLocation(getBoidHashProgram, "dimension"), dimension);
	glUniform1f(glGetUniformLocation(getBoidHashProgram, "confinementRadius"), confinementRadius);
	glDispatchCompute(numBoids / 10, 1, 1);
	glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

	printf("Hashes:\n");
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, hashBuffer);
	GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
	memcpy(boidHashes, p, sizeof(boidHashes));
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
	for (int& hash : boidHashes) {
		printf("%d, ", hash);
	}
	printf("\nEnd of hashes.\n\n");

	//Initialize the spaital data structures
	calculateCellNeighbours();

	t = SDL_GetTicks();	// Set start value for timer.

	const int precision = 10;
	int avgNum[precision];
	int i = 0;
	while (NoQuitMessageSDL())
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

	printf("Average simulate time: %d", sum / precision);

	SDL_SaveBMP( screen, "screenshot.bmp" );
	return 0;
}

// Returns the index of the cell in which the boid is for each dimension
ivec3 getCellPos(const vec3& pos){
	//Calculate how much the coordinates should be scaled to transformed into cells
	const float scale = dimension / (confinementRadius * 2);
	//Calculate the per axis cell indexes
	ivec3 offPos = (ivec3) ((pos + vec3(confinementRadius)) * scale);

	//Check if any pos was the maximum of said dimension or outside, and then
	//treat it as being in the last cell of that dimension
	for(int i = 0; i < 3; i++){
		if(offPos[i] >= dimension) offPos[i] = dimension - 1;
		else if(offPos[i] < 0) offPos[i] = 0;
	}

	return offPos;
}

// Returns the index in spatialCells for which the boid is in
int spatialCellsIndex(const vec3& pos){
	ivec3 offPos = getCellPos(pos);

	return offPos.x + offPos.y * dimension + offPos.z * dimension * dimension;
}

void adjacentCells(int index, ivec3 cellPos, vector<vector<Boid *> *>& neigh){
	const int zOffset = dimension * dimension;

	for(int z = -1; z <= 1; z++){
		if(cellPos.z + z > dimension - 1) continue;
		else if(cellPos.z + z < 0) continue;

		for(int y = -1; y <= 1; y++){
			if(cellPos.y + y > dimension - 1) continue;
			else if(cellPos.y + y < 0) continue;

			for(int x = -1; x <= 1; x++){
				if(cellPos.x + x > dimension - 1) continue;
				else if(cellPos.x + x < 0) continue;

				//The cell is valid, get its index
				int neighIndex = index;
				neighIndex += z * zOffset + y * dimension + x;

				//Add the list of the cell to neighbours worthy of looking at
				neigh.push_back(&spatialCells[neighIndex]);
			}
		}
	}
}

// Returns a list of lists containing the neighbours of the boid
void getNeighbours(const vec3& pos, vector<vector<Boid *> *>& neigh){
	int index = spatialCellsIndex(pos);
	ivec3 cellPos = getCellPos(pos);

	adjacentCells(index, cellPos, neigh);
}

void calculateCellNeighbours(){
	ivec3 cellPos(0, 0, 0);
	for(int i = 0; i < dimension * dimension * dimension; i++){
		cellPos.x = i % dimension;
		cellPos.y = (i / dimension) % dimension;
		cellPos.z = i / (dimension * dimension);

		adjacentCells(i, cellPos, neighbours[i]);
	}
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

	mat4 trans(vec4(1, 0, 0, 0), vec4(0, 1, 0, 0), vec4(0, 0, 1, 0), cameraPos);
	cameraMatrix = trans * R /* * glm::rotate(mat4(1), 180.0f, vec3(1, 0, 0))*/;
}



void simulateBoid(float dt){
	//Clear the spatial partition and re-insert the Boids at the correct space
	/*#pragma omp parallel for
	for(int i = 0; i < dimension * dimension * dimension; i++){
		spatialCells[i].clear();
	}*/
	for (int i = 0; i < dimension * dimension * dimension; i++) {
		spatialCells[i].clear();
	}
	#pragma omp parallel for
	for (int i = 0; i < boids.size(); i++) {
		Boid& b = boids[i];
		//printf("(%f, %f, %f): %d\n", b.pos.x, b.pos.y, b.pos.z, spatialCellsIndex(b.pos));
		int index = spatialCellsIndex(b.pos);
		omp_set_lock(&writelock[index]);
		spatialCells[index].push_back(&b);
		omp_unset_lock(&writelock[index]);
	}
	/*for (Boid& b : boids) {
		//printf("(%f, %f, %f): %d\n", b.pos.x, b.pos.y, b.pos.z, spatialCellsIndex(b.pos));
		spatialCells[spatialCellsIndex(b.pos)].push_back(&b);
	}*/



	/*
		For each boid, calulate the vector for each of the three rules
			Simulate a fourth bounds rule to keep the boids in line
		Then sum these vectors (without prioritising them, which differs from the paper)
		Add this sum to the velocity of the boid
		Update the position by the given velocity	

		Sources: https://vergenet.net/~conrad/boids/pseudocode.html, https://dl.acm.org/doi/10.1145/37402.37406 
	*/
}

void Update()
{
	// Compute frame time:
	int t2 = SDL_GetTicks();
	float dt = float(t2-t);
	t = t2;
	//cout << "Render time: " << dt << " ms." << endl;

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
