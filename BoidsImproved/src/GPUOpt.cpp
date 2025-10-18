#include<iostream>
#include<sstream>
#include<fstream>
#include<string>
#include<glm.hpp>
#include<algorithm>
#include<numeric>
#include<omp.h>
#include"SDL.h"
#include"headers/SDLauxiliary.h"
#include<gtx/constants.hpp>
#include<gtc/matrix_transform.hpp>
#include<gtx/rotate_vector.hpp>
#include<gtx/euler_angles.hpp>
#include"headers/TestModel.h"
//#include"headers/glad.h"

#include"../include/glad/wgl.h"

#include<CL/opencl.hpp>
#include<CL/cl_gl.h>
#include<CL/cl_gl_ext.h>

#include"../clError.h";


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
const vec3 boidForward = vec3(0.5, 1, 1.0f / (2.0f * glm::sqrt(3.0f))) 
	- vec3(1.0f / 2.0f, glm::sqrt(2.0f / 3.0f) / 2.0f, 1.0f / (2.0f * glm::sqrt(3.0f)));

unsigned int shader;
GLuint getBoidHashProgram;

unsigned int VAO;
GLuint posBuffer;
GLuint velBuffer;
GLuint hashBuffer;
GLuint idsBuffer;

omp_lock_t writelock[dimension * dimension * dimension];

unsigned int modelBuffer;
glm::mat4 boidModels[numBoids];

size_t GlobalWorkSize[3] = { numBoids, 1, 1 };
size_t LocalWorkSize[3] = { 1024, 1, 1 };
cl_device_id clDeviceId;
cl_kernel clKernel;
cl_platform_id clPlatformId;
cl_program clProgram;
cl_context clContext;
cl_command_queue clQueue;
cl_kernel computeBoid;

int clNeighsNumCap = numBoids;

cl_mem clBoids;
cl_mem clNeighs;
cl_mem clNeighOffset;
cl_mem clNeighSize;
cl_mem clBoidModels;

cl_int clErr = 0;

struct GPUBoid {
	cl_float3 pos;
	cl_float3 vel;
};

struct NeighbourWatchlistEntry {
	int index;
	float dist;
};

GPUBoid boids[numBoids];

// Defines colors:
vec3 red(0.75f, 0.15f, 0.15f);

//Spatial Partitioning
//vector<GPUBoid*> spatialCells[dimension * dimension * dimension];
//vector<vector<GPUBoid*>*> neighbours[dimension * dimension * dimension];
vector<int> spatialCells[dimension * dimension * dimension];
vector<vector<int>*> neighbours[dimension * dimension * dimension];
vector<int> neighboursVGPU[dimension * dimension * dimension];

vector<NeighbourWatchlistEntry> neighs;
vector<int> neighOffset(numBoids);
vector<int> neighSize(numBoids);

// ----------------------------------------------------------------------------
// FUNCTIONS
void Update();
void Draw();
void calculateCellNeighbours();

bool IsCLExtensionSupported(const char* extension) {
	// see if the extension is bogus:
	if (extension == NULL || extension[0] == '\0')
		return false;
	char* where = (char*)strchr(extension, ' ');
	if (where != NULL)
		return false;
	// get the full list of extensions:
	size_t extensionSize;
	clGetDeviceInfo(clDeviceId, CL_DEVICE_EXTENSIONS, 0, NULL, &extensionSize);
	char* extensions = new char[extensionSize];
	clGetDeviceInfo(clDeviceId, CL_DEVICE_EXTENSIONS, extensionSize, extensions, NULL);
	for (char* start = extensions; ; )
	{
		where = (char*)strstr((const char*)start, extension);
		if (where == 0)
		{
			delete[] extensions;
			return false;
		}
		char* terminator = where + strlen(extension); // points to what should be the separator
		if (*terminator == ' ' || *terminator == '\0' || *terminator == '\r' || *terminator == '\n')
		{
			delete[] extensions;
			return true;
		}
		start = terminator;
	}
}

void initKernels() {
	const unsigned int numKernels = 1;

	string kernelStrings[numKernels];
	ifstream kernelFiles[numKernels];
	std::stringstream kernelStreams[numKernels];

	for (int i = 0; i < numKernels; i++)
	{
		kernelFiles[i].exceptions(std::ifstream::failbit | std::ifstream::badbit);
	}


	try {
		kernelFiles[0].open("src/shaders/computeBoid.cl");

		for (int i = 0; i < numKernels; i++)
		{
			kernelStreams[i] << kernelFiles[i].rdbuf();
			kernelStrings[i] = kernelStreams[i].str();
			kernelFiles[i].close();
		}

	}
	catch (std::ifstream::failure& e) {
		std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ: " << e.what() << std::endl;
	}

	const char* codes[numKernels];

	for (int i = 0; i < numKernels; i++)
	{
		std::cout << "Code " << i << ": " << kernelStrings[i] << std::endl;
		codes[i] = kernelStrings[i].c_str();
	}

	clProgram = clCreateProgramWithSource(clContext, numKernels, codes, NULL, &clErr);
	clError(__LINE__, clErr);

	// Build options
	char options[2048] = { 0 };
	snprintf(options, 2048, "-D CONFINERAD=%.8ff -D COHESIONRAD=%.8ff -D AVOIDRAD=%.8ff -D CONFORMRAD=%.8ff -D DIM=%d -cl-single-precision-constant -cl-fast-relaxed-math",
		confinementRadius,
		cohesionRadius,
		avoidanceRadius,
		conformanceRadius,
		dimension);
	clErr = clBuildProgram(clProgram, 1, &clDeviceId, options, NULL, NULL);
	if (clErr)
	{
		char buf[4096] = { 0 };
		clGetProgramBuildInfo(clProgram, clDeviceId, CL_PROGRAM_BUILD_LOG, 4096, buf, NULL);
		printf("Error compiling the kernel %i: - %s\n", clErr, buf);
	}

	computeBoid = clCreateKernel(clProgram, "simulateBoid", &clErr);
	clError(__LINE__, clErr);

	clBoids = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(GPUBoid) * numBoids, NULL, &clErr);
	clError(__LINE__, clErr);

	clNeighs = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(NeighbourWatchlistEntry) * numBoids, NULL, &clErr);
	clError(__LINE__, clErr);

	clNeighOffset = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(int) * numBoids, NULL, &clErr);
	clError(__LINE__, clErr);

	clNeighSize = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(int) * numBoids, NULL, &clErr);
	clError(__LINE__, clErr);

	clBoidModels = clCreateFromGLBuffer(clContext, CL_MEM_WRITE_ONLY, modelBuffer, &clErr);
	clError(__LINE__, clErr);

	clErr = clSetKernelArg(computeBoid, 0, sizeof(cl_mem), &clBoids);
	clError(__LINE__, clErr);

	clErr = clSetKernelArg(computeBoid, 1, sizeof(cl_mem), &clNeighs);
	clError(__LINE__, clErr);

	clErr = clSetKernelArg(computeBoid, 2, sizeof(cl_mem), &clNeighOffset);
	clError(__LINE__, clErr);

	clErr = clSetKernelArg(computeBoid, 3, sizeof(cl_mem), &clNeighSize);
	clError(__LINE__, clErr);

	clErr = clSetKernelArg(computeBoid, 4, sizeof(cl_mem), &clBoidModels);
	clError(__LINE__, clErr);
}

void initCL() {
	cl_uint clRetSize;
	cl_platform_id all_platforms[3];
	clErr = clGetPlatformIDs(3, all_platforms, &clRetSize);
	clError(__LINE__, clErr);
	if (clRetSize == 0) {
		std::cout << " No platforms found. Check OpenCL installation!\n";
		exit(1);
	}
	clPlatformId = all_platforms[0];
	char clNameRet[128];
	clErr = clGetPlatformInfo(clPlatformId, CL_PLATFORM_NAME, 128, clNameRet, &clRetSize);
	clError(__LINE__, clErr);
	std::cout << "Using platform: " << clNameRet << "\n";

	//get default device of the default platform
	cl_device_id all_devices[2];
	clErr = clGetDeviceIDs(clPlatformId, CL_DEVICE_TYPE_GPU, 2, all_devices, &clRetSize);
	clError(__LINE__, clErr);
	if (clRetSize == 0) {
		std::cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}
	clDeviceId = all_devices[0];
	clErr = clGetDeviceInfo(clDeviceId, CL_DEVICE_NAME, 128, clNameRet, &clRetSize);
	clError(__LINE__, clErr);
	std::cout << "Using device: " << clNameRet << "\n";

	// since this is an opengl interoperability program,
	// check if the opengl sharing extension is supported
	// (no point going on if it isn’t):
	// (we need the Device in order to ask, so we can't do it any sooner than right here)
	if( IsCLExtensionSupported( "cl_khr_gl_sharing" ) )
	{
		fprintf(stderr, "cl_khr_gl_sharing is supported.\n");
	}
	else
	{
		fprintf(stderr, "cl_khr_gl_sharing is not supported -- sorry.\n");
		return;
	}

	HGLRC glContext = wglGetCurrentContext();

	cl_context_properties props[] =
	{
		CL_GL_CONTEXT_KHR, (cl_context_properties)glContext,
		CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
		CL_CONTEXT_PLATFORM, (cl_context_properties)clPlatformId, 0
	};

	clContext = clCreateContext(props, 1, all_devices, NULL, NULL, &clErr);
	clError(__LINE__, clErr);

	clQueue = clCreateCommandQueue(clContext, clDeviceId, 0, &clErr);
	clError(__LINE__, clErr);

	initKernels();
}

void updateShaders(vec3 objectColor) {
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
		vertexFile.open("src/shaders/vertexGPUShader.vert");
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

void GPULoadLevel(std::vector<Object>& objects, GPUBoid* boids)
{
	using glm::vec3;

	objects.clear();
	objects.resize(1);

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

	for (int i = 0; i < numBoids; i++) {
		vec3 pos = sphericalRand(linearRand(0.0f, confinementRadius));
		vec3 vel = sphericalRand(0.001f);

		boids[i].pos.x = pos.x;
		boids[i].pos.y = pos.y;
		boids[i].pos.z = pos.z;
		boids[i].vel.x = vel.x;
		boids[i].vel.y = vel.y;
		boids[i].vel.z = vel.z;

	}
}

int main(int argc, char* argv[])
{
	for (int i = 0; i < dimension * dimension * dimension; i++)
	{
		omp_init_lock(&writelock[i]);
	}

	GPULoadLevel(objects, boids);

	screen = InitializeSDL(SCREEN_WIDTH, SCREEN_HEIGHT);

	if (!gladLoadGL((GLADloadfunc)SDL_GL_GetProcAddress))
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
	glBindVertexArray(0);

	glGenBuffers(1, &modelBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, modelBuffer);
	glBufferData(GL_ARRAY_BUFFER, numBoids * sizeof(glm::mat4), &boidModels[0], GL_DYNAMIC_DRAW);

	glBindVertexArray(VAO);

	// model instance attribute
	std::size_t vec4Size = sizeof(glm::vec4);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4 * vec4Size, (void*)0);
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 4 * vec4Size, (void*)(1 * vec4Size));
	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 4 * vec4Size, (void*)(2 * vec4Size));
	glEnableVertexAttribArray(5);
	glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 4 * vec4Size, (void*)(3 * vec4Size));

	glVertexAttribDivisor(2, 1);
	glVertexAttribDivisor(3, 1);
	glVertexAttribDivisor(4, 1);
	glVertexAttribDivisor(5, 1);

	glBindVertexArray(0);

	initCL();

	/*
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

	*/

	//Initialize the spaital data structures
	calculateCellNeighbours();

	clErr = clEnqueueWriteBuffer(clQueue, clBoids, CL_TRUE, 0, sizeof(GPUBoid) * numBoids, boids, 0, NULL, NULL);
	clError(__LINE__, clErr);

	t = SDL_GetTicks();	// Set start value for timer.

	const int precision = 222;
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
int spatialCellsIndex(const cl_float3& pos){
	ivec3 offPos = getCellPos(vec3(pos.x, pos.y, pos.z));

	return offPos.x + offPos.y * dimension + offPos.z * dimension * dimension;
}

void adjacentCellsGPU(int index, ivec3 cellPos, vector<int>& neigh) {
	const int zOffset = dimension * dimension;

	for (int z = -1; z <= 1; z++) {
		if (cellPos.z + z > dimension - 1) continue;
		else if (cellPos.z + z < 0) continue;

		for (int y = -1; y <= 1; y++) {
			if (cellPos.y + y > dimension - 1) continue;
			else if (cellPos.y + y < 0) continue;

			for (int x = -1; x <= 1; x++) {
				if (cellPos.x + x > dimension - 1) continue;
				else if (cellPos.x + x < 0) continue;

				//The cell is valid, get its index
				int neighIndex = index;
				neighIndex += z * zOffset + y * dimension + x;

				//Add the list of the cell to neighbours worthy of looking at
				neigh.push_back(neighIndex);
			}
		}
	}
}

void adjacentCells(int index, ivec3 cellPos, vector<vector<int> *>& neigh){
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
//void getNeighbours(const vec3& pos, vector<vector<GPUBoid*> *>& neigh){
//	int index = spatialCellsIndex(pos);
//	ivec3 cellPos = getCellPos(pos);
//
//	adjacentCells(index, cellPos, neigh);
//}

void calculateCellNeighboursGPU() {
	ivec3 cellPos(0, 0, 0);
	for (int i = 0; i < dimension * dimension * dimension; i++) {
		cellPos.x = i % dimension;
		cellPos.y = (i / dimension) % dimension;
		cellPos.z = i / (dimension * dimension);

		adjacentCellsGPU(i, cellPos, neighboursVGPU[i]);
	}
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

void getNeighbourListsV3() {
	const float radius = largestRadius;

	int simT = SDL_GetTicks();
	#pragma omp parallel for
	for (int i = 0; i < numBoids; i++) {
		GPUBoid& current = boids[i];
		vec3 currPos = vec3(current.pos.x, current.pos.y, current.pos.z);
		vector<vector<int>*>& neigh = neighbours[spatialCellsIndex(current.pos)];

		int neighbourCount = 0;

		for (vector<int>* neighList : neigh) {
			for (int b : *neighList) {
				if (b == i) continue;
				GPUBoid neighbour = boids[b];
				vec3 neighPos = vec3(neighbour.pos.x, neighbour.pos.y, neighbour.pos.z);

				float d = glm::distance(currPos, neighPos);
				if (d < radius) {
					neighbourCount++;
				}
			}
		}

		neighSize[i] = neighbourCount;
	}
	int simT2 = SDL_GetTicks();
	int timeDiff = simT2 - simT;
	cout << "Neighbour count time: " << timeDiff << " ms." << endl;

	simT = SDL_GetTicks();

	exclusive_scan(neighSize.begin(), neighSize.end(), neighOffset.begin(), 0);
	int numNeighs = neighSize[numBoids - 1] + neighOffset[numBoids - 1];

	if (neighs.size() < numNeighs) neighs.resize(numNeighs);

	simT2 = SDL_GetTicks();
	timeDiff = simT2 - simT;
	cout << "Exclusive scan time: " << timeDiff << " ms." << endl;

	simT = SDL_GetTicks();
	#pragma omp parallel for
	for (int i = 0; i < numBoids; i++) {
		GPUBoid& current = boids[i];
		vec3 currPos = vec3(current.pos.x, current.pos.y, current.pos.z);
		vector<vector<int>*>& neigh = neighbours[spatialCellsIndex(current.pos)];

		int neighbourCount = 0;

		for (vector<int>* neighList : neigh) {
			for (int b : *neighList) {
				if (b == i) continue;
				GPUBoid neighbour = boids[b];
				vec3 neighPos = vec3(neighbour.pos.x, neighbour.pos.y, neighbour.pos.z);

				float d = glm::distance(currPos, neighPos);
				if (d < radius) {
					neighs[neighOffset[i] + neighbourCount] = { b, d };
					neighbourCount++;
				}
			}
		}
	}
	simT2 = SDL_GetTicks();
	timeDiff = simT2 - simT;
	cout << "Neighbour insertion time: " << timeDiff << " ms." << endl;
}

void getNeighbourListsV2() {
	const float radius = largestRadius;

	#pragma omp parallel for
	for (int i = 0; i < numBoids; i++) {
		GPUBoid& current = boids[i];
		vec3 currPos = vec3(current.pos.x, current.pos.y, current.pos.z);
		vector<vector<int>*>& neigh = neighbours[spatialCellsIndex(current.pos)];

		int neighbourCount = 0;

		for (vector<int>* neighList : neigh) {
			for (int b : *neighList) {
				if (b == i) continue;
				GPUBoid neighbour = boids[b];
				vec3 neighPos = vec3(neighbour.pos.x, neighbour.pos.y, neighbour.pos.z);

				float d = glm::distance(currPos, neighPos);
				if (d < radius) {
					neighbourCount++;
				}
			}
		}

		neighSize[i] = neighbourCount;
	}

	exclusive_scan(neighSize.begin(), neighSize.end(), neighOffset.begin(), 0);
	int numNeighs = neighSize[numBoids - 1] + neighOffset[numBoids - 1];
	neighs.resize(numNeighs);

	#pragma omp parallel for
	for (int i = 0; i < numBoids; i++) {
		GPUBoid& current = boids[i];
		vec3 currPos = vec3(current.pos.x, current.pos.y, current.pos.z);
		vector<vector<int>*>& neigh = neighbours[spatialCellsIndex(current.pos)];

		int neighbourCount = 0;

		for (vector<int>* neighList : neigh) {
			for (int b : *neighList) {
				if (b == i) continue;
				GPUBoid neighbour = boids[b];
				vec3 neighPos = vec3(neighbour.pos.x, neighbour.pos.y, neighbour.pos.z);

				float d = glm::distance(currPos, neighPos);
				if (d < radius) {
					neighs[neighOffset[i] + neighbourCount] = { b, d };
					neighbourCount++;
				}
			}
		}
	}

	for (int i = 0; i < numBoids; i++) {
		sort(neighs.begin() + neighOffset[i], neighs.begin() + neighOffset[i] + neighSize[i],
			[](NeighbourWatchlistEntry a, NeighbourWatchlistEntry b) {
			return a.dist < b.dist;
		});
	}
	
}

void getNeighbourLists() {
	vector<NeighbourWatchlistEntry> tempWatchList[numBoids];

	const float radius = largestRadius;

	#pragma omp parallel for
	for (int i = 0; i < numBoids; i++) {
		GPUBoid& current = boids[i];
		vec3 currPos = vec3(current.pos.x, current.pos.y, current.pos.z);
		vector<vector<int>*>& neigh = neighbours[spatialCellsIndex(current.pos)];

		for (vector<int>* neighList : neigh) {
			for (int b : *neighList) {
				if (b == i) continue;
				GPUBoid neighbour = boids[b];
				vec3 neighPos = vec3(neighbour.pos.x, neighbour.pos.y, neighbour.pos.z);

				float d = glm::distance(currPos, neighPos);
				if (d < radius) {
					tempWatchList[i].push_back({b, d});
				}
			}
		}

		neighSize[i] = tempWatchList[i].size();

		sort(tempWatchList[i].begin(), tempWatchList[i].end(), 
			[](NeighbourWatchlistEntry a, NeighbourWatchlistEntry b) {
			return a.dist < b.dist;
		});
	}

	exclusive_scan(neighSize.begin(), neighSize.end(), neighOffset.begin(), 0);
	int numNeighs = neighSize[numBoids - 1] + neighOffset[numBoids - 1];
	neighs.resize(numNeighs);

	//#pragma omp parallel for (Also just makes it way slower)
	for (int i = 0; i < numBoids; i++) {
		copy(tempWatchList[i].begin(), tempWatchList[i].end(), neighs.begin() + neighOffset[i]);
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

vec3 cohesion(int index, vector<vector<int>*>& neigh) {
	/*
		Find the center of mass amongst nearby boids, with nearby beind defined
		as those boids within a sphere centered on this boid with a given radius.
		Weight the boids effect on the center of mass by their inverse distance²

		Lastly calculate the vector required to move to said center
	*/
	const float radius = cohesionRadius;
	const float strength = 0.1f;
	/*const float epsilon = 1.0f / 10.0f;
	const float epsInvSqr = 1.0f / glm::pow(epsilon, 2.0f);*/

	GPUBoid& current = boids[index];

	vec3 currPos = vec3(current.pos.x, current.pos.y, current.pos.z);

	vec3 center(0, 0, 0);
	int numNear = 0;
	for (vector<int>* neighList : neigh) {
		for (int b : *neighList) {
			if (b == index) continue;

			GPUBoid neighbour = boids[b];
			vec3 neighPos = vec3(neighbour.pos.x, neighbour.pos.y, neighbour.pos.z);

			float d = glm::distance(currPos, neighPos);
			if (d < radius) {
				//TODO: This should probably be weighted by the inv square dist
				//float w = (d > epsilon) ? (1.0f / glm::pow(d, 2.0f)) / epsInvSqr : 1.0f;

				center += neighPos /* ((radius - d) / d)*/;
				numNear++;
			}
		}
	}

	//Find the averaged center, or self if no neighbours
	center = (numNear > 0) ? center / (float)numNear - currPos : center;

	return center * strength;
}

vec3 avoidance(int index, vector<vector<int>*>& neigh) {
	const float radius = avoidanceRadius;
	const float strength = 0.4f;

	vec3 res(0, 0, 0);

	GPUBoid& current = boids[index];
	vec3 currPos = vec3(current.pos.x, current.pos.y, current.pos.z);

	for (vector<int>* neighList : neigh) {
		for (int b : *neighList) {
			if (b == index) continue;

			GPUBoid neighbour = boids[b];
			vec3 neighPos = vec3(neighbour.pos.x, neighbour.pos.y, neighbour.pos.z);

			float dist = glm::distance(currPos, neighPos);

			if (dist < radius) {
				res -= (neighPos - currPos) * ((radius - dist) / dist);
			}
		}
	}

	return res * strength;
}

vec3 conformance(int index, vector<vector<int>*>& neigh) {
	const float radius = conformanceRadius;
	const float strength = 0.2f;

	GPUBoid& current = boids[index];

	vec3 currPos = vec3(current.pos.x, current.pos.y, current.pos.z);
	vec3 currVel = vec3(current.vel.x, current.vel.y, current.vel.z);

	vec3 velocity(0, 0, 0);
	int numNear = 0;
	for (vector<int>* neighList : neigh) {
		for (int b : *neighList) {
			if (b == index) continue;

			GPUBoid neighbour = boids[b];
			vec3 neighPos = vec3(neighbour.pos.x, neighbour.pos.y, neighbour.pos.z);

			if (glm::distance(currPos, neighPos) < radius) {
				//TODO: This should probably be weighted by the inv square dist
				velocity += vec3(neighbour.vel.x, neighbour.vel.y, neighbour.vel.z);
				numNear++;
			}
		}
	}

	velocity = (numNear > 0) ? velocity / (float)numNear : currVel;

	return (velocity - currVel) * strength;
}

vec3 confinement(GPUBoid& current) {
	const float strength = 0.1f;

	vec3 v(0, 0, 0);

	/*if(glm::length(current.pos) > confinementRadius){
		v = glm::normalize(-current.pos);
	}*/

	vec3 currPos = vec3(current.pos.x, current.pos.y, current.pos.z);

	for (int i = 0; i < 3; i++) {
		if (currPos[i] < -confinementRadius) v[i] = 1;
		else if (currPos[i] > confinementRadius) v[i] = -1;
	}

	if (v != vec3(0, 0, 0)) v = glm::normalize(v);

	return v * strength;
}

vec3 clamp(vec3& original, vec3& increment, const float normalizer, const float dt) {
	const float speedLimitUpper = 1.0f * normalizer;
	const float speedLimitLower = 0.1f * normalizer;

	vec3 newBoidVel = original + increment * normalizer * dt;
	vec3 newVel = 0.5f * increment * dt * normalizer + newBoidVel;
	if (glm::length(newVel) > speedLimitUpper) {
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

vec3 drag(GPUBoid& current) {
	const float drag = 0.1f;
	vec3 currVel = vec3(current.vel.x, current.vel.y, current.vel.z);
	return -currVel * drag;
}

void simulateBoidGPU(float dt) {
	//Clear the spatial partition and re-insert the Boids at the correct space
	//#pragma omp parallel for (Parallelizing this part is slower than not parallelizing it.
	int simT = SDL_GetTicks();
	for (int i = 0; i < dimension * dimension * dimension; i++) {
		spatialCells[i].clear();
	}
	#pragma omp parallel for
	for (int i = 0; i < numBoids; i++) {
		GPUBoid& b = boids[i];
		//printf("(%f, %f, %f): %d\n", b.pos.x, b.pos.y, b.pos.z, spatialCellsIndex(b.pos));
		int index = spatialCellsIndex(b.pos);
		omp_set_lock(&writelock[index]);
		spatialCells[index].push_back(i);
		omp_unset_lock(&writelock[index]);
	}
	getNeighbourListsV3();

	if (neighs.size() > clNeighsNumCap) {
		clErr = clReleaseMemObject(clNeighs);
		clError(__LINE__, clErr);

		clNeighsNumCap = neighs.size();
		clNeighs = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(NeighbourWatchlistEntry) * clNeighsNumCap, NULL, &clErr);
		clError(__LINE__, clErr);

		clErr = clSetKernelArg(computeBoid, 1, sizeof(cl_mem), &clNeighs);
		clError(__LINE__, clErr);
	}

	clErr = clEnqueueWriteBuffer(clQueue, clNeighs, CL_TRUE, 0, sizeof(NeighbourWatchlistEntry) * neighs.size(), neighs.data(), 0, NULL, NULL);
	clError(__LINE__, clErr);

	clErr = clEnqueueWriteBuffer(clQueue, clNeighOffset, CL_TRUE, 0, sizeof(int) * numBoids, neighOffset.data(), 0, NULL, NULL);
	clError(__LINE__, clErr);

	clErr = clEnqueueWriteBuffer(clQueue, clNeighSize, CL_TRUE, 0, sizeof(int) * numBoids, neighSize.data(), 0, NULL, NULL);
	clError(__LINE__, clErr);

	int simT2 = SDL_GetTicks();
	int timeDiff = simT2 - simT;
	cout << "Neighboursearch time: " << timeDiff << " ms." << endl;

	/*
		For each boid, calulate the vector for each of the three rules
			Simulate a fourth bounds rule to keep the boids in line
		Then sum these vectors (without prioritising them, which differs from the paper)
		Add this sum to the velocity of the boid
		Update the position by the given velocity

		Sources: https://vergenet.net/~conrad/boids/pseudocode.html, https://dl.acm.org/doi/10.1145/37402.37406
	*/

	simT = SDL_GetTicks();

	clErr = clSetKernelArg(computeBoid, 5, sizeof(float), &dt);
	clError(__LINE__, clErr);

	clErr = clEnqueueAcquireGLObjects(clQueue, 1, &clBoidModels, 0, NULL, NULL);
	clError(__LINE__, clErr);

	clErr = clEnqueueNDRangeKernel(clQueue, computeBoid, 1, NULL, GlobalWorkSize, NULL, 0, NULL, NULL);
	clError(__LINE__, clErr);
	clErr = clFinish(clQueue);
	clError(__LINE__, clErr);

	clErr = clEnqueueReleaseGLObjects(clQueue, 1, &clBoidModels, 0, NULL, NULL);
	clError(__LINE__, clErr);
	clErr = clFinish(clQueue);
	clError(__LINE__, clErr);

	clErr = clEnqueueReadBuffer(clQueue, clBoids, CL_TRUE, 0, sizeof(GPUBoid) * numBoids, boids, 0, NULL, NULL);
	clError(__LINE__, clErr);

	clErr = clFinish(clQueue);
	clError(__LINE__, clErr);

	simT2 = SDL_GetTicks();
	timeDiff = simT2 - simT;
	cout << "Computation time: " << timeDiff << " ms." << endl;
}


void simulateBoidV4(float dt) {
	//Clear the spatial partition and re-insert the Boids at the correct space
	//#pragma omp parallel for (Parallelizing this part is slower than not parallelizing it.
	int simT = SDL_GetTicks();
	for (int i = 0; i < dimension * dimension * dimension; i++) {
		spatialCells[i].clear();
	}
	#pragma omp parallel for
	for (int i = 0; i < numBoids; i++) {
		GPUBoid& b = boids[i];
		//printf("(%f, %f, %f): %d\n", b.pos.x, b.pos.y, b.pos.z, spatialCellsIndex(b.pos));
		int index = spatialCellsIndex(b.pos);
		omp_set_lock(&writelock[index]);
		spatialCells[index].push_back(i);
		omp_unset_lock(&writelock[index]);
	}
	getNeighbourListsV3();
	int simT2 = SDL_GetTicks();
	int timeDiff = simT2 - simT;
	cout << "Neighboursearch time: " << timeDiff << " ms." << endl;

	/*
		For each boid, calulate the vector for each of the three rules
			Simulate a fourth bounds rule to keep the boids in line
		Then sum these vectors (without prioritising them, which differs from the paper)
		Add this sum to the velocity of the boid
		Update the position by the given velocity

		Sources: https://vergenet.net/~conrad/boids/pseudocode.html, https://dl.acm.org/doi/10.1145/37402.37406
	*/
	const float normalizer = 1.0f / 1000.f;

	//Avoidance
	const float avoidanceStrength = 0.4f;

	//Cohesion
	const float coheshionStrength = 0.1f;
	/*const float coheshionEpsilon = 1.0f / 10.0f;
	const float coheshionEpsInvSqr = 1.0f / glm::pow(coheshionEpsilon, 2.0f);*/

	//Conformance
	const float conformanceStrength = 0.2f;

	simT = SDL_GetTicks();
	#pragma omp parallel for
	for (int i = 0; i < numBoids; i++) {
		GPUBoid& b = boids[i];
		int numNeighs = neighSize[i];
		int offset = neighOffset[i];

		vec3 currPos = vec3(b.pos.x, b.pos.y, b.pos.z);
		vec3 currVel = vec3(b.vel.x, b.vel.y, b.vel.z);

		vec3 a = vec3(0, 0, 0);

		//Avoidance
		vec3 avoidanceRes(0, 0, 0);

		//Cohesion
		vec3 coheshionCenter(0, 0, 0);
		int coheshionNumNear = 0;

		//Conformance
		vec3 conformanceVelocity(0, 0, 0);
		int conformanceNumNear = 0;

		for (int neighIndex = 0; neighIndex < numNeighs; neighIndex++) {
			NeighbourWatchlistEntry entry = neighs[offset + neighIndex];
			GPUBoid neighbour = boids[entry.index];
			vec3 neighPos = vec3(neighbour.pos.x, neighbour.pos.y, neighbour.pos.z);
			vec3 neighVel = vec3(neighbour.vel.x, neighbour.vel.y, neighbour.vel.z);

			float d = entry.dist;
			if (d < avoidanceRadius) {
				//Avoidance
				avoidanceRes -= (neighPos - currPos) * ((avoidanceRadius - d) / d);

				//Cohesion
				coheshionCenter += neighPos;
				coheshionNumNear++;

				//Conformance
				conformanceVelocity += neighVel;
				conformanceNumNear++;
			}
			else if (d < cohesionRadius) {
				//Cohesion
				coheshionCenter += neighPos;
				coheshionNumNear++;

				//Conformance
				conformanceVelocity += neighVel;
				conformanceNumNear++;
			}
			else {
				//Conformance
				conformanceVelocity += neighVel;
				conformanceNumNear++;
			}
		}

		//Avoidance
		a += avoidanceRes * avoidanceStrength;

		//Cohesion
		coheshionCenter = (coheshionNumNear > 0) ? coheshionCenter / (float)coheshionNumNear - currPos : coheshionCenter;
		a += coheshionCenter * coheshionStrength;

		//Conformance
		conformanceVelocity = (conformanceNumNear > 0) ? conformanceVelocity / (float)conformanceNumNear : currVel;
		a += (conformanceVelocity - currVel) * conformanceStrength;

		a += confinement(b);
		a += drag(b);

		currPos += clamp(currVel, a, normalizer, dt) * dt;
		b.pos.x = currPos.x;
		b.pos.y = currPos.y;
		b.pos.z = currPos.z;
		b.vel.x = currVel.x;
		b.vel.y = currVel.y;
		b.vel.z = currVel.z;

		vec3 ref = glm::normalize(glm::cross(boidForward, currVel));
		float angle = glm::orientedAngle(glm::normalize(boidForward), glm::normalize(currVel), ref);

		glm::mat4 model = glm::translate(glm::mat4(1.0f), currPos);

		boidModels[i] = glm::rotate(model, angle, ref);
		/*printf(
			"%f, %f, %f, %f\n"
			"%f, %f, %f, %f\n"
			"%f, %f, %f, %f\n"
			"%f, %f, %f, %f\n\n\n",
			boidModels[i][0][0], boidModels[i][0][1], boidModels[i][0][2], boidModels[i][0][3],
			boidModels[i][1][0], boidModels[i][1][1], boidModels[i][1][2], boidModels[i][1][3],
			boidModels[i][2][0], boidModels[i][2][1], boidModels[i][2][2], boidModels[i][2][3],
			boidModels[i][3][0], boidModels[i][3][1], boidModels[i][3][2], boidModels[i][3][3]
		);*/
	}
	simT2 = SDL_GetTicks();
	timeDiff = simT2 - simT;
	cout << "Computation time: " << timeDiff << " ms." << endl;
}

void simulateBoidV3(float dt) {
	//Clear the spatial partition and re-insert the Boids at the correct space
	//#pragma omp parallel for (Parallelizing this part is slower than not parallelizing it.
	int simT = SDL_GetTicks();
	for (int i = 0; i < dimension * dimension * dimension; i++) {
		spatialCells[i].clear();
	}
	#pragma omp parallel for
	for (int i = 0; i < numBoids; i++) {
		GPUBoid& b = boids[i];
		//printf("(%f, %f, %f): %d\n", b.pos.x, b.pos.y, b.pos.z, spatialCellsIndex(b.pos));
		int index = spatialCellsIndex(b.pos);
		omp_set_lock(&writelock[index]);
		spatialCells[index].push_back(i);
		omp_unset_lock(&writelock[index]);
	}
	int simT2 = SDL_GetTicks();
	int timeDiff = simT2 - simT;
	cout << "Neighboursearch time: " << timeDiff << " ms." << endl;

	/*
		For each boid, calulate the vector for each of the three rules
			Simulate a fourth bounds rule to keep the boids in line
		Then sum these vectors (without prioritising them, which differs from the paper)
		Add this sum to the velocity of the boid
		Update the position by the given velocity

		Sources: https://vergenet.net/~conrad/boids/pseudocode.html, https://dl.acm.org/doi/10.1145/37402.37406
	*/
	const float normalizer = 1.0f / 1000.f;

	//Avoidance
	const float avoidanceStrength = 0.4f;

	//Cohesion
	const float coheshionStrength = 0.1f;
	/*const float coheshionEpsilon = 1.0f / 10.0f;
	const float coheshionEpsInvSqr = 1.0f / glm::pow(coheshionEpsilon, 2.0f);*/

	//Conformance
	const float conformanceStrength = 0.2f;

	simT = SDL_GetTicks();
	#pragma omp parallel for
	for (int i = 0; i < numBoids; i++) {
		GPUBoid& b = boids[i];
		int numNeighs = neighSize[i];

		vec3 currPos = vec3(b.pos.x, b.pos.y, b.pos.z);
		vec3 currVel = vec3(b.vel.x, b.vel.y, b.vel.z);

		vec3 a = vec3(0, 0, 0);

		//Avoidance
		vec3 avoidanceRes(0, 0, 0);

		//Cohesion
		vec3 coheshionCenter(0, 0, 0);
		int coheshionNumNear = 0;

		//Conformance
		vec3 conformanceVelocity(0, 0, 0);
		int conformanceNumNear = 0;

		vector<vector<int>*>& neigh = neighbours[spatialCellsIndex(b.pos)];

		for (vector<int>* neighList : neigh) {
			for (int b : *neighList) {
				if (b == i) continue;
				GPUBoid neighbour = boids[b];
				vec3 neighPos = vec3(neighbour.pos.x, neighbour.pos.y, neighbour.pos.z);
				vec3 neighVel = vec3(neighbour.vel.x, neighbour.vel.y, neighbour.vel.z);

				float d = glm::distance(currPos, neighPos);
				if (d < avoidanceRadius) {
					//Avoidance
					avoidanceRes -= (neighPos - currPos) * ((avoidanceRadius - d) / d);

					//Cohesion
					coheshionCenter += neighPos;
					coheshionNumNear++;

					//Conformance
					conformanceVelocity += neighVel;
					conformanceNumNear++;
				} else if (d < cohesionRadius) {
					//Cohesion
					coheshionCenter += neighPos;
					coheshionNumNear++;

					//Conformance
					conformanceVelocity += neighVel;
					conformanceNumNear++;
				} else if (d < conformanceRadius) {
					//Conformance
					conformanceVelocity += neighVel;
					conformanceNumNear++;
				}
			}
		}

		//Avoidance
		a += avoidanceRes * avoidanceStrength;

		//Cohesion
		coheshionCenter = (coheshionNumNear > 0) ? coheshionCenter / (float)coheshionNumNear - currPos : coheshionCenter;
		a += coheshionCenter * coheshionStrength;

		//Conformance
		conformanceVelocity = (conformanceNumNear > 0) ? conformanceVelocity / (float)conformanceNumNear : currVel;
		a += (conformanceVelocity - currVel) * conformanceStrength;

		a += confinement(b);
		a += drag(b);

		currPos += clamp(currVel, a, normalizer, dt) * dt;
		b.pos.x = currPos.x;
		b.pos.y = currPos.y;
		b.pos.z = currPos.z;
		b.vel.x = currVel.x;
		b.vel.y = currVel.y;
		b.vel.z = currVel.z;

		vec3 ref = glm::normalize(glm::cross(boidForward, currVel));
		float angle = glm::orientedAngle(glm::normalize(boidForward), glm::normalize(currVel), ref);

		glm::mat4 model = glm::translate(glm::mat4(1.0f), currPos);

		boidModels[i] = glm::rotate(model, angle, ref);
	}
	simT2 = SDL_GetTicks();
	timeDiff = simT2 - simT;
	cout << "Computation time: " << timeDiff << " ms." << endl;
}

void simulateBoidV2(float dt) {
	//Clear the spatial partition and re-insert the Boids at the correct space
	//#pragma omp parallel for (Parallelizing this part is slower than not parallelizing it.
	int simT = SDL_GetTicks();
	for (int i = 0; i < dimension * dimension * dimension; i++) {
		spatialCells[i].clear();
	}
	#pragma omp parallel for
	for (int i = 0; i < numBoids; i++) {
		GPUBoid& b = boids[i];
		//printf("(%f, %f, %f): %d\n", b.pos.x, b.pos.y, b.pos.z, spatialCellsIndex(b.pos));
		int index = spatialCellsIndex(b.pos);
		omp_set_lock(&writelock[index]);
		spatialCells[index].push_back(i);
		omp_unset_lock(&writelock[index]);
	}

	//Generates our neighbourlists and watchlist for the neighbours
	getNeighbourListsV2();
	int simT2 = SDL_GetTicks();
	int timeDiff = simT2 - simT;
	cout << "Neighboursearch time: " << timeDiff << " ms." << endl;

	/*
		For each boid, calulate the vector for each of the three rules
			Simulate a fourth bounds rule to keep the boids in line
		Then sum these vectors (without prioritising them, which differs from the paper)
		Add this sum to the velocity of the boid
		Update the position by the given velocity

		Sources: https://vergenet.net/~conrad/boids/pseudocode.html, https://dl.acm.org/doi/10.1145/37402.37406
	*/
	const float normalizer = 1.0f / 1000.f;

	//Avoidance
	const float avoidanceStrength = 0.4f;

	//Cohesion
	const float coheshionStrength = 0.1f;
	/*const float coheshionEpsilon = 1.0f / 10.0f;
	const float coheshionEpsInvSqr = 1.0f / glm::pow(coheshionEpsilon, 2.0f);*/

	//Conformance
	const float conformanceStrength = 0.2f;
	
	simT = SDL_GetTicks();
	#pragma omp parallel for
	for (int i = 0; i < numBoids; i++) {
		GPUBoid& b = boids[i];
		int numNeighs = neighSize[i];

		vec3 currPos = vec3(b.pos.x, b.pos.y, b.pos.z);
		vec3 currVel = vec3(b.vel.x, b.vel.y, b.vel.z);

		vec3 a = vec3(0, 0, 0);

		//Avoidance
		vec3 avoidanceRes(0, 0, 0);

		//Cohesion
		vec3 coheshionCenter(0, 0, 0);
		int coheshionNumNear = 0;

		//Conformance
		vec3 conformanceVelocity(0, 0, 0);
		int conformanceNumNear = 0;

		int neighIndex = 0;
		NeighbourWatchlistEntry entry;

		while (neighIndex < numNeighs && neighs[neighOffset[i] + neighIndex].dist < avoidanceRadius) {
			entry = neighs[neighOffset[i] + neighIndex];
			GPUBoid neighbour = boids[entry.index];
			vec3 neighPos = vec3(neighbour.pos.x, neighbour.pos.y, neighbour.pos.z);
			vec3 neighVel = vec3(neighbour.vel.x, neighbour.vel.y, neighbour.vel.z);

			//Avoidance
			avoidanceRes -= (neighPos - currPos) * ((avoidanceRadius - entry.dist) / entry.dist);

			//Cohesion
			coheshionCenter += neighPos;
			coheshionNumNear++;

			//Conformance
			conformanceVelocity += neighVel;
			conformanceNumNear++;

			neighIndex++;
		}

		//Avoidance
		a += avoidanceRes * avoidanceStrength;

		while (neighIndex < numNeighs && neighs[neighOffset[i] + neighIndex].dist < cohesionRadius) {
			entry = neighs[neighOffset[i] + neighIndex];
			GPUBoid neighbour = boids[entry.index];
			vec3 neighPos = vec3(neighbour.pos.x, neighbour.pos.y, neighbour.pos.z);
			vec3 neighVel = vec3(neighbour.vel.x, neighbour.vel.y, neighbour.vel.z);

			//Cohesion
			coheshionCenter += neighPos;
			coheshionNumNear++;

			//Conformance
			conformanceVelocity += neighVel;
			conformanceNumNear++;

			neighIndex++;
		}

		//Cohesion
		coheshionCenter = (coheshionNumNear > 0) ? coheshionCenter / (float)coheshionNumNear - currPos : coheshionCenter;
		a += coheshionCenter * coheshionStrength;

		while (neighIndex < numNeighs) {
			entry = neighs[neighOffset[i] + neighIndex];
			GPUBoid neighbour = boids[entry.index];
			vec3 neighVel = vec3(neighbour.vel.x, neighbour.vel.y, neighbour.vel.z);

			//Conformance
			conformanceVelocity += neighVel;
			conformanceNumNear++;

			neighIndex++;
		}

		//Conformance
		conformanceVelocity = (conformanceNumNear > 0) ? conformanceVelocity / (float)conformanceNumNear : currVel;
		a += (conformanceVelocity - currVel) * conformanceStrength;

		a += confinement(b);
		a += drag(b);

		currPos += clamp(currVel, a, normalizer, dt) * dt;
		b.pos.x = currPos.x;
		b.pos.y = currPos.y;
		b.pos.z = currPos.z;
		b.vel.x = currVel.x;
		b.vel.y = currVel.y;
		b.vel.z = currVel.z;

		vec3 ref = glm::normalize(glm::cross(boidForward, currVel));
		float angle = glm::orientedAngle(glm::normalize(boidForward), glm::normalize(currVel), ref);

		glm::mat4 model = glm::translate(glm::mat4(1.0f), currPos);

		boidModels[i] = glm::rotate(model, angle, ref);
	}
	simT2 = SDL_GetTicks();
	timeDiff = simT2 - simT;
	cout << "Computation time: " << timeDiff << " ms." << endl;
}

void simulateBoid(float dt) {
	//Clear the spatial partition and re-insert the Boids at the correct space
	//#pragma omp parallel for (Parallelizing this part is slower than not parallelizing it.

	int simT = SDL_GetTicks();
	for (int i = 0; i < dimension * dimension * dimension; i++) {
		spatialCells[i].clear();
	}
	#pragma omp parallel for
	for (int i = 0; i < numBoids; i++) {
		GPUBoid& b = boids[i];
		//printf("(%f, %f, %f): %d\n", b.pos.x, b.pos.y, b.pos.z, spatialCellsIndex(b.pos));
		int index = spatialCellsIndex(b.pos);
		omp_set_lock(&writelock[index]);
		spatialCells[index].push_back(i);
		omp_unset_lock(&writelock[index]);
	}
	int simT2 = SDL_GetTicks();
	int timeDiff = simT2 - simT;
	cout << "Neighboursearch time: " << timeDiff << " ms." << endl;

	/*
		For each boid, calulate the vector for each of the three rules
			Simulate a fourth bounds rule to keep the boids in line
		Then sum these vectors (without prioritising them, which differs from the paper)
		Add this sum to the velocity of the boid
		Update the position by the given velocity	

		Sources: https://vergenet.net/~conrad/boids/pseudocode.html, https://dl.acm.org/doi/10.1145/37402.37406 
	*/
	const float normalizer = 1.0f / 1000.f;

	simT = SDL_GetTicks();
	#pragma omp parallel for
	for (int i = 0; i < numBoids; i++) {
		GPUBoid& b = boids[i];
		vector<vector<int>*>& neigh = neighbours[spatialCellsIndex(b.pos)];

		vec3 a = avoidance(i, neigh); 
		a += cohesion(i, neigh);
		a += conformance(i, neigh);
		a += confinement(b);
		a += drag(b);

		vec3 currPos = vec3(b.pos.x, b.pos.y, b.pos.z);
		vec3 currVel = vec3(b.vel.x, b.vel.y, b.vel.z);
		currPos += clamp(currVel, a, normalizer, dt) * dt;
		b.pos.x = currPos.x;
		b.pos.y = currPos.y;
		b.pos.z = currPos.z;
		b.vel.x = currVel.x;
		b.vel.y = currVel.y;
		b.vel.z = currVel.z;

		vec3 ref = glm::normalize(glm::cross(boidForward, currVel));
		float angle = glm::orientedAngle(glm::normalize(boidForward), glm::normalize(currVel), ref);

		glm::mat4 model = glm::translate(glm::mat4(1.0f), currPos);

		boidModels[i] = glm::rotate(model, angle, ref);
	}
	simT2 = SDL_GetTicks();
	timeDiff = simT2 - simT;
	cout << "Computation time: " << timeDiff << " ms." << endl;
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
	simulateBoidGPU(dt);
	int simT2 = SDL_GetTicks();
	dSimT = simT2 - simT;

	cout << "Prev Tot time: " << dt << " ms." << " Simulation time: " << dSimT << " ms." << endl;
}

void Draw(){

	glClearColor(0.0f, 0.0f, 0.0f, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//Update our shaders based on the camera positions and stuff
	glUseProgram(shader); //TODO: Do once

	//glBufferSubData(GL_ARRAY_BUFFER, 0, numBoids * sizeof(glm::mat4), &boidModels[0]);


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
	int numtri = 4 * 3;
	updateShaders(red); //TODO: By copy or by reference?
	glDrawArraysInstanced(GL_TRIANGLES, 0, numtri, numBoids);
	glBindVertexArray(0);

	SDL_GL_SwapBuffers();
}
