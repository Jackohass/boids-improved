# Optimizing Boids
This is based on a group project I did for the course "DH2323 Computer Graphics and Interaction" where we implemented 3D Boids with Spatial Partitioning in 2022 programming all the graphics code from scratch with OpenGL.
We handle all rendering ourself. So we render and handle all triangles and handle light calculations ourself using OpenGL fragment and vertex shaders.
I have decided to extend this project, with consent from my group partner, to try to use parallelism to improve the speed of the simulation.
I have also added instancing to better render the thousands of boids without any extra draw calls.
I have also moved this project to Visual Studios so it is self contained there.
All the libraries and such should be included in the project for ease of use.
But more work will be done to make it more accessible and easy to use. 

The original report can be found in the root folder.
The 2 orignal implementations are also included as 2 different projects in the solution.
I have made 2 extra projects in the solution as my extension.
One is utlizing OpenMP.
The other one is trying to utilize the GPU to do the Boid simulation using OpenCL and includes OpenCL & OpenGL interoperability functions to directly allow the OpenCL output the necessary matrices so OpenGL can render without having to go through the CPU.
It will be further extended to include the neighbour search on the GPU. 
The neighbour search algorithm will be based on the paper "*Particle Simulation using CUDA*" (Green, S. 2010. *Particle Simulation using CUDA*).
The GPU version is in need of some deseperate cleaning, since I wrote the whole thing over a weekend. Too many functions in the same file and way too many global variables. Also mutliple versions of functions as I was testing different implementations to see what was more optimal. 
There is also some room for improvments speed-wise. The kernel has some places where I am sure we could avoid branching, but some issues surfaced up when I made it branchless so I had to resort to go back to branches.
The workgroup division could also be applied better I am sure, to achieve a better occupancy on the GPU.

All of the projects share a lot of code through headers and shaders.
The biggest difference between them is the cpp file with the main method.
So there is a lot of code duplication that could be removed which shall be a future milestone.

I beg to the coding gods and ask my previous programming teachers for forgiveness for my improper use of, well more like lack of, proper build tools.  

### Prerequists:
- Visual Studio 2017 or later.
- OpenGL 4.6 capable GPU
- OpenCL capable GPU and drivers.

### Compiling:
There are 4 different projects that you can build.
Choose 1 and compile and run it.

In the file TestModel.h, you can change the parameters for the simulation for all versions.

### Controls
The arrows allows you to control the camera in the cardinal directions.
WASD allows you to control the light source in the cardinal directions.

Q & E allows you to move the light source up and down.


