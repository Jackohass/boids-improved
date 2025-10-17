# Optimizing Boids
This is based on a group project I did for the course "DH2323 Computer Graphics and Interaction Implementing 3D Boids with Spatial Partitioning" in 2022.
I have decided to extend this project, with consent from my group partner, to try to use parallelism to improve the speed of the simulation.
I have also moved this project to Visual Studios so it is self contained there.
I believe there should be no other prerequisits other than Visual Studio.
All the libraries and such should be included in the project for ease of use.
I moved it so it would be easier for someone else to download it and compile it without having to jump through hoops to get it to work.
Because I had to spend for too many hours to make it work for my machine.
Such hours shall not be wasted by another person again if I have any say.

The original report can be found in the root folder.
The 2 orignal implementations are also included as 2 different projects in the solution.
I have made 2 extra projects in the solution as my extension.
One is utlizing OpenMP.
The other one is trying to utilize the GPU. However, that one hasn't been finished due to illness.
But it is based on the paper "*Particle Simulation using CUDA*" (Green, S. 2010. *Particle Simulation using CUDA*).

All of them share a lot of code through headers and shaders.
The biggest difference between them is the cpp file with the main method.
So there is a lot of code duplication that could be removed which shall be a future milestone.

I beg to the coding gods and ask my previous programming teachers for forgiveness for my improper use of, well more like lack of, proper build tools.  

### Prerequists:
- Visual Studio 2017 or later.
- OpenGL 4.6 capable GPU

### Compiling:
There are 4 different projects that you can build.
Choose 1 and compile and run it.

In the file TestModel.h, you can change the parameters for the simulation for all versions.

### Controls
The arrows allows you to control the camera in the cardinal directions.
WASD allows you to control the light source in the cardinal directions.

Q & E allows you to move the light source up and down.

