#version 330 core
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;

out vec3 Normal;
out vec3 pos3D;

uniform mat4 model;
uniform mat4 camera;
uniform mat4 projection;

void main()
{
    pos3D = vec3(model * vec4(pos, 1.0));
    Normal = mat3(transpose(inverse(model))) * normal;  
    
    gl_Position = projection * camera * vec4(pos3D, 1.0);
}