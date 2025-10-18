#version 330 core
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;
layout (location = 2) in mat4 instanceModel;

out vec3 Normal;
out vec3 pos3D;

uniform mat4 camera;
uniform mat4 projection;

void main()
{
    pos3D = vec3(instanceModel * vec4(pos, 1.0));
    Normal = mat3(transpose(inverse(instanceModel))) * normal;  
    
    gl_Position = projection * camera * vec4(pos3D, 1.0);
}