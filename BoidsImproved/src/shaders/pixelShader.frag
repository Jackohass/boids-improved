#version 330 core
out vec4 illumination;

in vec3 Normal;  
in vec3 pos3D;  
  
uniform vec3 lightPos;
uniform vec3 lightPower;
uniform vec3 objectColor;
uniform	vec3 indirectLightPowerPerArea;

#define PI 3.141592f

void main()
{
	vec3 r = lightPos - pos3D;
	float sphereArea = 4.0f * PI * dot(r, r);
	vec3 rNorm = normalize(r);
	vec3 D = (lightPower * max(dot(rNorm, Normal), 0.0f)) / sphereArea;
	vec3 temp = objectColor * (D + indirectLightPowerPerArea);
	illumination = vec4(temp, 1.0f);
} 