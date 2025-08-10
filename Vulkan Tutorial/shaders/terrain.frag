#version 450

// Tessellation Evaluation Shader inputs
layout (location = 0) in vec3 inNormal;
layout (location = 3) in vec3 inLightVec;

layout (location = 0) out vec4 outFragColor;

float fog(float density)
{
	const float LOG2 = -1.442695;
	float dist = gl_FragCoord.z / gl_FragCoord.w * 0.1;
	float d = density * dist;
	return 1.0 - clamp(exp2(d * d * LOG2), 0.0, 1.0);
}

void main()
{
	vec3 N = normalize(inNormal);
	vec3 L = normalize(inLightVec);
	vec3 ambient = vec3(0.3, 0.3, 0.4); 
	vec3 diffuse = max(dot(N, L), 0.0) * vec3(0.9, 0.85, 0.8); 

	vec3 baseColor = vec3(0.5, 0.45, 0.4); 

	vec4 color = vec4((ambient + diffuse) * baseColor, 1.0);

	const vec4 fogColor = vec4(0.47, 0.5, 0.67, 0.0);
	outFragColor = mix(color, fogColor, fog(0.25));
}