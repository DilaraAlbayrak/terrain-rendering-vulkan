#version 450

layout(location = 0) out vec4 outColor;

layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in float inHeight;

layout(binding = 1) uniform sampler2D heightMap;

vec3 calculateNormal()
{
    float left  = textureOffset(heightMap, inTexCoord, ivec2(-1, 0)).r;
    float right = textureOffset(heightMap, inTexCoord, ivec2(1, 0)).r;
    float down  = textureOffset(heightMap, inTexCoord, ivec2(0, -1)).r;
    float up    = textureOffset(heightMap, inTexCoord, ivec2(0, 1)).r;

    return normalize(vec3(left - right, down - up, 2.0));
}

void main()
{
    vec3 normal = calculateNormal();
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    float diffuse = max(dot(normal, lightDir), 0.2);
    vec3 baseColour = vec3(inHeight);
    outColor = vec4(baseColour * diffuse, 1.0);
}