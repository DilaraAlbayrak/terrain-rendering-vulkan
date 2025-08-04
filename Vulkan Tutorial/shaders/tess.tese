#version 450

//layout(quads, equal_spacing, ccw) in;
layout(quads) in;

layout(location = 0) in vec3 inPosition_TES[];
layout(location = 1) in vec2 inTexCoord_TES[];

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec2 outTexCoord;
layout(location = 2) out float outHeight;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 cameraPosition;
    float minZ;
    float maxZ;
} ubo;

layout(binding = 1) uniform sampler2D heightMap;

void main()
{
    float u = gl_TessCoord.x;
    float v = gl_TessCoord.y;

    vec2 texBottom = mix(inTexCoord_TES[0], inTexCoord_TES[1], u);
    vec2 texTop    = mix(inTexCoord_TES[2], inTexCoord_TES[3], u);
    vec2 texCoord  = mix(texBottom, texTop, v);
    outTexCoord = texCoord;

    float heightSample = texture(heightMap, texCoord).r;
    outHeight = heightSample;

    float heightWorld = mix(ubo.minZ, ubo.maxZ, heightSample);

    vec3 posBottom = mix(inPosition_TES[0], inPosition_TES[1], u);
    vec3 posTop    = mix(inPosition_TES[2], inPosition_TES[3], u);
    vec3 basePos   = mix(posBottom, posTop, v);
    basePos.z += heightWorld;

    outWorldPos = (ubo.model * vec4(basePos, 1.0)).xyz;
    gl_Position = ubo.proj * ubo.view * vec4(outWorldPos, 1.0);
}