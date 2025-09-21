#version 450

layout(quads) in;

layout(location = 0) in vec3 inPosition_TES[];
layout(location = 1) in vec2 inTexCoord_TES[];

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec3 fragPosition;
layout(location = 2) out vec3 fragViewDir;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 cameraPosition;
    float minZ;
    float maxZ;
} ubo;

layout(binding = 1) uniform sampler2D heightMap;

vec3 calculateTangentSpaceNormal(vec2 texCoord)
{
    vec2 offset = 1.0 / textureSize(heightMap, 0);
    float hL = texture(heightMap, texCoord - vec2(offset.x, 0)).r;
    float hR = texture(heightMap, texCoord + vec2(offset.x, 0)).r;
    float hD = texture(heightMap, texCoord - vec2(0, offset.y)).r;
    float hU = texture(heightMap, texCoord + vec2(0, offset.y)).r;
    
    float scale = ubo.maxZ - ubo.minZ;
    
    //vec3 N = vec3(scale * (hL - hR), scale * (hD - hU), 2.0 * offset.x);
    vec3 N = vec3(scale * (hL - hR), scale * (hD - hU), 2.0);
    return normalize(N);
}

void main()
{
    float u = gl_TessCoord.x;
    float v = gl_TessCoord.y;
    vec2 texCoord = mix(mix(inTexCoord_TES[0], inTexCoord_TES[1], u), mix(inTexCoord_TES[2], inTexCoord_TES[3], u), v);
    vec3 basePos = mix(mix(inPosition_TES[0], inPosition_TES[1], u), mix(inPosition_TES[2], inPosition_TES[3], u), v);

    float heightSample = texture(heightMap, texCoord).r;
    basePos.z += mix(ubo.minZ, ubo.maxZ, heightSample);

    vec3 worldPos = vec3(ubo.model * vec4(basePos, 1.0));
    fragPosition = worldPos;

    fragViewDir = normalize(ubo.cameraPosition.xyz - worldPos);

    vec3 tangentNormal = calculateTangentSpaceNormal(texCoord);
    fragNormal = normalize(mat3(transpose(inverse(ubo.model))) * tangentNormal);
    
    gl_Position = ubo.proj * ubo.view * vec4(worldPos, 1.0);
}