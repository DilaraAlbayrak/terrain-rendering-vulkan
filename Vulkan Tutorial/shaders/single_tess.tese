#version 450
#extension GL_EXT_multiview : require

layout(quads) in;

layout(location = 0) in vec3 inPosition_TES[];
layout(location = 1) in vec2 inTexCoord_TES[];

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec3 fragPosition;
layout(location = 2) out vec3 fragViewDir;

layout(std140, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view[2];
    mat4 proj[2];
    vec4 cameraPosition[2];
    vec4 minMaxZ;
} ubo;

layout(binding = 1) uniform sampler2D heightMap;

vec3 calculateTangentSpaceNormal(vec2 texCoord)
{
    vec2 offset = 1.0 / textureSize(heightMap, 0);
    float hL = texture(heightMap, texCoord - vec2(offset.x, 0)).r;
    float hR = texture(heightMap, texCoord + vec2(offset.x, 0)).r;
    float hD = texture(heightMap, texCoord - vec2(0, offset.y)).r;
    float hU = texture(heightMap, texCoord + vec2(0, offset.y)).r;
    
    float scale = ubo.minMaxZ.y - ubo.minMaxZ.x;
    
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
    basePos.z += mix(ubo.minMaxZ.x, ubo.minMaxZ.y, heightSample);

    vec3 worldPos = vec3(ubo.model * vec4(basePos, 1.0));
    fragPosition = worldPos;

    // Use gl_ViewIndex instead of pushConstants.viewIndex
    fragViewDir = normalize(ubo.cameraPosition[gl_ViewIndex].xyz - worldPos);

    vec3 tangentNormal = calculateTangentSpaceNormal(texCoord);
    fragNormal = normalize(mat3(transpose(inverse(ubo.model))) * tangentNormal);
    
    // Use gl_ViewIndex instead of pushConstants.viewIndex
    gl_Position = ubo.proj[gl_ViewIndex] * ubo.view[gl_ViewIndex] * ubo.model * vec4(basePos, 1.0);
}