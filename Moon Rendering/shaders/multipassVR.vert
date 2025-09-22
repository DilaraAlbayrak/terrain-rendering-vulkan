#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec3 fragPosition;
layout(location = 2) out vec3 fragViewDir;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view[2];
    mat4 proj[2];
    vec4 cameraPosition[2];
} ubo;

layout(push_constant) uniform PushConstants {
    int viewIndex;
} pushConstants;

void main() {
    vec4 worldPosition = ubo.model * vec4(inPosition, 1.0);
    fragPosition = worldPosition.xyz;

    // world-space normal
    mat3 normalMatrix = transpose(inverse(mat3(ubo.model)));
    fragNormal = normalize(normalMatrix * inNormal);

    // Camera view direction in world space
    fragViewDir = normalize(ubo.cameraPosition[pushConstants.viewIndex].xyz - worldPosition.xyz);

    // clip-space position
    gl_Position = ubo.proj[pushConstants.viewIndex] * ubo.view[pushConstants.viewIndex] * worldPosition;
}