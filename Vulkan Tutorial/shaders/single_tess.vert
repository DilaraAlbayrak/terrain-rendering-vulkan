#version 450
#extension GL_EXT_multiview : require

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view[2];
    mat4 proj[2];
    vec4 cameraPosition[2];
    vec4 minMaxZ;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 3) in vec2 inTexCoord;

layout(location = 0) out vec3 outPosition;
layout(location = 1) out vec2 outTexCoord;

void main() {
    outPosition = inPosition;
    outTexCoord = inTexCoord;
    gl_Position = ubo.proj[gl_ViewIndex] * ubo.view[gl_ViewIndex] * ubo.model * vec4(inPosition, 1.0);
}