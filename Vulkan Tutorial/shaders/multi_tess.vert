#version 450

// Push constant block to receive the view index from the C++ code
layout(push_constant) uniform PushConstants {
    int viewIndex;
} pushConstants;

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
    gl_Position = ubo.proj[pushConstants.viewIndex] * ubo.view[pushConstants.viewIndex] * ubo.model * vec4(inPosition, 1.0);
}