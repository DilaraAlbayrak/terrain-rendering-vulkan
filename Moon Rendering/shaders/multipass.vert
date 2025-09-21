#version 450

// Push constant block to receive the view index from the C++ code
layout(push_constant) uniform PushConstants {
    int viewIndex;
} pushConstants;

// The UBO structure remains the same
layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view[2];
    mat4 proj[2];
    vec4 cameraPosition[2];
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec3 fragPosition;

void main() {
    fragNormal = normalize(inNormal); 
    fragPosition = vec3(ubo.model * vec4(inPosition, 1.0));

    // Use the viewIndex from the push constant instead of gl_ViewIndex
    gl_Position = ubo.proj[pushConstants.viewIndex] * ubo.view[pushConstants.viewIndex] * ubo.model * vec4(inPosition, 1.0);
}