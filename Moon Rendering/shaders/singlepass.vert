#version 450
#extension GL_EXT_multiview : require

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
layout(location = 2) out vec3 fragViewDir;

void main() {
    fragNormal = normalize(inNormal); 
    fragPosition = vec3(ubo.model * vec4(inPosition, 1.0));
    fragViewDir = normalize(ubo.cameraPosition[gl_ViewIndex].xyz - fragPosition); 
    gl_Position = ubo.proj[gl_ViewIndex] * ubo.view[gl_ViewIndex] * ubo.model * vec4(inPosition, 1.0);
}