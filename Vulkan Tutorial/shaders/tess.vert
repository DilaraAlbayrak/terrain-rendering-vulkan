#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 3) in vec2 inTexCoord;

layout(location = 0) out vec3 outPosition;
layout(location = 1) out vec2 outTexCoord;

void main() {
    outPosition = inPosition;
    outTexCoord = inTexCoord;
    gl_Position = vec4(inPosition, 1.0);
}