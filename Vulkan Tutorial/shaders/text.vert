#version 450
layout(location = 0) in vec2 inPos;

void main() {
    gl_Position = vec4((inPos.x / 1200.0) * 2.0 - 1.0, 1.0 - (inPos.y / 900.0) * 2.0, 0.0, 1.0);
}