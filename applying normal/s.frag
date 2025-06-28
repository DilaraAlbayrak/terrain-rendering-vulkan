#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragPosition;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 normal = normalize(fragNormal);
    vec3 lightDir = normalize(vec3(0.4, 0.8, 0.6));

    float diff = max(dot(normal, lightDir), 0.0);

    vec3 ambient = vec3(0.15);          
    vec3 lit = ambient + diff * vec3(0.55);

    outColor = vec4(normal, 1.0);
}