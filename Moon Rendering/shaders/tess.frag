#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragPosition;
layout(location = 2) in vec3 fragViewDir;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 normal = normalize(fragNormal);
    vec3 lightDir = normalize(vec3(0.0, 4.0, 0.0));
    vec3 viewDir = normalize(fragViewDir);
    vec3 reflectDir = reflect(-lightDir, normal);

    float diff = max(dot(normal, lightDir), 0.0);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);

    vec3 ambient = vec3(0.05);
    vec3 diffuse = diff * vec3(0.25);
    vec3 specular = spec * vec3(0.4);

    vec3 finalColour = ambient + diffuse + specular;

    outColor = vec4(finalColour, 1.0);
}