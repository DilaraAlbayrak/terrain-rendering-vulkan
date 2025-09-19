#version 450

layout(vertices = 4) out;

layout(location = 0) in vec3 inPosition[];
layout(location = 1) in vec2 inTexCoord[];

layout(location = 0) out vec3 outPosition[];
layout(location = 1) out vec2 outTexCoord[];

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 cameraPosition;
    float minZ;
    float maxZ;
} ubo;

const float MIN_DISTANCE = 1.0;
const float MAX_DISTANCE = 600.0;
const float MIN_TESS_LEVEL = 1.0;
const float MAX_TESS_LEVEL = 16.0;

void main() {
    outPosition[gl_InvocationID] = inPosition[gl_InvocationID];
    outTexCoord[gl_InvocationID] = inTexCoord[gl_InvocationID];
    gl_out[gl_InvocationID].gl_Position = vec4(inPosition[gl_InvocationID], 1.0);

    if (gl_InvocationID == 0)
    {
        const float MIN_DISTANCE_SQ = MIN_DISTANCE * MIN_DISTANCE;
        const float MAX_DISTANCE_SQ = MAX_DISTANCE * MAX_DISTANCE;

        // dot() instead of length() for distance squared
        vec4 v0 = ubo.view * vec4(inPosition[0], 1.0);
        vec4 v1 = ubo.view * vec4(inPosition[1], 1.0);
        vec4 v2 = ubo.view * vec4(inPosition[2], 1.0);
        vec4 v3 = ubo.view * vec4(inPosition[3], 1.0);

        float d0_sq = dot(v0.xyz, v0.xyz);
        float d1_sq = dot(v1.xyz, v1.xyz);
        float d2_sq = dot(v2.xyz, v2.xyz);
        float d3_sq = dot(v3.xyz, v3.xyz);

        float dist0 = clamp((d0_sq - MIN_DISTANCE_SQ) / (MAX_DISTANCE_SQ - MIN_DISTANCE_SQ), 0.0, 1.0);
        float dist1 = clamp((d1_sq - MIN_DISTANCE_SQ) / (MAX_DISTANCE_SQ - MIN_DISTANCE_SQ), 0.0, 1.0);
        float dist2 = clamp((d2_sq - MIN_DISTANCE_SQ) / (MAX_DISTANCE_SQ - MIN_DISTANCE_SQ), 0.0, 1.0);
        float dist3 = clamp((d3_sq - MIN_DISTANCE_SQ) / (MAX_DISTANCE_SQ - MIN_DISTANCE_SQ), 0.0, 1.0);
    
        float tessLevel0 = mix(MAX_TESS_LEVEL, MIN_TESS_LEVEL, min(dist2, dist0));
        float tessLevel1 = mix(MAX_TESS_LEVEL, MIN_TESS_LEVEL, min(dist0, dist1));
        float tessLevel2 = mix(MAX_TESS_LEVEL, MIN_TESS_LEVEL, min(dist1, dist3));
        float tessLevel3 = mix(MAX_TESS_LEVEL, MIN_TESS_LEVEL, min(dist3, dist2));

        gl_TessLevelOuter[0] = tessLevel0;
        gl_TessLevelOuter[1] = tessLevel1;
        gl_TessLevelOuter[2] = tessLevel2;
        gl_TessLevelOuter[3] = tessLevel3;
    
        gl_TessLevelInner[0] = max(tessLevel1, tessLevel3);
        gl_TessLevelInner[1] = max(tessLevel0, tessLevel2);
    }
}