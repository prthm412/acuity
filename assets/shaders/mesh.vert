#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

// Push Constants
layout(push_constant) uniform PushConstants {
    mat4 model;         // object-to-world transform
    mat4 view;          // world-to-camera transform
    mat4 projection;    // camera-to-clip transform (perspective)
    int  lodVisualization;
} pc;

// Outputs (passed to fragment shader)
layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec3 fragColor;
layout(location = 2) out vec3 fragWorldPos;

void main() {
    // Transform vertex position through MVP matrices
    // gl_Position is the built-in output: final clip-space position
    vec4 worldPos = pc.model * vec4(inPosition, 1.0);
    gl_Position   = pc.projection * pc.view * worldPos;

    // Pass to fragment shader
    fragWorldPos = worldPos.xyz;
    fragNormal   = mat3(transpose(inverse(pc.model))) * inNormal;
    
    // In normal mode use flat grey, in LOD mode use baked vertex color
    if (pc.lodVisualization == 1) {
        fragColor = inColor;
    } else {
        fragColor = vec3(0.35, 0.35, 0.35);
    }
}