#version 450

// Input from vertex shader
layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragColor;
layout(location = 2) in vec3 fragWorldPos;

// Output
layout(location = 0) out vec4 outColor;

void main() {
    // Normalize interpolated normal (interpolation can denormalize it)
    vec3 normal = normalize(fragNormal);

    // Simple directional light
    // Light coming from upper-right-front direction
    vec3 lightDir   = normalize(vec3(1.0, 2.0, 1.0));
    vec3 lightColor = vec3(1.0, 1.0, 1.0);

    // Ambient: base illumination so no part is completely black
    float ambientStrength   = 0.2;
    vec3 ambient            = ambientStrength * lightColor;

    // Diffuse: Lambertian shading - brightness depends on angle to light
    float diff      = max(dot(normal, lightDir), 0.0);
    vec3 diffuse    = diff * lightColor;

    // Final color
    vec3 result = (ambient + diffuse) * fragColor;
    outColor    = vec4(result, 1.0);
}