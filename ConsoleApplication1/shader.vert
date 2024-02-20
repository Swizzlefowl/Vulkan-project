#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(binding = 2) uniform  lightman{vec3 light;} lightpos;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inNormal;
layout(location = 4) in vec3 instance;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 normal;
layout(location = 3) out float vert_color;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4((inPosition + instance), 1.0);
    normal = mat3(transpose(inverse(ubo.view * ubo.model))) * inNormal;
    float vertColor = max( 0.0, dot( normal, lightpos.light ) ) + 0.1;

    vert_color = vertColor;
    fragColor = inColor;
    fragTexCoord = inTexCoord;
}