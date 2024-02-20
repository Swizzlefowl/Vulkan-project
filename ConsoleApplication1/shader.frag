#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 normal;
layout(location = 3) in float vert_color;

layout(set = 0, binding = 1) uniform sampler2D texSampler;
layout(location = 0) out vec4 outColor;

void main() {
     //outColor = vec4(normal, 1.0f);
     //outColor = vec4(vert_color * fragColor, 1.0);
     outColor = vert_color * texture(texSampler, fragTexCoord);
}