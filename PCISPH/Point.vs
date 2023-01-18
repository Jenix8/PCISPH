#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec4 aColor;

out vec4 color;
out float fragDepth;

uniform mat4 view;
uniform mat4 projection;

void main() {
    color = aColor;

    gl_Position = projection * view * vec4(aPos, 1.0f);
    fragDepth = gl_Position.z / gl_Position.w;
    gl_PointSize = (1.0 - fragDepth) * 250.0;
}