#version 330 core
out vec4 FragColor;
//uniform vec3 objectColor;

in vec4 color;

void main() {
    vec2 Point = gl_PointCoord - vec2(0.5f, 0.5f);
    float d = Point.x * Point.x + Point.y * Point.y;
    if (d > 0.25f) discard;
    if (color.a < 0.1f) discard;
    //FragColor = vec4(objectColor, 1.0f);
    FragColor = color;
}