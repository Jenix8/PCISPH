#version 330 core
out vec4 FragColor;

in vec4 color;
in float fragDepth;

void main() {
    //vec2 Point = gl_PointCoord - vec2(0.5f, 0.5f);
    //float d = Point.x * Point.x + Point.y * Point.y;
    //if (d > 0.25f) discard;

    vec3 N;
    N.xy = gl_PointCoord.xy * 2.0f - 1.0f;
    float r2 = dot(N.xy, N.xy);
    if(r2 > 1.0f) discard;
    N.z = sqrt(1.0 - r2);

    if (color.a < 0.1f) discard;

    FragColor = color;
}