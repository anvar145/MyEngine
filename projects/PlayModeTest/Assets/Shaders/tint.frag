#version 330 core
in vec2 v_uv;
out vec4 frag_color;
uniform vec4 u_tint;
void main() { frag_color = u_tint * vec4(v_uv, 0.5, 1.0); }
