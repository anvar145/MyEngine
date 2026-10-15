#version 330 core

// Увеличение меньшего уровня bloom; аддитивно смешивается с большим уровнем.

uniform sampler2D u_color;

in vec2 v_uv;
out vec4 out_color;

void main() {
    out_color = vec4(texture(u_color, v_uv).rgb, 1.0);
}
