#version 330 core

#pragma once

// Фрагментный шейдер спрайтов: текстура атласа, умноженная на цвет вершины.
// Атлас хранится в sRGB: железо линеаризует выборку, поэтому смешивание цветов
// идёт в линейном пространстве, как и у остальных материалов.

in vec2 v_uv;
in vec4 v_color;

uniform sampler2D u_texture;
uniform float u_alpha_cutoff;

out vec4 out_color;

void main() {
    vec4 texel = texture(u_texture, v_uv) * v_color;
    if (texel.a < u_alpha_cutoff) {
        discard;
    }
    out_color = texel;
}
