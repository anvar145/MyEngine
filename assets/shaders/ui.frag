#version 330 core

#pragma once

// Фрагментный шейдер UI: текстура (для простых квадов — белый пиксель),
// умноженная на цвет вершины. Смешивание — SRC_ALPHA / ONE_MINUS_SRC_ALPHA.

in vec2 v_uv;
in vec4 v_color;

uniform sampler2D u_texture;

out vec4 out_color;

void main() {
    out_color = texture(u_texture, v_uv) * v_color;
}
