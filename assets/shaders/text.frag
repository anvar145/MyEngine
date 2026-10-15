#version 330 core

#pragma once

// Фрагментный шейдер текста: атлас глифов (белый RGB, альфа — покрытие),
// умноженный на цвет вершины и общую прозрачность. Смешивание — обычное
// SRC_ALPHA / ONE_MINUS_SRC_ALPHA, запись глубины выключена.

in vec2 v_uv;
in vec4 v_color;

uniform sampler2D u_atlas;
uniform float u_alpha;

out vec4 out_color;

void main() {
    out_color = texture(u_atlas, v_uv) * v_color * u_alpha;
}
