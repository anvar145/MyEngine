#version 330 core

#pragma once

// Отладочные линии: цвет вершины без освещения.

in vec3 v_color;

out vec4 out_color;

void main() {
    out_color = vec4(v_color, 1.0);
}
