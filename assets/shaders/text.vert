#version 330 core

#pragma once

// Вершинный шейдер текста: вершины квадов уже в мировых координатах (их
// собирает TextBatch на CPU, в том числе billboard-ориентацию), u_model —
// дополнительная матрица (по умолчанию единичная).

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec4 in_color;

uniform mat4 u_view_projection;
uniform mat4 u_model;

out vec2 v_uv;
out vec4 v_color;

void main() {
    v_uv = in_uv;
    v_color = in_color;
    gl_Position = u_view_projection * u_model * vec4(in_position, 1.0);
}
