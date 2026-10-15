#version 330 core

#pragma once

// Отладочные линии: вершины уже в мировых координатах, цвет — на вершину.

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

uniform mat4 u_view_projection;

out vec3 v_color;

void main() {
    v_color = in_color;
    gl_Position = u_view_projection * vec4(in_position, 1.0);
}
