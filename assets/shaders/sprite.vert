#version 330 core

#pragma once

// Вершинный шейдер спрайтов: вершины квадов уже в мировых координатах
// (их собирает SpriteBatch на CPU одним numpy-вызовом), цвет — на вершину.

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec4 in_color;

uniform mat4 u_view_projection;

out vec2 v_uv;
out vec4 v_color;

void main() {
    v_uv = in_uv;
    v_color = in_color;
    gl_Position = u_view_projection * vec4(in_position, 1.0);
}
