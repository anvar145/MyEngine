#version 330 core

#pragma once

// Вершинный шейдер без освещения: используется для вспомогательной геометрии
// (сетка редактора, gizmos, отладочные линии).

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat4 u_view_projection;

out vec3 v_world_position;
out vec3 v_view_position;
out vec3 v_world_normal;
out vec2 v_uv;

void main() {
    vec4 world_position = u_model * vec4(in_position, 1.0);

    // Вспомогательная геометрия тоже получает полный набор данных: так один
    // и тот же вершинный шейдер подходит и линиям гизмо, и сетке редактора.
    v_world_position = world_position.xyz;
    v_view_position = (u_view * world_position).xyz;
    v_world_normal = mat3(u_model) * in_normal;
    v_uv = in_uv;

    gl_Position = u_view_projection * world_position;
}
