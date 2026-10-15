#version 330 core

#pragma once

// Вершинный шейдер UI: вершины квадов приходят в UI-единицах (начало слева
// сверху, ось Y вниз). u_scale и u_offset переводят их в пиксели экрана, затем
// x_ndc = 2*x/width - 1, y_ndc = 1 - 2*y/height.

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec4 in_color;

uniform vec2 u_screen;
uniform float u_scale;
uniform vec2 u_offset;

out vec2 v_uv;
out vec4 v_color;

void main() {
    vec2 pixel = in_position * u_scale + u_offset;
    v_uv = in_uv;
    v_color = in_color;
    gl_Position = vec4(
        2.0 * pixel.x / u_screen.x - 1.0,
        1.0 - 2.0 * pixel.y / u_screen.y,
        0.0,
        1.0
    );
}
