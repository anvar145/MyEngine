#version 330 core

#pragma once

// Полноэкранный треугольник без VBO: вершины берутся из gl_VertexID.
// z = 1 (дальняя плоскость): небо рисуется только там, где нет геометрии.

out vec2 v_ndc;

void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_ndc = p * 2.0 - 1.0;
    gl_Position = vec4(v_ndc, 1.0, 1.0);
}
