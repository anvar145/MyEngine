#version 330 core

#pragma once

// Фрагментный шейдер без освещения: сплошной цвет (u_base_color), при
// необходимости с альфой. Совместим по uniform-переменным с обычным материалом,
// поэтому один и тот же Material подходит разным программам.

in vec3 v_world_position;
in vec3 v_view_position;
in vec3 v_world_normal;
in vec2 v_uv;

uniform vec3 u_base_color;
uniform float u_alpha;

out vec4 out_color;

void main() {
    // v_view_position — позиция во view-space: заготовка для эффектов
    // глубины (туман, контур выделения) во вспомогательной геометрии.
    out_color = vec4(u_base_color, u_alpha);
}
