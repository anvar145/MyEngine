#version 330 core

#pragma once

// Вершинный шейдер 3D-моделей MyEngine.
//
// Поддерживает два режима:
//   * обычный — матрица модели приходит uniform-переменной u_model;
//   * инстансный (ключевое слово INSTANCED) — матрица модели читается из
//     атрибутов экземпляра (locations 8..11), что позволяет рисовать тысячи
//     объектов одним вызовом отрисовки.

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

// Вершинный цвет: у процедурных мешей его нет, поэтому атрибут объявлен только
// в варианте VERTEX_COLOR. Иначе драйвер отдавал бы в шейдер произвольные
// данные (атрибут не включён) и запечённый свет «портил» бы картинку.
#ifdef VERTEX_COLOR
layout(location = 3) in vec4 in_color;
#endif

#ifdef INSTANCED
layout(location = 8) in vec4 in_instance_row0;
layout(location = 9) in vec4 in_instance_row1;
layout(location = 10) in vec4 in_instance_row2;
layout(location = 11) in vec4 in_instance_row3;
#endif

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat4 u_view_projection;

out vec3 v_world_position;
out vec3 v_view_position;
out vec3 v_world_normal;
out vec2 v_uv;
//: Запечённый свет (вершинные цвета). Белый цвет означает «свет не запечён»,
//: поэтому шейдер может умножать albedo на этот множитель без условий.
out vec4 v_color;

/// Собрать матрицу модели: из uniform-переменной или из атрибутов экземпляра.
mat4 model_matrix() {
#ifdef INSTANCED
    return mat4(
        in_instance_row0,
        in_instance_row1,
        in_instance_row2,
        in_instance_row3
    );
#else
    return u_model;
#endif
}

void main() {
    mat4 model = model_matrix();
    vec4 world_position = model * vec4(in_position, 1.0);

    v_world_position = world_position.xyz;
    // Позиция в пространстве камеры: нужна эффектам глубины, туману и
    // отладочной визуализации — считается отдельной матрицей вида.
    v_view_position = (u_view * world_position).xyz;
    // Нормаль преобразуется обратной транспонированной матрицей; для
    // однородного масштаба достаточно верхней части 3x3.
    v_world_normal = mat3(model) * in_normal;
    v_uv = in_uv;
#ifdef VERTEX_COLOR
    v_color = in_color;
#else
    v_color = vec4(1.0);
#endif

    gl_Position = u_view_projection * world_position;
}
