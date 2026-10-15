#version 330 core

// Вершинный шейдер прохода теней (directional light, одна карта на сцену).
//
// Задача прохода — только записать глубину со стороны источника света, поэтому
// шейдер предельно дешёвый: ни нормалей, ни UV, ни вычислений освещения. На
// слабых GPU это важно: карта 512×512 рисуется второй раз по всей геометрии,
// и лишняя работа в вершинном шейдере умножается на число вершин.
//
// Поддерживает тот же вариант INSTANCED, что и основной шейдер: инстансные
// пачки (трава, деревья, пропсы) обязаны отбрасывать тень без распада на
// отдельные вызовы отрисовки.

#pragma once

layout(location = 0) in vec3 in_position;

#ifdef INSTANCED
layout(location = 8) in vec4 in_instance_row0;
layout(location = 9) in vec4 in_instance_row1;
layout(location = 10) in vec4 in_instance_row2;
layout(location = 11) in vec4 in_instance_row3;
#endif

uniform mat4 u_model;
// Мир → пространство источника: даёт clip-координаты и глубину в 0..1.
uniform mat4 u_light_space_matrix;

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
    gl_Position = u_light_space_matrix * model_matrix() * vec4(in_position, 1.0);
}
