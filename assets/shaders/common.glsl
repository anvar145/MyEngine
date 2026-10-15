// Общие функции шейдеров MyEngine.
// Подключается загрузчиком через #include и вставляется в исходник.

#ifndef MYENGINE_COMMON
#define MYENGINE_COMMON

const float PI = 3.14159265359;

/// Преобразовать линейный цвет в гамма-пространство (sRGB).
vec3 linear_to_srgb(vec3 color) {
    return pow(color, vec3(1.0 / 2.2));
}

/// Простая функция насыщенности для пост-обработки.
vec3 saturate_color(vec3 color) {
    return clamp(color, 0.0, 1.0);
}

/// Отсечь только отрицательные значения (верх не клампится — HDR-кадр;
/// в RGBA8-цель аппаратная запись всё равно обрежет до 1).
vec3 clamp_low(vec3 color) {
    return max(color, vec3(0.0));
}

#endif
