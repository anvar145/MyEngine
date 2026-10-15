#version 330 core

// Фрагментный шейдер 3D-моделей MyEngine: forward-освещение (Блинн—Фонг).
//
// Сам расчёт живёт в подключаемой библиотеке освещения:
//
//     #include "lighting.frag"
//
// Она объявляет uniform-массивы источников (структура массивов: позиции,
// направления, цвета, параметры, типы) и функции shade_light/compute_lighting.
// Здесь остаётся то, что относится к материалу: базовый цвет, диффузная
// текстура, запечённый свет и прозрачность.
//
// Варианты шейдера:
//   * DEFAULT — обычный объект;
//   * INSTANCED — матрица модели из атрибутов экземпляра (см. simple.vert);
//   * VERTEX_COLOR — меш принёс вершинные цвета (запечённый свет);
//   * USE_LIGHTMAP — освещение берётся из лайтмапы с UV2-координатами.

#include "common.glsl"

// Позицию камеры объявляем ДО подключения библиотеки освещения: она нужна и
// функции расчёта света, и материалам без освещения. GLSL не допускает
// использования идентификатора до его объявления, поэтому порядок здесь важен.
uniform vec3 u_camera_position;

#include "lighting.frag"

//: Динамическое освещение можно выключить целиком: тогда свет берётся из
//: вершинных цветов или лайтмапы, а кадр не платит за расчёт источников.
#ifndef DYNAMIC_LIGHTING
#define DYNAMIC_LIGHTING 1
#endif

in vec3 v_world_position;
in vec3 v_view_position;
in vec3 v_world_normal;
in vec2 v_uv;
in vec4 v_color;

uniform vec3 u_base_color;
uniform float u_roughness;
uniform float u_metallic;

uniform bool u_use_texture;
uniform sampler2D u_diffuse_texture;
uniform float u_alpha;

#ifdef USE_LIGHTMAP
//: Лайтмапа: запечённое освещение в текстуре. Множитель «белый = не запечено»
//: сохраняется, поэтому смешивание с динамикой не требует ветвлений.
uniform sampler2D u_lightmap;
uniform float u_lightmap_strength;
#endif

out vec4 out_color;

/// Запечённый свет фрагмента: вершинные цвета плюс (опционально) лайтмапа.
vec3 baked_light() {
    vec3 baked = v_color.rgb;
#ifdef USE_LIGHTMAP
    vec3 sampled = texture(u_lightmap, v_uv).rgb;
    baked *= mix(vec3(1.0), sampled, clamp(u_lightmap_strength, 0.0, 1.0));
#endif
    return baked;
}

void main() {
    vec3 albedo = u_base_color;
    if (u_use_texture) {
        albedo *= texture(u_diffuse_texture, v_uv).rgb;
    }

    // Запечённый свет умножается на albedo: белый множитель означает
    // «свет не запечён», поэтому формула работает в обоих режимах.
    albedo *= baked_light();

    vec3 lit;
#if DYNAMIC_LIGHTING
    lit = compute_lighting(
        v_world_position,
        // Двусторонние материалы: у задней грани нормаль смотрит к камере.
        gl_FrontFacing ? v_world_normal : -v_world_normal,
        albedo,
        specular_power(u_roughness),
        specular_strength(u_metallic)
    );
#else
    // Запечённый путь: динамических источников нет вовсе, поэтому кадр не
    // платит ни за цикл по источникам, ни за выборки карты теней. Фоновая
    // подсветка уже учтена при запекании (вершинные цвета/лайтмапа), поэтому
    // u_ambient здесь не добавляется — иначе она применялась бы дважды.
    lit = albedo;
#endif

    out_color = vec4(linear_to_srgb(clamp_low(lit)), u_alpha);
}
