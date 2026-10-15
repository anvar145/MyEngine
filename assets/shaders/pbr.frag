#version 330 core

// Фрагментный шейдер PBR-материалов MyEngine: металличность/шероховатость.
//
// Модель: Cook—Torrance (GGX + геометрия Смита—Шлика + Френель—Шлик) для
// прямых источников и дешёвая аналитическая аппроксимация для фоновой
// подсветки (вместо IBL: на слабых GPU карты окружения не по карману).
// Вершинный шейдер общий с simple.frag (``simple.vert``).
//
// Источники света и тени — из той же библиотеки, что у Блинна—Фонга:
//
//     #include "lighting.frag"
//
// Текстуры материала (юниты 0, 3, 4, 5; юнит 1 — карта теней):
//   * u_diffuse_texture — base color (RGBA, sRGB декодируется железом);
//   * u_normal_texture  — карта нормалей (касательное пространство,
//                         строится по производным, вершинные касательные
//                         не нужны);
//   * u_orm_texture     — R = occlusion, G = roughness, B = metallic
//                         (линейная; её собирает material cooker);
//   * u_emissive_texture — свечение (sRGB).
//
// Варианты шейдера:
//   * INSTANCED     — матрица модели из атрибутов экземпляра;
//   * VERTEX_COLOR  — вершинный цвет умножает base color;
//   * ALPHA_MASK    — отсечение фрагментов с альфой ниже u_alpha_cutoff;
//   * DYNAMIC_LIGHTING=0 — без динамических источников (только фон).

#include "common.glsl"

uniform vec3 u_camera_position;

#include "lighting.frag"

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
uniform float u_alpha;

uniform bool u_use_texture;
uniform sampler2D u_diffuse_texture;
uniform bool u_has_normal_map;
uniform sampler2D u_normal_texture;
uniform bool u_has_orm_map;
uniform sampler2D u_orm_texture;
uniform bool u_has_emissive_map;
uniform sampler2D u_emissive_texture;

uniform vec3 u_emissive;
uniform float u_normal_scale;
uniform float u_occlusion_strength;
uniform int u_alpha_mode;          // 0 — непрозрачный, 1 — маска, 2 — смешивание
uniform float u_alpha_cutoff;
uniform vec2 u_uv_scale;
uniform vec2 u_uv_offset;

out vec4 out_color;

//: Нижняя граница шероховатости: при нуле GGX вырождается в точку.
const float MIN_ROUGHNESS = 0.045;
const vec3 DIELECTRIC_F0 = vec3(0.04);

/// Нормаль с учётом карты нормалей.
///
/// Касательный базис строится по производным позиции и UV (Шюлер, 2013):
/// меши не хранят касательных, а базис по производным даёт ту же картинку.
vec3 perturbed_normal(vec3 normal, vec2 uv) {
    if (!u_has_normal_map) {
        return normal;
    }
    vec3 dp1 = dFdx(v_world_position);
    vec3 dp2 = dFdy(v_world_position);
    vec2 duv1 = dFdx(uv);
    vec2 duv2 = dFdy(uv);
    vec3 dp2_perp = cross(dp2, normal);
    vec3 dp1_perp = cross(normal, dp1);
    vec3 tangent = dp2_perp * duv1.x + dp1_perp * duv2.x;
    vec3 bitangent = dp2_perp * duv1.y + dp1_perp * duv2.y;
    float scale = inversesqrt(max(max(dot(tangent, tangent), dot(bitangent, bitangent)), 1e-12));

    vec3 sampled = texture(u_normal_texture, uv).xyz * 2.0 - 1.0;
    sampled.xy *= u_normal_scale;
    mat3 basis = mat3(tangent * scale, bitangent * scale, normal);
    return normalize(basis * sampled);
}

/// Распределение микрограней GGX (Trowbridge—Reitz).
float distribution_ggx(float n_dot_h, float alpha) {
    float a2 = alpha * alpha;
    float d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

/// Видимость Смита—Шлика (геометрия, уже делённая на 4·N·L·N·V).
float visibility_smith_schlick(float n_dot_v, float n_dot_l, float roughness) {
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float gv = n_dot_v * (1.0 - k) + k;
    float gl = n_dot_l * (1.0 - k) + k;
    return 0.25 / max(gv * gl, 1e-4);
}

/// Приближение Френеля Шлика.
vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(1.0 - clamp(cos_theta, 0.0, 1.0), 5.0);
}

/// Аналитический «BRDF окружения» (Лазарев, 2013): масштаб и смещение F0.
vec3 environment_brdf(vec3 f0, float roughness, float n_dot_v) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    vec2 ab = vec2(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

void main() {
    vec2 uv = v_uv * u_uv_scale + u_uv_offset;

    vec4 base = vec4(u_base_color, u_alpha) * v_color;
    if (u_use_texture) {
        base *= texture(u_diffuse_texture, uv);
    }
#ifdef ALPHA_MASK
    if (base.a < u_alpha_cutoff) {
        discard;
    }
#endif

    float occlusion = 1.0;
    float roughness = u_roughness;
    float metallic = u_metallic;
    if (u_has_orm_map) {
        vec3 orm = texture(u_orm_texture, uv).rgb;
        occlusion = mix(1.0, orm.r, clamp(u_occlusion_strength, 0.0, 1.0));
        roughness *= orm.g;
        metallic *= orm.b;
    }
    roughness = clamp(roughness, MIN_ROUGHNESS, 1.0);
    metallic = clamp(metallic, 0.0, 1.0);

    vec3 geometric_normal = normalize(v_world_normal);
    if (!gl_FrontFacing) {
        geometric_normal = -geometric_normal;  // двусторонние материалы
    }
    vec3 normal = perturbed_normal(geometric_normal, uv);
    vec3 view_dir = normalize(u_camera_position - v_world_position);
    float n_dot_v = max(dot(normal, view_dir), 1e-4);

    vec3 albedo = base.rgb;
    vec3 f0 = mix(DIELECTRIC_F0, albedo, metallic);
    vec3 diffuse_color = albedo * (1.0 - metallic);
    float alpha = roughness * roughness;

    // Фон: диффузная часть и грубая зеркальная (без карты окружения).
#if DYNAMIC_LIGHTING
    vec3 lit = u_ambient * occlusion
        * (diffuse_color + environment_brdf(f0, roughness, n_dot_v));
#else
    // Запечённый свет (включая ambient) уже в v_color, а значит в albedo.
    vec3 lit = occlusion * diffuse_color;
#endif

#if DYNAMIC_LIGHTING
    ShadowConfig shadow = shadow_config();
    float shadow_term = shadow_factor(u_shadow_map, shadow, v_world_position, geometric_normal);

    for (int index = 0; index < MAX_LIGHTS; ++index) {
        if (index >= u_light_count) {
            break;
        }
        vec3 to_light;
        vec3 radiance;
        if (!light_incidence(index, v_world_position, geometric_normal, to_light, radiance)) {
            continue;
        }
        float n_dot_l = max(dot(normal, to_light), 0.0);
        if (n_dot_l <= 0.0) {
            continue;
        }
        vec3 half_dir = normalize(to_light + view_dir);
        float n_dot_h = max(dot(normal, half_dir), 0.0);
        float v_dot_h = max(dot(view_dir, half_dir), 0.0);

        vec3 fresnel = fresnel_schlick(v_dot_h, f0);
        float d = distribution_ggx(n_dot_h, alpha);
        float vis = visibility_smith_schlick(n_dot_v, n_dot_l, roughness);
        vec3 specular = fresnel * (d * vis);
        vec3 diffuse = (1.0 - fresnel) * diffuse_color;

        // Яркости источников в движке заданы под модель Ламберта без деления
        // на пи (albedo * свет), поэтому диффузная часть тоже без 1/пи, а
        // зеркальная умножена на пи: так PBR и Блинн—Фонг одинаково ярки.
        vec3 contribution = (diffuse + PI * specular) * radiance * n_dot_l;
        if (u_light_type[index] == LIGHT_DIRECTIONAL) {
            contribution *= shadow_term;
        }
        lit += contribution;
    }
#endif

    vec3 emissive = u_emissive;
    if (u_has_emissive_map) {
        emissive *= texture(u_emissive_texture, uv).rgb;
    }
    lit += emissive;

    float out_alpha = (u_alpha_mode == 2) ? base.a : 1.0;
    out_color = vec4(linear_to_srgb(clamp_low(lit)), out_alpha);
}
