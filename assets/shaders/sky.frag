#version 330 core

#pragma once

// Процедурное небо: градиент горизонт -> зенит, земля под горизонтом,
// диск солнца и его свечение. Без текстур и спектральных моделей: дёшево
// для интегрированной графики.

in vec2 v_ndc;

uniform mat4 u_inv_view_projection;
uniform vec3 u_sun_direction;   // нормализовано: куда светит солнце
uniform vec3 u_sun_color;
uniform vec3 u_zenith_color;
uniform vec3 u_horizon_color;
uniform vec3 u_ground_color;
uniform float u_sun_size;       // угловой диаметр диска, градусы
uniform float u_sun_glow;       // степень свечения
uniform float u_exposure;
uniform float u_intensity;
uniform int u_mode;             // 0 - процедурное, 1 - сплошной цвет

out vec4 frag_color;

void main() {
    vec4 near_point = u_inv_view_projection * vec4(v_ndc, -1.0, 1.0);
    vec4 far_point = u_inv_view_projection * vec4(v_ndc, 1.0, 1.0);
    vec3 dir = normalize(far_point.xyz / far_point.w - near_point.xyz / near_point.w);

    vec3 sky;
    if (u_mode == 1) {
        sky = u_zenith_color;
    } else {
        float h = dir.y;
        sky = mix(u_horizon_color, u_zenith_color, pow(max(h, 0.0), 0.5));
        sky = mix(u_ground_color, sky, smoothstep(-0.05, 0.05, h));

        float cos_angle = dot(dir, -u_sun_direction);
        float angle = degrees(acos(clamp(cos_angle, -1.0, 1.0)));
        float disk = 1.0 - smoothstep(u_sun_size * 0.5, u_sun_size, angle);
        float glow = pow(max(cos_angle, 0.0), u_sun_glow);
        sky += u_sun_color * (disk * 8.0 + glow * 0.6);
    }

    sky = vec3(1.0) - exp(-sky * u_exposure);
    frag_color = vec4(sky * u_intensity, 1.0);
}
