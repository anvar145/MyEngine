#version 330 core

// Fused-проход: bloom + экспозиция + tonemap (+ LUT) + виньетка за один
// полноэкранный проход. HDR-буфер хранит гамма-кодированные значения
// (gamma 2.2) без верхнего клампа, поэтому сжатие идёт в линейном
// пространстве и результат кодируется обратно. Режим linear только
// масштабирует и обрезает. Bloom складывается с кадром до tonemap (в HDR),
// FXAA идёт следом уже по LDR-результату.

uniform sampler2D u_color;
uniform int u_mode;       // 0 linear, 1 reinhard, 2 aces
uniform float u_exposure;

uniform sampler2D u_bloom;       // линейное свечение (RGBA16F)
uniform int u_bloom_on;
uniform float u_bloom_intensity;

uniform int u_use_lut;
uniform sampler2D u_lut;         // полоса 256x16 (16 срезов 16x16)
uniform float u_lut_intensity;   // 0..1: mix(исходный, LUT, intensity)

uniform int u_vignette_on;
uniform float u_inner;
uniform float u_outer;
uniform float u_vignette;        // сила затемнения краёв

in vec2 v_uv;
out vec4 out_color;

const float GAMMA = 2.2;

vec3 aces_fit(vec3 x) {
    // Аппроксимация ACES (Narkowicz).
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

vec3 apply_lut(vec3 color) {
    // Выбор между двумя ближайшими срезами по синему каналу.
    vec3 c = clamp(color, 0.0, 1.0) * 15.0;
    float slice = floor(c.b);
    float next = min(slice + 1.0, 15.0);
    vec2 uv_a = vec2((slice * 16.0 + c.r + 0.5) / 256.0, (c.g + 0.5) / 16.0);
    vec2 uv_b = vec2((next * 16.0 + c.r + 0.5) / 256.0, (c.g + 0.5) / 16.0);
    return mix(texture(u_lut, uv_a).rgb, texture(u_lut, uv_b).rgb, c.b - slice);
}

void main() {
    vec3 encoded = texture(u_color, v_uv).rgb;
    if (u_bloom_on != 0) {
        vec3 glow = texture(u_bloom, v_uv).rgb * u_bloom_intensity;
        vec3 lin = pow(max(encoded, vec3(0.0)), vec3(GAMMA)) + glow;
        encoded = pow(lin, vec3(1.0 / GAMMA));
    }
    vec3 rgb;
    if (u_mode == 0) {
        // Экспозиция на гамма-кодированных значениях: (x^2.2 * e)^(1/2.2) = x * e^(1/2.2).
        rgb = clamp(encoded * pow(u_exposure, 1.0 / GAMMA), 0.0, 1.0);
    } else {
        vec3 linear_color = pow(max(encoded, vec3(0.0)), vec3(GAMMA)) * u_exposure;
        if (u_mode == 1) {
            linear_color = linear_color / (1.0 + linear_color);
        } else {
            linear_color = aces_fit(linear_color);
        }
        rgb = pow(linear_color, vec3(1.0 / GAMMA));
    }
    if (u_use_lut != 0) {
        rgb = mix(rgb, apply_lut(rgb), u_lut_intensity);
    }
    if (u_vignette_on != 0) {
        // r: 0 в центре, 1 в углу кадра.
        float r = length((v_uv - 0.5) * 2.0) * 0.70710678;
        float edge = smoothstep(u_inner, max(u_outer, u_inner + 1e-4), r);
        rgb *= mix(1.0, 1.0 - u_vignette, edge);
    }
    out_color = vec4(rgb, 1.0);
}
