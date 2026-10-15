#version 330 core

// Bright-pass bloom: из HDR-цели в половинное разрешение. HDR-буфер хранит
// гамма-кодированные значения, порог задаётся в линейных единицах, результат
// пишется в линейном пространстве. Четыре билинейных выборки усредняют блок
// 4x4 исходных пикселей (меньше мерцания «светлячков»).

uniform sampler2D u_color;
uniform vec2 u_texel;       // 1 / размер исходной цели
uniform float u_threshold;  // линейная яркость, с которой начинается свечение
uniform float u_soft_knee;  // ширина плавного перехода

in vec2 v_uv;
out vec4 out_color;

const float GAMMA = 2.2;

vec3 bright(vec2 uv) {
    vec3 c = pow(max(texture(u_color, uv).rgb, vec3(0.0)), vec3(GAMMA));
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    return c * smoothstep(u_threshold, u_threshold + max(u_soft_knee, 1e-4), luma);
}

void main() {
    vec2 o = u_texel;
    vec3 sum = bright(v_uv + vec2(-o.x, -o.y)) + bright(v_uv + vec2(o.x, -o.y))
             + bright(v_uv + vec2(-o.x, o.y)) + bright(v_uv + vec2(o.x, o.y));
    out_color = vec4(sum * 0.25, 1.0);
}
