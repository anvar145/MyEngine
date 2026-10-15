#version 330 core

// FXAA 3.11 (упрощённый вариант quality-пресета) над LDR-кадром после tonemap.
// Яркость считается по зелёному-красному-синему без альфа-трюка, поэтому
// входная цель — обычный RGBA8.

uniform sampler2D u_color;
uniform vec2 u_texel;  // 1 / размер входной текстуры

in vec2 v_uv;
out vec4 out_color;

const float EDGE_THRESHOLD = 0.125;
const float EDGE_THRESHOLD_MIN = 0.0312;
const float SUBPIX_QUALITY = 0.75;
const int SEARCH_STEPS = 10;

float luma(vec3 rgb) {
    return dot(rgb, vec3(0.299, 0.587, 0.114));
}

float step_for(int i) {
    // Шаги поиска конца ребра: 1, 1, 1, 1, 1.5, 2, 2, 2, 4, 8.
    if (i < 4) return 1.0;
    if (i == 4) return 1.5;
    if (i < 8) return 2.0;
    if (i == 8) return 4.0;
    return 8.0;
}

void main() {
    vec3 center = texture(u_color, v_uv).rgb;
    float l_c = luma(center);
    float l_s = luma(textureOffset(u_color, v_uv, ivec2(0, -1)).rgb);
    float l_n = luma(textureOffset(u_color, v_uv, ivec2(0, 1)).rgb);
    float l_w = luma(textureOffset(u_color, v_uv, ivec2(-1, 0)).rgb);
    float l_e = luma(textureOffset(u_color, v_uv, ivec2(1, 0)).rgb);

    float l_min = min(l_c, min(min(l_s, l_n), min(l_w, l_e)));
    float l_max = max(l_c, max(max(l_s, l_n), max(l_w, l_e)));
    float range = l_max - l_min;
    if (range < max(EDGE_THRESHOLD_MIN, l_max * EDGE_THRESHOLD)) {
        out_color = vec4(center, 1.0);
        return;
    }

    float l_sw = luma(textureOffset(u_color, v_uv, ivec2(-1, -1)).rgb);
    float l_se = luma(textureOffset(u_color, v_uv, ivec2(1, -1)).rgb);
    float l_nw = luma(textureOffset(u_color, v_uv, ivec2(-1, 1)).rgb);
    float l_ne = luma(textureOffset(u_color, v_uv, ivec2(1, 1)).rgb);

    float l_ns = l_n + l_s;
    float l_we = l_w + l_e;
    float l_corners = l_sw + l_se + l_nw + l_ne;

    // Горизонтальное или вертикальное ребро.
    float edge_h = abs(-2.0 * l_w + (l_nw + l_sw)) + abs(-2.0 * l_c + l_ns) * 2.0
                 + abs(-2.0 * l_e + (l_ne + l_se));
    float edge_v = abs(-2.0 * l_n + (l_nw + l_ne)) + abs(-2.0 * l_c + l_we) * 2.0
                 + abs(-2.0 * l_s + (l_sw + l_se));
    bool is_horizontal = edge_h >= edge_v;

    float l_neg = is_horizontal ? l_s : l_w;
    float l_pos = is_horizontal ? l_n : l_e;
    float grad_neg = abs(l_neg - l_c);
    float grad_pos = abs(l_pos - l_c);
    bool negative_side = grad_neg >= grad_pos;
    float gradient = max(grad_neg, grad_pos);
    float l_edge_avg = 0.5 * ((negative_side ? l_neg : l_pos) + l_c);

    float step_len = is_horizontal ? u_texel.y : u_texel.x;
    if (negative_side) step_len = -step_len;

    // Субпиксельное сглаживание.
    float l_avg = (2.0 * (l_ns + l_we) + l_corners) / 12.0;
    float sub_offset = clamp(abs(l_avg - l_c) / range, 0.0, 1.0);
    sub_offset = smoothstep(0.0, 1.0, sub_offset);
    sub_offset = sub_offset * sub_offset * SUBPIX_QUALITY;

    // Поиск концов ребра в обе стороны вдоль него.
    vec2 uv_edge = v_uv;
    if (is_horizontal) uv_edge.y += step_len * 0.5; else uv_edge.x += step_len * 0.5;
    vec2 along = is_horizontal ? vec2(u_texel.x, 0.0) : vec2(0.0, u_texel.y);

    float end_neg = 0.0;
    float end_pos = 0.0;
    float delta_neg = 0.0;
    float delta_pos = 0.0;
    bool done_neg = false;
    bool done_pos = false;
    float quarter = gradient * 0.25;
    for (int i = 0; i < SEARCH_STEPS; ++i) {
        float s = step_for(i);
        if (!done_neg) {
            end_neg = luma(texture(u_color, uv_edge - along * (delta_neg + s)).rgb) - l_edge_avg;
            done_neg = abs(end_neg) >= quarter;
            delta_neg += s;
        }
        if (!done_pos) {
            end_pos = luma(texture(u_color, uv_edge + along * (delta_pos + s)).rgb) - l_edge_avg;
            done_pos = abs(end_pos) >= quarter;
            delta_pos += s;
        }
        if (done_neg && done_pos) break;
    }

    float dist_neg = delta_neg;
    float dist_pos = delta_pos;
    bool closer_neg = dist_neg < dist_pos;
    float dist = min(dist_neg, dist_pos);
    float span = dist_neg + dist_pos;
    bool center_smaller = l_c < l_edge_avg;
    bool good_span = ((closer_neg ? end_neg : end_pos) < 0.0) != center_smaller;
    float pixel_offset = good_span ? (0.5 - dist / span) : 0.0;
    float final_offset = max(pixel_offset, sub_offset);

    vec2 uv = v_uv;
    if (is_horizontal) uv.y += final_offset * step_len; else uv.x += final_offset * step_len;
    out_color = vec4(texture(u_color, uv).rgb, 1.0);
}
