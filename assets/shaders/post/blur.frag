#version 330 core

// Направленное гауссово размытие (9 тапов через 5 билинейных выборок).
// Источник может быть вдвое больше цели: шаг берётся от размера цели, и
// билинейная выборка заодно усредняет блок 2x2 (понижение разрешения).

uniform sampler2D u_color;
uniform vec2 u_direction;  // шаг в UV: (1/w, 0) или (0, 1/h) целевого уровня

in vec2 v_uv;
out vec4 out_color;

void main() {
    vec3 sum = texture(u_color, v_uv).rgb * 0.2270270270;
    sum += texture(u_color, v_uv + u_direction * 1.3846153846).rgb * 0.3162162162;
    sum += texture(u_color, v_uv - u_direction * 1.3846153846).rgb * 0.3162162162;
    sum += texture(u_color, v_uv + u_direction * 3.2307692308).rgb * 0.0702702703;
    sum += texture(u_color, v_uv - u_direction * 3.2307692308).rgb * 0.0702702703;
    out_color = vec4(sum, 1.0);
}
