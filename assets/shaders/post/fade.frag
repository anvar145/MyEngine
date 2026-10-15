#version 330 core

// Fade/flash: плоский цвет с альфой, смешивается поверх кадра.

uniform vec4 u_color;

out vec4 out_color;

void main() {
    out_color = u_color;
}
