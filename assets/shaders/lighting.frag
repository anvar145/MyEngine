// Освещение MyEngine: forward-путь, до MAX_LIGHTS источников на объект.
//
// Файл подключается фрагментными шейдерами материалов:
//
//     #include "lighting.frag"
//
// и добавляет к ним uniform-массивы источников и функции освещения. Здесь нет
// функции main() и нет директивы #version: включаемый файл подставляется после
// строки версии шейдера-хозяина (GLSL требует, чтобы #version был первым
// токеном). Шейдер материала сам решает, что делать с посчитанным светом
// (текстура, прозрачность, запечённый свет).
//
// Этот файл — единственный источник правды для GLSL: копия в
// ``assets/shaders/lighting.frag`` синхронизируется скриптом
// ``tools/sync_shaders.py`` и проверяется тестом.
//
// Освещение — Блинн—Фонг (дешевле Фонга: не нужен reflect).
//
// Данные источников передаются «структурой массивов»: отдельный uniform-массив
// на каждый атрибут, и рендерер грузит его одним вызовом (glUniform3fv на весь
// массив) вместо поштучной установки переменных. Число активных источников
// задаёт u_light_count: цикл идёт до MAX_LIGHTS, лишние итерации отсекаются.
//
// Типы источников: 0 — направленный, 1 — точечный, 2 — прожектор,
// 3 — фоновый (участвует только яркостью через u_ambient).

#ifndef MYENGINE_LIGHTING
#define MYENGINE_LIGHTING

#define LIGHT_DIRECTIONAL 0
#define LIGHT_POINT 1
#define LIGHT_SPOT 2
#define LIGHT_AMBIENT 3

//: Максимальное число источников на объект (совпадает с размером массивов).
#ifndef MAX_LIGHTS
#define MAX_LIGHTS 8
#endif

// --------------------------------------------------------------------------- //
// Uniform-массивы источников (заполняет LightManager)
// --------------------------------------------------------------------------- //
uniform vec3 u_light_position[MAX_LIGHTS];   // позиция (точечный, прожектор)
uniform vec3 u_light_direction[MAX_LIGHTS];  // направление, КУДА светит источник
uniform vec3 u_light_color[MAX_LIGHTS];      // цвет, умноженный на яркость
uniform vec4 u_light_params[MAX_LIGHTS];     // (радиус, cos внутр., cos внеш., затухание)
uniform int u_light_type[MAX_LIGHTS];        // тип источника
uniform int u_light_count;                   // сколько источников активно
uniform int u_max_lights;                    // размер массивов (для отладки)
uniform vec3 u_ambient;                      // суммарная фоновая подсветка
//: Позицию камеры объявляет шейдер-хозяин (он использует её и для блика в
//: материала без освещения), поэтому здесь она не переобъявляется.

// --------------------------------------------------------------------------- //
// Карта теней направленного источника
// --------------------------------------------------------------------------- //
uniform sampler2D u_shadow_map;      // карта глубины (512×512 на слабых GPU)
uniform mat4 u_light_space_matrix;   // мир → пространство источника
uniform vec2 u_shadow_texel;         // 1 / разрешение карты
uniform float u_shadow_bias;         // смещение против «акне» теней
uniform float u_shadow_normal_bias;  // смещение по нормали
uniform float u_shadow_strength;     // 0 — теней нет, 1 — полностью тёмные
uniform int u_shadow_enabled;        // 0 — карта не используется
uniform int u_no_receive_shadow;     // 1 — объект не принимает чужие тени

//: Настройки теней. Сэмплер в структуру положить нельзя (GLSL в 3.30 требует,
//: чтобы сэмплеры были отдельными uniform-переменными), поэтому структура
//: хранит только числа, а карта передаётся в функции отдельным аргументом.
struct ShadowConfig {
    mat4 light_space;
    vec2 texel;
    float bias;
    float normal_bias;
    float strength;
    int enabled;
};

/// Собрать настройки теней из uniform-переменных.
ShadowConfig shadow_config() {
    ShadowConfig shadow;
    shadow.light_space = u_light_space_matrix;
    shadow.texel = u_shadow_texel;
    shadow.bias = u_shadow_bias;
    shadow.normal_bias = u_shadow_normal_bias;
    shadow.strength = u_shadow_strength;
    shadow.enabled = (u_no_receive_shadow == 0) ? u_shadow_enabled : 0;
    return shadow;
}

// --------------------------------------------------------------------------- //
// Затухание
// --------------------------------------------------------------------------- //
/// Линейное затухание до нуля на границе радиуса влияния.
///
/// Именно оно позволяет отсекать источники по сфере: за радиусом вклад ровно
/// ноль, поэтому отброшенный свет не оставляет швов на геометрии.
float light_range_attenuation(float distance_to_light, float range, float falloff) {
    float ratio = clamp(distance_to_light / max(range, 1e-4), 0.0, 1.0);
    return pow(1.0 - ratio, max(falloff, 1e-4));
}

/// Мягкий край конуса прожектора: между внутренним и внешним углами.
float spot_cone_attenuation(float cosine, float inner_cosine, float outer_cosine) {
    float denominator = max(inner_cosine - outer_cosine, 1e-4);
    return clamp((cosine - outer_cosine) / denominator, 0.0, 1.0);
}

// --------------------------------------------------------------------------- //
// Тени
// --------------------------------------------------------------------------- //
/// Доля света после PCF 3×3: каждая выборка сравнивается с глубиной точки
/// отдельно, а усредняется уже результат сравнения. Усреднять глубину до
/// сравнения нельзя: на краях теней это даёт утечки света.
float shadow_visibility(sampler2D shadow_map, ShadowConfig shadow, vec2 uv,
                        float reference_depth) {
    float lit = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float closest =
                texture(shadow_map, uv + vec2(float(x), float(y)) * shadow.texel).r;
            lit += step(reference_depth, closest);
        }
    }
    return lit / 9.0;
}

/// Множитель тени: 1 — свет доходит полностью, 0 — точка в тени.
float shadow_factor(sampler2D shadow_map, ShadowConfig shadow,
                    vec3 world_position, vec3 normal) {
    if (shadow.enabled == 0) {
        return 1.0;
    }
    // Смещение по нормали снимает самозатенение на наклонных поверхностях:
    // оно дешевле большого bias, который «отклеивает» тень от объекта.
    vec3 offset_position = world_position + normal * shadow.normal_bias;
    vec4 light_position = shadow.light_space * vec4(offset_position, 1.0);
    // Матрица даёт NDC, перспективного деления нет (источник направленный,
    // матрица ортографическая, поэтому w == 1). Приводим к оконным
    // координатам: xy — в 0..1, глубина — по той же формуле (z_ndc + 1) / 2,
    // которой драйвер пользовался при записи карты.
    vec3 coordinate = light_position.xyz;
    vec2 uv = coordinate.xy * 0.5 + 0.5;
    float current_depth = coordinate.z * 0.5 + 0.5;
    if (uv.x < 0.0 || uv.x > 1.0
        || uv.y < 0.0 || uv.y > 1.0
        || current_depth > 1.0) {
        // Вне карты теней считаем, что свет доходит: иначе за границей карты
        // появлялась бы ложная тень.
        return 1.0;
    }
    float lit = shadow_visibility(
        shadow_map, shadow, uv, current_depth - shadow.bias
    );
    // strength управляет жёсткостью: 1 — тень полностью перекрывает свет.
    return mix(1.0, lit, clamp(shadow.strength, 0.0, 1.0));
}

// --------------------------------------------------------------------------- //
// Падающий свет одного источника
// --------------------------------------------------------------------------- //
/// Направление на источник и сила падающего света в точке.
///
/// Общая часть Блинна—Фонга и PBR: тип источника, затухание по радиусу и
/// конус прожектора считаются в одном месте.
///
/// Args:
///   index: Номер источника в массивах.
///   world_position: Позиция фрагмента в мире.
///   normal: Нормаль поверхности (нормализованная).
///   to_light: (выход) единичный вектор от фрагмента к источнику.
///   radiance: (выход) цвет источника с учётом затухания.
///
/// Returns:
///   ``false``, если источник не даёт света (фоновый или за радиусом).
bool light_incidence(
    int index,
    vec3 world_position,
    vec3 normal,
    out vec3 to_light,
    out vec3 radiance
) {
    to_light = normal;
    radiance = vec3(0.0);
    int light_type = u_light_type[index];
    if (light_type == LIGHT_AMBIENT) {
        // Фоновая подсветка приходит через u_ambient: слот пропускаем,
        // чтобы не считать её дважды.
        return false;
    }

    float attenuation = 1.0;
    if (light_type == LIGHT_DIRECTIONAL) {
        // В массиве лежит направление, КУДА светит источник, а для расчёта
        // нужен вектор НА источник.
        to_light = -u_light_direction[index];
    } else {
        vec3 delta = u_light_position[index] - world_position;
        float distance_to_light = length(delta);
        if (distance_to_light < 1e-5) {
            to_light = normal;
        } else {
            to_light = delta / distance_to_light;
        }
        float range = u_light_params[index].x;
        float falloff = u_light_params[index].w;
        attenuation = light_range_attenuation(distance_to_light, range, falloff);
        if (attenuation <= 0.0) {
            return false;
        }
        if (light_type == LIGHT_SPOT) {
            // Косинус угла между осью конуса и направлением на фрагмент.
            float cosine = dot(-to_light, u_light_direction[index]);
            attenuation *= spot_cone_attenuation(
                cosine, u_light_params[index].y, u_light_params[index].z
            );
            if (attenuation <= 0.0) {
                return false;
            }
        }
    }
    radiance = u_light_color[index] * attenuation;
    return true;
}

// --------------------------------------------------------------------------- //
// Вклад одного источника
// --------------------------------------------------------------------------- //
/// Освещение от одного источника (Блинн—Фонг).
///
/// Args:
///   index: Номер источника в массивах.
///   world_position: Позиция фрагмента в мире.
///   normal: Нормаль поверхности (нормализованная).
///   view_dir: Направление от фрагмента к камере.
///   albedo: Цвет поверхности.
///   shininess: Степень блика (из шероховатости).
///   specular_level: Сила блика (из металличности).
///
/// Returns:
///   Освещение этого источника.
vec3 shade_light(
    int index,
    vec3 world_position,
    vec3 normal,
    vec3 view_dir,
    vec3 albedo,
    float shininess,
    float specular_level
) {
    vec3 to_light;
    vec3 radiance;
    if (!light_incidence(index, world_position, normal, to_light, radiance)) {
        return vec3(0.0);
    }

    float lambert = max(dot(normal, to_light), 0.0);
    if (lambert <= 0.0) {
        return vec3(0.0);
    }

    vec3 diffuse = albedo * lambert;

    // Блик Блинна—Фонга: half-вектор дешевле отражения.
    vec3 half_dir = normalize(to_light + view_dir);
    float specular = pow(max(dot(normal, half_dir), 0.0), shininess);
    specular *= specular_level;

    return radiance * (diffuse + vec3(specular));
}

// --------------------------------------------------------------------------- //
// Полный расчёт освещения
// --------------------------------------------------------------------------- //
/// Forward-освещение фрагмента со всеми активными источниками.
///
/// Args:
///   world_position: Позиция фрагмента в мире.
///   world_normal: Нормаль (до нормализации).
///   albedo: Цвет поверхности (уже с текстурой и запечённым светом).
///   shininess: Степень блика.
///   specular_level: Сила блика.
///
/// Returns:
///   Линейный цвет до гамма-коррекции.
vec3 compute_lighting(
    vec3 world_position,
    vec3 world_normal,
    vec3 albedo,
    float shininess,
    float specular_level
) {
    vec3 normal = normalize(world_normal);
    vec3 view_dir = normalize(u_camera_position - world_position);

    // Фоновая подсветка — один раз, без учёта геометрии.
    vec3 result = u_ambient * albedo;

    ShadowConfig shadow = shadow_config();
    // Множитель тени считается один раз на фрагмент: от него зависит вклад
    // направленного источника.
    float shadow_term = shadow_factor(
        u_shadow_map, shadow, world_position, normal
    );

    // Цикл с постоянной границей: драйвер разворачивает его без ветвлений.
    for (int index = 0; index < MAX_LIGHTS; ++index) {
        if (index >= u_light_count) {
            break;
        }
        vec3 contribution = shade_light(
            index, world_position, normal, view_dir,
            albedo, shininess, specular_level
        );
        if (u_light_type[index] == LIGHT_DIRECTIONAL) {
            contribution *= shadow_term;
        }
        result += contribution;
    }

    return result;
}

/// Степень блика из шероховатости (8 — матовое, 128 — глянцевое).
float specular_power(float roughness) {
    return mix(8.0, 128.0, clamp(1.0 - roughness, 0.0, 1.0));
}

/// Сила блика из металличности.
float specular_strength(float metallic) {
    return mix(0.04, 1.0, clamp(metallic, 0.0, 1.0));
}

#endif
