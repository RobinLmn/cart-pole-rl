#pragma once

#include <glm/glm.hpp>

struct material
{
    glm::vec<4, uint8_t> fill_color;
    glm::vec<4, uint8_t> outline_color;
    float outline_thickness;
};

struct rectangle
{
    glm::vec2 size;
    material material;
};

struct circle
{
    float radius;
    material material;
};
