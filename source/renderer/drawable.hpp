#pragma once

#include <SFML/System/Vector2.hpp>
#include <SFML/Graphics/Color.hpp>

struct material
{
    sf::Color fill_color;
    sf::Color outline_color;
    float outline_thickness;
};

struct rectangle
{
    sf::Vector2f size;
    material material;
};

struct circle
{
    float radius;
    material material;
};
