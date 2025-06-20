#pragma once

#include "world/world.hpp"

#include <SFML/graphics.hpp>

class renderer
{
public:
    renderer(sf::RenderTarget& target);

public:
    void draw(const world& world);

private:
    sf::RenderTarget* target;
};