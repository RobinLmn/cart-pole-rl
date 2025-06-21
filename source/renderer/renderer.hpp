#pragma once

#include "world/world.hpp"

#include <SFML/Graphics/View.hpp>

namespace sf
{
    class RenderTarget;
}

class renderer
{
public:
    explicit renderer(sf::RenderTarget& target);

public:
    void draw(const world& world) const;

private:
    sf::RenderTarget* target;
    sf::View view;
};