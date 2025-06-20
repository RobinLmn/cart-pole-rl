#pragma once

#include <SFML/Graphics.hpp>
#include "renderer/renderer.hpp"

class environment;
class agent;
class renderer;

class simulation
{
public:
    simulation(environment* env);

public:
    void run(const int episodes);

private:
    environment* env;
    sf::RenderWindow window;
    renderer renderer;
};