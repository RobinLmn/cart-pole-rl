#pragma once

#include <SFML/Graphics.hpp>
#include "renderer/renderer.hpp"
#include <memory>

class environment;
class agent;
class renderer;

class simulation
{
public:
    simulation(environment* env, const bool headless);

public:
    void run(const int episodes);

private:
    const bool headless;
    environment* env;
    std::unique_ptr<sf::RenderWindow> window_ptr;
    std::unique_ptr<renderer> renderer_ptr;
};