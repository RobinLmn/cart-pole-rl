#pragma once

#include <functional>

class world;

class app
{
public:
    static void run(const world& world, const float dt, const std::function<void()>& update);
};
