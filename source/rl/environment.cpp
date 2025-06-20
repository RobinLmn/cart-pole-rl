#include "rl/environment.hpp"

void environment::reset()
{
    world.clear();
}

const world& environment::get_world() const
{
    return world;
}