#include "world/world.hpp"

entity world::create_entity()
{
    return registry.create();
}

void world::destroy_entity(const entity entity)
{
    registry.destroy(entity);
}

void world::clear()
{
    registry.clear();
}