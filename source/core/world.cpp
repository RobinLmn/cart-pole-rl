#include "core/world.hpp"

entity world::create_entity()
{
    return registry.create();
}

void world::destroy_entity(entity entity)
{
    registry.destroy(entity);
}