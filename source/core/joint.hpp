#pragma once

#include "core/world.hpp"

#include <glm/glm.hpp>

struct joint
{
    entity a;
    entity b;

    glm::vec2 anchor_a_local;
    glm::vec2 anchor_b_local;
};

