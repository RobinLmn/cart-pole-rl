#pragma once

#include <glm/glm.hpp>

struct uses_gravity {};

struct rigidbody
{
    float mass;
    glm::vec2 force;
    glm::vec2 velocity;

    float inertia;
    float torque;
    float angular_velocity;

    float friction;
};