#pragma once

#include "rl/agent.hpp"
#include "world/world.hpp"

#include <vector>

class environment
{
public:
    virtual void reset();

    [[nodiscard]] virtual bool is_done() const = 0;
    [[nodiscard]] virtual float step(const float dt, const std::vector<float>& actions) = 0;
    [[nodiscard]] virtual std::vector<float> get_state() = 0;
    [[nodiscard]] virtual agent create_agent() = 0;

    const world& get_world() const;

protected:
    world world;
};
