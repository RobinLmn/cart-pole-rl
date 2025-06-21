#pragma once

#include "rl/agent.hpp"
#include "world/world.hpp"

#include <vector>

class environment
{
public:
    virtual void reset() = 0;

    [[nodiscard]] virtual bool is_done() const = 0;
    [[nodiscard]] virtual float step(const float dt, const int action) = 0;
    [[nodiscard]] virtual std::vector<float> get_state() const = 0;
    [[nodiscard]] virtual agent create_agent() const = 0;

    [[nodiscard]] const world& get_world() const;

protected:
    world world;
};
