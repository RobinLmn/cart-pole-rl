#pragma once

#include "world/world.hpp"

#include "rl/transition.hpp"
#include "rl/reinforce_agent.hpp"

class cartpole_environment
{
public:
    float step(const float dt, const action& action);

    void reset();

    [[nodiscard]] bool is_done() const;
    [[nodiscard]] std::vector<float> get_state() const;

    const world& get_world() const { return world; }

private:
    world world;

    entity cart;
    entity pole;
};

reinforce_agent create_reinforce_cartpole_agent();
