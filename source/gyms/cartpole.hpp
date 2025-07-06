#pragma once

#include "rl/reinforce_agent.hpp"
#include "world/world.hpp"

#include <Eigen/Dense>

class cartpole_environment
{
public:
    float step(const float dt, const int action);

    void reset();

    [[nodiscard]] bool is_done() const;
    [[nodiscard]] Eigen::VectorXf get_state() const;

    const world& get_world() const { return world; }

private:
    world world;

    entity cart;
    entity pole;

    int step_index;
};

reinforce_agent create_reinforce_cartpole_agent();
