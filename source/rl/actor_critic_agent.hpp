#pragma once

#include "rl/transition.hpp"
#include "rl/neural_network.hpp"
#include "rl/adam_optimizer.hpp"

#include <vector>

class actor_critic_agent
{
public:
    actor_critic_agent(const neural_network& actor, const neural_network& critic, const float gamma = 0.99f);
    
public:
    [[nodiscard]] int act(const Eigen::VectorXf& state) const;
    void learn(const std::vector<episode>& episodes);

    void save(const char* filename) const;
    void load(const char* filename);

private:
    neural_network actor;
    neural_network critic;

    adam_optimizer actor_optimizer;
    adam_optimizer critic_optimizer;

    const float gamma; // discount factor
};