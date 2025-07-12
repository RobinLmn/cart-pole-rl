#pragma once

#include "rl/transition.hpp"
#include "rl/neural_network.hpp"
#include "rl/adam_optimizer.hpp"
#include "rl/action.hpp"

#include <vector>

class ppo_agent
{
public:
    ppo_agent(const neural_network& actor, const neural_network& critic, const action_space space, const float gamma = 0.99f, const float epsilon = 0.2f, const int ppo_epochs = 1);
    
public:
    [[nodiscard]] action act(const Eigen::VectorXf& state) const;
    void learn(const std::vector<episode>& episodes);

    void save(const char* filename) const;
    void load(const char* filename);

private:
    neural_network actor;
    neural_network critic;

    adam_optimizer actor_optimizer;
    adam_optimizer critic_optimizer;

    const action_space space;
    const float gamma;   // discount factor
    const float epsilon; // PPO clipping parameter
    const int ppo_epochs; // number of PPO update epochs per batch
};