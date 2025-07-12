#pragma once

#include "rl/transition.hpp"
#include "rl/neural_network.hpp"
#include "rl/adam_optimizer.hpp"
#include "rl/action.hpp"

#include <vector>

class reinforce_agent
{
public:
    reinforce_agent(const neural_network& policy, const action_space space, const float gamma = 0.99f, const float baseline_decay = 0.99f);
    
public:
    [[nodiscard]] action act(const Eigen::VectorXf& state) const;
    void learn(const std::vector<episode>& episodes);

    void save(const char* filename) const;
    void load(const char* filename);

private:
    neural_network policy;
    adam_optimizer optimizer;

    const action_space space;
    const float gamma; // discount factor
    const float baseline_decay;
    
    float running_baseline;
};
