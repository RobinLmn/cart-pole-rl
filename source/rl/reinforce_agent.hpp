#pragma once

#include "rl/transition.hpp"
#include "rl/neural_network.hpp"

#include <vector>

class reinforce_agent
{
public:
    reinforce_agent(const neural_network& policy, const float gamma = 0.99f, const float learning_rate = 0.01f, const float baseline_decay = 0.99f);
    
public:
    [[nodiscard]] int act(const Eigen::VectorXf& state) const;
    void learn(const std::vector<episode>& episodes);

    void save(const char* filename) const;
    void load(const char* filename);

private:
    neural_network policy;

    const float gamma; // discount factor
    const float learning_rate;
    const float baseline_decay;
    
    float running_baseline;
};
