#pragma once

#include "rl/transition.hpp"
#include "rl/neural_network.hpp"

#include <vector>

class reinforce_agent
{
public:
    reinforce_agent(const neural_network& policy, const float gamma, const float learning_rate);
    
public:
    [[nodiscard]] int act(const Eigen::VectorXf& state) const;
    void learn(const std::vector<episode>& episodes);

    void save(const char* filename) const;
    void load(const char* filename);

private:
    neural_network policy;

    float gamma; // discount factor
    float learning_rate;
};
