#pragma once

#include "rl/transition.hpp"
#include "rl/neural_network.hpp"

#include <vector>

class reinforce_agent
{
public:
    reinforce_agent(const neural_network& policy, const float gamma, const float learning_rate, const bool use_baseline = true);
    
public:
    [[nodiscard]] action act(const std::vector<float>& state) const;
    void learn(const std::vector<transition>& transitions);

    void save(const char* filename) const;
    void load(const char* filename);

private:
    bool use_baseline;

    neural_network policy;

    float gamma; // discount factor
    float learning_rate;
};
