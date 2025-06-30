#pragma once

#include "rl/neural_network.hpp"

#include <vector>

struct transition
{
    std::vector<float> state;
    int action;
    float reward;
    bool done;
};

class agent
{
public:
    agent(const neural_network& policy, const float gamma, const float learning_rate);
    
public:
    [[nodiscard]] int act(const std::vector<float>& state) const;

    void learn();
    void store_transition(const transition& transition);

    void save(const char* filename) const;
    void load(const char* filename);

private:
    neural_network policy;
    std::vector<transition> transitions;

    float gamma; // discount factor
    float learning_rate;
};
