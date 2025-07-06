#pragma once

#include "rl/neural_network.hpp"

#include <vector>

class sgd_optimizer
{
public:
    sgd_optimizer(neural_network& policy, const float learning_rate, const float momentum = 0.9f);

public:
    void step(const std::vector<parameters>& gradients);

private:
    const float learning_rate;
    const float momentum;

    neural_network& policy;

    std::vector<parameters> velocities;
};