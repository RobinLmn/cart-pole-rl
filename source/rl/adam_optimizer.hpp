#pragma once

#include "rl/neural_network.hpp"

#include <vector>

class adam_optimizer
{
public:
    adam_optimizer(neural_network& policy, const float learning_rate = 0.0005f, const float beta1 = 0.9f, const float beta2 = 0.999f, const float epsilon = 1e-8f);

public:
    void step(const std::vector<parameters>& gradients);

private:
    const float learning_rate;
    const float beta1;
    const float beta2;
    const float epsilon;

    int timestep;

    neural_network& policy;

    std::vector<parameters> moments;
    std::vector<parameters> variances;
};