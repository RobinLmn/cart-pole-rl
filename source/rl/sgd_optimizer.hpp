#pragma once

#include "rl/neural_network.hpp"

#include <vector>

class sgd_optimizer
{
public:
    sgd_optimizer(const float learning_rate);

public:
    void step(neural_network& policy, const std::vector<gradient>& gradients) const;

private:
    const float learning_rate;
};