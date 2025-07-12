#pragma once

#include "rl/action.hpp"

#include <vector>
#include <Eigen/Dense>

struct transition
{
    bool done;
    action action;
    float reward;
    Eigen::VectorXf state;
    Eigen::VectorXf next_state;
};

using episode = std::vector<transition>;