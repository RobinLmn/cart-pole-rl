#pragma once

#include <vector>

#include <Eigen/Dense>

struct transition
{
    bool done;
    int action;
    float reward;
    Eigen::VectorXf state;
    Eigen::VectorXf next_state;
};

using episode = std::vector<transition>;