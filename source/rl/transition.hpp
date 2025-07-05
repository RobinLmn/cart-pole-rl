#pragma once

#include <vector>

#include <Eigen/Dense>

struct transition
{
    bool done;
    int action;
    float reward;
    Eigen::VectorXf state;
};

using episode = std::vector<transition>;