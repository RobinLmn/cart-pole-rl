#pragma once

#include <Eigen/Dense>

#include <string>

using activation_function_signature = std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)>;

struct activation_function
{
    activation_function(const char* name, const activation_function_signature& func, const activation_function_signature& derivative);
    activation_function(const char* name);

    std::string name;
    activation_function_signature func;
    activation_function_signature derivative;
};