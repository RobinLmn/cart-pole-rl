#pragma once

#include "rl/activation_functions.hpp"

#include <Eigen/Dense>
#include <functional>
#include <vector>

struct parameters
{
    Eigen::MatrixXf weights;
    Eigen::VectorXf biases;
};

struct layer
{
public:
    layer(const int input_dim, const int output_dim, const char* activation_name);
    layer(const Eigen::MatrixXf& weights, const Eigen::VectorXf& biases, const char* activation_name);

public:
    [[nodiscard]] Eigen::VectorXf forward(const Eigen::VectorXf& input) const;
    [[nodiscard]] std::pair<parameters, Eigen::VectorXf> backward(const Eigen::VectorXf& gradient) const;

public:
    parameters params;

    const activation_function activation;

private:
    mutable Eigen::MatrixXf inputs;
    mutable Eigen::MatrixXf pre_activation;
};

struct neural_network
{
public:
    void add_layer(const layer& layer);
    
    [[nodiscard]] Eigen::VectorXf forward(const Eigen::VectorXf& input) const;
    [[nodiscard]] std::vector<parameters> backward(const Eigen::VectorXf& gradients) const;

    void save(const char* filename) const;
    void load(const char* filename);

public:
    std::vector<layer> layers;
};

parameters& operator+=(parameters& lhs, const parameters& rhs);
parameters& operator*=(parameters& lhs, float rhs);
parameters& operator/=(parameters& lhs, float rhs);
parameters operator*(const parameters& lhs, float rhs);
parameters operator*(float lhs, const parameters& rhs);
parameters operator-(const parameters& lhs, const parameters& rhs);
