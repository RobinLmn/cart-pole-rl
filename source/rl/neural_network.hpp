#pragma once

#include "rl/activation_functions.hpp"

#include <Eigen/Dense>
#include <functional>
#include <vector>

class layer
{
public:
    layer(const int input_dim, const int output_dim, const char* activation_name);
    layer(const Eigen::MatrixXf& weights, const Eigen::VectorXf& biases, const char* activation_name);

public:
    [[nodiscard]] Eigen::MatrixXf forward(const Eigen::MatrixXf& input) const;
    [[nodiscard]] Eigen::MatrixXf backward(const Eigen::MatrixXf& gradients, const float learning_rate);

    Eigen::MatrixXf get_weights() const;
    Eigen::VectorXf get_biases() const;
    const char* get_activation_name() const;

private:
    const activation_function activation;

    Eigen::MatrixXf weights;
    Eigen::VectorXf biases;

    mutable Eigen::MatrixXf inputs;
    mutable Eigen::MatrixXf pre_activation;
};

class neural_network
{
public:
    void add_layer(const layer& layer);
    
    [[nodiscard]] Eigen::VectorXf forward(const Eigen::VectorXf& input) const;
    void backward(const Eigen::VectorXf& gradients, const float learning_rate);

    void save(const char* filename) const;
    void load(const char* filename);

private:
    std::vector<layer> layers;
};
