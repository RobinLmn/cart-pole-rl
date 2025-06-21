#pragma once

#include <Eigen/Dense>

#include <functional>
#include <vector>

using activation_function = std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)>;

class layer
{
public:
    layer(const int input_dim, const int output_dim, const activation_function& activation, const activation_function& activation_derivative);

public:
    [[nodiscard]] Eigen::MatrixXf forward(const Eigen::MatrixXf& input) const;
    [[nodiscard]] Eigen::MatrixXf backward(const Eigen::MatrixXf& gradients, const float learning_rate);

private:
    Eigen::MatrixXf weights;
    Eigen::VectorXf biases;

    mutable Eigen::MatrixXf inputs;
    mutable Eigen::MatrixXf pre_activation;

    const activation_function activation;
    const activation_function activation_derivative;
};

class neural_network
{
public:
    void add_layer(const layer& layer);
    
    [[nodiscard]] Eigen::VectorXf forward(const Eigen::VectorXf& input) const;
    void backward(const Eigen::VectorXf& gradients, const float learning_rate);

private:
    std::vector<layer> layers;
};
