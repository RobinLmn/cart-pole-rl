#pragma once

#include "rl/activation_functions.hpp"

#include <Eigen/Dense>
#include <functional>
#include <vector>

struct gradient
{
    Eigen::MatrixXf dW;
    Eigen::VectorXf dB;

    gradient& operator+=(const gradient& rhs);
    gradient& operator/=(const float rhs);
};

class layer
{
public:
    layer(const int input_dim, const int output_dim, const char* activation_name);
    layer(const Eigen::MatrixXf& weights, const Eigen::VectorXf& biases, const char* activation_name);

public:
    [[nodiscard]] Eigen::VectorXf forward(const Eigen::VectorXf& input) const;
    [[nodiscard]] std::pair<gradient, Eigen::VectorXf> backward(const Eigen::VectorXf& gradient) const;

    void gradient_descent(const gradient& gradient, const float learning_rate);

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
    [[nodiscard]] std::vector<gradient> backward(const Eigen::VectorXf& gradients) const;

    void gradient_descent(const std::vector<gradient>& gradients, const float learning_rate);

    void save(const char* filename) const;
    void load(const char* filename);

private:
    std::vector<layer> layers;
};
