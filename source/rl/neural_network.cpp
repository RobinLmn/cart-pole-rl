#include "rl/neural_network.hpp"

#include "core/log.hpp"

layer::layer(const int input_dim, const int output_dim, const activation_function& activation, const activation_function& activation_derivative)
    : activation(std::move(activation))
    , activation_derivative(std::move(activation_derivative))
    , weights{ Eigen::MatrixXf::Random(output_dim, input_dim) * std::sqrt(6.f / (input_dim + output_dim)) } // (output_dim x input_dim)
    , biases{ Eigen::VectorXf::Zero(output_dim) } // (output_dim x 1)
{
}

Eigen::MatrixXf layer::forward(const Eigen::MatrixXf& input) const
{
    inputs = input;
    pre_activation = weights * input + biases; // (output_dim x input_dim) * (input_dim x 1) + (output_dim x 1) = (output_dim x 1)
    return activation(pre_activation);
}

Eigen::MatrixXf layer::backward(const Eigen::MatrixXf& dA, const float learning_rate)
{
    const Eigen::MatrixXf dZ = dA.array() * activation_derivative(pre_activation).array();  // (output_dim × 1)
    const Eigen::MatrixXf dW = dZ * inputs.transpose();  // (output_dim × input_dim)
    const Eigen::VectorXf dB = dZ.col(0);  // single sample

    const Eigen::MatrixXf dInput = weights.transpose() * dZ;  // (input_dim × 1)

    weights -= learning_rate * dW;
    biases -= learning_rate * dB;

    return dInput;
}

void neural_network::add_layer(const layer& layer)
{
    layers.emplace_back(layer);
}

Eigen::VectorXf neural_network::forward(const Eigen::VectorXf& input) const
{
    Eigen::MatrixXf output_matrix = input;
    for (const layer& layer : layers)
    {
        output_matrix = layer.forward(output_matrix);
    }

    return output_matrix.col(0);
}

void neural_network::backward(const Eigen::VectorXf& gradients, const float learning_rate)
{
    Eigen::MatrixXf gradient_matrix = gradients;
    for (auto layer_it = layers.rbegin(); layer_it != layers.rend(); ++layer_it)
    {
        gradient_matrix = layer_it->backward(gradient_matrix, learning_rate);
    }
}
