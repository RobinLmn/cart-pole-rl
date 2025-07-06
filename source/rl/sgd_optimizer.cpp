#include "rl/sgd_optimizer.hpp"

sgd_optimizer::sgd_optimizer(neural_network& policy, const float learning_rate, const float momentum)
    : learning_rate{ learning_rate }
    , momentum{ momentum }
    , policy{ policy }
    , velocities(policy.layers.size())
{
    for (size_t i = 0; i < policy.layers.size(); ++i)
    {
        const parameters& layer_params = policy.layers[i].params;
        velocities[i].weights = Eigen::MatrixXf::Zero(layer_params.weights.rows(), layer_params.weights.cols());
        velocities[i].biases = Eigen::VectorXf::Zero(layer_params.biases.size());
    }
}

void sgd_optimizer::step(const std::vector<parameters>& gradients)
{
    for (int i = 0; i < policy.layers.size(); ++i)
    {
        layer& layer = policy.layers[i];
        const parameters& gradient = gradients[i];
        parameters& velocity = velocities[i];

        velocity = momentum * velocity - learning_rate * gradient;
        layer.params += velocity;
    }
}