#include "rl/adam_optimizer.hpp"

adam_optimizer::adam_optimizer(neural_network& policy,  const float learning_rate, const float beta1, const float beta2, const float epsilon)
    : learning_rate{ learning_rate }
    , beta1{ beta1 }
    , beta2{ beta2 }
    , epsilon{ epsilon}
    , timestep{ 1 }
    , policy{ policy }
    , moments(policy.layers.size())
    , variances(policy.layers.size())
{
    for (size_t i = 0; i < policy.layers.size(); ++i)
    {
        const parameters& layer_params = policy.layers[i].params;

        moments[i].weights = Eigen::MatrixXf::Zero(layer_params.weights.rows(), layer_params.weights.cols());
        variances[i].weights = Eigen::MatrixXf::Zero(layer_params.weights.rows(), layer_params.weights.cols());

        moments[i].biases = Eigen::VectorXf::Zero(layer_params.biases.size());
        variances[i].biases = Eigen::VectorXf::Zero(layer_params.biases.size());
    }
}

void adam_optimizer::step(const std::vector<parameters>& gradients)
{
    for (size_t i = 0; i < policy.layers.size(); ++i)
    {
        layer& layer = policy.layers[i];
        const parameters& gradient = gradients[i];
        parameters& momentum = moments[i];
        parameters& variance = variances[i];

        momentum.weights = beta1 * momentum.weights + (1.f - beta1) * gradient.weights;
        momentum.biases = beta1 * momentum.biases + (1.f - beta1) * gradient.biases;

        variance.weights = beta2 * variance.weights + (1.f - beta2) * gradient.weights.array().square().matrix();
        variance.biases = beta2 * variance.biases + (1.f - beta2) * gradient.biases.array().square().matrix();

        const float bias1 = 1.0f / (1.0f - static_cast<float>(std::pow(beta1, timestep)));
        const float bias2 = 1.0f / (1.0f - static_cast<float>(std::pow(beta2, timestep)));

        layer.params.weights -= (learning_rate * ((momentum.weights * bias1).array() / ((variance.weights * bias2).array().sqrt() + epsilon))).matrix();
        layer.params.biases -= (learning_rate * ((momentum.biases * bias1).array() / ((variance.biases * bias2).array().sqrt() + epsilon))).matrix();
    }

    timestep++;
}