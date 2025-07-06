#include "rl/sgd_optimizer.hpp"

sgd_optimizer::sgd_optimizer(const float learning_rate)
    : learning_rate{ learning_rate }
{
}

void sgd_optimizer::step(neural_network& policy, const std::vector<gradient>& gradients) const
{
    for (int i = 0; i < policy.layers.size(); ++i)
    {
        layer& layer = policy.layers[i];
        const gradient& gradient = gradients[i];

        layer.weights -= learning_rate * gradient.dW;
        layer.biases -= learning_rate * gradient.dB;
    }
}