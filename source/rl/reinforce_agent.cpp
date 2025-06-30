#include "rl/reinforce_agent.hpp"

#include "core/log.hpp"
#include "core/utils.hpp"

#include <glm/glm.hpp>
#include <Eigen/Dense>

reinforce_agent::reinforce_agent(const neural_network& policy, const float gamma, const float learning_rate)
    : policy{ std::move(policy) }
    , gamma{ gamma }
    , learning_rate{ learning_rate }
{
}

action reinforce_agent::act(const std::vector<float>& state) const
{
    const Eigen::VectorXf& input = eigen(state);
    const Eigen::VectorXf& logits = policy.forward(input);
    const Eigen::VectorXf& probs = softmax(logits);

    const int action = sample_from_distribution(probs);
    return action;
}

void reinforce_agent::learn()
{
    const int transition_size = static_cast<int>(transitions.size());
    ASSERT(transition_size != 0, return, "No episodes to learn from.");

    // Compute the return for each episode: G_t = r_t + γ * r_{t+1} + γ^2 * r_{t+2} + ... + γ^{T-t} * r_T
    Eigen::VectorXf returns(transition_size);
    float G = 0;
    for (int t = transition_size - 1; t >= 0; --t) 
    {
        if (transitions[t].done)
        {
            G = 0;
        }
        
        G = transitions[t].reward + gamma * G;
        returns[t] = G;
    }

    // Calculate advantages by subtracting a baseline (the mean of returns)
    const float baseline = returns.mean();
    returns = returns.array() - baseline;

    // Normalize the returns
    const float stddev = glm::sqrt((returns.array()).square().mean());
    static constexpr float epsilon = 1e-8f;
    if (stddev > epsilon) 
    {
        returns /= stddev;
    }

    // Train with REINFORCE using advantages
    for (int t = 0; t < transition_size; ++t)
    {
        const transition& transition = transitions[t];
        const float Gt = returns[t];

        const Eigen::VectorXf& state = eigen(transition.state);
        const Eigen::VectorXf& logits = policy.forward(state);
        const Eigen::VectorXf& probs = softmax(logits);

        ASSERT(std::holds_alternative<int>(transition.action), return, "REINFORCE only implements discrete actions for now.");
        const int action = std::get<int>(transition.action);

        // Policy gradient
        Eigen::VectorXf gradient = -Gt * (Eigen::VectorXf::Unit(probs.size(), action) - probs);

        // Gradient norm clipping
        static constexpr float max_grad_norm = 0.5f;
        const float norm = gradient.norm();
        if (norm > max_grad_norm) 
        {
            gradient = gradient * (max_grad_norm / norm);
        }

        policy.backward(gradient, learning_rate);
    }

    transitions.clear();
}

void reinforce_agent::save(const char* filename) const
{
    policy.save(filename);
}

void reinforce_agent::load(const char* filename)
{
    policy.load(filename);
}
