#include "rl/reinforce_agent.hpp"

#include "core/log.hpp"
#include "core/utils.hpp"

#include <glm/glm.hpp>
#include <Eigen/Dense>

reinforce_agent::reinforce_agent(const neural_network& policy, const float gamma, const float learning_rate)
    : policy{ policy }
    , gamma{ gamma }
    , learning_rate{ learning_rate }
{
}

int reinforce_agent::act(const Eigen::VectorXf& state) const
{
    const Eigen::VectorXf& logits = policy.forward(state);
    const Eigen::VectorXf& probs = softmax(logits);

    const int action = sample_from_distribution(probs);
    return action;
}

void reinforce_agent::learn(const std::vector<episode>& episodes)
{
    const int episode_count = static_cast<int>(episodes.size());
    ASSERT(episode_count > 0, return, "REINFORCE requires at least one episode to learn.");

    std::vector<Eigen::VectorXf> returns;
    returns.reserve(episode_count);

    for (const episode& episode : episodes)
    {
        const int episode_length = static_cast<int>(episode.size());
        ASSERT(episode_length > 0, continue, "REINFORCE episode is empty.");
        ASSERT(episode.back().done, continue, "REINFORCE must learn with full episodes.");

        float G = 0.0f;
        Eigen::VectorXf episode_return(episode_length);

        for (int t = episode_length - 1; t >= 0; --t)
        {
            G = episode[t].reward + gamma * G;
            episode_return[t] = G;
        }

        returns.push_back(std::move(episode_return));
    }

    // Normalize returns and substract baseline
    {
        const Eigen::VectorXf& flatten_returns = flatten(returns);

        const float mean = flatten_returns.mean();
        const float stddev = glm::sqrt((flatten_returns.array() - mean).square().mean());
        const float norm = std::max(stddev, 1e-8f);

        for (Eigen::VectorXf& episode_return : returns)
        {
            episode_return = episode_return.array() - mean;
            episode_return /= norm;
        }
    }

    for (int e = 0; e < episode_count; ++e)
    {
        const episode& episode = episodes[e];

        for (int t = 0; t < static_cast<int>(episode.size()); ++t)
        {
            const int action = episode[t].action;
            const float advantage = returns[e][t];
            const Eigen::VectorXf& state = episode[t].state;
            const Eigen::VectorXf& logits = policy.forward(state);
            const Eigen::VectorXf& probs = softmax(logits);

            Eigen::VectorXf gradient = -advantage * (Eigen::VectorXf::Unit(probs.size(), action) - probs);

            static constexpr float max_grad_norm = 0.5f;
            const float norm = gradient.norm();
            if (norm > max_grad_norm)
            {
                gradient = gradient * (max_grad_norm / norm);
            }

            policy.backward(gradient, learning_rate);
        }
    }
}

void reinforce_agent::save(const char* filename) const
{
    policy.save(filename);
}

void reinforce_agent::load(const char* filename)
{
    policy.load(filename);
}
