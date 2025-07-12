#include "rl/actor_critic_agent.hpp"

#include "core/log.hpp"
#include "core/utils.hpp"

#include <glm/glm.hpp>
#include <Eigen/Dense>

actor_critic_agent::actor_critic_agent(const neural_network& actor, const neural_network& critic, const float gamma)
    : actor{ actor }
    , critic{ critic }
    , actor_optimizer{ this->actor, 0.0005f }
    , critic_optimizer{ this->critic, 0.001f }
    , gamma{ gamma }
{
}

int actor_critic_agent::act(const Eigen::VectorXf& state) const
{
    const Eigen::VectorXf& logits = actor.forward(state);
    const Eigen::VectorXf& probs = softmax(logits);

    const int action = sample_from_distribution(probs);
    return action;
}

void actor_critic_agent::learn(const std::vector<episode>& episodes)
{
    const int episode_count = static_cast<int>(episodes.size());
    ASSERT(episode_count > 0, return, "Actor-Critic requires at least one episode to learn.");

    // Compute TD errors
    std::vector<Eigen::VectorXf> td_errors(episode_count);
    for (int e = 0; e < episode_count; ++e)
    {
        const episode& episode = episodes[e];
        const int episode_length = static_cast<int>(episode.size());
        ASSERT(episode_length > 0, continue, "Actor-Critic episode is empty.");

        Eigen::VectorXf td_error(episode_length);
        for (int t = 0; t < episode_length; t++)
        {
            const float value = critic.forward(episode[t].state)(0);
            const float value_next = episode[t].done ? 0.0f : critic.forward(episode[t].next_state)(0);
            td_error[t] = episode[t].reward + gamma * value_next - value;
        }

        td_errors[e] = td_error;
    }

    std::vector<parameters> actor_gradients;
    std::vector<parameters> critic_gradients;
    float steps = 0.0f;

    // Accumulate gradients with Actor-Critic
    for (int e = 0; e < episode_count; ++e)
    {
        const episode& episode = episodes[e];
        for (size_t t = 0; t < episode.size(); ++t)
        {
            const int action = episode[t].action;
            const Eigen::VectorXf& state = episode[t].state;
            const float td_error = td_errors[e][t];

            // Critic update
            const Eigen::VectorXf critic_step_gradient = Eigen::VectorXf::Constant(1, td_error);
            const std::vector<parameters>& critic_step_gradients = critic.backward(critic_step_gradient);
            accumulate(critic_gradients, critic_step_gradients);

            // Actor update
            const Eigen::VectorXf& logits = actor.forward(state);
            const Eigen::VectorXf& probs = softmax(logits);
            const Eigen::VectorXf actor_step_gradient = -td_error * (Eigen::VectorXf::Unit(probs.size(), action) - probs);
            const std::vector<parameters>& actor_step_gradients = actor.backward(actor_step_gradient);
            accumulate(actor_gradients, actor_step_gradients);

            steps += 1.0f;
        }
    }

    normalize(actor_gradients, steps);
    normalize(critic_gradients, steps);

    clip_norm(actor_gradients, 0.5f);
    clip_norm(critic_gradients, 0.5f);

    // Update the weights
    actor_optimizer.step(actor_gradients);
    critic_optimizer.step(critic_gradients);
}

void actor_critic_agent::save(const char* filename) const
{
    actor.save(filename);
}

void actor_critic_agent::load(const char* filename)
{
    actor.load(filename);
}
