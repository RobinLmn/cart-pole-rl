#include "rl/ppo_agent.hpp"

#include "core/log.hpp"
#include "core/utils.hpp"

#include <glm/glm.hpp>
#include <Eigen/Dense>

#include <numeric>

ppo_agent::ppo_agent(const neural_network& actor, const neural_network& critic, const action_space space, const float gamma, const float epsilon, const int ppo_epochs)
    : actor{ actor }
    , critic{ critic }
    , actor_optimizer{ this->actor, 0.0005f }
    , critic_optimizer{ this->critic, 0.001f }
    , space{ space }
    , gamma{ gamma }
    , epsilon{ epsilon }
    , ppo_epochs{ ppo_epochs }
{
}

action ppo_agent::act(const Eigen::VectorXf& state) const
{
    if (space == action_space::DISCRETE)
    {
        const Eigen::VectorXf& logits = actor.forward(state);
        const Eigen::VectorXf& probs = softmax(logits);
        const int discrete_action = sample_from_distribution(probs);
        return action(discrete_action);
    }
    else
    {
        const Eigen::VectorXf& actions = actor.forward(state);

        // Add exploration noise
        Eigen::VectorXf noisy_actions = actions;
        for (int i = 0; i < actions.size(); ++i)
        {
            noisy_actions[i] += normal_distribution(0.0f, 0.1f);
        }

        return action(noisy_actions);
    }
}

void ppo_agent::learn(const std::vector<episode>& episodes)
{
    const int episode_count = static_cast<int>(episodes.size());
    ASSERT(episode_count > 0, return, "PPO requires at least one episode to learn.");

    // Compute returns, values, log probabilities and advantages for each episode
    std::vector<Eigen::VectorXf> returns(episode_count);
    std::vector<Eigen::VectorXf> old_values(episode_count);
    std::vector<Eigen::VectorXf> old_log_probs(episode_count);
    std::vector<Eigen::VectorXf> advantages(episode_count);
    for (int e = 0; e < episode_count; ++e)
    {
        const episode& episode = episodes[e];
        const int episode_length = static_cast<int>(episode.size());
        ASSERT(episode_length > 0, continue, "PPO episode is empty.");
        
        // Compute returns
        float G = 0.0f;
        Eigen::VectorXf episode_returns(episode_length);
        for (int t = episode_length - 1; t >= 0; --t)
        {
            G = episode[t].reward + gamma * G;
            episode_returns[t] = G;
        }

        // Compute old values and log probabilities
        Eigen::VectorXf episode_old_values(episode_length);
        Eigen::VectorXf episode_old_log_probs(episode_length);
        for (int t = 0; t < episode_length; ++t)
        {
            const transition& step = episode[t];
            
            episode_old_values[t] = critic.forward(step.state)(0);

            const Eigen::VectorXf& logits = actor.forward(step.state);
            const Eigen::VectorXf& probs = softmax(logits);
            if (space == action_space::DISCRETE)
            {
                episode_old_log_probs[t] = std::log(std::max(probs(step.action.as_discrete()), 1e-8f));
            }
            else
            {
                // For continuous actions, assume Gaussian policy with unit variance
                const Eigen::VectorXf& continuous_action = step.action.as_continuous();
                float log_prob = 0.0f;
                for (int i = 0; i < continuous_action.size(); ++i)
                {
                    const float diff = continuous_action(i) - logits(i);
                    log_prob += -0.5f * diff * diff;  // Gaussian log probability (ignoring constant terms)
                }
                episode_old_log_probs[t] = log_prob;
            }
        }

        returns[e] = std::move(episode_returns);
        old_values[e] = std::move(episode_old_values);
        old_log_probs[e] = std::move(episode_old_log_probs);
        advantages[e] = returns[e] - old_values[e];
    }

    // Normalize advantages and subtract baseline
    {
        const Eigen::VectorXf& flatten_advantages = flatten(advantages);
        const float mean = flatten_advantages.mean();
        const float stddev = std::sqrt((flatten_advantages.array() - mean).square().mean());
        const float norm = std::max(stddev, 1e-8f);

        for (Eigen::VectorXf& episode_advantages : advantages)
        {
            episode_advantages = episode_advantages.array() - mean;
            episode_advantages /= norm;
        }
    }

    // PPO updates for multiple epochs
    for (int epoch = 0; epoch < ppo_epochs; ++epoch)
    {
        std::vector<parameters> actor_gradients;
        std::vector<parameters> critic_gradients;
        float steps = 0.0f;

        // Accumulate gradients with PPO
        for (int e = 0; e < episode_count; ++e)
        {
            const episode& episode = episodes[e];
            for (int t = 0; t < static_cast<int>(episode.size()); ++t)
            {
                const Eigen::VectorXf& state = episode[t].state;
                const float advantage = advantages[e][t];
                const float return_value = returns[e][t];
                const float old_log_prob = old_log_probs[e][t];

                // Critic update
                const float value = critic.forward(state)(0);
                const float value_loss = return_value - value;
                const Eigen::VectorXf critic_gradient = Eigen::VectorXf::Constant(1, value_loss);
                const std::vector<parameters>& critic_step_gradients = critic.backward(critic_gradient);
                accumulate(critic_gradients, critic_step_gradients);

                // Actor update with PPO clipping
                const Eigen::VectorXf& logits = actor.forward(state);
                float new_log_prob;
                Eigen::VectorXf actor_gradient;
                
                if (space == action_space::DISCRETE)
                {
                    const int action = episode[t].action.as_discrete();
                    const Eigen::VectorXf& probs = softmax(logits);
                    new_log_prob = std::log(std::max(probs(action), 1e-8f));
                    
                    const float ratio = std::exp(new_log_prob - old_log_prob);
                    const float clipped_ratio = std::clamp(ratio, 1.0f - epsilon, 1.0f + epsilon);
                    const float policy_loss = -std::min(ratio * advantage, clipped_ratio * advantage);
                    
                    actor_gradient = policy_loss * (Eigen::VectorXf::Unit(probs.size(), action) - probs);
                }
                else
                {
                    const Eigen::VectorXf& continuous_action = episode[t].action.as_continuous();
                    new_log_prob = 0.0f;
                    for (int i = 0; i < continuous_action.size(); ++i)
                    {
                        const float diff = continuous_action(i) - logits(i);
                        new_log_prob += -0.5f * diff * diff;
                    }
                    
                    const float ratio = std::exp(new_log_prob - old_log_prob);
                    const float clipped_ratio = std::clamp(ratio, 1.0f - epsilon, 1.0f + epsilon);
                    const float policy_loss = -std::min(ratio * advantage, clipped_ratio * advantage);
                    
                    actor_gradient = Eigen::VectorXf::Zero(logits.size());
                    for (int i = 0; i < continuous_action.size(); ++i)
                    {
                        actor_gradient(i) = policy_loss * (continuous_action(i) - logits(i));
                    }
                }
                
                const std::vector<parameters>& actor_step_gradients = actor.backward(actor_gradient);
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
}

void ppo_agent::save(const char* filename) const
{
    actor.save(filename);
}

void ppo_agent::load(const char* filename)
{
    actor.load(filename);
}