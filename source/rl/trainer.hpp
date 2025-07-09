#pragma once

#include "rl/transition.hpp"
#include "core/log.hpp"

#include <vector>

template<typename environment_type>
concept environment_concept = std::default_initializable<environment_type> && requires(environment_type environment) {
    { environment.reset() } -> std::same_as<void>;
    { environment.is_done() } -> std::convertible_to<bool>;
    { environment.get_state() } -> std::same_as<Eigen::VectorXf>;
    { environment.step(std::declval<float>(), std::declval<int>()) } -> std::convertible_to<float>;
};

template<typename agent_type>
concept agent_concept = requires(agent_type agent) {
    { agent.act(std::declval<Eigen::VectorXf>()) } -> std::same_as<int>;
    { agent.learn(std::declval<std::vector<episode>>()) } -> std::same_as<void>;
};

class trainer
{
public:
    template<environment_concept environment_type, agent_concept agent_type, typename functor>
    static void train(agent_type& agent, const float dt, const int batch_count, const int episodes_per_batch, const int learning_step_batch_size, functor&& on_batch_complete);
};

template<environment_concept environment_type, agent_concept agent_type, typename functor>
void trainer::train(agent_type& agent, const float dt, const int batch_count, const int episodes_per_batch, const int learning_step_batch_size, functor&& on_batch_complete)
{
    const auto are_environments_done = [](const std::vector<environment_type>& environments) -> bool
    {
        return std::ranges::all_of(environments, &environment_type::is_done);
    };

    std::vector<environment_type> environments(episodes_per_batch);

    for (int batch_index = 0; batch_index < batch_count; ++batch_index)
    {    
        for (int episode_index = 0; episode_index < episodes_per_batch; ++episode_index)
        {
            environments[episode_index].reset();
        }

        int learning_step = 0;
        float total_reward = 0.0f;

        std::vector<episode> episodes(episodes_per_batch);

        while (!are_environments_done(environments))
        {
            for (int episode_index = 0; episode_index < episodes_per_batch; ++episode_index)
            {
                environment_type& environment = environments[episode_index];

                if (environment.is_done())
                {
                    continue;
                }

                const Eigen::VectorXf& state = environment.get_state();
                const int action = agent.act(state);

                const float reward = environment.step(dt, action);
                const Eigen::VectorXf& next_state = environment.get_state();

                episodes[episode_index].emplace_back(environment.is_done(), action, reward, state, next_state);

                total_reward += reward;
            }
        
            learning_step++;

            if (learning_step_batch_size > 0 && learning_step % learning_step_batch_size == 0)
            {
                agent.learn(episodes);

                for (episode& episode : episodes)
                {
                    episode.clear();
                }
            }
        }

        if (learning_step_batch_size <= 0 || learning_step % learning_step_batch_size != 0)
        {
            agent.learn(episodes);

            for (episode& episode : episodes)
            {
                episode.clear();
            }
        }

        const float average_reward = total_reward / static_cast<float>(episodes_per_batch);
        on_batch_complete(batch_index, average_reward);
    }
}
