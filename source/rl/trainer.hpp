#pragma once

#include "rl/transition.hpp"
#include "core/log.hpp"

#include <vector>
#include <thread>

template<typename environment_type>
concept environment_concept = std::default_initializable<environment_type> && requires(environment_type environment) {
    { environment.reset() } -> std::same_as<void>;
    { environment.is_done() } -> std::convertible_to<bool>;
    { environment.get_state() } -> std::same_as<std::vector<float>>;
    { environment.step(std::declval<float>(), std::declval<action>()) } -> std::convertible_to<float>;
};

template<typename agent_type>
concept agent_concept = requires(agent_type agent) {
    { agent.act(std::declval<std::vector<float>>()) } -> std::same_as<action>;
    { agent.learn(std::declval<std::vector<transition>>()) } -> std::same_as<void>;
};

class trainer
{
public:
    template<environment_concept environment_type, agent_concept agent_type>
    static void step(environment_type& environment, agent_type& agent, const float dt, std::vector<transition>& transitions);

    template<environment_concept environment_type, agent_concept agent_type, typename functor>
    static void train(agent_type& agent, const float dt, const int batch_count, const int episodes_per_batch, const int learning_step_batch_size, functor&& on_learn);
};

template<environment_concept environment_type, agent_concept agent_type, typename functor>
void trainer::train(agent_type& agent, const float dt, const int batch_count, const int episodes_per_batch, const int learning_step_batch_size, functor&& on_learn)
{
    const auto are_environments_done = [](const std::vector<environment_type>& environments) -> bool
    {
        return std::all_of(environments.begin(), environments.end(), [](const environment_type& environment)
        {
            return environment.is_done();
        });
    };

    std::vector<environment_type> environments(episodes_per_batch);

    for (int batch_index = 0; batch_index < batch_count; ++batch_index)
    {
        for (int episode_index = 0; episode_index < episodes_per_batch; ++episode_index)
        {
            environments[episode_index].reset();
        }

        int learning_step = 0;

        std::vector<std::vector<transition>> transitions_per_episode(episodes_per_batch);

        const auto merge_transitions = [&transitions_per_episode]() -> std::vector<transition> 
        {
            std::vector<transition> transitions;
            for (const std::vector<transition>& episode_transitions : transitions_per_episode) 
            {
                transitions.insert(transitions.end(), episode_transitions.begin(), episode_transitions.end());
            }
            
            transitions_per_episode.clear();
            return transitions;
        };

        while (!are_environments_done(environments))
        {
            std::vector<std::thread> threads;
            threads.reserve(episodes_per_batch);

            for (int episode_index = 0; episode_index < episodes_per_batch; ++episode_index)
            {
                if (environments[episode_index].is_done())
                {
                    continue;
                }

                threads.emplace_back([&, episode_index]() { step(environments[episode_index], agent, dt, transitions_per_episode[episode_index]); });
            }
        
            for (std::thread& thread : threads)
            {
                thread.join();
            }

            learning_step++;

            if (learning_step_batch_size > 0 && learning_step % learning_step_batch_size == 0)
            {
                const std::vector<transition>& transitions = merge_transitions();
                on_learn(transitions, batch_index, learning_step);
                agent.learn(transitions);
            }
        }

        if (learning_step_batch_size <= 0 || learning_step % learning_step_batch_size != 0)
        {
                const std::vector<transition>& transitions = merge_transitions();
                on_learn(transitions, batch_index, learning_step);
                agent.learn(transitions);
        }
    }
}

template<environment_concept environment_type, agent_concept agent_type>
void trainer::step(environment_type& environment, agent_type& agent, const float dt, std::vector<transition>& transitions)
{
    const std::vector<float>& state = environment.get_state();
    const action& action = agent.act(state);

    const float reward = environment.step(dt, action);

    transitions.emplace_back(state, action, reward, environment.is_done());
}
