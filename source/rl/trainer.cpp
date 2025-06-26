#include "rl/trainer.hpp"

#include "core/log.hpp"

#include "rl/environment.hpp"

#include <numeric>

trainer::trainer(environment* env, const int episodes, const int batch_size)
    : episode_count{ episodes }
    , batch_size{ batch_size }
    , agent{ env->create_agent() }
    , current_episode{ 1 }
    , episode_reward{ 0.f }
    , batch_rewards{}
    , env{ env }
{
    env->reset();
}

void trainer::train(const float dt)
{
    while (!step(dt));
}

bool trainer::step(const float dt)
{
    const std::vector<float>& state = env->get_state();
    const int action = agent.act(state);

    const float reward = env->step(dt, action);
    episode_reward += reward;

    agent.store_transition({ state, action, reward, env->is_done() });

    if (!env->is_done())
    {
        return false;
    }

    batch_rewards.push_back(episode_reward);

    if (current_episode % batch_size == 0)
    {
        agent.learn();
        const float avg_reward = std::accumulate(batch_rewards.begin(), batch_rewards.end(), 0.f) / batch_rewards.size();
        LOG_INFO("Batch {} (Episode {}): avg reward = {:.2f}", current_episode / batch_size, current_episode, avg_reward);
        batch_rewards.clear();
    }

    current_episode++;

    env->reset();
    episode_reward = 0;

    return current_episode > episode_count;
}