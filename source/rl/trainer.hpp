#pragma once

#include "rl/agent.hpp"

#include <vector>

class environment;

class trainer
{
public:
    trainer(environment* env, agent* agent, const int episodes, const int batch_size);

public:
    bool step(const float dt);
    void train(const float dt);

private:
    const int episode_count;
    const int batch_size;

    int current_episode;
    float episode_reward;

    std::vector<float> batch_rewards;

    agent* agent;
    environment* environment;
};