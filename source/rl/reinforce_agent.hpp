#pragma once

#include "rl/agent.hpp"
#include "rl/neural_network.hpp"

class reinforce_agent final : public agent
{
public:
    reinforce_agent(const neural_network& policy, const float gamma, const float learning_rate);
    
public:
    [[nodiscard]] virtual action act(const std::vector<float>& state) const final override;
    virtual void learn() final override;

    void save(const char* filename) const;
    void load(const char* filename);

private:
    neural_network policy;

    float gamma; // discount factor
    float learning_rate;
};
