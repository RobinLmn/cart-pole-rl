#pragma once

#include "rl/environment.hpp"
#include "rl/reinforce_agent.hpp"

class cartpole final : public environment
{
public:
    float step(const float dt, const action& action) override;

    void reset() override;

    [[nodiscard]] bool is_done() const override;
    [[nodiscard]] std::vector<float> get_state() const override;

private:
    entity cart;
    entity pole;
};

reinforce_agent create_reinforce_cartpole_agent();
