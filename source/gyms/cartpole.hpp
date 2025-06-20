#pragma once

#include "rl/environment.hpp"
#include "renderer/renderer.hpp"

#include <SFML/Window.hpp>

class cartpole final : public environment
{
public:
    [[nodiscard]] virtual float step(const float dt, const std::vector<float>& actions) override;

    virtual void reset() override;

    [[nodiscard]] virtual bool is_done() const override;
    [[nodiscard]] virtual std::vector<float> get_state() override;
    [[nodiscard]] virtual agent create_agent() override;

private:
    entity cart;
    entity pole;
};
