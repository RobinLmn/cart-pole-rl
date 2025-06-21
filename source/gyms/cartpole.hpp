#pragma once

#include "rl/environment.hpp"
#include "renderer/renderer.hpp"

#include <SFML/Window.hpp>

class cartpole final : public environment
{
public:
    [[nodiscard]] float step(const float dt, const int action) override;

    void reset() override;

    [[nodiscard]] bool is_done() const override;
    [[nodiscard]] std::vector<float> get_state() const override;
    [[nodiscard]] agent create_agent() const override;

private:
    entity cart;
    entity pole;
};
