#pragma once

#include <vector>
#include <variant>

using action = std::variant<int, std::vector<float>>;

struct transition
{
    std::vector<float> state;
    action action;
    float reward;
    bool done;
};

class agent
{
public:
    [[nodiscard]] virtual action act(const std::vector<float>& state) const = 0;
    virtual void learn() = 0;

    void store_transition(const transition& transition);

protected:
    std::vector<transition> transitions;
};
