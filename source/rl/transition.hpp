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
