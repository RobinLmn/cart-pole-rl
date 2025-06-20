#pragma once

#include <vector>

class agent
{
public:
   [[nodiscard]] std::vector<float> act(const std::vector<float>& state) const;
    void store_episode(const std::vector<float>& state, const std::vector<float> actions, const float reward);
    void learn();
};
