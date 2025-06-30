#include "rl/agent.hpp"

void agent::store_transition(const transition& transition)
{
    transitions.emplace_back(transition);
}
