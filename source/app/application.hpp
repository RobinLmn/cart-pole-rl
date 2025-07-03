#pragma once

class environment;
class agent;

class application
{
public:
    void train(environment* environment, agent* agent, const int episodes, const int batch_size, const float dt);
    void replay(environment* environment, agent* agent, const float dt);
};