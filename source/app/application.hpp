#pragma once

class environment;
class agent;

class application
{
public:
    void run(environment* environment, agent* agent, const int episodes, const int batch_size, const float dt);
};