#pragma once

class environment;

class application
{
public:
    void run(environment* env, const int episodes, const int batch_size, const float dt);
};