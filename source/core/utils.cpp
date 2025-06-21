#include "core/utils.hpp"

#include <random>

Eigen::VectorXf eigen(const std::vector<float>& v)
{
    Eigen::VectorXf out(v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        out[i] = v[i];
    }

    return out;
}

std::vector<float> vector(const Eigen::VectorXf& v)
{
    std::vector<float> out(v.size());
    for (size_t i = 0; i < out.size(); ++i)
    {
        out[i] = v[i];
    }

    return out;
}

float random_uniform(const float min, const float max) 
{
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(min, max);
    return dist(rng);
}

float cross(const glm::vec2& a, const glm::vec2& b)
{
    return a.x * b.y - a.y * b.x;
}

glm::vec2 cross(const float s, const glm::vec2& v)
{
    return glm::vec2(-s * v.y, s * v.x);
}

Eigen::VectorXf softmax(const Eigen::VectorXf& x) 
{
    const Eigen::VectorXf exps = (x.array() - x.maxCoeff()).exp();
    return exps / exps.sum();
}

int sample_from_distribution(const Eigen::VectorXf& probs) 
{
    static thread_local std::mt19937 gen(std::random_device{}());
    std::discrete_distribution<int> dist(probs.data(), probs.data() + probs.size());
    return dist(gen);
}