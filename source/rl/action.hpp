#pragma once

#include <Eigen/Dense>
#include <variant>

enum class action_space
{
    DISCRETE,
    CONTINUOUS
};

struct action
{
    action(int discrete_action) : data(discrete_action) {}
    action(const Eigen::VectorXf& continuous_action) : data(continuous_action) {}
    
public:
    [[nodiscard]] bool is_discrete() const { return std::holds_alternative<int>(data); }
    [[nodiscard]] bool is_continuous() const { return std::holds_alternative<Eigen::VectorXf>(data); }
    
    [[nodiscard]] int as_discrete() const { return std::get<int>(data); }
    [[nodiscard]] const Eigen::VectorXf& as_continuous() const { return std::get<Eigen::VectorXf>(data); }

private:
    std::variant<int, Eigen::VectorXf> data;
};