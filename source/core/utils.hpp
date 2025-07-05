#pragma once

#include <Eigen/Dense>
#include <glm/glm.hpp>
#include <cmath>

#include <vector>

[[nodiscard]] Eigen::VectorXf eigen(const std::vector<float>& v);
[[nodiscard]] std::vector<float> vector(const Eigen::VectorXf& v);

[[nodiscard]] float random_uniform(const float min, const float max);

[[nodiscard]] float cross(const glm::vec2& a, const glm::vec2& b);
[[nodiscard]] glm::vec2 cross(const float s, const glm::vec2& v);

[[nodiscard]] Eigen::VectorXf softmax(const Eigen::VectorXf& x);

[[nodiscard]] int sample_from_distribution(const Eigen::VectorXf& probs);

[[nodiscard]] Eigen::VectorXf flatten(const std::vector<Eigen::VectorXf>& data);