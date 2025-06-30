#include "rl/activation_functions.hpp"

#include "core/log.hpp"

activation_function_signature relu = [](const Eigen::MatrixXf& x) { return x.array().cwiseMax(0.0f); };
activation_function_signature relu_derivative = [](const Eigen::MatrixXf& x) { return (x.array() > 0.0f).cast<float>(); };

activation_function_signature identity = [](const Eigen::MatrixXf& x) { return x; };
activation_function_signature identity_derivative = [](const Eigen::MatrixXf& x) { return Eigen::MatrixXf::Ones(x.rows(), x.cols()); };

std::unordered_map<std::string, activation_function> activation_name_map = 
{
    { "relu", { "relu", relu, relu_derivative } },
    { "identity", { "identity", identity, identity_derivative } }
};

activation_function::activation_function(const char* name, const activation_function_signature& func, const activation_function_signature& derivative)
    : name{ name }
    , func{ func }
    , derivative{ derivative }
{
}

activation_function::activation_function(const char* name)
{
    const auto& it = activation_name_map.find(name);
    ASSERT(it != activation_name_map.end(), "Activation name {} is unrecognized.", name);

    *this = it->second;
}
