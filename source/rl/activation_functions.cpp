#include "rl/activation_functions.hpp"

#include "core/log.hpp"

static const activation_function_signature fn_relu = [](const Eigen::MatrixXf& x) { return x.array().cwiseMax(0.0f); };
static const activation_function_signature fn_relu_derivative = [](const Eigen::MatrixXf& x) { return (x.array() > 0.0f).cast<float>(); };

static const activation_function_signature fn_identity = [](const Eigen::MatrixXf& x) { return x; };
static const activation_function_signature fn_identity_derivative = [](const Eigen::MatrixXf& x) { return Eigen::MatrixXf::Ones(x.rows(), x.cols()); };

static const activation_function_signature fn_tanh = [](const Eigen::MatrixXf& x) { return x.array().tanh(); };
static const activation_function_signature fn_tanh_derivative = [](const Eigen::MatrixXf& x) { const Eigen::MatrixXf& tanh_x = x.array().tanh(); return 1.0f - tanh_x.array().square(); };

std::unordered_map<std::string, activation_function> activation_name_map = 
{
    { "relu", { "relu", fn_relu, fn_relu_derivative } },
    { "identity", { "identity", fn_identity, fn_identity_derivative } },
    { "tanh", { "tanh", fn_tanh, fn_tanh_derivative } }
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
