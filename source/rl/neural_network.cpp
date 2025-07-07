#include "rl/neural_network.hpp"

#include "core/log.hpp"

#include <fstream>

namespace
{

Eigen::MatrixXf random_weights(const int input_dim, const int output_dim)
{
    return Eigen::MatrixXf::Random(output_dim, input_dim) * std::sqrt(6.f / (input_dim + output_dim));
}

}

layer::layer(const Eigen::MatrixXf& weights, const Eigen::VectorXf& biases, const char* activation_name)
    : activation{ activation_name }
    , params{ weights, biases }
{
}

layer::layer(const int input_dim, const int output_dim, const char* activation_name)
    : activation{ activation_name }
    , params{ random_weights(input_dim, output_dim), Eigen::VectorXf::Zero(output_dim) }
{
}

Eigen::VectorXf layer::forward(const Eigen::VectorXf& input) const
{
    inputs = input;
    pre_activation = params.weights * input + params.biases;

    return activation.func(pre_activation);
}

std::pair<parameters, Eigen::VectorXf> layer::backward(const Eigen::VectorXf& dA) const
{
    const Eigen::VectorXf dZ = dA.array() * activation.derivative(pre_activation).array();
    const Eigen::MatrixXf dW = dZ * inputs.transpose();
    const Eigen::VectorXf dB = dZ;
    const Eigen::VectorXf dInput = params.weights.transpose() * dZ;

    return { { dW, dB }, dInput };
}

void neural_network::add_layer(const layer& layer)
{
    layers.emplace_back(layer);
}

Eigen::VectorXf neural_network::forward(const Eigen::VectorXf& input) const
{
    Eigen::VectorXf output = input;
    for (const layer& layer : layers)
    {
        output = layer.forward(output);
    }

    return output;
}

std::vector<parameters> neural_network::backward(const Eigen::VectorXf& input_gradient) const
{
    std::vector<parameters> gradients(layers.size());

    Eigen::VectorXf dA = input_gradient;
    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i)
    {
        const auto& [gradient, dInput] = layers[i].backward(dA);
        gradients[i] = gradient;

        dA = dInput;
    }

    return gradients;
}

void neural_network::save(const char* filename) const
{
    std::ofstream out{ filename };

    out << "# layers\n";
    out << layers.size() << "\n";

    for (const auto& layer : layers)
    {
        out << "# activation\n";
        out << layer.activation.name << "\n";

        const Eigen::MatrixXf& weights = layer.params.weights;
        out << "# weights\n";
        out << weights.innerSize() << " " << weights.outerSize() << "\n";
        out << weights << "\n";

        const Eigen::VectorXf& biases = layer.params.biases;
        out << "# biases\n";
        out << biases.size() << "\n";
        out << biases.transpose() << "\n";
    }
}

void neural_network::load(const char* filename)
{
    layers.clear();

    std::ifstream in{ filename };

    std::string skip_line;

     std::getline(in, skip_line);
    int layer_count;
    in >> layer_count;
    in.ignore();

    for (int layer_index = 0; layer_index < layer_count; ++layer_index)
    {
        std::string activation_name;
        std::getline(in, skip_line);
        std::getline(in, activation_name);

        std::getline(in, skip_line);
        int rows, cols;
        in >> rows >> cols;
        in.ignore();

        Eigen::MatrixXf weights(rows, cols);
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                in >> weights(i, j);
            }
        }
           
        in.ignore();

        std::getline(in, skip_line);
        int bias_size;
        in >> bias_size;
        in.ignore();

        Eigen::VectorXf biases(bias_size);
        for (int i = 0; i < bias_size; ++i)
        {
            in >> biases(i);
        }
        in.ignore();

        add_layer({ weights, biases, activation_name.c_str() });
    }
}

parameters operator+(const parameters& lhs, const parameters& rhs)
{
    return { lhs.weights + rhs.weights, lhs.biases + rhs.biases };
}

parameters operator*(const parameters& lhs, float rhs)
{
    return { lhs.weights * rhs, lhs.biases * rhs };
}

parameters operator*(float lhs, const parameters& rhs)
{
    return { lhs * rhs.weights, lhs * rhs.biases };
}

parameters operator/(const parameters& lhs, float rhs)
{
    return { lhs.weights / rhs, lhs.biases / rhs };
}

parameters& operator+=(parameters& lhs, const parameters& rhs)
{
    lhs.weights += rhs.weights;
    lhs.biases += rhs.biases;
    return lhs;
}

parameters& operator*=(parameters& lhs, float rhs)
{
    lhs.weights *= rhs;
    lhs.biases *= rhs;
    return lhs;
}

parameters& operator/=(parameters& lhs, float rhs)
{
    lhs.weights /= rhs;
    lhs.biases /= rhs;
    return lhs;
}

parameters operator-(const parameters& lhs, const parameters& rhs)
{
    return { lhs.weights - rhs.weights, lhs.biases - rhs.biases };
}

void accumulate(std::vector<parameters>& params, const std::vector<parameters>& new_params)
{
    if (params.empty()) 
    {
        params = new_params;
    } 
    else
    {
        for (int i = 0; i < params.size(); ++i)
        {
            params[i] += new_params[i];
        }
    }
}

void normalize(std::vector<parameters>& params, const float count)
{
    for (int i = 0; i < params.size(); ++i)
    {
        params[i] /= count;
    }
}