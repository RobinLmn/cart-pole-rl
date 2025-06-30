#include "rl/neural_network.hpp"

#include "core/log.hpp"

#include <fstream>

layer::layer(const Eigen::MatrixXf& weights, const Eigen::VectorXf& biases, const char* activation_name)
    : activation{ activation_name }
    , weights{ weights }
    , biases{ biases }
{
}

layer::layer(const int input_dim, const int output_dim, const char* activation_name)
    : activation{ activation_name }
    , weights{ Eigen::MatrixXf::Random(output_dim, input_dim) * std::sqrt(6.f / (input_dim + output_dim)) } // (output_dim x input_dim)
    , biases{ Eigen::VectorXf::Zero(output_dim) } // (output_dim x 1)
{
}

Eigen::MatrixXf layer::forward(const Eigen::MatrixXf& input) const
{
    inputs = input;
    pre_activation = weights * input + biases; // (output_dim x input_dim) * (input_dim x 1) + (output_dim x 1) = (output_dim x 1)
    return activation.func(pre_activation);
}

Eigen::MatrixXf layer::backward(const Eigen::MatrixXf& dA, const float learning_rate)
{
    const Eigen::MatrixXf dZ = dA.array() * activation.derivative(pre_activation).array();  // (output_dim × 1)
    const Eigen::MatrixXf dW = dZ * inputs.transpose();  // (output_dim × input_dim)
    const Eigen::VectorXf dB = dZ.col(0);

    const Eigen::MatrixXf dInput = weights.transpose() * dZ;  // (input_dim × 1)

    weights -= learning_rate * dW;
    biases -= learning_rate * dB;

    return dInput;
}

Eigen::MatrixXf layer::get_weights() const
{
    return weights;
}

Eigen::VectorXf layer::get_biases() const
{
    return biases;
}

const char* layer::get_activation_name() const
{
    return activation.name.c_str();
}

void neural_network::add_layer(const layer& layer)
{
    layers.emplace_back(layer);
}

Eigen::VectorXf neural_network::forward(const Eigen::VectorXf& input) const
{
    Eigen::MatrixXf output_matrix = input;
    for (const layer& layer : layers)
    {
        output_matrix = layer.forward(output_matrix);
    }

    return output_matrix.col(0);
}

void neural_network::backward(const Eigen::VectorXf& gradients, const float learning_rate)
{
    Eigen::MatrixXf gradient_matrix = gradients;
    for (auto layer_it = layers.rbegin(); layer_it != layers.rend(); ++layer_it)
    {
        gradient_matrix = layer_it->backward(gradient_matrix, learning_rate);
    }
}

void neural_network::save(const char* filename) const
{
    std::ofstream out{ filename };

    out << "# layers\n";
    out << layers.size() << "\n";

    for (const auto& layer : layers)
    {
        out << "# activation\n";
        out << layer.get_activation_name() << "\n";

        const Eigen::MatrixXf& weights = layer.get_weights();
        out << "# weights\n";
        out << weights.innerSize() << " " << weights.outerSize() << "\n";
        out << weights << "\n";

        const Eigen::VectorXf& biases = layer.get_biases();
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
