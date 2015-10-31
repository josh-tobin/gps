#include "gps_agent_pkg/neuralnetwork.h"

using namespace gps_control;

// Destruct all objects.
NeuralNetwork::~NeuralNetwork()
{
    // Nothing to do here.
}

void NeuralNetwork::forward(const Eigen::VectorXd &input, Eigen::VectorXd &output)
{
    // Nothing to do here.
}

// Set the scales and biases, also preallocate any internal temporaries for fast evaluation.
void NeuralNetwork::set_scalebias(const std::vector<double> data, const std::vector<int> dims)
{
    // Note that we expect there to be two things in data:
    // 0: number of entries in scaling matrix
    // 1: number of entries in bias vector

    // Check validity.
    assert(dims[0] == dims[1]*dims[1]);

    // Initialize readback index
    int idx = 0;

    // Unpack the scaling matrix (stored in column major order)
    scale_.resize(dims[1],dims[1]);
    for (int j = 0; j < dims[1]; ++j)
    {
        for (int i = 0; i < dims[1]; ++i)
        {
            scale_(i,j) = data[idx];
            idx++;
        }
    }

    // Unpack the bias vector
    bias_.resize(dims[1]);
    for (int i = 0; i < dims[1]; ++i)
    {
        bias_(i) = data[idx];
        idx++;
    }

    // Preallocate temporaries
    input_scaled_.resize(dims[1]);
    ROS_INFO("Scale and bias set successfully");
    scale_bias_set_ = true;
}

void NeuralNetwork::set_weights(void *weights_ptr)
{
    // Nothing to do here.
}
