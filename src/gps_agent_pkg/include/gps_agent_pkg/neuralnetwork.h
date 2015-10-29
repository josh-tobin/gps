/*
This is the base class for the neural network object. The neural network object
is a helper class for running neural networks on the robot.
*/
#pragma once

// Headers
#include <Eigen/Dense>
#include <vector>
#include <ros/ros.h>

namespace gps_control
{

class NeuralNetwork {
protected:
    // Internal scales and biases.
    Eigen::MatrixXd scale_;
    Eigen::VectorXd bias_;

public:
    bool scale_bias_set_, weights_set_;
    // Constructor -- doesn't do much, since most of the action is in setting parameters and scales & biases.
    NeuralNetwork(int num_layers, std::vector<int> link_functions, std::vector<int> layer_size, int output_dims, const std::vector<double> data);

    virtual ~NeuralNetwork();

    // Function that takes in an input state and outputs the neural network output action.
    // We will likely want our neural network functionality to be more general than this,
    // but this is the most basic functionality.
    // This should be implemented in the subclass.
    virtual void forward(const Eigen::VectorXd &input, Eigen::VectorXd &output);

    // Set the scales and biases, also preallocate any internal temporaries for fast evaluation.
    // This is implemented in neural_network.cpp
    virtual void set_scalebias(const std::vector<double> data, const std::vector<int> dims);

    // Set the weights of the neural network (and create it if necessary)
    virtual void set_weights(void *weights_ptr);
};

}
