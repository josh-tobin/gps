/*
Helper class for running caffe neural networks on the robot.
*/
#pragma once

// Headers
#include <Eigen/Dense>
#include <ros/ros.h>
#include <vector>
#include "caffe/caffe.hpp"
#include "google/protobuf/text_format.h"
#include "gps_agent_pkg/neuralnetwork.h"

// TODO - two namespaces okay?
using namespace caffe;

namespace gps_control
{

class NeuralNetworkCaffe : public NeuralNetwork {
protected:
    shared_ptr<Net<float> > net_;
    std::vector<shared_ptr<Blob<float> > > input_blobs_;  // preallocated input blobs

public:
    // Constructs caffe network using the specified model file
    NeuralNetworkCaffe(const char *model_file, Phase phase);
    // Constructs caffe network using the specified NetParameter
    NeuralNetworkCaffe(NetParameter& model_param);

    virtual ~NeuralNetworkCaffe();
    virtual void forward(const Eigen::VectorXd &input, Eigen::VectorXd &output);
    virtual void forward(const Eigen::VectorXd &input, std::vector<float> &feat_input, Eigen::VectorXd &output);
    virtual void set_weights(void *weights_ptr);
};

}
