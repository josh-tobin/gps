#include "gps_agent_pkg/neuralnetworkcaffe.h"

using namespace gps_control;

NeuralNetworkCaffe::NeuralNetworkCaffe(const char *model_file, Phase phase)
{
    scale_bias_set_ = false;
    weights_set_ = false;

    ROS_INFO("Constructing Caffe net from file %f", model_file);
    net_.reset(new Net<float>(model_file, phase));

    // If we're not in CPU_ONLY mode, use the GPU
#ifndef CPU_ONLY
    Caffe::set_mode(Caffe::GPU);
#endif
    ROS_INFO("Net constructed");
}

NeuralNetworkCaffe::NeuralNetworkCaffe(NetParameter& model_param)
{
    scale_bias_set_ = false;
    weights_set_ = false;

    ROS_INFO("Constructing Caffe net from net param");
    net_.reset(new Net<float>(model_param));
    // If we're not in CPU_ONLY mode, use the GPU
#ifndef CPU_ONLY
    Caffe::set_mode(Caffe::GPU);
#endif
    ROS_INFO("Net constructed");
}

// Destructor -- free up memory here.
NeuralNetworkCaffe::~NeuralNetworkCaffe()
{
}

// This function computes the action from rgb features and joint states.
void NeuralNetworkCaffe::forward(const Eigen::VectorXd &input, std::vector<float> &feat_input, Eigen::VectorXd &output)
{
    // Transform the input by scale and bias.
    // Note that this assumes that all state information that we don't want to feed to the network is stored at the end of the state vector.
    assert(x.rows() >= input_scaled_.rows());
    input_scaled_ = scale_*input.segment(0, input_scaled_.rows()) + bias_;

    vector<float*> inputs;
    inputs.push_back(feat_input.data());
    inputs.push_back((float*) input_scaled_.data());

    // Put input blobs into network.
    shared_ptr<MemoryDataLayer<float> > md_layer =
      boost::dynamic_pointer_cast<MemoryDataLayer<float> >(net_->layers()[0]);
    md_layer->Reset(inputs, 1);  // Batch size of 1

    // Call net forward.
    float initial_loss;
    const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled(&initial_loss);

    // Copy output blob to u.
    for (int i = 0; i < output.rows(); ++i) {
        output[i] = (double) output_blobs[0]->cpu_data()[i];
    }
}

// F
void NeuralNetworkCaffe::forward(const Eigen::VectorXd &input, Eigen::VectorXd &output)
{
    // Transform the input by scale and bias.
    // Note that this assumes that all state information that we don't want to feed to the network is stored at the end of the state vector.
    assert(x.rows() >= input_scaled_.rows());
    input_scaled_ = scale_*input.segment(0, input_scaled_.rows()) + bias_;

    // Initialize input blobs and copy data to input blobs.
    vector<float*> inputs;
    inputs.push_back((float*) input_scaled_.data());

    // Put input blobs into network.
    shared_ptr<MemoryDataLayer<float> > md_layer =
      boost::dynamic_pointer_cast<MemoryDataLayer<float> >(net_->layers()[0]);
    md_layer->Reset(inputs, 1);

    // Call net forward.
    float initial_loss;
    const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled(&initial_loss);

    // Copy output blob to u.
    for (int i = 0; i < output.rows(); ++i) {
        output[i] = (double) output_blobs[0]->cpu_data()[i];
    }

}

// Set the weights on the network/
void NeuralNetworkCaffe::set_weights(void *weights_ptr)
{
    ROS_INFO("Reading model weights");
    NetParameter net_param;
    std::string *weights = static_cast<std::string*>(weights_ptr);
    const std::string weights_string = *weights;  // Make a copy
    delete weights;

    google::protobuf::TextFormat::ParseFromString(weights_string, &net_param);
    net_->CopyTrainedLayersFrom(net_param);
    ROS_INFO("NN weights set successfully");
    weights_set_ = true;
}
