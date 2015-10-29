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

// Destructor -- free up memory here.
NeuralNetworkCaffe::~NeuralNetworkCaffe()
{
}

// Create networks.
void NeuralNetworkCaffe::create_nets()
{
    if (split_net_) {
        // Assume rgb and joint state data in second joint net.
        input_blobs_.resize(2);
    } else {
        // Assume rgb and joint state data.
        input_blobs_.resize(2);
        ROS_INFO("Constructor finished");
    }
}

// Load an rgb neural network from a file.
void NeuralNetworkCaffe::load_rgb_net_from_file(const std::string rgb_caffemodel_file, unsigned width, unsigned height)
{
    // Delete and recreate network.
    create_nets();

    rgb_net_from_file_ = true;

    rgb_net_->CopyTrainedLayersFrom(rgb_caffemodel_file);
    params_set_ = true;

    // WARNING: The following variables are not initialized (don't know their dimension),
    // so get_action_mean will pass the original x parameter into the neural network.
    // scale_
    // bias_
    // x_scaled_
    meanrgb_ = Eigen::VectorXd::Zero(3*width*height);
    meandepth_ = Eigen::VectorXd::Zero(3*width*height);
    scale_bias_set_ = true;
}

// This function returns the number of features extracted from the image.
unsigned NeuralNetworkCaffe::get_feat_count()
{
   const shared_ptr<Blob<float> > last_blob = rgb_net_->blobs().back();
   return (unsigned)(last_blob->channels()*last_blob->width()*last_blob->height());
}

// This function computes rgb features from rgb image.
void NeuralNetworkCaffe::get_rgb_feats(const std::vector<uint8_t> &rgb,unsigned width, unsigned height, vector<float> &feat)
{
    // Check if initialized.
    if (!scale_bias_set_ || !params_set_)
        return;

    // TODO: Don't assume rgb/depth to always be present.
    // Initialize pointer to the rgb data and convert rgb to float.
    // meanrgb_ and meandepth_ are stored as [WIDTH x HEIGHT x CHANNEL]
    // the input (rgb and depth) are stored as [CHANNEL x WIDTH x HEIGHT] (in the format they come from the Kinect)
    float rgb_ptr[3*width*height];
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; (unsigned) h < height; ++h) {
            for (int w = 0; (unsigned) w < width; ++w) {
                int caffe_index = c * height * width + h * width + w;
                int kinect_index = h*width*3 + w*3 + (2-c);
                rgb_ptr[caffe_index] = ((float)rgb[kinect_index]) - ((float)meanrgb_[caffe_index]);
            }
        }
    }

    vector<float*> inputs;
    inputs.push_back(rgb_ptr);

    // Put input blobs into network.
    shared_ptr<MemoryDataLayer<float> > md_layer =
      boost::dynamic_pointer_cast<MemoryDataLayer<float> >(rgb_net_->layers()[0]);
    md_layer->Reset(inputs, 1);

    // Call net forward.
    float initial_loss;
    const vector<Blob<float>*>& output_blobs = rgb_net_->ForwardPrefilled(&initial_loss);

    // Copy output blob to feat.
    for (int i = 0; (unsigned) i < feat.size(); ++i) {
        feat[i] = output_blobs[0]->cpu_data()[i];
    }

    // TODO: try to minimize memory allocations here -- preallocate as much as possible
}

// This function computes feature presence indicators based on the most recent rgb features
// calculated by the neural network. To be called right after get_rgb_feats.
void NeuralNetworkCaffe::get_feat_presence(const std::vector<float> &feat, std::vector<bool> &presence)
{
    ROS_INFO("Getting feat presence");
    // Check softmax output is called conv3
    if (!rgb_net_->has_blob("conv3")) {
        ROS_WARN("Could not find conv3 layer");
        return;
    }

    // Pull it out and compute feature presence indicators for each feature
    shared_ptr<Blob<float> > softmax_blob = rgb_net_->blob_by_name("conv3");
    int sfx_width = softmax_blob->shape(-1);
    int sfx_height = softmax_blob->shape(-2);
    // TODO - Add check for gpu vs. cpu.
    const float* softmax_data = softmax_blob->cpu_data();

    for (int f=0; (unsigned) f < feat.size()/2; ++f) {
        // Convert feature point from range [-1,1] to softmax size range
        int feat_x = (int) ((feat[f*2 + 0]+1.0)*sfx_width/2.0);
        int feat_y = (int) ((feat[f*2 + 1]+1.0)*sfx_height/2.0);
        float prob_val = 0;
        for (int x_ind = std::max(0,feat_x-1); x_ind <= std::min(sfx_width-1,feat_x+1); ++x_ind) {
            for (int y_ind = std::max(0,feat_y-1); y_ind <= std::min(sfx_height-1,feat_y+1); ++y_ind) {
                // TODO - call offset function with vector of indices (semi-deprecated)
                int index = softmax_blob->offset(0, f, y_ind, x_ind);
                prob_val += softmax_data[index];
            }
        }
        if (prob_val > 1.0001) ROS_ERROR("Prob value is greater than 1, is softmax output not called conv3?");
        // TODO - don't hardcode this threshold
        presence[f] = (prob_val > feat_presence_thresh_);
    }
}

// This function computes the action from rgb features and joint states.
void NeuralNetworkCaffe::get_action_mean(const Eigen::VectorXd &x, std::vector<float> &rgb_feat, const std::vector<uint8_t> &depth, unsigned width, unsigned height, Eigen::VectorXd &u)
{
    // TODO: Don't assume rgb/depth to always be present.
    // Transform the input by scale and bias.
    // Note that this assumes that all state information that we don't want to feed to the network is stored at the end of the state vector.
    if (!rgb_net_from_file_) {
        assert(x.rows() >= x_scaled_.rows());
        x_scaled_ = scale_*x.segment(0, x_scaled_.rows()) + bias_;
    } else {
        x_scaled_ = x;
    }

    // Initialize pointer to the joint data and convert x_scaled_ to float.
    float x_ptr[x_scaled_.rows()];
    for (int i = 0; i < x_scaled_.rows(); ++i) {
        x_ptr[i] = (float) x_scaled_[i];
    }
    // Initialize pointer to the rgb data and convert rgb to float.
    float* rgb_feat_ptr = rgb_feat.data();

    vector<float*> inputs;
    inputs.push_back(rgb_feat_ptr);
    inputs.push_back(x_ptr);

    // Put input blobs into network.
    shared_ptr<MemoryDataLayer<float> > md_layer =
      boost::dynamic_pointer_cast<MemoryDataLayer<float> >(joint_net_->layers()[0]);
    md_layer->Reset(inputs, 1);

    // Call net forward.
    float initial_loss;
    const vector<Blob<float>*>& output_blobs = joint_net_->ForwardPrefilled(&initial_loss);

    // Copy output blob to u.
    for (int i = 0; i < u.rows(); ++i) {
        u[i] = (double) output_blobs[0]->cpu_data()[i];
    }

    // TODO: try to minimize memory allocations here -- preallocate as much as possible
}


// This function actually computes the action.
void NeuralNetworkCaffe::get_action_mean(const Eigen::VectorXd &x, const std::vector<uint8_t> &rgb, const std::vector<uint8_t> &depth, unsigned width, unsigned height, Eigen::VectorXd &u)
{
    // Caffe::set_phase(Caffe::TEST);

    // TODO: Don't assume rgb/depth to always be present.
    // Transform the input by scale and bias.
    // Note that this assumes that all state information that we don't want to feed to the network is stored at the end of the state vector.
    if (!rgb_net_from_file_) {
        assert(x.rows() >= x_scaled_.rows());
        x_scaled_ = scale_*x.segment(0, x_scaled_.rows()) + bias_;
    } else {
        x_scaled_ = x;
    }

    // Initialize pointer to the joint data and convert x_scaled_ to float.
    float x_ptr[x_scaled_.rows()];
    for (int i = 0; i < x_scaled_.rows(); ++i) {
        x_ptr[i] = (float) x_scaled_[i];
    }
    // Initialize pointer to the rgb data and convert rgb to float.
    // meanrgb_ and meandepth_ are stored as [WIDTH x HEIGHT x CHANNEL]
    // the input (rgb and depth) are stored as [CHANNEL x WIDTH x HEIGHT] (in the format they come from the Kinect)
    float rgb_ptr[3*width*height];
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; (unsigned) h < height; ++h) {
            for (int w = 0; (unsigned) w < width; ++w) {
                int caffe_index = c * height * width + h * width + w;
                int kinect_index = h*width*3 + w*3 + (2-c);
                rgb_ptr[caffe_index] = ((float)rgb[kinect_index]) - ((float)meanrgb_[caffe_index]);
            }
        }
    }

    // Initialize input blobs and copy data to input blobs.
    vector<float*> inputs;
    inputs.push_back(rgb_ptr);
    inputs.push_back(x_ptr);

    // Put input blobs into network.
    shared_ptr<MemoryDataLayer<float> > md_layer =
      boost::dynamic_pointer_cast<MemoryDataLayer<float> >(net_->layers()[0]);
    md_layer->Reset(inputs, 1);

    // Call net forward.
    float initial_loss;
    const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled(&initial_loss);

    // Copy output blob to u.
    for (int i = 0; i < u.rows(); ++i) {
        u[i] = (double) output_blobs[0]->cpu_data()[i];
    }

    // TODO: try to minimize memory allocations here -- preallocate as much as possible
}

// Set the parameters on the network (and create it as necessary).
void NeuralNetworkCaffe::set_params(const std::string proto_string)
{
    // Delete and recreate network.
    create_nets();

    // When we have data:
    ROS_INFO("Reading model weights");
    if (!split_net_)
    {
        ROS_INFO("Reading weights for single network");
        NetParameter net_param;
        google::protobuf::TextFormat::ParseFromString(proto_string, &net_param);
        net_->CopyTrainedLayersFrom(net_param);
    } else {
        ROS_INFO("Reading weights for network and copied to both");
        NetParameter net_param;
        google::protobuf::TextFormat::ParseFromString(proto_string, &net_param);
        rgb_net_->CopyTrainedLayersFrom(net_param);
        joint_net_->CopyTrainedLayersFrom(net_param);
    }
    ROS_INFO("Parameters set successfully");
    // ROS_INFO("Loading Caffe model");
    // ROS_INFO("String: %s",string.c_str());
    // net_->CopyTrainedLayersFrom("/u/svlevine/matnet_iter_500.caffemodel");
    weights_set_ = true;
}
