#include "gps_agent_pkg/sample.h"
#include "gps/proto/gps.pb.h"
#include "ros/ros.h"

using namespace gps_control;

Sample::Sample(int T)
{
	ROS_INFO("Initializing Sample with T=%d", T);
	T_ = T;
	internal_data_size_.resize((int)gps::TOTAL_DATA_TYPES);
	internal_data_format_.resize((int)gps::TOTAL_DATA_TYPES);
	//Fill in all possible sample types
	for(int i=0; i<gps::TOTAL_DATA_TYPES; i++){
		SampleList samples_list;
		samples_list.resize(T);
		internal_data_[(gps::SampleType)i] = samples_list;
		internal_data_size_[i] = -1; //initialize to -1
	}
}

Sample::~Sample()
{
}

void* Sample::get_data_pointer(int t, gps::SampleType type)
{
    return NULL;
}

void Sample::set_data(int t, gps::SampleType type, SampleVariant data, int data_size, SampleDataFormat data_format)
{
	if(t >= T_) ROS_ERROR("Out of bounds t: %d/%d", t, T_);
    SampleList samples_list = internal_data_[type];
    samples_list[t] = data;
    internal_data_[type] = samples_list; //TODO: This is probably inefficient. Try to pass pointers
    return;
}

void Sample::get_data(int t, gps::SampleType type, void *data, int data_size, SampleDataFormat data_format) const
{
    return;
}


void Sample::set_meta_data(gps::SampleType type, int data_size, SampleDataFormat data_format, OptionsMap meta_data_)
{
    int type_key = (int) type;
    internal_data_size_[type_key] = data_size;
    internal_data_format_[type_key] = data_format;
    return;
}

void Sample::get_available_dtypes(std::vector<gps::SampleType> &types){
	for(int i=0; i<gps::TOTAL_DATA_TYPES; i++){
		if(internal_data_size_[i] != -1){
			types.push_back((gps::SampleType)i);
		}
	}
}

void Sample::get_meta_data(gps::SampleType type, int &data_size, SampleDataFormat &data_format, OptionsMap &meta_data_) const
{
    return;
}

void Sample::get_state(int t, Eigen::VectorXd &x) const
{
	x.fill(0.0);
    return;
}

void Sample::get_obs(int t, Eigen::VectorXd &obs) const
{
	obs.fill(0.0);
    return;
}

void Sample::get_data_all_timesteps(Eigen::VectorXd &data, gps::SampleType datatype){
	int size = internal_data_size_[(int)datatype];
	data.resize(size*T_);
	std::vector<gps::SampleType> dtype_vector;
	dtype_vector.push_back(datatype);

	Eigen::VectorXd tmp_data;
	for(int t=0; t<T_; t++){
		get_data(t, tmp_data, dtype_vector);
		//Fill in original data
		for(int i=0; i<size; i++){
			data[t*size+i] = tmp_data[i];
		}
	}
}

void Sample::get_data(int t, Eigen::VectorXd &data, std::vector<gps::SampleType> datatypes)
{
	if(t >= T_) ROS_ERROR("Out of bounds t: %d/%d", t, T_);
	//ROS_INFO("Getting data");
    //Calculate size
    int total_size = 0;
	for(int i=0; i<datatypes.size(); i++){
		int dtype = (int)datatypes[i];
		if(dtype >= internal_data_size_.size()){
			ROS_ERROR("Requested size of dtype %d, but internal_data_size_ only has %d elements", dtype,
				internal_data_size_.size());
		}
		total_size += internal_data_size_[dtype];
	}
	ROS_INFO("Total get_data() size:%d", total_size);

	data.resize(total_size);
	data.fill(0.0);

    //Fill in data
    int current_idx = 0;
	for(int i=0; i<datatypes.size(); i++){
		int dtype = (int)datatypes[i];
		if(dtype >= internal_data_.size()){
			ROS_ERROR("Requested internal data of dtype %d, but internal_data_ only has %d elements", dtype,
				internal_data_.size());
		}
		int size = internal_data_size_[dtype];
		//Check that format is a vector
		if(internal_data_format_[dtype] != SampleDataFormatEigenVector){
			ROS_ERROR("Datatypes currently must be in Eigen::Vector format. Offender: dtype=%d", dtype);
		}

		SampleList sample_list = internal_data_[datatypes[i]];
		SampleVariant sample_variant = sample_list[t];
		//Hardcoded Eigen::VectorXd. TODO: Add support for other types?
		Eigen::VectorXd sensor_data = boost::get<Eigen::VectorXd>(sample_variant);
		data.segment(current_idx, size) = sensor_data;
		current_idx += size;
	}

	return;
}

int Sample::get_T(){
	return T_;
}


void Sample::get_action(int, Eigen::VectorXd &u) const
{
    return;
}

