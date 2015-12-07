#include "gps_agent_pkg/robotplugin.h"
#include "gps_agent_pkg/sensor.h"
#include "gps_agent_pkg/controller.h"
#include "gps_agent_pkg/positioncontroller.h"
#include "gps_agent_pkg/lingausscontroller.h"
#include "gps_agent_pkg/trialcontroller.h"
#include "gps_agent_pkg/LinGaussParams.h"
#include "gps_agent_pkg/ControllerParams.h"
#include "gps_agent_pkg/util.h"
#include "gps/proto/gps.pb.h"
#include <vector>

using namespace gps_control;

// Plugin constructor.
RobotPlugin::RobotPlugin()
{
    // Nothing to do here, since all variables are initialized in initialize(...)
}

// Destructor.
RobotPlugin::~RobotPlugin()
{
    // Nothing to do here, since all instance variables are destructed automatically.
}

// Initialize everything.
void RobotPlugin::initialize(ros::NodeHandle& n)
{
    ROS_INFO_STREAM("Initializing RobotPlugin");
    data_request_waiting_ = false;

    // Initialize all ROS communication infrastructure.
    initialize_ros(n);

    // Initialize all sensors.
    initialize_sensors(n);

    // Initialize the position controllers.
    // Note that the trial controllers are created from scratch for each trial.
    // However, the position controllers persist, since there is only one type.
    initialize_position_controllers(n);

    // After this, we still need to create the kinematics solvers. How these are
    // created depends on the particular robot, and should be implemented in a
    // subclass.
}

// Initialize ROS communication infrastructure.
void RobotPlugin::initialize_ros(ros::NodeHandle& n)
{
    ROS_INFO_STREAM("Initializing ROS subs/pubs");
    // Create subscribers.
    position_subscriber_ = n.subscribe("/gps_controller_position_command", 1, &RobotPlugin::position_subscriber_callback, this);
    trial_subscriber_ = n.subscribe("/gps_controller_trial_command", 1, &RobotPlugin::trial_subscriber_callback, this);
    test_sub_ = n.subscribe("/test_sub", 1, &RobotPlugin::test_callback, this);
    relax_subscriber_ = n.subscribe("/gps_controller_relax_command", 1, &RobotPlugin::relax_subscriber_callback, this);
    data_request_subscriber_ = n.subscribe("/gps_controller_data_request", 1, &RobotPlugin::data_request_subscriber_callback, this);

    // Create publishers.
    report_publisher_.reset(new realtime_tools::RealtimePublisher<gps_agent_pkg::SampleResult>(n, "/gps_controller_report", 1));
}

// Initialize all sensors.
void RobotPlugin::initialize_sensors(ros::NodeHandle& n)
{
    // Clear out the old sensors.
    sensors_.clear();

    // Create all sensors.
    for (int i = 0; i < 1; i++)
    // TODO: readd this when more sensors work
    //for (int i = 0; i < TotalSensorTypes; i++)
    {
        ROS_INFO_STREAM("creating sensor: " + to_string(i));
        boost::shared_ptr<Sensor> sensor(Sensor::create_sensor((SensorType)i,n,this));
        sensors_.push_back(sensor);
    }

    // Create current state sample and populate it using the sensors.
    current_time_step_sample_.reset(new Sample(MAX_TRIAL_LENGTH));
    initialize_sample(current_time_step_sample_);
}


// Helper method to configure all sensors
void RobotPlugin::configure_sensors(OptionsMap &opts)
{
    ROS_INFO("configure sensors");
    for (int i = 0; i < 1; i++)
    // TODO: readd this when more sensors work
    //for (int i = 0; i < TotalSensorTypes; i++)
    {
        sensors_[i]->configure_sensor(opts);
        sensors_[i]->set_sample_data_format(current_time_step_sample_);
    }
}

// Initialize position controllers.
void RobotPlugin::initialize_position_controllers(ros::NodeHandle& n)
{
    // Create passive arm position controller.
    // TODO: fix this to be something that comes out of the robot itself
    passive_arm_controller_.reset(new PositionController(n, gps::AUXILIARY_ARM, 7));

    // Create active arm position controller.
    active_arm_controller_.reset(new PositionController(n, gps::TRIAL_ARM, 7));
}

// Helper function to initialize a sample from the current sensors.
void RobotPlugin::initialize_sample(boost::scoped_ptr<Sample>& sample)
{
    // Go through all of the sensors and initialize metadata.
    // TODO ZDM :uncomment the following to account for more than joint sensors
    //for (int i = 0; i < TotalSensorTypes; i++)
    for (int i = 0; i < 1; i++)
    {
        sensors_[i]->set_sample_data_format(sample);
    }
    ROS_INFO("set sample data format");
}

// Update the sensors at each time step.
void RobotPlugin::update_sensors(ros::Time current_time, bool is_controller_step)
{
    //if(!is_controller_step){ //TODO: Remove this
        //return;
    //}
    // Update all of the sensors and fill in the sample.
    // TODO ZDM :uncomment the following to account for more than joint sensors
    //for (int sensor = 0; sensor < TotalSensorTypes; sensor++)
    for (int sensor = 0; sensor < 1; sensor++)
    {
        sensors_[sensor]->update(this, current_time, is_controller_step);
        if (trial_controller_ != NULL){
            sensors_[sensor]->set_sample_data(current_time_step_sample_,
                trial_controller_->get_step_counter());
        }
        else {
            sensors_[sensor]->set_sample_data(current_time_step_sample_, 0);
        }
    }

    // If a data request is waiting, publish the sample.
    if (data_request_waiting_) {
        publish_sample_report(current_time_step_sample_);
        data_request_waiting_ = false;
    }
}

// Update the controllers at each time step.
void RobotPlugin::update_controllers(ros::Time current_time, bool is_controller_step)
{
    // Update passive arm controller.
    // TODO - don't pass in wrong sample if used
    passive_arm_controller_->update(this, current_time, current_time_step_sample_, passive_arm_torques_);

    bool trial_init = trial_controller_ != NULL && trial_controller_->is_configured();
    if(!is_controller_step && trial_init){
        return;
    }
    // If we have a trial controller, update that, otherwise update position controller.
    if (trial_init) trial_controller_->update(this, current_time, current_time_step_sample_, active_arm_torques_);
    else active_arm_controller_->update(this, current_time, current_time_step_sample_, active_arm_torques_);

    // Check if the trial controller finished and delete it.
    if (trial_init && trial_controller_->is_finished()) {

        // Publish sample after trial completion
        publish_sample_report(current_time_step_sample_, trial_controller_->get_trial_length());
        //Clear the trial controller.
        trial_controller_->reset(current_time);
        trial_controller_.reset(NULL);

        //Reset the active arm controller.
        active_arm_controller_->reset(current_time);

        // Switch the sensors to run at full frequency.
        for (int sensor = 0; sensor < TotalSensorTypes; sensor++)
        {
            //sensors_[sensor]->set_update(active_arm_controller_->get_update_delay());
        }
    }
    if (active_arm_controller_->report_waiting){
        if (active_arm_controller_->is_finished()){
            publish_sample_report(current_time_step_sample_);
            active_arm_controller_->report_waiting = false;
        }
    }
    if (passive_arm_controller_->report_waiting){
        if (passive_arm_controller_->is_finished()){
            publish_sample_report(current_time_step_sample_);
            passive_arm_controller_->report_waiting = false;
        }
    }

    /* TODO: check is_finished for passive_arm_controller and active_arm_controller */
    /* publish message when finished */
}

void RobotPlugin::publish_sample_report(boost::scoped_ptr<Sample>& sample, int T /*=1*/){
    while(!report_publisher_->trylock());
    std::vector<gps::SampleType> dtypes;
    sample->get_available_dtypes(dtypes);

    report_publisher_->msg_.sensor_data.resize(dtypes.size());
    for(int d=0; d<dtypes.size(); d++){ //Fill in each sample type
        report_publisher_->msg_.sensor_data[d].data_type = dtypes[d];
        Eigen::VectorXd tmp_data;
        sample->get_data(T, tmp_data, (gps::SampleType)dtypes[d]);
        report_publisher_->msg_.sensor_data[d].data.resize(tmp_data.size());


        std::vector<int> shape;
        sample->get_shape((gps::SampleType)dtypes[d], shape);
        shape.insert(shape.begin(), T);
        report_publisher_->msg_.sensor_data[d].shape.resize(shape.size());
        int total_expected_shape = 1;
        for(int i=0; i< shape.size(); i++){
            report_publisher_->msg_.sensor_data[d].shape[i] = shape[i];
            total_expected_shape *= shape[i];
        }
        if(total_expected_shape != tmp_data.size()){
            ROS_ERROR("Data stored in sample has different length than expected (%d vs %d)",
                    tmp_data.size(), total_expected_shape);
        }
        for(int i=0; i<tmp_data.size(); i++){
            report_publisher_->msg_.sensor_data[d].data[i] = tmp_data[i];
        }
    }
    report_publisher_->unlockAndPublish();
}

void RobotPlugin::position_subscriber_callback(const gps_agent_pkg::PositionCommand::ConstPtr& msg){

    ROS_INFO_STREAM("received position command");
    OptionsMap params;
    int8_t arm = msg->arm;
    params["mode"] = msg->mode;
    Eigen::VectorXd data;
    data.resize(msg->data.size());
    for(int i=0; i<data.size(); i++){
        data[i] = msg->data[i];
    }
    params["data"] = data;

    Eigen::MatrixXd pd_gains;
    pd_gains.resize(msg->pd_gains.size() / 4, 4);
    for(int i=0; i<pd_gains.rows(); i++){
        for(int j=0; j<4; j++){
            pd_gains(i, j) = msg->pd_gains[i * 4 + j];
            ROS_INFO("pd_gain[%f]", pd_gains(i, j));
        }
    }
    params["pd_gains"] = pd_gains;

    if(arm == gps::TRIAL_ARM){
        active_arm_controller_->configure_controller(params);
    }else if (arm == gps::AUXILIARY_ARM){
        passive_arm_controller_->configure_controller(params);
    }else{
        ROS_ERROR("Unknown position controller arm type");
    }
}

void RobotPlugin::trial_subscriber_callback(const gps_agent_pkg::TrialCommand::ConstPtr& msg){

    OptionsMap controller_params;
    ROS_INFO_STREAM("received trial command");

    //Read out trial information
    uint32_t T = msg->T;  // Trial length
    if (T > MAX_TRIAL_LENGTH) {
        ROS_FATAL("Trial length specified is longer than maximum trial length (%d vs %d)",
                T, MAX_TRIAL_LENGTH);
    }

    // TODO - it seems like the below could cause a race condition seg fault,
    // but I haven't seen one happen yet...
    initialize_sample(current_time_step_sample_);

    float frequency = msg->frequency;  // Controller frequency

    // Update sensor frequency
    //for (int sensor = 0; sensor < TotalSensorTypes; sensor++)
    for (int sensor = 0; sensor < 1; sensor++)
    {
        sensors_[sensor]->set_update(1.0/frequency);
    }

    std::vector<int> state_datatypes, obs_datatypes;
    state_datatypes.resize(msg->state_datatypes.size());
    for(int i=0; i<state_datatypes.size(); i++){
        state_datatypes[i] = msg->state_datatypes[i];
    }
    controller_params["state_datatypes"] = state_datatypes;
    obs_datatypes.resize(msg->obs_datatypes.size());
    for(int i=0; i<obs_datatypes.size(); i++){
        obs_datatypes[i] = msg->obs_datatypes[i];
    }
    controller_params["obs_datatypes"] = obs_datatypes;

    if(msg->controller.controller_to_execute == gps::LIN_GAUSS_CONTROLLER){
        //
        gps_agent_pkg::LinGaussParams lingauss = msg->controller.lingauss;
        trial_controller_.reset(new LinearGaussianController());
        int dX = (int) lingauss.dX;
        int dU = (int) lingauss.dU;
        //Prepare options map
        controller_params["T"] = (int)lingauss.T;
        controller_params["dX"] = dX;
        controller_params["dU"] = dU;
        for(int t=0; t<(int)lingauss.T; t++){
            Eigen::MatrixXd K;
            K.resize(dU, dX);
            for(int u=0; u<dU; u++){
                for(int x=0; x<dX; x++){
                    K(u,x) = lingauss.K_t[x+u*dX+t*dU*dX];
                }
            }
            Eigen::VectorXd k;
            k.resize(dU);
            for(int u=0; u<dU; u++){
                k(u) = lingauss.k_t[u+t*dU];
            }
            controller_params["K_"+to_string(t)] = K; //TODO: Does this copy or will all values be the same?
            controller_params["k_"+to_string(t)] = k;
        }
        trial_controller_->configure_controller(controller_params);
    }else{
        ROS_ERROR("Unknown trial controller arm type");
    }

    // Configure sensor for trial
    OptionsMap sensor_params;

    // Feed EE points/sites to sensors
    Eigen::MatrixXd ee_points;
    if( msg->ee_points.size() % 3 != 0){
        ROS_ERROR("Got %d ee_points (must be multiple of 3)", (int)msg->ee_points.size());
    }
    int n_points = msg->ee_points.size()/3;
    ee_points.resize(n_points, 3);
    for(int i=0; i<n_points; i++){
        for(int j=0; j<3; j++){
            ee_points(i, j) = msg->ee_points[j+3*i];
        }
    }
    sensor_params["ee_sites"] = ee_points;
    configure_sensors(sensor_params);
}

void RobotPlugin::test_callback(const std_msgs::Empty::ConstPtr& msg){
    ROS_INFO_STREAM("Received test message");
}

void RobotPlugin::relax_subscriber_callback(const gps_agent_pkg::RelaxCommand::ConstPtr& msg){

    ROS_INFO_STREAM("received relax command");
    OptionsMap params;
    int8_t arm = msg->arm;
    params["mode"] = gps::NO_CONTROL;

    if(arm == gps::TRIAL_ARM){
        active_arm_controller_->configure_controller(params);
    }else if (arm == gps::AUXILIARY_ARM){
        passive_arm_controller_->configure_controller(params);
    }else{
        ROS_ERROR("Unknown position controller arm type");
    }
}

void RobotPlugin::data_request_subscriber_callback(const gps_agent_pkg::DataRequest::ConstPtr& msg) {
    ROS_INFO_STREAM("received data request");
    data_request_waiting_ = true;
}

// Get sensor.
Sensor *RobotPlugin::get_sensor(SensorType sensor)
{
    assert(sensor < TotalSensorTypes);
    // TODO: does this need to be a raw pointer?
    return sensors_[sensor].get();
}

// Get forward kinematics solver.
void RobotPlugin::get_fk_solver(boost::shared_ptr<KDL::ChainFkSolverPos> &fk_solver, boost::shared_ptr<KDL::ChainJntToJacSolver> &jac_solver, gps::ActuatorType arm)
{
    //TODO: compile errors related to boost::scoped_ptr
    if (arm == gps::AUXILIARY_ARM)
    {
        fk_solver = passive_arm_fk_solver_;
        jac_solver = passive_arm_jac_solver_;
    }
    else if (arm == gps::TRIAL_ARM)
    {
        fk_solver = active_arm_fk_solver_;
        jac_solver = active_arm_jac_solver_;
    }
    else
    {
        ROS_ERROR("Unknown ArmType %i requested for joint encoder readings!",arm);
    }
}
