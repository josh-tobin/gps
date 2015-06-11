/*
How the sensor objects are supposed to work:
- each sensor object may include subscribers
- each sensor object has an update function
- this update function includes the robot's state
- a joint angle sensor will just store the latest input
- a vision sensor will store the latest image from the subscriber
- the vision sensor does pretty much nothing during the update
- this is also true for other subscriber-based sensor, such as F/T sensor
- be sure to add warnings if a subscriber-based sensor has not updated recently
- this is done by storing time stamps, and checking how old the newest message is
- note that updates will happen at up to 1000 Hz, so we won't get messages for each update
- however, we can have the RobotPlugin tell us the frequency of the current controller, and check we have at least that frequency
*/
