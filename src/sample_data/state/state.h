/*
Here is how I think the state assembler should work at a high level:
1. take in a list of things to include in the state, e.g.
"joint angles"
"joint angle velocities"
"previous joint angles"
"previous joint angle velocities"
"FK"
"FK velocities"

2. Given sensors from the robot (see controllers dir), assemble X and phi using this list, but store *all* sensors unless explicitly told not to (e.g. to avoid storing images)

3. Some functionality to be able to take the full sensor state and assemble X and phi on demand, in case we change the state representation later...
*/

// TODO: create an enum here that contains all of the available sensors.
// To create a new sensor:
// 1. add it to the enum
// 2. create corresponding sensor object
// 3. Add a call to the constructor of this object in the RobotPlugin
// This should be sufficient, because the number of sensors will be obtained
// automatically from this enum, and the state assembler will automatically
// assemble the state (X and phi) from what is in these enums.
// Each sensor will always have a little bit of meta-data defined, which
// might include things like the dimensions of the image. This meta-data
// needs to be passed around along with the sample... should figure out
// a good way to do this.

