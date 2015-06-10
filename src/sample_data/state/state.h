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
