
- This file contains class definitions for CostMap, CostMapWithTime, Vehicle, Lane, and Barrier objects. Instructions for their use are shown below:

# ```CostMap(x_0, x_f, y_0, y_f)```:
Class for creating and manipulating a scenario's cost distribution map. This represents a single time "layer." A CostMapWithTime contains a list of these CostMap objects.

## Constructor:
- x_0, x_f: Map x range, in meters
- y_0, y_f: Map y range, in meters

## Functions:
- Both functions within CostMap are not used in the final RRT modification. They were made for testing purposes.


# ``` CostMapWithTime(x_0, x_f, y_0, y_f, t_step=0.01):```
Creates a layered list of CostMap objects
## Constructor:
- x_0, x_f: Map x range, in meters
- y_0, y_f: Map y range, in meters
- t_step: Time step, in seconds. Default is 0.01 seconds but I normally use 0.25 or 0.5 seconds.

## Functions:
### vehicle_collision(my_vehicle, x, y, t, 
Function for detecting whether there exists a "collision" at the specified point at (x, y, t).
- my_vehicle: MyCar object defined in t_rrt_time_varied.py. Used for accessing the car's length and width.
- x, y, t: positional and time coordinates you'd like to test.
- threshold: Cost value which returns True for a collision. (entire Cost Map should be normalized to a maximum cost of 1). Default is 0.5.

### ```append_time_layer(map)```:
Used for constructing the CostMapWithTime. Not needed for normal setup.

### ``` update_time(t_in)```:
With specified time, updates the current time "t" of the object.

# ``` Barrier(x_0, y_0, x_f, y_f, cost_map) ```
## constructor:
- ```x_0, y_0, x_f, y_f```: x and y ranges, in meters, which define the barrier.
- ```cost_map```: CostMap object you want to add the Barrier to.

## Functions:
- No functions associated with Barrier class

# ```Vehicle(x, y, vel, psi, psi_dot, grid_map)```
- Vehicle class with cost defined by Evan's lognormal distribution.

## Constructor:
- ```x, y```: Starting x and y positions, in meters.
- ```vel, psi```: Starting velocity and heading angle (m/s and radians)
- ```psi_dot```: Heading anglular rate, rad/s
- ```grid_map```: The CostMap object you'd like to insert the Vehicle in. 

## Functions:
### ``` project_vehicle_cost```:
- Used to add the vehicle's cost model into the associated CostMap object.

### ``` get_future_position(grid_map, t_step)```:
- Updates the vehicle's position given its motion parameters specified in the constructor.
- **Note**: This is only needed if the vehicle is not following a saved vehicle trajectory, found in the ```/car_info``` directory.
















