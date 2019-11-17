"""

Time-Varying T_RRT

Author: Jack O'Neill (jroneill@wpi.edu)

References:
    PythonRobotics - https://github.com/AtsushiSakai/PythonRobotics - Atsushi Sakai(@Atsushi_twi)
    "Transition-based  RRT  for  Path  Planning  in  Continuous  Cost  Spaces" - L ́eonard Jaillet et. al.
    "Dynamic Path Planning and Replanning for Mobile Robots using RRT*" - Devin Connell et. al.

"""

import random
from cost_map import *
import matplotlib.pyplot as plt
import json
from t_rrt import TRRT
from actor_motion import *

show_animation = True


class TRRT_TV(TRRT):

    class MyCar:
        def __init__(self):
            self.length = 4
            self.width = 1.75

    class Node:
        def __init__(self, x, y, speed=0.0, accel=0.0, psi=0.0, steer_rate=0.0):
            self.x = x
            self.y = y
            self.t = None  # s
            self.speed = speed  # m/s
            self.accel = accel  # acceleration (m/s^2)
            self.psi = psi  # heading (rad)
            self.steer_rate = steer_rate  # rad/sec

            self.cost = 0.0  # Node cost initially set to 0.0 (max cost normalized to 1.0)
            self.parent = None  # Node initially has no parents
            self.n_children = 0  # Node initially has no children

            self.r = []  # Upper and Lower radial expansion limits for this node [r_min, r_max]
            self.exp_angle = []  # Angular expansion limits for this node (current heading +/- max steer angle)

    def __init__(self, start, goal, obstacle_list, rand_area,
                 expand_dis=0.125,
                 goal_sample_rate=0,
                 max_iter=100000,
                 connect_circle_dist=1.0,
                 map=CostMapWithTime(0, 50, 0, 50, t_step=0.1),
                 speed_range=None,
                 accel_range=None,
                 steer_range=None,
                 steer_rate_range=None,
                 children_per_node=None
                 ):
        if steer_rate_range is None:
            steer_rate_range = [-0.2, 0.2]
        if steer_range is None:
            steer_range = [-0.610865, 0.610865]
        if accel_range is None:
            accel_range = [-8, 4]
        if speed_range is None:
            speed_range = [15, 27]
        if children_per_node is None:
            children_per_node = 1

        self.children_per_node = children_per_node
        self.speed_range = speed_range
        self.accel_range = accel_range
        self.steer_range = steer_range
        self.steer_rate_range = steer_rate_range
        self.expand_range = [map.t_step * speed_range[0], map.t_step * speed_range[1]]

        super().__init__(start, goal, obstacle_list,
                         rand_area, expand_dis, goal_sample_rate, max_iter,
                         connect_circle_dist, CostMap(rand_area[0], rand_area[1], rand_area[2], rand_area[3]))
        self.expand_dis = np.sum(self.expand_range) / 3
        self.connect_circle_dist = connect_circle_dist
        self.map = map
        self.path = []
        self.node_list_min_child = []
        self.goal_difference = [10, 6]  # Allowable area for goal to be met
        self.goal = goal

    ''' Main path planning function'''
    def planning(self, start_speed=15.0, start_psi=0.0, start_throttle=0.0, animation=True,
                 search_until_maxiter=False):
        my_car = self.MyCar()
        self.start.t = 0.0
        self.start.speed = start_speed
        self.start.psi = start_psi
        self.start.throttle = start_throttle
        self.get_r_bounds(self.start)
        self.get_expansion_angle(self.start)
        self.node_list = [self.start]
        self.node_list_min_child = self.node_list

        for i in range(self.max_iter):
            ''' Find the nearest node in the node list to a random node'''
            rnd = self.get_random_point_sector()
            new_node = self.Node(rnd[0], rnd[1])

            ''' Important: sets new node time'''
            nearest_node = self.get_best_node(new_node)
            ref_control = self.refinement_control(nearest_node, self.children_per_node)
            if nearest_node is not None and ref_control is True:

                '''Perform the transition test for the two nodes. Note: adjust k to adjust willingness to change lane'''
                trans_test = self.linear_transition_test(nearest_node, new_node, cmax=0.75, k=0.005, my_vehicle=my_car)

                collision = self.map.vehicle_collision(my_car, new_node.x, new_node.y, new_node.t, threshold=0.75)
                if trans_test and not collision:
                    new_node.parent = nearest_node
                    nearest_node.n_children += 1
                    d, _ = self.calc_distance_and_angle(nearest_node, new_node)
                    self.get_r_bounds(new_node)
                    self.get_expansion_angle(new_node)
                    self.node_list.append(new_node)

                    #  Add the new node to the list of children if the number of children is less than the limit
                    self.node_list_min_child.append(new_node)
                    if nearest_node.n_children > self.children_per_node:
                        self.node_list_min_child.pop(self.node_list_min_child.index(nearest_node))
                    self.end.t = new_node.t + self.map.t_step
                    if self.get_constraint_satisfication(new_node, self.end, goal_check=True):
                        return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 1000 == 0:  # draw after every 1000 iterations
                print(i)
                self.draw_graph(t=0.0, rnd=rnd)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            return None

        return None

    ''' Steer rate minimization function'''
    def minimize_steering_rate(self, d_psi, k=0.05):
        d_t = self.map.t_step

        p = math.exp((-abs(d_psi)/d_t) / k)
        if random.uniform(0, 1) < p:
            return True
        return False

    ''' Jerk minimization function'''
    def minimize_jerk(self, node, accel, k=1.5):  # K = 41.7 to make jerk = 6 a 75% chance of passing
        d_t = self.map.t_step
        jerk = (accel - node.accel) / d_t
        p = math.exp(-abs(jerk) / k)
        if random.uniform(0, 1) < p:
            return True
        return False

    ''' Limits the amount of "refinement" nodes in the tree to encourage exploration rather than refinement'''
    @staticmethod
    def refinement_control(node, n_children):
        # num_ref_nodes = 0
        # dt = self.map.t_step
        # for node in self.node_list:
        #     # if node.n_children > 3:
        #     if node.n_children > 3:
        #         for d in node.children_distances:
        #             if d < node.speed*dt + (0.1*node.throttle)*dt**2:
        #                 num_ref_nodes += 1
        # if num_ref_nodes > ratio*len(self.node_list):
        if node is not None:
            if node.n_children > n_children:
                return False
        return True

    ''' Determines whether the motion constraints are met between two nodes'''
    def get_constraint_satisfication(self, node, new_node, goal_check=False):
        d_t = new_node.t - node.t
        if min(self.map.t_array, key=lambda temp: abs(d_t - temp)) != self.map.t_step or \
                d_t < 0:
            return False
        d, psi_new = self.calc_distance_and_angle(node, new_node)
        speed = d / d_t
        accel = (speed - node.speed) / d_t

        #  Calculate steering angle
        d_psi = psi_new - node.psi
        d_psi_d = d_psi/d_t

        #  If the node is not being checked as a goal node, perform this check
        within_sector = node.exp_angle[0]/2 <= psi_new < node.exp_angle[1]/2
        if not goal_check and \
                self.minimize_steering_rate(d_psi, k=0.05) and \
                self.minimize_jerk(node, accel) and \
                self.steer_rate_range[0] < d_psi_d < self.steer_rate_range[1]:
            new_node.speed = speed
            new_node.psi = psi_new
            new_node.throttle = accel
            return True

        #  If the node being checked is the goal node, perform this check instead of the first one
        if goal_check and \
                ((self.goal[0] - self.goal_difference[0]) <= node.x <= (self.goal_difference[0] + self.goal[0]) and
                 (self.goal[1] - self.goal_difference[1]) <= node.y <= (self.goal_difference[1] + self.goal[1])) and \
                within_sector:
            new_node.speed = speed
            new_node.psi = psi_new
            new_node.throttle = accel
            return True

        return False

    ''' Sets the node's expansion angle given its heading angle'''
    def get_expansion_angle(self, node):
        node.exp_angle = [
            node.psi + self.steer_range[0],
            node.psi + self.steer_range[1]
        ]

    ''' Sets the node's expansion range given its speed and throttle (acceleration)'''
    def get_r_bounds(self, node):
        dt = self.map.t_step
        a = self.accel_range

        r_accel = [
            node.speed*dt + 0.5*a[0]*dt**2,
            node.speed*dt + 0.5*a[1]*dt**2
        ]
        r_speed = [
            self.speed_range[0]*dt,
            self.speed_range[1]*dt
        ]
        node.r = [max([r_accel[0], r_speed[0]]), min([r_accel[1], r_speed[1]])]
        if node.r[0] > node.r[1]:
            node.r = np.flip(node.r)

    ''' Get a random point within any of the sectors in the tree'''
    def get_random_point_sector(self):
        is_outside = True
        while is_outside:
            node_idx = random.randint(0, len(self.node_list_min_child)-1)
            node_rnd = self.node_list_min_child[node_idx]
            r_rnd = random.uniform(node_rnd.r[0], node_rnd.r[1])
            exp_angle_rnd = random.uniform(node_rnd.exp_angle[0], node_rnd.exp_angle[1])

            rnd = [
                node_rnd.x + r_rnd*math.cos(exp_angle_rnd),
                node_rnd.y + r_rnd*math.sin(exp_angle_rnd)
            ]
            if self.map.x_0 < rnd[0] < self.map.x_f and self.map.y_0 < rnd[1] < self.map.y_f:
                is_outside = False

        return rnd

    ''' Determines the closest node to a given node which satisfies the motion constraints'''
    def get_best_node(self, new_node):
        near_nodes = []
        for node in self.node_list:
            new_node.t = node.t + self.map.t_step
            if self.get_constraint_satisfication(node, new_node):
                near_nodes.append(node)

        dlist = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2 for node in near_nodes]

        if dlist:
            minind = dlist.index(min(dlist))
            new_node.t = near_nodes[minind].t + self.map.t_step  # Sets the time for the new node
            return near_nodes[minind]
        else:
            return None

    ''' Distance and angle between two nodes'''
    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.sqrt(dx ** 2 + dy ** 2)
        theta = math.atan2(dy, dx)
        return d, theta

    ''' Method which generates the final path'''
    def generate_final_course(self, goal_ind):
        path = []  # Not adding goal node since it now accepts a range of locations as goal
        # path = [[self.end.x, self.end.y, self.end.t, self.end.psi, 0.0]]
        node = self.node_list[goal_ind]

        while node.parent is not None:
            path.append([node.x, node.y, node.t, node.psi, node.throttle, node.speed])
            node = node.parent
        path.append([node.x, node.y, node.t, node.psi, node.throttle, node.speed])
        self.path = path
        return path

    ''' Path visualization method'''
    def draw_graph(self, t=0.0, rnd=None, projection='2d'):
        if projection == '2d':
            plt.clf()
            plt.ion()
            t_idx = list(self.map.t_array).index(t)
            # plt.plot(rnd[0], rnd[1], "^k")
            plt.contourf(self.map.mesh_grid[0], self.map.mesh_grid[1],
                         self.map.cost_map3d[t_idx][0], 100, cmap='terrain')
            for node in self.node_list:
                if node.parent:
                    plt.plot([node.x, node.parent.x],
                             [node.y, node.parent.y],
                             "-k")
            plt.axis([self.min_rand_x, self.max_rand_x, self.min_rand_y, self.max_rand_y])
            plt.grid(True)
            plt.xlabel('x (meters)')
            plt.ylabel('y (meters)')
            plt.draw()
            plt.pause(0.01)
            plt.grid(False)
            plt.show()
        else:
            plt.clf()
            ax = plt.axes(projection='3d')
            plt.ion()
            t_idx = list(self.map.t_array).index(t)

            # if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")
            plt.contour(self.map.mesh_grid[0], self.map.mesh_grid[1], self.map.cost_map3d[t_idx][0], 20,
                         cmap='terrain')
            for node in self.node_list:
                if node.parent:
                    ax.plot3D([node.x, node.parent.x],
                              [node.y, node.parent.y],
                              [node.t, node.parent.t],
                              "-y")

            plt.axis([self.min_rand_x, self.max_rand_x, self.min_rand_y, self.max_rand_y])
            ax.set_zlim3d(0, self.map.t_array[-1])
            plt.grid(True)
            plt.xlabel('x (meters)')
            plt.ylabel('y (meters)')
            plt.draw()
            plt.pause(0.01)
            plt.show()

    ''' Outputs json path and map data to '/out/path_information.txt' '''
    def write_to_file(self, map3d):
        waypoints = []
        t = []
        heading = []
        throttle = []
        speed = []
        temp_path = self.path
        temp_path.reverse()

        for point in temp_path:
            waypoints.append([point[0], point[1]])
            t.append(point[2])
            heading.append(point[3])
            throttle.append(point[4])
            speed.append(point[5])

        path = {'waypoints': waypoints, 'heading': heading, 'throttle': throttle, 'speed': speed}
        cost = []
        for idx in range(0, len(map3d.cost_map3d)):
            cost.append(map3d.cost_map3d[idx][0].tolist())

        mesh_grid = []
        for idx in range(0, len(map3d.mesh_grid)):
            mesh_grid.append(map3d.mesh_grid[idx].tolist())

        with open('./out/path_information.txt', 'w') as data_file:
            json.dump({'t': t, 'path': path, 'cost': cost, 'mesh_grid': mesh_grid}, data_file,
                      separators=(',', ':'), sort_keys=True, indent=4)


def main():
    t_span = [0, 10]
    t_step = 0.5
    lane_cost = 0.3  # lane line cost (usually between 0.25 to 0.5)
    lane_width = 2  # m
    map_bounds = [0, 130, 0, 6]  # [x_min, x_max, y_min, y_max]

    children_per_node = 1  # The maximum allowable number of children per node on the tree
    starting_speed = 10.0  # Starting speed (m/s)

    initial_map = CostMap(map_bounds[0], map_bounds[1], map_bounds[2], map_bounds[3])

    ''' Get car information from the car_info folder'''
    car_info_1 = ActorMotion(1)
    car_info_2 = ActorMotion(2)
    car_info_3 = ActorMotion(3)

    ''' Vehicle initial conditions setup'''
    car1 = Vehicle(50, 1.5*lane_width, car_info_1.v[0]-5, 0, 0, initial_map)
    car2 = Vehicle(20, 1.5*lane_width, car_info_2.v[0]+1, 0, 0, initial_map)
    car3 = Vehicle(0,  2.5*lane_width, car_info_3.v[0], 0, 0, initial_map)

    ''' Initializing lane lines and barriers'''
    Lane(0, 0*lane_width, 300, 1*lane_width, initial_map, lane_cost)  # lanes are now the actual length of the lane
    Lane(0, 1*lane_width, 300, 2*lane_width, initial_map, lane_cost)
    Lane(0, 2*lane_width, 300, 3*lane_width, initial_map, lane_cost)

    Barrier(0, 0, 300, 0.25, initial_map)
    Barrier(0, 11.75, 300, 12, initial_map)

    map3d = CostMapWithTime(map_bounds[0], map_bounds[1], map_bounds[2], map_bounds[3], t_step=t_step)

    for t in np.arange(t_span[0], t_span[1], map3d.t_step):
        t = round(t, 3)
        print(t)
        map3d.update_time(t)

        '''Add on lanes and barriers'''
        temp_map = CostMap(map_bounds[0], map_bounds[1], map_bounds[2], map_bounds[3])

        Lane(0, 0 * lane_width, 300, 1 * lane_width, temp_map, lane_cost)
        Lane(0, 1 * lane_width, 300, 2 * lane_width, temp_map, lane_cost)
        Lane(0, 2 * lane_width, 300, 3 * lane_width, temp_map, lane_cost)

        Barrier(0, 0, 300, 0.25, temp_map)
        Barrier(0, 11.75, 300, 12, temp_map)

        car1.get_future_position(temp_map, map3d.t_step)
        car2.get_future_position(temp_map, map3d.t_step)
        car3.get_future_position(temp_map, map3d.t_step)

        map3d.append_time_layer(temp_map)

        '''Update car velocities and heading angles from the car_info files'''
        # car1.speed, car1.psi = car_info_1.get_motion_at_t(t)
        car2.speed, car2.psi = car_info_2.get_motion_at_t(t)
        car2.speed *= 1.5
        car3.speed, car3.psi = car_info_3.get_motion_at_t(t)

    path = None
    while not path:
        time_rrt = TRRT_TV(start=[0, 1.5*lane_width],
                           goal=[130, 1.5*lane_width],
                           rand_area=map_bounds,
                           obstacle_list=[],
                           map=map3d,
                           children_per_node=children_per_node)
        path = time_rrt.planning(starting_speed, animation=True, search_until_maxiter=False)

    ''' Output the path and map information to ./out/path_information.txt'''
    time_rrt.write_to_file(map3d)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

    ''' Show resulting path and save the image to ./out/path_output.png'''
    plt.clf()
    time_rrt.draw_graph(t=t, rnd=None)
    plt.plot([x for (x, y, t, psi, throttle, speed) in path], [y for (x, y, t, psi, throttle, speed) in path], '-r')
    plt.pause(t_step / 2)
    plt.show(block=True)
    # fig.savefig('./out/path_output.png')


if __name__ == '__main__':
    main()

