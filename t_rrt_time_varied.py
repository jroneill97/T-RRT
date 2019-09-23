"""

Time-Varying T_RRT

Author: Jack O'Neill (jroneill@wpi.edu)

References:
    PythonRobotics - https://github.com/AtsushiSakai/PythonRobotics - Atsushi Sakai(@Atsushi_twi)
    "Transition-based  RRT  for  Path  Planning  in  Continuous  Cost  Spaces" - L ́eonard Jaillet et. al.
    "Dynamic Path Planning and Replanningfor Mobile Robots using RRT*" - Devin Connell et. al.

"""

import random
from cost_map import *
import matplotlib.pyplot as plt
import json
from t_rrt import TRRT

show_animation = True


class TRRT_TV(TRRT):

    class MyCar:
        def __init__(self):
            self.length = 3
            self.width = 1.5



    class Node:
        def __init__(self, x, y, speed=0.0, psi=0.0, steer_rate=0.0):
            self.x = x
            self.y = y
            self.t = 0.0  # s
            self.speed = speed  # m/s
            self.accel = 0  # m/s^2
            self.psi = psi
            self.steer_rate = steer_rate  # rad/sec

            self.cost = 0.0
            self.parent = None
            self.n_children = 0
            self.goals = []

    def __init__(self, start, goal, obstacle_list, rand_area,
                 expand_dis=0.125,
                 goal_sample_rate=0,
                 max_iter=10000000,
                 connect_circle_dist=1.0,
                 map=CostMapWithTime(0, 50, 0, 50, t_step=0.1),
                 speed_range=[0, 27],
                 accel_range=[-8, 4],
                 steer_range=[-0.610865, 0.610865],
                 steer_rate_range=[-0.2, 0.2]
                 ):
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
        self.goal_difference = [5, 7]  # Allowable area for goal to be met

    def planning(self, animation=True, search_until_maxiter=False):
        """
        rrt star path planning

        animation: flag for animation on or off
        search_until_maxiter: search until max iteration for path improving or not
        """
        n_fail = 0
        my_car = self.MyCar()
        self.start.t = 0.0
        self.start.speed = 20.0
        self.start.psi = 0.0
        self.start.throttle = 0.0
        self.node_list = [self.start]

        for i in range(self.max_iter):
            ''' Find the nearest node in the node list to a random node'''
            rnd = self.get_random_point()
            nearest_ind = self.get_nearest_list_index(self.node_list, rnd)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(rnd, nearest_node)
            new_node.t = nearest_node.t + self.map.t_step

            ref_control = self.refinement_control(ratio=0.5)
            if ref_control:
                [trans_test, n_fail] = self.linear_transition_test(nearest_node, new_node, cmax=1, k=2,
                                                                   n_fail=n_fail, my_vehicle=my_car)
                if trans_test and not self.map.vehicle_collision(my_car, new_node.x, new_node.y, new_node.t,
                                                                 threshold=0.5):
                    near_inds = self.find_near_nodes(new_node)
                    new_node = self.choose_parent(new_node, near_inds)
                    if new_node:
                        self.node_list.append(new_node)
                        self.rewire(new_node, near_inds)
                        self.end.t = new_node.t + self.map.t_step
                        if self.get_constraint_satisfication(new_node, self.end, goal_check=True):
                            return self.generate_final_course(len(self.node_list) - 1)
                else:
                    n_fail += 1

            if animation and i % 1000 == 0:  # draw after every 5 iterations
                print(i)
                self.draw_graph(t=0.0, rnd=rnd)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            return self.generate_final_course(last_index)

        return None

    def minimze_jerk(self, node, accel, k=6):  # K = 41.7 to make jerk = 6 a 75% chance of passing
        d_t = self.map.t_step
        jerk = (accel - node.accel) / d_t
        if abs(jerk) < 10:
            return True
        return False
        # p = math.exp((-abs(accel - node.accel) / d_t) / k)
        # if random.uniform(0, 1) < p:
        #     return True
        # return False

    def refinement_control(self, ratio=0.1):
        num_ref_nodes = 0
        for node in self.node_list:
            if node.n_children > 2:
                num_ref_nodes += 1
        if num_ref_nodes > ratio*len(self.node_list):
            return False
        return True

    def steer(self, rnd, nearest_node):
        new_node = self.Node(rnd[0], rnd[1])
        d, theta = self.calc_distance_and_angle(nearest_node, new_node)
        # if d <= self.expand_dis:  # I set expand_dis to the mean of the expand range BTW
        #     new_node.x = nearest_node.x + self.expand_dis * math.cos(theta)
        #     new_node.y = nearest_node.y + self.expand_dis * math.sin(theta)

        return new_node

    def get_point_cost(self, x, y, t):
        t_idx = list(self.map.t_array).index(min(self.map.t_array, key=lambda temp: abs(temp - t)))
        j = list(self.map.x_span).index(min(self.map.x_span, key=lambda temp: abs(temp - x)))
        i = list(self.map.y_span).index(min(self.map.y_span, key=lambda temp: abs(temp - y)))
        return self.map.cost_map3d[t_idx][0][i, j]  # This is the cost at the specified time at i and j

    def get_constraint_satisfication(self, node, new_node, goal_check=False):
        d_t = new_node.t - node.t
        if min(self.map.t_array, key=lambda temp: abs(d_t - temp)) != self.map.t_step or \
                d_t < 0:
            return False

        d, psi_new = self.calc_distance_and_angle(node, new_node)
        speed = d / d_t
        accel = (speed - node.speed) / d_t
        #  Calculating steering and steer rate
        d_psi = psi_new - node.psi
        d_psi_dot = d_psi / d_t
        if not goal_check and \
                self.minimze_jerk(node, accel) and \
                self.speed_range[0] <= speed <= self.speed_range[1] and \
                self.accel_range[0] <= accel <= self.accel_range[1] and \
                self.steer_rate_range[0] <= d_psi_dot <= self.steer_rate_range[1] and \
                self.steer_range[0] <= d_psi <= self.steer_range[1]:


            new_node.speed = speed
            new_node.psi = psi_new
            new_node.throttle = accel
            return True

        #  If the node being checked is the goal node, perform this check instead of the first one
        if goal_check and \
                ((new_node.x - self.goal_difference[0]) <= node.x <= (self.goal_difference[0] + new_node.x) and
                 (new_node.y - self.goal_difference[1]) <= node.y <= (self.goal_difference[1] + new_node.y)):
            # self.speed_range[0] <= speed <= self.speed_range[1] and \
            # self.accel_range[0] <= accel <= self.accel_range[1] and \
            # self.steer_range[0] <= d_psi <= self.steer_range[1] and \
            new_node.speed = speed
            new_node.psi = psi_new
            new_node.throttle = accel
            return True

        return False

    def find_near_nodes(self, new_node):
        near_inds = []
        for i in range(0, len(self.node_list)):
            if self.get_constraint_satisfication(self.node_list[i], new_node):
                near_inds.append(i)
        return near_inds

    def get_random_point(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = [round(random.uniform(self.min_rand_x, self.max_rand_x), 3),
                   round(random.uniform(self.min_rand_y, self.max_rand_y), 3)]
        else:  # goal point sampling
            rnd = [self.end.x, self.end.y]
        return rnd

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return None

        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            d, theta = self.calc_distance_and_angle(self.node_list[i], new_node)
            if self.check_collision_extend(self.node_list[i], theta, d):
                costs.append(self.node_list[i].cost + d)
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        new_node.cost = min_cost
        min_ind = near_inds[costs.index(min_cost)]
        new_node.parent = self.node_list[min_ind]
        self.node_list[min_ind].n_children += 1

        return new_node

    @staticmethod
    def get_nearest_list_index(node_list, rnd):
        dlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.sqrt(dx ** 2 + dy ** 2)
        theta = math.atan2(dy, dx)
        return d, theta

    def generate_final_course(self, goal_ind):
        path = []  # Not adding goal node since it now accepts a range of locations as goal
        # path = [[self.end.x, self.end.y, self.end.t, self.end.psi, 0.0]]
        node = self.node_list[goal_ind]

        while node.parent is not None:
            path.append([node.x, node.y, node.t, node.psi, node.throttle])
            node = node.parent
        path.append([node.x, node.y, node.t, node.psi, node.throttle])
        self.path = path
        return path

    def draw_graph(self, t=0.0, rnd=None, projection='2d'):
        if projection == '2d':
            plt.clf()
            plt.ion()
            t_idx = list(self.map.t_array).index(t)

            plt.contourf(self.map.mesh_grid[0], self.map.mesh_grid[1], self.map.cost_map3d[t_idx][0], 20, cmap='viridis')
            for node in self.node_list:
                if node.parent:
                    plt.plot([node.x, node.parent.x],
                             [node.y, node.parent.y],
                             "-y")
            plt.axis([self.min_rand_x, self.max_rand_x, self.min_rand_y, self.max_rand_y])
            plt.grid(True)
            plt.xlabel('x (meters)')
            plt.ylabel('y (meters)')
            plt.draw()
            plt.pause(0.01)
            plt.show()
        else:
            plt.clf()
            ax = plt.axes(projection='3d')
            plt.ion()
            t_idx = list(self.map.t_array).index(t)

            # if rnd is not None:
            #     plt.plot(rnd[0], rnd[1], "^k")
            plt.contour(self.map.mesh_grid[0], self.map.mesh_grid[1], self.map.cost_map3d[t_idx][0], 20,
                         cmap='viridis')
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

    def write_to_file(self, map3d):
        waypoints = []
        t = []
        heading = []
        throttle = []
        temp_path = self.path
        temp_path.reverse()

        for point in temp_path:
            waypoints.append([point[0], point[1]])
            t.append(point[2])
            heading.append(point[3])
            throttle.append(point[4])

        path = {'waypoints': waypoints, 'heading': heading, 'throttle': throttle}
        cost = []
        for idx in range(0, len(map3d.cost_map3d)):
            cost.append(map3d.cost_map3d[idx][0].tolist())

        mesh_grid = []
        for idx in range(0, len(map3d.mesh_grid)):
            mesh_grid.append(map3d.mesh_grid[idx].tolist())

        with open('path_information.txt', 'w') as data_file:
            json.dump({'t': t, 'path': path, 'cost': cost, 'mesh_grid': mesh_grid}, data_file,
                      separators=(',', ':'), sort_keys=True, indent=4)


def main():
    map_bounds = [0, 75, 0, 7]  # [x_min, x_max, y_min, y_max]
    t_span = [0, 10]
    t_step = 0.5

    initial_map = CostMap(map_bounds[0], map_bounds[1], map_bounds[2], map_bounds[3])
    car1 = Vehicle(75, 5, -10, 0, 0, initial_map)
    # car2 = Vehicle(50, 5, 5, -np.pi/5, 0, initial_map)
    # Lane(0, 5, 300, 8, initial_map, lane_cost=0.25)
    # Lane(0, -1, 300, 2, initial_map, lane_cost=0.25)
    Barrier(0, 0, 300, 0.25, initial_map)
    Barrier(0, 6.75, 300, 7, initial_map)
    Lane(0, 3.25, 300, 3.75, initial_map, lane_cost=0)


    map3d = CostMapWithTime(map_bounds[0], map_bounds[1], map_bounds[2], map_bounds[3], t_step=t_step)

    for t in np.arange(t_span[0], t_span[1], map3d.t_step):
        print(t)
        map3d.update_time(t)
        temp_map = CostMap(map_bounds[0], map_bounds[1], map_bounds[2], map_bounds[3])

        # Lane(0, 5, 300, 8, temp_map, lane_cost=0.
        # Lane(0, -1, 300, 2, temp_map, lane_cost=0.25)
        Barrier(0, 0, 300, 0.25, temp_map)
        Barrier(0, 6.75, 300, 7, temp_map)
        Lane(0, 3, 300, 4, temp_map, lane_cost=0.)
        car1.get_future_position(temp_map, map3d.t_step)
        # car2.get_future_position(temp_map, map3d.t_step)
        map3d.append_time_layer(temp_map)

    time_rrt = TRRT_TV(start=[0, 5],
                       goal=[[75, 5]],
                       rand_area=map_bounds,
                       obstacle_list=[],
                       map=map3d)

    path = time_rrt.planning(animation=show_animation, search_until_maxiter=False)

    time_rrt.write_to_file(map3d)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

    my_car = TRRT_TV.MyCar()
    if show_animation:
        # for t in map3d.t_array:
        # for (x, y, t, psi, throttle) in reversed(path):
        plt.clf()
        fig = plt.figure()
        time_rrt.draw_graph(t=t, rnd=None)
        plt.plot([x for (x, y, t, psi, throttle) in path], [y for (x, y, t, psi, throttle) in path], '-r')
        # my_car.plot(x, y, psi)
        plt.pause(t_step / 2)
        plt.show(block=True)
        fig.savefig('path_output.png')


if __name__ == '__main__':
    main()

