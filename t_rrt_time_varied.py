"""

Time-Varying T_RRT

Author: Jack O'Neill (jroneill@wpi.edu)

References:
    PythonRobotics - https://github.com/AtsushiSakai/PythonRobotics - Atsushi Sakai(@Atsushi_twi)
    "Transition-based  RRT  for  Path  Planning  in  Continuous  Cost  Spaces" - L ́eonard Jaillet et. al.
    "Dynamic Path Planning and Replanningfor Mobile Robots using RRT*" - Devin Connell et. al.

"""

import copy
import random
from cost_map import *
import matplotlib.pyplot as plt
from rrt import RRT
from t_rrt import TRRT

show_animation = True


class TRRT_TV(TRRT):

    class MyCar:
        def __init__(self):
            self.length = 5
            self.width = 2

    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.t = []
            self.speed = []

            self.cost = 0.0
            self.parent = None
            self.goals = []

    def __init__(self, start, goal, obstacle_list, rand_area,
                 expand_dis=0.5,
                 goal_sample_rate=20,
                 max_iter=10000,
                 connect_circle_dist=10.0,
                 map=CostMapWithTime(0, 50, 0, 50, t_step=0.1),
                 speed_range=[0, 0.1],
                 accel_range=[-9.8, 9.8]
                 ):
        self.speed_range = speed_range
        self.accel_range = accel_range
        super().__init__(start, goal, obstacle_list,
                         rand_area, expand_dis, goal_sample_rate, max_iter,
                         connect_circle_dist, CostMap(rand_area[0], rand_area[1], rand_area[2], rand_area[3]))
        self.connect_circle_dist = connect_circle_dist
        self.map = map

    def planning(self, animation=True, search_until_maxiter=True):
        """
        rrt star path planning

        animation: flag for animation on or off
        search_until_maxiter: search until max iteration for path improving or not
        """
        n_fail = 0
        T = 1
        my_car = self.MyCar()
        print(my_car.length)

        self.node_list = [self.start]
        for t_idx in range(0, len(self.map.t_array) - 1):
            t_now = self.map.t_array[t_idx]
            t_next = self.map.t_array[t_idx + 1]

            for i in range(self.max_iter):
                rnd = self.get_random_point()
                nearest_ind = self.get_nearest_list_index(self.node_list, rnd)
                nearest_node = self.node_list[nearest_ind]
                new_node = self.steer(rnd, nearest_node)

                d, _ = self.calc_distance_and_angle(new_node, nearest_node)
                c_near = self.get_point_cost(nearest_node.x, nearest_node.y, t_now)
                c_new = self.get_point_cost(new_node.x, new_node.y, t_next)
                [trans_test, n_fail, T] = self.transition_test(c_near, c_new, d, cmax=0.5, k=2, t=T, nFail=n_fail)

                if self.check_collision(new_node, self.obstacleList) and trans_test: # and \
                        # not self.map.vehicle_collision(my_car, new_node.x, new_node.y, threshold=0.5):
                    near_inds = self.find_near_nodes(new_node)
                    new_node = self.choose_parent(new_node, near_inds)

                    if new_node:
                        self.node_list.append(new_node)
                        self.rewire(new_node, near_inds)
                else:
                    n_fail += 1

                if animation and i % 1000 == 0:  # draw after every 5 iterations
                    self.draw_graph(t_now, rnd)

                if not search_until_maxiter and new_node:  # check reaching the goal
                    d, _ = self.calc_dist_to_end(new_node)
                    if d <= self.expand_dis:
                        return self.generate_final_course(len(self.node_list) - 1)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            return self.generate_final_course(last_index)

        return None

    def steer(self, rnd, nearest_node):
        new_node = self.Node(rnd[0], rnd[1])
        d, theta = self.calc_distance_speed_and_angle(nearest_node, new_node)
        if d > self.expand_dis and self.speed_range[0] < nearest_node.speed < self.speed_range[1]:
            new_node.x = nearest_node.x + self.expand_dis * math.cos(theta)
            new_node.y = nearest_node.y + self.expand_dis * math.sin(theta)

        return new_node

    def get_point_cost(self, x, y, t):
        t_idx = list(self.map.t_array).index(t)
        j = list(self.map.x_span).index(min(self.map.x_span, key=lambda temp: abs(temp - x)))
        i = list(self.map.y_span).index(min(self.map.y_span, key=lambda temp: abs(temp - y)))
        return self.map.cost_map3d[t_idx][0][i, j]  # This is the cost at the specified time at i and j

    def calc_distance_speed_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.sqrt(dx ** 2 + dy ** 2)
        from_node.speed = d / self.map.t_step
        theta = math.atan2(dy, dx)
        return d, theta

    def draw_graph(self, t, rnd=None):
        plt.clf()
        t_idx = list(self.map.t_array).index(t)

        plt.contourf(self.map.mesh_grid[0], self.map.mesh_grid[1], self.map.cost_map3d[t_idx][0])
        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot([node.x, node.parent.x],
                         [node.y, node.parent.y],
                         "-y")

        for (ox, oy, size) in self.obstacleList:
            plt.plot(ox, oy, "ok", ms=30 * size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis([self.min_rand_x, self.max_rand_x, self.min_rand_y, self.max_rand_y])
        plt.axis('equal')
        plt.grid(True)
        plt.pause(0.01)


def main():
    map_bounds = [0, 20, 0, 20]  # [x_min, x_max, y_min, y_max]
    t_span = [0, 1]

    initial_map = CostMap(map_bounds[0], map_bounds[1], map_bounds[2], map_bounds[3])
    car1 = Vehicle(0, 15, 10, 0, -0.5, initial_map)
    car2 = Vehicle(60, 2, 1, 0, 0, initial_map)
    car3 = Vehicle(20, 10, 5, 0, 0, initial_map)
    # right_barrier = Barrier(0, 2.5, 100, 5, map)
    # left_barrier = Barrier(0, 22.5, 100, 25, map)
    Lane(0, 3.5, 100, 4.5, initial_map, lane_cost=0.25)
    Lane(0, 7.5, 100, 8.5, initial_map, lane_cost=0.25)

    map3d = CostMapWithTime(map_bounds[0], map_bounds[1], map_bounds[2], map_bounds[3], t_step=0.1)

    for t in np.arange(t_span[0], t_span[1], map3d.t_step):
        map3d.update_time(t)
        temp_map = CostMap(map_bounds[0], map_bounds[1], map_bounds[2], map_bounds[3])
        car1.get_future_position(temp_map, map3d.t_step)
        # car2.get_future_position(temp_map, map3d.t_step)
        # car3.get_future_position(temp_map, map3d.t_step)
        map3d.append_time_layer(temp_map)
        print(t)

    time_rrt = TRRT_TV(start=[0, 0],
                       goal=[[20, 20]],
                       rand_area=map_bounds,
                       obstacle_list=[],
                       map=map3d)

    path = time_rrt.planning(animation=show_animation, search_until_maxiter=False)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # Draw final path
        if show_animation:
            for t in time_rrt.map.t_array:
                plt.clf()
                time_rrt.draw_graph(t, rnd=None)
                plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
                plt.grid(True)
                plt.pause(0.5)  # Need for Mac


if __name__ == '__main__':
    main()

