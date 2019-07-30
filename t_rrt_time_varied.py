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
            self.t = 0.0  # s
            self.speed = 0  # m/s
            self.accel = 0  # m/s^2
            self.steer_rate = 0  # rad/sec

            self.cost = 0.0
            self.parent = None
            self.goals = []

    def __init__(self, start, goal, obstacle_list, rand_area,
                 expand_dis=0.5,
                 goal_sample_rate=20,
                 max_iter=100000000,
                 connect_circle_dist=1.0,
                 map=CostMapWithTime(0, 50, 0, 50, t_step=0.1),
                 speed_range=[0, 5],
                 accel_range=[-9.8, 9.8]
                 ):
        self.speed_range = speed_range
        self.accel_range = accel_range
        self.expand_range = [map.t_step * speed_range[0], map.t_step * speed_range[1]]

        super().__init__(start, goal, obstacle_list,
                         rand_area, expand_dis, goal_sample_rate, max_iter,
                         connect_circle_dist, CostMap(rand_area[0], rand_area[1], rand_area[2], rand_area[3]))
        self.connect_circle_dist = connect_circle_dist
        self.map = map

    def planning(self, animation=True, search_until_maxiter=False):
        """
        rrt star path planning

        animation: flag for animation on or off
        search_until_maxiter: search until max iteration for path improving or not
        """
        n_fail = 0
        T = 1
        my_car = self.MyCar()
        self.start.t = 0.0
        t_current = 0.0
        self.node_list = [self.start]

        for i in range(self.max_iter):

            ''' Find the nearest node in the node list to a random node'''
            rnd = self.get_random_point()
            nearest_ind = self.get_nearest_list_index(self.node_list, rnd)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(rnd, nearest_node)
            new_node.t = nearest_node.t + self.map.t_step

            d, _ = self.calc_distance_and_angle(new_node, nearest_node)
            c_near = self.get_point_cost(nearest_node.x, nearest_node.y, nearest_node.t)
            c_new = self.get_point_cost(new_node.x, new_node.y, new_node.t)
            [trans_test, n_fail, T] = self.transition_test(c_near, c_new, d, cmax=0.5, k=2, t=T, nFail=n_fail)
            if trans_test:  # and not self.map.vehicle_collision(my_car, new_node.x, new_node.y, threshold=0.5):
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_inds)

                    d, _ = self.calc_dist_to_end(new_node)
                    if d <= self.expand_range[1]:
                        self.end.t = new_node.t + self.map.t_step
                        return self.generate_final_course(len(self.node_list) - 1)
            else:
                n_fail += 1

            if animation and i % 10000 == 0:  # draw after every 5 iterations
                self.draw_graph(t=0.0, rnd=rnd)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            return self.generate_final_course(last_index)

        return None

    def steer(self, rnd, nearest_node):
        new_node = self.Node(rnd[0], rnd[1])
        d, theta = self.calc_distance_and_angle(nearest_node, new_node)
        if d <= self.expand_dis:
            new_node.x = nearest_node.x + self.expand_dis * math.cos(theta)
            new_node.y = nearest_node.y + self.expand_dis * math.sin(theta)

        return new_node

    def get_point_cost(self, x, y, t):
        t_idx = list(self.map.t_array).index(min(self.map.t_array, key=lambda temp: abs(temp - t)))
        j = list(self.map.x_span).index(min(self.map.x_span, key=lambda temp: abs(temp - x)))
        i = list(self.map.y_span).index(min(self.map.y_span, key=lambda temp: abs(temp - y)))
        return self.map.cost_map3d[t_idx][0][i, j]  # This is the cost at the specified time at i and j

    def find_near_nodes(self, new_node):
        dist_list = [math.sqrt((node.x - new_node.x) ** 2 +
                     (node.y - new_node.y) ** 2) for node in self.node_list]
        time_list = [new_node.t - node.t for node in self.node_list]
        speed_list = [dist_list[i] / self.map.t_step for i in range(0, len(dist_list))]
        accel_list = []
        for node in self.node_list:
            d_2, _ = self.calc_distance_and_angle(node, new_node)
            speed_2 = d_2 / self.map.t_step
            if node.parent:
                d_1, _ = self.calc_distance_and_angle(node.parent, node)
                speed_1 = d_1 / self.map.t_step
                accel_list.append((speed_2 - speed_1) / (2 * self.map.t_step))
            else:
                accel_list.append(speed_2 / (2 * self.map.t_step))
        near_inds = []
        for i in range(0, len(dist_list)):
            if self.speed_range[0] <= speed_list[i] <= self.speed_range[1] and \
                    min(self.map.t_array, key=lambda temp: abs(temp - time_list[i])) == self.map.t_step:
                print(speed_list[i])
                near_inds.append(i)

        return near_inds

    def get_random_point(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = [round(random.uniform(self.min_rand_x, self.max_rand_x), 3),
                   round(random.uniform(self.min_rand_y, self.max_rand_y), 3)]
        else:  # goal point sampling
            rnd = [self.end.x, self.end.y, random.uniform(self.map.t_array[0], self.map.t_array[1])]  # Need to find a way to allow any end time
        return rnd

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
        path = [[self.end.x, self.end.y, self.end.t]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y, node.t])
            node = node.parent
        path.append([node.x, node.y, node.t])

        return path

    def draw_graph(self, t=0.0, rnd=None):
        plt.clf()
        ax = plt.axes(projection='3d')
        plt.ion()
        t_idx = list(self.map.t_array).index(t)

        plt.contour(self.map.mesh_grid[0], self.map.mesh_grid[1], self.map.cost_map3d[t_idx][0], 20, cmap='RdGy')
        # if rnd is not None:
        #     plt.plot(rnd[0], rnd[1], "^k")
        for node in self.node_list:
            if node.parent:
                # plt.plot([node.x, node.parent.x],
                #          [node.y, node.parent.y],
                #          "-y")
                ax.plot3D([node.x, node.parent.x],
                          [node.y, node.parent.y],
                          [node.t, node.parent.t],
                          "-y")
                ax.view_init(45, ax.azim + 0.0001)

        plt.axis([self.min_rand_x, self.max_rand_x, self.min_rand_y, self.max_rand_y])
        ax.set_zlim3d(0, self.map.t_array[-1])
        plt.grid(True)
        plt.draw()
        plt.pause(0.01)
        plt.show()


def main():
    map_bounds = [0, 5, 0, 5]  # [x_min, x_max, y_min, y_max]
    t_span = [0, 1]
    t_step = 0.1

    initial_map = CostMap(map_bounds[0], map_bounds[1], map_bounds[2], map_bounds[3])
    # car1 = Vehicle(10, 2.5, 5, 0, 0, initial_map)
    # car2 = Vehicle(60, 2, 1, 0, 0, initial_map)
    # car3 = Vehicle(20, 10, 5, 0, 0, initial_map)
    # right_barrier = Barrier(0, 2.5, 100, 5, map)
    # left_barrier = Barrier(0, 22.5, 100, 25, map)
    Lane(0, 7.5, 100, 8.5, initial_map, lane_cost=0.25)

    map3d = CostMapWithTime(map_bounds[0], map_bounds[1], map_bounds[2], map_bounds[3], t_step=t_step)

    for t in np.arange(t_span[0], t_span[1], map3d.t_step):
        map3d.update_time(t)
        temp_map = CostMap(map_bounds[0], map_bounds[1], map_bounds[2], map_bounds[3])
        # car1.get_future_position(temp_map, map3d.t_step)
        # car2.get_future_position(temp_map, map3d.t_step)
        # car3.get_future_position(temp_map, map3d.t_step)
        map3d.append_time_layer(temp_map)
        print(t)

    time_rrt = TRRT_TV(start=[0, 2.5],
                       goal=[[5, 2.5]],
                       rand_area=map_bounds,
                       obstacle_list=[],
                       map=map3d)

    path = time_rrt.planning(animation=show_animation, search_until_maxiter=False)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

    print(path)
    if show_animation:
        for t in map3d.t_array:
            plt.clf()
            time_rrt.draw_graph(t=t, rnd=None)
            plt.plot([x for (x, y, t) in path], [y for (x, y, t) in path], [t for (x, y, t) in path], '-r')
            plt.pause(t_step / 2)
        plt.show(block=True)


if __name__ == '__main__':
    main()

