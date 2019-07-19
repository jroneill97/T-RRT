"""

Path planning Sample Code with RRT*

author: Atsushi Sakai(@Atsushi_twi)

"""

import copy
import math
import os
import sys
from RRTStar.cost_map import CostMap, Vehicle
import numpy as np

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../Software/PythonRobotics/PathPlanning/RRT/")

try:
    from rrt import RRT
except ImportError:
    raise

show_animation = True


class RRTStar(RRT):
    """
    Class for RRT Star planning
    """

    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.cost = 0.0
            self.parent = None

    def __init__(self, start, goal, obstacle_list, rand_area,
                 expand_dis=0.5,
                 goal_sample_rate=20,
                 max_iter=500,
                 connect_circle_dist=50.0
                 ):
        super().__init__(start, goal, obstacle_list,
                         rand_area, expand_dis, goal_sample_rate, max_iter)
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]

        """

        self.connect_circle_dist = connect_circle_dist

    def planning(self, animation=True, search_until_maxiter=True):
        """
        rrt star path planning

        animation: flag for animation on or off
        search_until_maxiter: search until max iteration for path improving or not
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd = self.get_random_point()
            nearest_ind = self.get_nearest_list_index(self.node_list, rnd)
            new_node = self.steer(rnd, self.node_list[nearest_ind])

            if self.check_collision(new_node, self.obstacleList):
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_inds)

            if animation and i % 5 == 0:
                self.draw_graph(rnd)

            if not search_until_maxiter and new_node:  # check reaching the goal
                d, _ = self.calc_distance_and_angle(new_node, self.end)
                if d <= self.expand_dis:
                    return self.generate_final_course(len(self.node_list) - 1)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            return self.generate_final_course(last_index)

        return None

    def min_expand_control(self, c_near, c_new, d_near_new):
        pass

    def transition_test(self, ci, cj, dij, cmax, k=1, t=1):
        """
        Note: This does not include nFail or auto-tuning of
        temperature. Refer to pg. 640 of "SAMPLING-BASED PATH PLANNING ON CONFIGURATION-SPACE COSTMAPS"
        to incorporate these features into this function
        """
        if cj > cmax:
            return False
        if cj < ci:
            return True
        p = math.exp((-(cj-ci)/dij)/(k*t))
        if math.random(0, 1) < p:
            return True
        else:
            return False

    def calculate_edge_cost(self, node_ind):
        pass

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

        return new_node

    def search_best_goal_node(self):
        dist_to_goal_list = [self.calc_dist_to_goal(n.x, n.y) for n in self.node_list]
        goal_inds = [dist_to_goal_list.index(i) for i in dist_to_goal_list if i <= self.expand_dis]

        if not goal_inds:
            return None

        min_cost = min([self.node_list[i].cost for i in goal_inds])
        for i in goal_inds:
            if self.node_list[i].cost == min_cost:
                return i

        return None

    def find_near_nodes(self, new_node):
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        dist_list = [(node.x - new_node.x) ** 2 +
                     (node.y - new_node.y) ** 2 for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r ** 2]
        return near_inds

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            d, theta = self.calc_distance_and_angle(near_node, new_node)
            new_cost = new_node.cost + d

            if near_node.cost > new_cost:
                if self.check_collision_extend(near_node, theta, d):
                    near_node.parent = new_node
                    near_node.cost = new_cost
                    self.propagate_cost_to_leaves(new_node)

    def propagate_cost_to_leaves(self, parent_node):
        for node in self.node_list:
            if node.parent == parent_node:
                d, _ = self.calc_distance_and_angle(parent_node, node)
                node.cost = parent_node.cost + d
                self.propagate_cost_to_leaves(node)

    def check_collision_extend(self, near_node, theta, d):

        tmp_node = copy.deepcopy(near_node)

        for i in range(int(d / self.expand_dis)):
            tmp_node.x += self.expand_dis * math.cos(theta)
            tmp_node.y += self.expand_dis * math.sin(theta)
            if not self.check_collision(tmp_node, self.obstacleList):
                return False

        return True

    @staticmethod
    def test_cost_distribution(x, y):
        return 


def main():
    print("Start " + __file__)

    # ====Search Path with RRT====
    obstacle_list = [
        (9, 5, 2)
    ]  # [x,y,size(radius)]

    # Define map and vehicle layout
    map = CostMap(-2, 15, -2, 15)
    car1 = Vehicle(6, 8, 0, math.pi/2, 0, map)


    rrt = RRTStar(start=[0, 0],
                  goal=[10, 10],
                  rand_area=[-2, 15],
                  obstacle_list=obstacle_list)
    path = rrt.planning(animation=show_animation, search_until_maxiter=False)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # Draw final path
        if show_animation:
            rrt.draw_graph()
            plt.contourf(map.mesh_grid[0], map.mesh_grid[1], map.cost_map)
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.pause(0.01)  # Need for Mac
            plt.show()


if __name__ == '__main__':
    main()
