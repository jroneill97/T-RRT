"""

Path planning Sample Code with RRT*

Original author: Atsushi Sakai(@Atsushi_twi)
Modified for T-RRT: Jack O'Neill (jroneill@wpi.edu)

"""
import copy
import random
from cost_map import *
import matplotlib.pyplot as plt
from rrt import RRT

show_animation = True


class TRRT(RRT):
    """
    Class for RRT Star planning
    """
    class MyCar:
        def __init__(self):
            self.length = 5
            self.width = 2

    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y

            self.cost = 0.0
            self.parent = None
            self.goals = []

    def __init__(self, start, goal, obstacle_list, rand_area,
                 expand_dis=0.5,
                 goal_sample_rate=20,
                 max_iter=100000,
                 connect_circle_dist=10.0,
                 map=CostMap(0, 50, 0, 50)
                 ):
        self.map = map
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
        n_fail = 0
        T = 1
        my_car = self.MyCar()
        print(my_car.length)

        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd = self.get_random_point()
            nearest_ind = self.get_nearest_list_index(self.node_list, rnd)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(rnd, nearest_node)

            d, _ = self.calc_distance_and_angle(new_node, nearest_node)
            c_near = self.get_point_cost(nearest_node.x, nearest_node.y)
            c_new = self.get_point_cost(new_node.x, new_node.y)
            [trans_test, n_fail, T] = self.transition_test(c_near, c_new, d, cmax=0.5, k=2, t=T, nFail=n_fail)

            if self.check_collision(new_node, self.obstacleList) and trans_test and \
                    not self.map.vehicle_collision(my_car, new_node.x, new_node.y, threshold=0.5):
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)

                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_inds)
            else:
                n_fail += 1

            if animation and i % 1 == 0:  # draw after every 5 iterations
                self.draw_graph(rnd)

            if not search_until_maxiter and new_node:  # check reaching the goal
                d, _ = self.calc_dist_to_end(new_node)
                if d <= self.expand_dis:
                    return self.generate_final_course(len(self.node_list) - 1)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            return self.generate_final_course(last_index)

        return None

    def min_expand_control(self, c_near, c_new, d_near_new):
        pass

    def transition_test(self, ci, cj, dij, cmax, k, t, n_fail):
        """
        Note: This does not include nFail or auto-tuning of
        temperature. Refer to pg. 640 of "SAMPLING-BASED PATH PLANNING ON CONFIGURATION-SPACE COSTMAPS"
        to incorporate these features into this function
        """
        alpha = 2
        n_fail_max = 100

        if cj > cmax:
            return [False, n_fail, t]
        if cj < ci:
            t /= alpha
            n_fail = 0
            return [True, n_fail, t]
        if t == 0:
            t = 0.0001
        if dij == 0:
            dij = 0.0001

        p = math.exp((-abs(cj-ci)/dij)/(k*t))
        if random.uniform(0, 1) < p:
            return [True, n_fail, t]
        else:
            if n_fail > n_fail_max:
                t *= alpha
                n_fail = 0
            else:
                n_fail += 1
            return [False, n_fail, t]

    def get_point_cost(self, x, y):
        j = list(self.map.x_span).index(min(self.map.x_span, key=lambda temp: abs(temp - x)))
        i = list(self.map.y_span).index(min(self.map.y_span, key=lambda temp: abs(temp - y)))
        return self.map.cost_map[i, j]

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

    map_bounds = [0, 25, 0, 25]  # [x_min, x_max, y_min, y_max]

    # Define map and vehicle layout
    map = CostMap(map_bounds[0], map_bounds[1], map_bounds[2], map_bounds[3])
    Vehicle(25/2, 25/2, 0, 0, 0, map)
    # # Vehicle(45, 2, 0, 0, 0, map)
    # Vehicle(20, 10, 0, 0, 0, map)
    # # right_barrier = Barrier(0, 2.5, 100, 5, map)
    # # left_barrier = Barrier(0, 22.5, 100, 25, map)
    # Lane(0, 3.75, 100, 4.25, map, lane_cost=0.5)
    # Lane(0, 7.75, 100, 8.25, map, lane_cost=0.5)

    rrt = TRRT(start=[0, 0],
               goal=[[25, 25]],
               rand_area=map_bounds,
               obstacle_list=[],
               map=map)
    path = rrt.planning(animation=show_animation, search_until_maxiter=False)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # Draw final path
        if show_animation:
            rrt.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.pause(0.01)  # Need for Mac
            plt.show()


if __name__ == '__main__':
    main()
