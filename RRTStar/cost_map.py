import copy
import math
import os
import sys
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class CostMap:
    """
    Class for creating and manipulating a scenario's cost distribution map
    """

    def __init__(self, x_0, x_f, y_0, y_f):
        self.x_0 = x_0
        self.x_f = x_f
        self.y_0 = y_0
        self.y_f = y_f

        self.x_span = np.linspace(x_0, x_f, 100)
        self.y_span = np.linspace(y_0, y_f, 100)

        self.cost_map = np.zeros((len(self.x_span), len(self.y_span)))
        self.mesh_grid = np.meshgrid(self.x_span, self.y_span)

    class Barrier:
        pass

    class Lane:
        pass

    def get_cost_at_point(self, x, y):
        pass

    def gaussian_raster(self, x, y):
        vol = 1
        sigma = 1
        raster = np.zeros((len(self.x_span), len(self.y_span)))
        for i in range(0, len(self.x_span)):
            for j in range(0, len(self.y_span)):
                raster[i, j] = math.exp(-(((self.x_span[i] - x)**2) / \
                                            (2*sigma**2) + ((self.y_span[j] - y) ** 2 / 2*sigma ** 2))) / vol
        # X, Y = np.meshgrid(self.x_span, self.y_span)
        # raster = math.exp(-(((X - x) ** 2) / (2 * sigma ** 2) + ((Y - y) ** 2 / 2 * sigma ** 2))) / vol
        self.cost_map += raster


class Vehicle:
    def __init__(self, x, y, vel, psi, psi_dot, grid_map):
        self.x = x
        self.y = y
        self.vel = vel
        self.psi = psi
        self.psi_dot = psi_dot

        self.update_vehicle_position(grid_map, self.x, self.y, self.psi)

    def update_vehicle_position(self, grid_map, x_in, y_in, psi_in):

        rotate = np.matrix([[math.cos(psi_in), -math.sin(psi_in)],
                            [math.sin(psi_in), math.cos(psi_in)]])

        px = x_in - 8
        py = y_in - 5
        px_0 = px
        py_0 = py - 10
        s = 1
        Vrel = 20
        P = 0.05 - 0.00025 * Vrel
        sigma_x = .1 + .0092 * Vrel
        sigma_y = .1

        X, Y = np.meshgrid(grid_map.x_span, grid_map.y_span)

        for i in range(0, len(grid_map.x_span)):
            for j in range(0, len(grid_map.y_span)):
                X[i, j] -= x_in
                Y[i, j] -= y_in
                temp = np.matmul(np.array([X[i, j], Y[i, j]]), rotate)
                X[i, j] = temp[0, 0] + x_in
                Y[i, j] = temp[0, 1] + y_in

        Prel = math.sqrt((px - px_0) ** 2 + (py - py_0) ** 2)

        def a(y):
            return (math.log(abs(0.2 * (y - py)))) / sigma_y

        def b(x):
            return (math.log(abs(0.1 * (s * (x - px))))) / sigma_x

        def c(x, y):
            return 4 * math.exp(-P * Prel) / ((1 + math.exp(-P * Prel)) ** 2) * \
                (1 / (2 * math.pi * sigma_y * sigma_x * (0.1 * (s * (x - px))) * (0.2 * (y - py)))) * \
                math.exp(-1 * (a(y) ** 2 + b(x) ** 2) / 2)

        x_min = px - 2
        y_min = py - 2
        max_height = 1.0

        for i in range(0, len(grid_map.x_span)):
            for j in range(0, len(grid_map.y_span)):
                if c(X[i, j], Y[i, j]) > max_height:
                    max_height = c(X[i, j], Y[i, j])

        for i in range(0, len(grid_map.x_span)):
            for j in range(0, len(grid_map.y_span)):

                if X[i, j] >= x_min:
                    if Y[i, j] >= y_min:
                        grid_map.cost_map[i, j] += c(X[i, j], Y[i, j]) / max_height
                        if grid_map.cost_map[i, j].imag != 0:
                            grid_map.cost_map[i, j] = 0
                        if grid_map.cost_map[i, j] < 0:
                            grid_map.cost_map[i, j] = 0

        return grid_map.cost_map


def main():
    pass
    map = CostMap(-20, 20, -20, 20)
    car1 = Vehicle(10, 10, 0, math.pi/3, 0, map)
    car2 = Vehicle(5, 5, 0, math.pi / 2, 0, map)

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(map.x_span, map.y_span)

    ax.contour3D(X, Y, map.cost_map, 50, cmap='jet')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


if __name__ == '__main__':
    main()

