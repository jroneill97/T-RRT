from t_rrt_time_varied import *

path_file = "lane_change_with_car.txt"

def main():
    with open('saved_paths/' + path_file) as json_file:
        data = json.load(json_file)
        cost_map = np.array(data['cost'])
        mesh_grid = np.array([data['mesh_grid'][0], data['mesh_grid'][1]])
        path = []
        for i in range(0, len(data['t'])):
            point = [data['path']['waypoints'][i][0], data['path']['waypoints'][i][1],
                     data['t'][i], data['path']['heading'][i], data['path']['throttle'][i]]
            path.append(point)

    plt.contourf(mesh_grid[0], mesh_grid[1], cost_map[0], 20, cmap='inferno')
    plt.plot([x for (x, y, t, psi, throttle) in path], [y for (x, y, t, psi, throttle) in path], '-r')
    plt.grid(True)
    plt.xlabel('x (meters)')
    plt.ylabel('y (meters)')
    plt.title(path_file)
    plt.show()


if __name__ == '__main__':
    main()

