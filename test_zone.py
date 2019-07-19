import sys
sys.path.append("./RRTStar/")



show_animation = True


def main():
    print("Start " + __file__)

    # ====Search Path with RRT====
    obstacle_list = [
        (5, 5, 1),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 2),
        (7, 5, 2),
        (9, 5, 2)
    ]  # [x,y,size(radius)]

    # Set Initial parameters
    rrt = RRT(start=[0, 0],
                  goal=[10, 10],
                  rand_area=[-2, 15],
                  obstacle_list=obstacle_list)
    path = rrt.planning(animation=show_animation, search_until_maxiter=False)


if __name__ == '__main__':
    main()
