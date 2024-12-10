"""

Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)

author: AtsushiSakai(@Atsushi_twi)

"""

import math
import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

show_animation = True


class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z # default z to 0 if not provided
            self.path_x = []
            self.path_y = []
            self.path_z = []
            self.parent = None
            self.p = np.array([x,y,z])

    class AreaBounds:

        def __init__(self, area):
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])
            self.zmin = float(area[4]) 
            self.zmax = float(area[5]) 


    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=3.0,
                 path_resolution=0.5,
                 goal_sample_rate=5,
                 max_iter=500,
                 play_area=None,
                 robot_radius=0.0,
                 ):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        play_area:stay inside this area [xmin,xmax,ymin,ymax]
        robot_radius: robot body modeled as circle with given radius

        """
        self.start = self.Node(start[0], start[1], start[2])
        self.end = self.Node(goal[0], goal[1], goal[2])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []
        self.robot_radius = robot_radius

    def planning(self, animation=True):
        """
        rrt path planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            # print(self.node_list)
           # print("##################################")
            rnd_node = self.get_random_node()
            
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            # print(nearest_ind)
            # print(rnd_node)
            nearest_node = self.node_list[nearest_ind]
            # print(nearest_node, rnd_node)
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)
            # print(self.check_if_outside_play_area(new_node, self.play_area), self.check_collision(
            #        nearest_node.p, new_node.p, self.obstacle_list))
            if self.check_if_outside_play_area(new_node, self.play_area) and \
               self.check_collision(
                   nearest_node.p, new_node.p, self.obstacle_list):
                print(self.obstacle_list)
                self.node_list.append(new_node)

            # if animation and i % 5 == 0:
            #     self.draw_graph(rnd_node)

                if self.calc_dist_to_goal(self.node_list[-1].x,
                                        self.node_list[-1].y,
                                        self.node_list[-1].z) <= self.expand_dis:
                    # final_node = self.steer(self.node_list[-1], self.end,
                    #                         self.expand_dis)
                    if self.check_collision(new_node.p, self.end.p, self.obstacle_list):
                        self.end.parent = new_node
                        return self.generate_final_course(len(self.node_list) - 1)

            # if animation and i % 5:
            #     # self.draw_graph(rnd_node)

        return None  # cannot find path

    def steer(self, from_node, to_node, extend_length=float(1)):



        dist = np.linalg.norm(from_node.p - to_node.p)

        # print(to_node, from_node.p)

        if dist > extend_length:
            diff = from_node.p - to_node.p
            to_node.p = from_node.p - diff / dist * extend_length
        to_node.parent = from_node

   



        return to_node


    def generate_final_course(self, goal_ind):

        path = []
        node = self.end

        if (node.p == node.parent.p).all(): node = node.parent
        while node.parent:
            path.append(node.p)
            node = node.parent
        path.append(self.start.p)
        return np.array(path[::-1])

        

        return path

    def calc_dist_to_goal(self, x, y, z):
        dist = np.sqrt(np.sum((np.array([x,y,z]) - self.end.p) ** 2))
        # Check cause it might be buggy
        return dist

    # def get_random_node(self):
    #     if random.randint(0, 100) > self.goal_sample_rate:
    #         rnd = self.Node(
    #             random.uniform(self.min_rand, self.max_rand),
    #             random.uniform(self.min_rand, self.max_rand))
    #     else:  # goal point sampling
    #         rnd = self.Node(self.end.x, self.end.y)
    #     return rnd
    
    def get_random_node(self):
        if random.random() > self.goal_sample_rate / 100.0:
            rnd = (
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand)
            )
        else:
            rnd = self.end.p  # goal sampling
        return self.Node(rnd[0], rnd[1], rnd[2])


    # def draw_graph(self, rnd=None):
    #     plt.clf()
    #     # for stopping simulation with the esc key.
    #     plt.gcf().canvas.mpl_connect(
    #         'key_release_event',
    #         lambda event: [exit(0) if event.key == 'escape' else None])
    #     if rnd is not None:
    #         plt.plot(rnd.x, rnd.y, "^k")
    #         if self.robot_radius > 0.0:
    #             self.plot_circle(rnd.x, rnd.y, self.robot_radius, '-r')
    #     for node in self.node_list:
    #         if node.parent:
    #             plt.plot(node.path_x, node.path_y, "-g")

    #     for (ox, oy, oz, size) in self.obstacle_list:
    #         self.plot_circle(ox, oy, size)

    #     if self.play_area is not None:
    #         plt.plot([self.play_area.xmin, self.play_area.xmax,
    #                   self.play_area.xmax, self.play_area.xmin,
    #                   self.play_area.xmin],
    #                  [self.play_area.ymin, self.play_area.ymin,
    #                   self.play_area.ymax, self.play_area.ymax,
    #                   self.play_area.ymin],
    #                  "-k")

    #     plt.plot(self.start.x, self.start.y, "xr")
    #     plt.plot(self.end.x, self.end.y, "xr")
    #     plt.axis("equal")
    #     plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
    #     plt.grid(True)
    #     plt.pause(0.01)

    # @staticmethod
    # def plot_circle(x, y, size, color="-b"):  # pragma: no cover
    #     deg = list(range(0, 360, 5))
    #     deg.append(0)
    #     xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
    #     yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
    #     plt.plot(xl, yl, color)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        # print(rnd_node, node_list)
        dlist = [((node.p)[0] - (rnd_node.p)[0])**2 + ((node.p)[1] - (rnd_node.p)[1])**2 + ((node.p)[2] + (rnd_node.p)[2])**2
                 for node in node_list]
        
        min_ind = dlist.index(min(dlist))
        # print(min_ind)

        return min_ind

    @staticmethod
    def check_if_outside_play_area(node, play_area):

        if play_area is None:
            return True  # no play_area was defined, every pos should be ok

        if node.x < play_area.xmin or node.x > play_area.xmax or \
           node.y < play_area.ymin or node.y > play_area.ymax:
            return False  # outside - bad
        else:
            return True  # inside - ok
        # NEED TO ADD Z

    # @staticmethod
    def check_collision(self, near_node, new_node, obs):
        """
        Check for collisions along the path from near_node to new_node.

        obs: List of obstacles where each obstacle is (x, y, z, radius).
        """

        dist = np.linalg.norm(near_node - new_node)
        n = max(2, int(dist / 1))  # Ensure at least two points
        points = np.linspace(near_node, new_node, n)

        for p in points:
            for ox, oy, oz, radius in obs:
                # Check if the point p is within the obstacle's radius
                if np.linalg.norm(np.array([ox, oy, oz]) - p) <= radius:
                    return False  # Collision detected

        return True  # No collision
        




        return True  # safe


    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dz = to_node.z - from_node.z  # Calculate difference in z direction
        d = math.hypot(math.hypot(dx, dy), dz)  # 3D distance
        theta = math.atan2(dy, dx)  # 2D angle
        phi = math.atan2(dz, math.hypot(dx, dy))  # 3D angle (vertical)
        return d, theta, phi


def main(gx=6.0, gy=10.0, gz = 11):
    print("start " + __file__)

    # ====Search Path with RRT====
    obstacleList = [
    (5, 5, 5, 1),  # [x, y, z, radius]
    (3, 6, 7, 2),
    (3, 8, 4, 2),
    (3, 10, 2, 2),
    (7, 5, 3, 2),
    (9, 5, 6, 2),
    (8, 10, 5, 1)
]

    # Set Initial parameters
    rrt = RRT(
        start=[0, 0, 0],
        goal=[gx, gy, gz],
        rand_area=[],
        obstacle_list=obstacleList,
        # play_area=[0, 10, 0, 14]
        robot_radius=0.8
        )
    path = rrt.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")
        print(path)

        # Draw final path
        # if show_animation:
        #     rrt.draw_graph()
        #     plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
        #     plt.grid(True)
        #     plt.pause(0.01)  # Need for Mac
        #     plt.show()


if __name__ == '__main__':
    main()