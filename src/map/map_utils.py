# for each component reference point is center of object
#! Todo finetune and fix the components
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection




class Ray:
    """
    Represents a ray in 3D space.

    Attributes:
        start (np.ndarray): The starting point of the ray.
        end (np.ndarray): The ending point of the ray.
    """

    def __init__(self, start: np.ndarray, end: np.ndarray):
        self.start = start
        self.end = end

    def __repr__(self):
        return f"Ray(start={self.start}, end={self.end})"
    
    def plot(self, ax):
        ax.plot([self.start[0], self.end[0]], [self.start[1], self.end[1]], [self.start[2], self.end[2]], color='r')


class OBB:
    def __init__(self, center: np.ndarray, half_sizes:np.ndarray, rotation_matrix:np.ndarray, type="collision"):
        self.center = center.astype(float)
        self.half_sizes = half_sizes.astype(float)
        self.rotation_matrix = rotation_matrix.astype(float)
        self.type = type

    def __repr__(self):
        return f"OBB(center={self.center}, half_sizes={self.half_sizes}, rotation={self.rotation_matrix}, type={self.type})"
    
    def check_collision_with_ray(self, ray: Ray, drone_radius):
        # Transform the ray into the OBB's local coordinate system
        local_start = np.dot(ray.start - self.center, self.rotation_matrix)
        local_end = np.dot(ray.end - self.center, self.rotation_matrix)
        local_direction = local_end - local_start
        
        t_min = 0 # start of the ray
        t_max = 1 # end of the ray

        if self.type == "filling":
            inflated_half_sizes = self.half_sizes
        else:
            inflated_half_sizes = self.half_sizes + drone_radius
        
        box_min = -inflated_half_sizes
        box_max = inflated_half_sizes

        # Check for collision with the OBB
        for i in range(3):
            if abs(local_direction[i]) < 1e-4:
                # Ray is parallel to slab. No hit if origin not within slab
                if local_start[i] < box_min[i] or local_start[i] > box_max[i]:
                    return False
            else:
                inv_d = 1.0 / local_direction[i]
                t1 = (box_min[i] - local_start[i]) * inv_d
                t2 = (box_max[i] - local_start[i]) * inv_d

                t_entry = min(t1, t2)
                t_exit = max(t1, t2)

                t_min = max(t_min, t_entry)
                t_max = min(t_max, t_exit)

                if t_min > t_max:
                    return False
        return 0 <= t_min <= 1 and 0 <= t_max <= 1
    
    def check_collision_with_point(self, point: np.ndarray):
        # Transform the point into the OBB's local coordinate system
        local_point = np.dot(point - self.center, self.rotation_matrix)
        for i in range(3):
            if abs(local_point[i]) > self.half_sizes[i]:
                return False
        return True


    def plot(self, ax):
        # Generate corners of a cuboid before rotation
        corners = np.array([[-1, -1, -1],
                            [1, -1, -1],
                            [1, 1, -1],
                            [-1, 1, -1],
                            [-1, -1, 1],
                            [1, -1, 1],
                            [1, 1, 1],
                            [-1, 1, 1]])  # shape (8,3)
        corners = corners * self.half_sizes  # Scale by half sizes
        corners = corners @ self.rotation_matrix.T  # Rotate the corners
        corners += self.center  # Translate to the center

        # List of sides' polygons
        edges = [[corners[i] for i in [0,1,2,3]], [corners[i] for i in [4,5,6,7]], 
                 [corners[i] for i in [0,1,5,4]], [corners[i] for i in [2,3,7,6]], 
                 [corners[i] for i in [1,2,6,5]], [corners[i] for i in [0,3,7,4]]]

        face_color = 'cyan' if self.type == "collision" else 'green'
        edge_color = 'r' if self.type == "collision" else 'g'
        
        # Plot each polygon of the box
        for edge in edges:
            xs, ys, zs = zip(*edge)
            ax.add_collection3d(Poly3DCollection([list(zip(xs, ys, zs))], facecolors=face_color, linewidths=1, edgecolors=edge_color, alpha=.25))


class Object:
    def __init__(self):
        self.obbs = []
        self.global_center = np.array([0, 0, 0], dtype=float)
        self.global_rotation = np.eye(3)
    
    @staticmethod
    def transform_urdf_component_into_object(component):
        object = Object()
        for _, part in component.items():
            center = np.array(part['position'])
            half_sizes = np.array(part['size']) / 2
            type = part['type']
            rotation_matrix = np.eye(3)  # No rotation initially
            object.add_obb(OBB(center, half_sizes, rotation_matrix, type=type))
        
        return object

    def add_obb(self, obb):
        self.obbs.append(obb)

    def translate(self, vector: np.ndarray):
        """Translates the whole object and its components."""
        self.global_center += vector
        for obb in self.obbs:
            obb.center += vector

    def rotate_z(self, angle, radian=True):
        """Rotates the whole object around the Z-axis at the object's center."""
        if not radian:
            theta = np.radians(angle)
        else:
            theta = angle
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        rotation_z = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])
        self.global_rotation = rotation_z @ self.global_rotation
        # Apply rotation to each OBB
        for obb in self.obbs:
            # Update center relative to global center
            relative_center = obb.center - self.global_center
            obb.center = np.dot(rotation_z, relative_center) + self.global_center
            # Update rotation
            obb.rotation_matrix = np.dot(rotation_z, obb.rotation_matrix)
    
    def check_collision_with_ray(self, ray: Ray):
        for obb in self.obbs:
            if obb.check_collision_with_ray(ray):
                return True
        return False

    def plot(self, ax):
        for obb in self.obbs:
            obb.plot(ax)
        

    def __repr__(self):
        return f"Object(Center={self.global_center}, Rotation={self.global_rotation}, OBBs={self.obbs})"