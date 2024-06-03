import numpy as np
from collections import deque

class StateEstimator:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.positions = deque(maxlen=self.buffer_size)
        self.timestamps = deque(maxlen=self.buffer_size)
    
    def reset(self):
        self.positions.clear()
        self.timestamps.clear()
    
    def add_measurement(self, pos: np.ndarray, timestamp: float):
        if len(self.timestamps)  == 0:
            self.positions.append(pos)
            self.timestamps.append(timestamp)
            return
        
        # remove duplicates
        if timestamp == self.timestamps[-1]:
            self.positions[-1] = pos
            return
        
        # Normal case
        self.positions.append(pos)
        self.timestamps.append(timestamp)

    def estimate_state(self):
        """
        Estimate velocity and acceleration from the position measurements.
        """
        if len(self.positions) < 2:
            return np.zeros(3), np.zeros(3)
        
        # Estimate velocity
        velocities = np.zeros((len(self.positions)-1, 3))
        velocity_measurement_timestamp = np.zeros(len(self.positions)-1)
        for i in range(1, len(self.positions)):
            pos1, pos2 = self.positions[i-1], self.positions[i]
            t1, t2 = self.timestamps[i-1], self.timestamps[i]
            if (t2-t1) == 0:
                raise ValueError("Timestamps must be strictly increasing.")
    
            velocities[i-1] = (pos2 - pos1) / (t2 - t1)
            velocity_measurement_timestamp[i-1] = (t1 + t2) / 2
        velocity = np.mean(velocities, axis=0)
        
        # Estimate accelleration
        if velocities.shape[0] < 2:
            acceleration = np.zeros(3)
        else:
            accelerations = np.zeros((velocities.shape[0]-1, 3))
            for i in range(1, velocities.shape[0]):
                vel1, vel2 = velocities[i-1], velocities[i]
                t1, t2 = velocity_measurement_timestamp[i-1], velocity_measurement_timestamp[i]
                if (t2-t1) == 0:
                    raise ValueError("Timestamps must be strictly increasing.")
                accelerations[i-1] = (vel2 - vel1) / (t2 - t1)
            acceleration = np.mean(accelerations, axis=0)
        
        return velocity, acceleration
