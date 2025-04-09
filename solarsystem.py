import matplotlib.pyplot as plt
import numpy as np


earth_radius = 6.371e6
sun_radius = 696340000
G = 6.6743e-11
mass_of_earth = 5.972e24
mass_of_moon = 7.347e22
earth_moon_distance = 384.4e6
mass_of_sun = 1.989e30
earth_sun_distance = 1.5e8


class Simulation:
    def __init__(self, initial_earth_position, initial_earth_velocity, initial_moon_position):
        self.initial_position = initial_earth_position
        self.initial_velocity = initial_earth_velocity
        self.initial_moon_position = initial_moon_position
        self.initial_angle = 0
    def derive_linear(self, position, velocity):
        r = np.linalg.norm(position)
        return velocity, -G * mass_of_sun * position/ r**3

    def derive_angular(self, beta):
        return np.sqrt(G*mass_of_earth/earth_moon_distance**3)

    @staticmethod
    def euler_step(position, velocity, k_position, k_velocity, beta, k_beta, time_step):
        position_prime = position + k_position * time_step
        velocity_prime = velocity + k_velocity * time_step
        beta_prime = beta + k_beta * time_step * 3600
        return position_prime, velocity_prime, beta_prime
    def improved_euler_step(self, position, velocity,beta, time_step):
        k1_position, k1_velocity = self.derive_linear(position, velocity)
        k1_beta = self.derive_angular(beta)
        position_2, velocity_2, beta_2 =  self.euler_step(position, velocity, k1_position, k1_velocity,beta,k1_beta,time_step/2)
        k2_position, k2_velocity = self.derive_linear(position_2, velocity_2)
        return self.euler_step(position, velocity, k2_position, k2_velocity,beta,k1_beta,time_step)
    def simulate(self,episodes, time_step):
        position = self.initial_position
        velocity = self.initial_velocity
        positions = np.empty((episodes + 1, 2))
        beta = self.initial_angle
        angles = np.empty((episodes + 1))
        angles[0] = beta
        positions[0] = position
        for i in range(1, episodes + 1):
            position, velocity, beta = self.improved_euler_step(position, velocity, beta ,time_step)
            positions[i] = position
            angles[i] = beta
        return positions,angles
    def get_circle(self, radius, pos, resolution=100):
        points = np.empty((resolution + 1, 2))
        theta = 0
        for i in range(resolution + 1):
            points[i] = np.array([radius * np.cos(theta), radius * np.sin(theta)])
            theta += np.radians(360) / resolution
        return points + pos

    def get_point_trajectory(self, angles, radius):
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        trajectory = np.stack((x, y), axis=1)
        return trajectory


r = (earth_sun_distance + sun_radius)
pos = np.array([0, r])
orbital_velocity = np.sqrt(G * mass_of_sun / r)

vel = np.array([orbital_velocity, 0])

s = Simulation( pos, vel, pos + earth_moon_distance + earth_radius)
positions, angles = s.simulate(10000, 10)


x = positions[:,0]
y = positions[:,1]
earth = s.get_circle(earth_radius, positions[0])
sun = s.get_circle(sun_radius/4, np.array([0,0]))

x_sun = sun[:,0]
y_sun = sun[:,1]
x_earth = earth[:,0]
y_earth = earth[:,1]
moon_pos = s.get_point_trajectory(angles, 100e6) + positions
x_moon = moon_pos[:,0]
y_moon = moon_pos[:,1]



plt.figure(figsize=(10, 10))
plt.plot(x, y, label='Earth orbit')
plt.plot(x_moon,y_moon, label='Moon orbit')
plt.plot(x_sun, y_sun, label='Sun')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Earth and moon trajectory not to scale')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()