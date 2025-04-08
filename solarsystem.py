import matplotlib.pyplot as plt
import numpy as np


earth_radius = 6.371e6
G = 6.6743e-11
mass_of_earth = 5.972e24
mass_moon = 7.347e22
moon_distance = 384.4e6

class Simulation:
    def __init__(self, mass,initial_position,initial_velocity):
        self.mass = mass
        self.initial_position = initial_position
        self.initial_velocity = initial_velocity
        self.initial_angle = 0
    def derive_linear(self, position, velocity):
        r = np.linalg.norm(position)
        return velocity, -G * mass_of_earth * position/ r**3

    def derive_angular(self, alpha):
        return np.sqrt(G*mass_of_earth/moon_distance**3)

    @staticmethod
    def euler_step(position, velocity, k_position, k_velocity,time_step):
        position_prime = position + k_position * time_step
        velocity_prime = velocity + k_velocity * time_step
        return position_prime, velocity_prime
    def improved_euler_step(self, position, velocity, time_step):
        k1_position, k1_velocity = self.derive_linear(position, velocity)
        position_2, velocity_2 =  self.euler_step(position, velocity, k1_position, k1_velocity,time_step/2)
        k2_position, k2_velocity = self.derive_linear(position_2, velocity_2)
        return self.euler_step(position, velocity, k2_position, k2_velocity,time_step)
    def simulate(self,episodes, time_step):
        position = self.initial_position
        velocity = self.initial_velocity
        positions = np.empty((episodes + 1, 2))
        positions[0] = position
        for i in range(1, episodes + 1):
            position, velocity = self.improved_euler_step(position, velocity, time_step)
            positions[i] = position
        return positions
    def get_circle(self, resolution, radius):
        points = np.empty((resolution + 1, 2))
        theta = 0
        for i in range(resolution + 1):
            points[i] = np.array([radius * np.cos(theta), radius * np.sin(theta)])
            theta += np.radians(360) / resolution
        return points

# Set up initial conditions for a circular orbit
r = moon_distance + earth_radius  # Altitude of 10,000 meters
pos = np.array([0, r])    # Start at the top of the orbit
# Calculate orbital velocity for circular orbit
orbital_velocity = np.sqrt(G * mass_of_earth / r)
# Set velocity purely in x-direction for circular orbit
vel = np.array([orbital_velocity, 0])

# Create simulation with smaller time step and more steps
s = Simulation(10, pos, vel)
positions = s.simulate(500000, 10)  # More steps, smaller time step

# Plot the results
x = positions[:,0]
y = positions[:,1]
earth = s.get_circle(1000, earth_radius)
x_earth = earth[:,0]
y_earth = earth[:,1]

plt.figure(figsize=(10, 10))
plt.plot(x, y, label='Satellite Orbit')
plt.plot(x_earth, y_earth, label='Earth')
plt.scatter(0, 0, color='blue', s=100, label='Earth Center')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Satellite Orbit Simulation')
plt.legend()
plt.grid(True)
plt.axis('equal')  # Make the plot square to show true circularity
plt.show()