import numpy as np
import math
import matplotlib.pyplot as plt
import json

# Constants
g = 9.81
air_density = 1.225


# Gun and bullet properties
gun_height = 1
initial_bullet_velocity = 350
bullet_mass = 0.117
bullet_diameter = 0.009
bullet_drag_coefficient = 0.05

# Target properties
target_size = 1

# Calculate the bullet's cross-sectional area
bullet_area = math.pi * (bullet_diameter / 2)**2

# Wind properties
wind_speeds = np.linspace(0, 20, 10)  # Range of wind speeds to simulate
wind_direction = math.radians(0)

# Firing angles to simulate
firing_angles = np.radians(np.linspace(0, 90, 20))  # Range of firing angles to simulate

# Time settings
dt = 0.01  # Time step

# Function to calculate the derivatives using the Runge-Kutta method
def calculate_derivatives(t, x, y, v, θ, wind_speed):
    wind_effect = wind_speed * math.cos(wind_direction - θ)
    dxdt = v * math.cos(θ) + wind_effect
    dydt = v * math.sin(θ)
    dvdt = -g * math.sin(θ) - 0.5 * air_density * v**2 * bullet_drag_coefficient * bullet_area / bullet_mass
    dθdt = -g * math.cos(θ) / v
    return dxdt, dydt, dvdt, dθdt

# Function to perform one step of the Runge-Kutta method
def runge_kutta_step(t, dt, x, y, v, θ, wind_speed):
    k1_x, k1_y, k1_v, k1_θ = calculate_derivatives(t, x, y, v, θ, wind_speed)
    k2_x, k2_y, k2_v, k2_θ = calculate_derivatives(t + dt/2, x + k1_x*dt/2, y + k1_y*dt/2, v + k1_v*dt/2, θ + k1_θ*dt/2, wind_speed)
    k3_x, k3_y, k3_v, k3_θ = calculate_derivatives(t + dt/2, x + k2_x*dt/2, y + k2_y*dt/2, v + k2_v*dt/2, θ + k2_θ*dt/2, wind_speed)
    k4_x, k4_y, k4_v, k4_θ = calculate_derivatives(t + dt, x + k3_x*dt, y + k3_y*dt, v + k3_v*dt, θ + k3_θ*dt, wind_speed)

    x = x + dt * (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
    y = y + dt * (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
    v = v + dt * (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
    θ = θ + dt * (k1_θ + 2*k2_θ + 2*k3_θ + k4_θ) / 6

    return x, y, v, θ

# Define a list to store the results and a list to store the simulations
results = []
simulations = []

# Target distances to simulate
target_distances = np.linspace(0, 1000, 100)  # From 0 to 1000 meters

for θ in firing_angles:
    for wind in wind_speeds:
        for target_distance in target_distances:
            x, y = 0, gun_height
            v = initial_bullet_velocity
            θ_temp = θ  # The firing angle
            t = 0  # Start time

            # Variables to store the trajectory of the bullet
            x_array = []
            y_array = []

            while y >= 0:
                x, y, v, θ_temp = runge_kutta_step(t, dt, x, y, v, θ_temp, wind)

                # Append the current x and y values to the trajectory arrays
                x_array.append(x)
                y_array.append(y)

                if target_distance - target_size/2 <= x <= target_distance + target_size/2 and 0 <= y <= target_size:
                    delta_t = 0.001
                    delta_p = bullet_mass * v
                    impact_force = delta_p / delta_t
                    results.append({
                        'firing_angle': θ,
                        'wind_speed': wind,
                        'target_distance': target_distance,
                        'impact_force': impact_force
                    })

                    # Store the trajectory for this simulation
                    simulations.append({
                        'firing_angle': θ,
                        'wind_speed': wind,
                        'target_distance': target_distance,
                        'x': x_array,
                        'y': y_array
                    })
                    break
                t = t + dt  # Update the time

# Find the result with the highest impact force
max_force_result = max(results, key=lambda r: r['impact_force'])

# Filter the results and the simulations to only include those within a 50% range of the maximum conditions
filtered_results = [data for data in results if data['impact_force'] > 0.5 * max_force_result['impact_force']]

# Extract relevant data for the filtered results
firing_angles_filtered = [math.degrees(data['firing_angle']) for data in filtered_results]
wind_speeds_filtered = [data['wind_speed'] for data in filtered_results]
target_distances_filtered = [data['target_distance'] for data in filtered_results]
impact_forces_filtered = [data['impact_force'] for data in filtered_results]

# Create plots
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Firing angle vs impact force
axs[0].scatter(firing_angles_filtered, impact_forces_filtered)
axs[0].set_xlabel('Firing angle (degrees)')
axs[0].set_ylabel('Impact force (N)')
axs[0].set_title('Firing angle vs Impact force')

# Wind speed vs impact force
axs[1].scatter(wind_speeds_filtered, impact_forces_filtered)
axs[1].set_xlabel('Wind speed (m/s)')
axs[1].set_ylabel('Impact force (N)')
axs[1].set_title('Wind speed vs Impact force')

# Target distance vs impact force
axs[2].scatter(target_distances_filtered, impact_forces_filtered)
axs[2].set_xlabel('Target distance (m)')
axs[2].set_ylabel('Impact force (N)')
axs[2].set_title('Target distance vs Impact force')

# Display plots
plt.tight_layout()
plt.show()

print(f"The highest impact force was {max_force_result['impact_force']} N.")
print(f"It was achieved with a firing angle of {math.degrees(max_force_result['firing_angle'])} degrees, a wind speed of {max_force_result['wind_speed']} m/s, and a target distance of {max_force_result['target_distance']} m.")