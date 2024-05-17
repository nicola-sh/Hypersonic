# # import math
# # import time
# # import numpy as np
# # from scipy.integrate import solve_ivp
# # import warnings
# # from matplotlib import pyplot as plt, MatplotlibDeprecationWarning
# #
# # warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
# #
# #
# # # Constants
# # g = 9.81  # acceleration due to gravity
# # R_earth = 6371000  # radius of the Earth
# # a = 331  # speed of sound
# # R = 8.31446261815324  # gas law constant
# #
# # # Integration parameters
# # t = 0.0  # initial time, sec
# # dt = 0.01  # time step, sec
# # t_end = 600  # end time, sec
# #
# # # Mass parameters
# # m_fuel = 450  # initial fuel mass, kg
# # m_ha = 1000  # mass of aircraft, kg
# # m_total = 1450  # initial total mass, kg
# # Gc = 3  # fuel consumption rate, kg/s
# #
# # length = 12.5  # length HA,m
# # diameter = 0.5  # diameter HA,m
# # radius = diameter / 2  # radius HA,m
# #
# # Ae = math.pi * math.pow(radius, 2)
# # A0 = 0.54 * Ae
# # Afin = 0.3
# # print(Ae)
# # print(A0)
# # print(Afin)
# #
# # k = 1.4  # gamma for air and 2h2o
# # Mmw = 0.02003  # 2H20, kg/mol \\ molecular weight
# # Tc = 1600  # Chamber Temp, Kelvin
# # Pc = 250000  # Chamber Pressure, Pa
# #
# # Cd = 0.3
# # Cl = 0.6
# #
# #
# # def temperature(h):
# #     """Calculates air temperature [Celsius] at altitude [m]"""
# #     # from equations at
# #     #   http://www.grc.nasa.gov/WWW/K-12/airplane/atmosmet.html
# #     if h <= 11000:
# #         # troposphere
# #         tval = 15.04 - .00649 * h
# #     elif h <= 25000:
# #         # lower stratosphere
# #         tval = -56.46
# #     elif h > 25000:
# #         tval = -131.21 + .00299 * h
# #     return tval
# #
# #
# # def pressure(h):
# #     """Calculates air pressure [Pa] at altitude [m]"""
# #     # from equations at
# #     #   http://www.grc.nasa.gov/WWW/K-12/airplane/atmosmet.html
# #
# #     tvalue = temperature(h)
# #
# #     if h <= 11000:
# #         # troposphere
# #         p = 101.29 * ((tvalue + 273.1) / 288.15) ** 5.256
# #     elif h <= 25000:
# #         # lower stratosphere
# #         p = 22.65 * np.exp(1.73 - .000157 * h)
# #     elif h > 25000:
# #         # upper stratosphere
# #         p = 2.488 * ((tvalue + 273.1) / 216.6) ** -11.388
# #     return p * 1000
# #
# #
# # def density(h):
# #     """Calculates air density at altitude"""
# #     rho0 = 1.225  # [kg/m^3] air density at sea level
# #     if h < 19200:
# #         # use barometric formula, where 8420 is effective height of atmosphere [m]
# #         rho = rho0 * np.exp(-h / 8420)
# #     elif 19200 < h < 47000:
# #         # use 1976 Standard Atmosphere model
# #         # http://modelweb.gsfc.nasa.gov/atmos/us_standard.html
# #         # from http://scipp.ucsc.edu/outreach/balloon/glost/environment3.html
# #         rho = rho0 * (.857003 + h / 57947) ** -13.201
# #     else:
# #         # vacuum
# #         rho = 1.e-6
# #     return rho
# #
# #
# # def massFlowRate(velocityAircraft):
# #     Pt = Pc * math.pow((1 + (k - 1) / 2), (-k / (k - 1)))  # N/m^2
# #     Tt = Tc * (1 / (1 + (k - 1) / 2))  # Kelvin
# #
# #     return (Ae * Pt / math.sqrt(Tt)) * math.sqrt(
# #         (k * Mmw) / R) * (velocityAircraft / a) * math.pow(
# #         (1 + ((k - 1) / 2) * math.pow(
# #             (velocityAircraft / a), 2)), (-(k + 1) / (2 * (k - 1))))
# #
# #
# # def velocityExhaustGas(altitude):
# #     return math.sqrt(((Tc * R) / Mmw) * ((2 * k) / (k - 1)) * (1 - math.pow((pressure(altitude) / Pc), ((k - 1) / k))))
# #
# #
# # def thrust(altitude, velocityAircraft, m_total):
# #     if m_total != m_ha:
# #         thrust = massFlowRate(velocityAircraft) * (velocityExhaustGas(altitude) - velocityAircraft)
# #     elif m_total == m_ha and velocityAircraft > 0:
# #         thrust = massFlowRate(velocityAircraft) * (a - velocityAircraft)
# #     elif velocityAircraft <= 0 and m_total == m_ha:
# #         thrust = 0
# #
# #     return thrust
# #
# #
# # def drag(h, velocityAircraft):
# #     return .5 * Cd * density(h) * velocityAircraft ** 2 * A0
# #
# #
# # def lift(h, velocityAircraft):
# #     return .5 * Cl * density(h) * velocityAircraft ** 2 * Afin
# #
# #
# # def aircraft_mass(t, m_total, m_fuel):
# #     """Функция для вычисления массы в момент времени t"""
# #     # Calculate fuel mass at each time
# #     m_fuel -= Gc * t
# #
# #     # Check conditions to update total mass
# #     if m_total > 1000 and m_fuel > 0:
# #         m_total -= Gc * t
# #     else:
# #         m_total = m_ha
# #
# #     return m_total
# #
# #
# # def hypersonic_aircraft_model(t, u):
# #     V, theta, x, y = u[:]
# #
# #     totalMass = aircraft_mass(t, m_total, m_fuel)
# #
# #     if totalMass > m_ha:
# #         P = thrust(y, V, totalMass)
# #     else:
# #         P = thrust(y, V, m_ha)
# #         totalMass = m_ha
# #
# #     Drag = drag(y, V)
# #     Lift = lift(y, V)
# #
# #     dV = (P * np.cos(alpha) - Drag - (totalMass * g * np.sin(theta))) / totalMass
# #
# #     dtheta = (P * np.sin(alpha) + Lift - totalMass * g * np.cos(theta) +
# #               (totalMass * V ** 2 * np.cos(theta)) / (R_earth + y)) / (totalMass * V)
# #
# #     dx = V * np.cos(theta) * R_earth / (R_earth + y)
# #
# #     dy = V * np.sin(theta)
# #
# #     return [dV, dtheta, dx, dy]
# #
# #
# # def rk4(f, u0, t0, tf , n):
# #     t = np.linspace(t0, tf, n+1)
# #     u = np.array((n+1)*[u0])
# #     h = t[1]-t[0]
# #     for i in range(n):
# #         k1 = h * f(u[i], t[i])
# #         k2 = h * f(u[i] + 0.5 * k1, t[i] + 0.5*h)
# #         k3 = h * f(u[i] + 0.5 * k2, t[i] + 0.5*h)
# #         k4 = h * f(u[i] + k3, t[i] + h)
# #         u[i+1] = u[i] + (k1 + 2*(k2 + k3 ) + k4) / 6
# #     return u, t
# #
# #
# # # Define initial conditions
# # V = 3 * a  # initial velocity
# # theta = np.radians(30)  # initial angle
# # x = 0  # initial x position
# # y = 3000  # initial y position
# # alpha = np.degrees(2)  # angle of attack
# #
# #
# # # Define time span
# # t_span = [0, t_end]
# #
# # u0 = [V, theta, x, y]  # initial conditions
# # t0 = 0
# # tf = 10.
# # n = 10000
# # u, t = rk4(hypersonic_aircraft_model, u0, t0, tf, n)
# #
# #
# # # # Event function to stop integration when y <= 0
# # # def hitEarth(t, y):
# # #     return y[3]
# # #
# # #
# # # hitEarth.terminal = True  # Stop integration when event is triggered
# # # hitEarth.direction = -1  # Trigger only when y decreases (crosses zero from positive to negative)
# # #
# # # # Solve the system of ODEs with event
# # # sol = solve_ivp(lambda t, y: hypersonic_aircraft_model(t, y),
# # #                 t_span=t_span,
# # #                 y0=y0,
# # #                 method='RK45',
# # #                 # t_eval=np.linspace(t_span[0], t_span[1], 1000),
# # #                 events=hitEarth,
# # #                 first_step=dt, max_step=dt,
# # #                 # rtol=1e-3, atol=1e-6)
# # #                 rtol=1, atol=1)
# # #
# # # delta = np.diff(sol.t)
# # # print(f'Количество выполненых шагов = {len(delta)}')
# # # print(f'Наименьший шаг = {min(delta)}, Наибольший шаг = {max(delta)}\n')
# # #
# # # # print(sol.t_events)
# # # # print(sol.t)
# # #
# # # # Extract values at the final time step
# # # final_values = sol.y[:, -1]
# # #
# # # # Unpack into individual variables
# # # final_V, final_theta, final_x, final_y = final_values
# # #
# # # # Print the final values
# # # print("Final velocity:", final_V)
# # # print("Final theta angle:", final_theta)
# # # print("Final x-coordinate:", final_x)
# # # print("Final y-coordinate:", final_y)
# # #
# # #
# # # # Create a figure and axes for subplots
# # # fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# # #
# # # # Plot x vs y
# # # axes[0, 0].plot(sol.y[2] / 1000, sol.y[3] / 1000)
# # # axes[0, 0].set_xlabel('x (km)')
# # # axes[0, 0].set_ylabel('y (km)')
# # # axes[0, 0].set_title('Trajectory of Hypersonic Aircraft')
# # # axes[0, 0].grid(True)
# # #
# # # # Plot thrust, drag, and lift
# # # axes[0, 1].plot(sol.t,
# # #                 [thrust(y, v, aircraft_mass(t, m_total, m_fuel)) for t, (y, v) in zip(sol.t, zip(sol.y[3], sol.y[0]))],
# # #                 label='Thrust')
# # # axes[0, 1].plot(sol.t, [drag(y, v) for y, v in zip(sol.y[3], sol.y[0])], label='Drag')
# # # axes[0, 1].plot(sol.t, [lift(y, v) for y, v in zip(sol.y[3], sol.y[0])], label='Lift')
# # # axes[0, 1].set_xlabel('Time (s)')
# # # axes[0, 1].set_ylabel('Force (N)')
# # # axes[0, 1].set_title('Thrust, Drag, and Lift vs Time')
# # # axes[0, 1].legend()
# # # axes[0, 1].grid(True)
# # #
# # # # Plot velocity by time
# # # axes[1, 0].plot(sol.t, sol.y[0] / a)
# # # axes[1, 0].set_xlabel('Time (s)')
# # # axes[1, 0].set_ylabel('Velocity (m/s)')
# # # axes[1, 0].set_title('Velocity vs Time')
# # # axes[1, 0].grid(True)
# # #
# # # # Plot theta by time
# # # axes[1, 1].plot(sol.t, np.degrees(sol.y[1]), label='Theta')
# # # axes[1, 1].plot(sol.t, np.degrees(alpha) * np.ones_like(sol.t), label='Alpha')
# # # axes[1, 1].set_xlabel('Time (s)')
# # # axes[1, 1].set_ylabel('Angle (degrees)')
# # # axes[1, 1].set_title('Angles vs Time')
# # # axes[1, 1].legend()
# # # axes[1, 1].grid(True)
# # #
# # # plt.tight_layout()
# # # plt.show()
# # #
# # #
# # # # Create a new figure and axes for subplots
# # # fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# # #
# # # # Plot altitude vs time
# # # axes[0, 0].plot(sol.t, sol.y[3])
# # # axes[0, 0].set_xlabel('Time (s)')
# # # axes[0, 0].set_ylabel('Altitude (m)')
# # # axes[0, 0].set_title('Altitude vs Time')
# # # axes[0, 0].grid(True)
# # #
# # # # Plot range vs time
# # # axes[0, 1].plot(sol.t, sol.y[2] / 1000)
# # # axes[0, 1].set_xlabel('Time (s)')
# # # axes[0, 1].set_ylabel('Range (km)')
# # # axes[0, 1].set_title('Range vs Time')
# # # axes[0, 1].grid(True)
# # #
# # # # Plot theta vs x
# # # axes[1, 0].plot(sol.y[2] / 1000, np.degrees(sol.y[1]))
# # # axes[1, 0].set_xlabel('Range (km)')
# # # axes[1, 0].set_ylabel('Theta (degrees)')
# # # axes[1, 0].set_title('Theta vs Range (X)')
# # # axes[1, 0].grid(True)
# # #
# # # # Plot theta vs y
# # # axes[1, 1].plot(sol.y[3], np.degrees(sol.y[1]))
# # # axes[1, 1].set_xlabel('Altitude (m)')
# # # axes[1, 1].set_ylabel('Theta (degrees)')
# # # axes[1, 1].set_title('Theta vs Altitude (Y)')
# # # axes[1, 1].grid(True)
# # #
# # # plt.tight_layout()
# # # plt.show()
# # #
# # # # # Plot x vs y
# # # # plt.plot(sol.y[2] / 1000, sol.y[3] / 1000)
# # # # plt.xlabel('x (km)')
# # # # plt.ylabel('y (km)')
# # # # plt.title('Trajectory of Hypersonic Aircraft')
# # # # plt.grid(True)
# # # # plt.show()
# # # #
# # # # # Plot thrust, drag, and lift on the same plot
# # # # plt.plot(sol.t,
# # # #          [thrust(y, v, aircraft_mass(t, m_total, m_fuel)) for t, (y, v) in zip(sol.t, zip(sol.y[3], sol.y[0]))],
# # # #          label='Thrust')
# # # # plt.plot(sol.t, [drag(y, v) for y, v in zip(sol.y[3], sol.y[0])], label='Drag')
# # # # plt.plot(sol.t, [lift(y, v) for y, v in zip(sol.y[3], sol.y[0])], label='Lift')
# # # # plt.xlabel('Time (s)')
# # # # plt.ylabel('Force (N)')
# # # # plt.title('Thrust, Drag, and Lift vs Time')
# # # # plt.legend()
# # # # plt.grid(True)
# # # # plt.show()
# # # #
# # # # # Create subplots
# # # # fig, axes = plt.subplots(2, 1, figsize=(8, 10))
# # # #
# # # # # Plot velocity by time
# # # # axes[0].plot(sol.t, sol.y[0] / a)
# # # # axes[0].set_xlabel('Time (s)')
# # # # axes[0].set_ylabel('Velocity (m/s)')
# # # # axes[0].set_title('Velocity vs Time')
# # # # axes[0].grid(True)
# # # #
# # # # # Plot theta by time
# # # # axes[1].plot(sol.t, np.degrees(sol.y[1]))
# # # # axes[1].set_xlabel('Time (s)')
# # # # axes[1].set_ylabel('Theta (degrees)')
# # # # axes[1].set_title('Theta vs Time')
# # # # axes[1].grid(True)
# # # #
# # # # plt.tight_layout()
# # # # plt.show()
#
# import numpy as np
# import math
# import matplotlib.pyplot as plt
# import json
#
# # Constants
# g = 9.81
# air_density = 1.225
#
#
# # Gun and bullet properties
# gun_height = 1
# initial_bullet_velocity = 350
# bullet_mass = 0.117
# bullet_diameter = 0.009
# bullet_drag_coefficient = 0.05
#
# # Target properties
# target_size = 1
#
# # Calculate the bullet's cross-sectional area
# bullet_area = math.pi * (bullet_diameter / 2)**2
#
# # Wind properties
# wind_speeds = np.linspace(0, 20, 10)  # Range of wind speeds to simulate
# wind_direction = math.radians(0)
#
# # Firing angles to simulate
# firing_angles = np.radians(np.linspace(0, 90, 20))  # Range of firing angles to simulate
#
# # Time settings
# dt = 0.01  # Time step
#
# # Function to calculate the derivatives using the Runge-Kutta method
# def calculate_derivatives(t, x, y, v, θ, wind_speed):
#     wind_effect = wind_speed * math.cos(wind_direction - θ)
#     dxdt = v * math.cos(θ) + wind_effect
#     dydt = v * math.sin(θ)
#     dvdt = -g * math.sin(θ) - 0.5 * air_density * v**2 * bullet_drag_coefficient * bullet_area / bullet_mass
#     dθdt = -g * math.cos(θ) / v
#     return dxdt, dydt, dvdt, dθdt
#
# # Function to perform one step of the Runge-Kutta method
# def runge_kutta_step(t, dt, x, y, v, θ, wind_speed):
#     k1_x, k1_y, k1_v, k1_θ = calculate_derivatives(t, x, y, v, θ, wind_speed)
#     k2_x, k2_y, k2_v, k2_θ = calculate_derivatives(t + dt/2, x + k1_x*dt/2, y + k1_y*dt/2, v + k1_v*dt/2, θ + k1_θ*dt/2, wind_speed)
#     k3_x, k3_y, k3_v, k3_θ = calculate_derivatives(t + dt/2, x + k2_x*dt/2, y + k2_y*dt/2, v + k2_v*dt/2, θ + k2_θ*dt/2, wind_speed)
#     k4_x, k4_y, k4_v, k4_θ = calculate_derivatives(t + dt, x + k3_x*dt, y + k3_y*dt, v + k3_v*dt, θ + k3_θ*dt, wind_speed)
#
#     x = x + dt * (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
#     y = y + dt * (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
#     v = v + dt * (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
#     θ = θ + dt * (k1_θ + 2*k2_θ + 2*k3_θ + k4_θ) / 6
#
#     return x, y, v, θ
#
# # Define a list to store the results and a list to store the simulations
# results = []
# simulations = []
#
# # Target distances to simulate
# target_distances = np.linspace(0, 1000, 100)  # From 0 to 1000 meters
#
# for θ in firing_angles:
#     for wind in wind_speeds:
#         for target_distance in target_distances:
#             x, y = 0, gun_height
#             v = initial_bullet_velocity
#             θ_temp = θ  # The firing angle
#             t = 0  # Start time
#
#             # Variables to store the trajectory of the bullet
#             x_array = []
#             y_array = []
#
#             while y >= 0:
#                 x, y, v, θ_temp = runge_kutta_step(t, dt, x, y, v, θ_temp, wind)
#
#                 # Append the current x and y values to the trajectory arrays
#                 x_array.append(x)
#                 y_array.append(y)
#
#                 if target_distance - target_size/2 <= x <= target_distance + target_size/2 and 0 <= y <= target_size:
#                     delta_t = 0.001
#                     delta_p = bullet_mass * v
#                     impact_force = delta_p / delta_t
#                     results.append({
#                         'firing_angle': θ,
#                         'wind_speed': wind,
#                         'target_distance': target_distance,
#                         'impact_force': impact_force
#                     })
#
#                     # Store the trajectory for this simulation
#                     simulations.append({
#                         'firing_angle': θ,
#                         'wind_speed': wind,
#                         'target_distance': target_distance,
#                         'x': x_array,
#                         'y': y_array
#                     })
#                     break
#                 t = t + dt  # Update the time
#
# # Find the result with the highest impact force
# max_force_result = max(results, key=lambda r: r['impact_force'])
#
# # Filter the results and the simulations to only include those within a 50% range of the maximum conditions
# filtered_results = [data for data in results if data['impact_force'] > 0.5 * max_force_result['impact_force']]
#
# # Extract relevant data for the filtered results
# firing_angles_filtered = [math.degrees(data['firing_angle']) for data in filtered_results]
# wind_speeds_filtered = [data['wind_speed'] for data in filtered_results]
# target_distances_filtered = [data['target_distance'] for data in filtered_results]
# impact_forces_filtered = [data['impact_force'] for data in filtered_results]
#
# # Create plots
# fig, axs = plt.subplots(3, 1, figsize=(10, 15))
#
# # Firing angle vs impact force
# axs[0].scatter(firing_angles_filtered, impact_forces_filtered)
# axs[0].set_xlabel('Firing angle (degrees)')
# axs[0].set_ylabel('Impact force (N)')
# axs[0].set_title('Firing angle vs Impact force')
#
# # Wind speed vs impact force
# axs[1].scatter(wind_speeds_filtered, impact_forces_filtered)
# axs[1].set_xlabel('Wind speed (m/s)')
# axs[1].set_ylabel('Impact force (N)')
# axs[1].set_title('Wind speed vs Impact force')
#
# # Target distance vs impact force
# axs[2].scatter(target_distances_filtered, impact_forces_filtered)
# axs[2].set_xlabel('Target distance (m)')
# axs[2].set_ylabel('Impact force (N)')
# axs[2].set_title('Target distance vs Impact force')
#
# # Display plots
# plt.tight_layout()
# plt.show()
#
# print(f"The highest impact force was {max_force_result['impact_force']} N.")
# print(f"It was achieved with a firing angle of {math.degrees(max_force_result['firing_angle'])} degrees, a wind speed of {max_force_result['wind_speed']} m/s, and a target distance of {max_force_result['target_distance']} m.")
#
#
#
# # # Function to perform one step of the Runge-Kutta method
# # def runge_kutta_step(t, dt, x, y, V, theta, m):
# #     k1_x, k1_y, k1_V, k1_theta, k1_m = calculate_derivatives(t,
# #                                                          x,
# #                                                          y,
# #                                                          V,
# #                                                          theta,
# #                                                          m)
# #     k2_x, k2_y, k2_V, k2_theta, k2_m = calculate_derivatives(t + dt/2,
# #                                                          x + k1_x*dt/2,
# #                                                          y + k1_y*dt/2,
# #                                                          V + k1_V*dt/2,
# #                                                          theta + k1_theta*dt/2,
# #                                                          m + k1_m*dt/2)
# #     k3_x, k3_y, k3_V, k3_theta, k3_m = calculate_derivatives(t + dt/2,
# #                                                          x + k2_x*dt/2,
# #                                                          y + k2_y*dt/2,
# #                                                          V + k2_V*dt/2,
# #                                                          theta + k2_theta*dt/2,
# #                                                          m + k2_m*dt/2)
# #     k4_x, k4_y, k4_V, k4_theta, k4_m = calculate_derivatives(t + dt,
# #                                                          x + k3_x*dt,
# #                                                          y + k3_y*dt,
# #                                                          V + k3_V*dt,
# #                                                          theta + k3_theta*dt,
# #                                                          m + k3_m*dt)
# #
# #     x = x + dt * (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
# #     y = y + dt * (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
# #     V = V + dt * (k1_V + 2*k2_V + 2*k3_V + k4_V) / 6
# #     theta = theta + dt * (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta) / 6
# #     m = m + dt * (k1_m + 2*k2_m + 2*k3_m + k4_m) / 6
# #
# #     return x, y, V, theta, m

import matplotlib.pyplot as plt
import numpy as np
# (H, M): ImpulseSpecific
SpecificImpulseData = {
    (2000, 3): 11300,
    (2000, 3.5): 11492,
    (2000, 4): 11338,
    (2000, 4.5): 10926,
    (2000, 5): 10322,
    (2000, 5.5): 9583.2,
    (2000, 6): 9000,

    (5000, 3): 11493,
    (5000, 3.5): 11760,
    (5000, 4): 11610,
    (5000, 4.5): 11302,
    (5000, 5): 10704,
    (5000, 5.5): 9984.9,
    (5000, 6): 9200.7,

    (11000, 3): 11848,
    (11000, 3.5): 12322,
    (11000, 4): 12253,
    (11000, 4.5): 12018,
    (11000, 5): 11511,
    (11000, 5.5): 10858,
    (11000, 6): 10105,

    (25000, 3): 11729,
    (25000, 3.5): 12230,
    (25000, 4): 12176,
    (25000, 4.5): 11925,
    (25000, 5): 11423,
    (25000, 5.5): 10744,
    (25000, 6): 9989.6,

    (30000, 3): 11554,
    (30000, 3.5): 11969,
    (30000, 4): 11870,
    (30000, 4.5): 11592,
    (30000, 5): 11053,
    (30000, 5.5): 10348,
    (30000, 6): 9568.6,

    (35000, 3): 11363,
    (35000, 3.5): 11700,
    (35000, 4): 11563,
    (35000, 4.5): 11251,
    (35000, 5): 10687,
    (35000, 5.5): 9957.4,
    (35000, 6): 9141.1,

    (40000, 3): 11000,
    (40000, 3.5): 11430,
    (40000, 4): 11258,
    (40000, 4.5): 10914,
    (40000, 5): 10321,
    (40000, 5.5): 9568.3,
    (40000, 6): 8729.9
}

CxData = {
    0: 0.32,
    0.5: 0.28,
    1: 0.24,
    1.5: 0.42,
    1.8: 0.3,
    2: 0.3,
    3: 0.26,
    4: 0.25,
    5: 0.23,
    6: 0.22,
    7: 0.21,
    8: 0.2
}


# Вычисление среднего удельного импульса
total_specific_impulse = sum(CxData.values())
average_specific_impulse = total_specific_impulse / len(CxData)

# Сохранение среднего удельного импульса для будущего использования
AverageSpecificImpulse = average_specific_impulse

print(AverageSpecificImpulse)

def get_specific_impulse(height, mach):
    # Check if exact values are present in the data
    if (height, mach) in SpecificImpulseData:
        return SpecificImpulseData[(height, mach)]
    else:
        # Find the nearest heights
        lower_height = max([h for h, _ in SpecificImpulseData if h < height])
        upper_height = min([h for h, _ in SpecificImpulseData if h > height])

        # Find the nearest mach numbers
        lower_mach = max([m for _, m in SpecificImpulseData if m < mach])
        upper_mach = min([m for _, m in SpecificImpulseData if m > mach])

        # Interpolate between the nearest values
        specific_impulse_lower = SpecificImpulseData.get((lower_height, lower_mach))
        specific_impulse_upper = SpecificImpulseData.get((upper_height, upper_mach))

        # If one of the heights or machs is not available
        if specific_impulse_lower is None or specific_impulse_upper is None:
            return None

        specific_impulse = (specific_impulse_lower + specific_impulse_upper) / 2
        return specific_impulse



fuel_mass_flow_rate = 3
mass_flow_rate = 80

def calculate_thrust(height, mach, fuel_mass_flow_rate):
    specific_impulse = get_specific_impulse(height, mach)

    return specific_impulse * fuel_mass_flow_rate

def calculate_specific_thrust(height, mach, fuel_mass_flow_rate):
    thrust = calculate_thrust(height, mach, fuel_mass_flow_rate)

    return thrust / mass_flow_rate

# Ma = velocityAircraft / a
# Pt = Pc * (1 + (k - 1) * Ma / 2) ** (k / (k - 1))  # N/m^2
# Tt = Tc * (1 / (1 + (k - 1) / 2))  # Kelvin
#
# mfr = (Ae * Pt / math.sqrt(Tt)) * \
#       math.sqrt((k * Mmw) / R) * Ma * \
#       (1 + ((k - 1) / 2) * Ma ** 2) ** (-(k + 1) / (2 * (k - 1)))

# mfr = (k/(((k+1)/2)**((k+1)/(2*(k-1))))) * ((Pt*A0)*math.sqrt(k*R*Tt))*f(Ma)












# def velocityExhaustGas(altitude):
#     global engine
#
#     Mmw = 0.02003  # 2H20, kg/mol \\ molecular weight
#     Tc = 540  # Chamber Temp, Kelvin
#     Pc = 200000  # Chamber Pressure, Pa
#
#     k1 = 2*k / (k-1)
#     k2 = (k-1)/k
#     Ve = math.sqrt(((Tc * R) / Mmw) * k1 * (1 - (pressure(altitude) / Pc) ** k2))
#
#     if engine:
#         Ve = math.sqrt(((Tc * R) / Mmw) * k1 * (1 - (pressure(altitude) / Pc) ** k2))
#     elif engine != True:
#         Ve = math.sqrt(((Tc * R) / Mmw) * k1 * (1 - (pressure(altitude) / Pc) ** k2))
#
#     else:
#         Ve = 0
#
#     return Ve


# def thrust(altitude, velocityAircraft, m):
#     global engine
#
#     if m > mha and engine == True:
#         thrust = massFlowRate(altitude, velocityAircraft) * (velocityExhaustGas(altitude) - velocityAircraft)
#     else:
#         # thrust = massFlowRate(velocityAircraft) * (a - velocityAircraft)
#         thrust = 0
#         engine = False
#
#     return thrust


# Cd = 0.3