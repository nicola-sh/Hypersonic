import warnings

import numpy as np
import scipy.integrate as spi
import seaborn as sns
from matplotlib import pyplot as plt, MatplotlibDeprecationWarning
from scipy.integrate import odeint
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

# Constants
g = 9.81  # gravitational acceleration, m/s^2
R_earth = 6371000.0  # radius of the Earth, m

# Initial condition of Hypersonic Aircraft
T = 200000.0  # thrust, N
m_ha = 1000.0  # initial full mass of aircraft, kg
m_fuel = 450.0  # initial full mass fuel of aircraft, kg
v = 5000.0  # initial velocity, m/s
x = 0.0  # initial Distance, m
h = 2000.0  # initial altitude, m
alpha = np.radians(30)  # initial angle of attack, radians
theta = np.radians(0)  # initial angle of inclination of the flight trajectory, radians
dt = 0.1  # time step, s
G_c = 100.0  # initial fuel burnout per 1 sec


def f(t):
    return G_c


def mass_after_fuel_burning(t, m_fuel):
    try:
        return m_fuel - spi.quad(f, 0, t)[0]
    except spi.IntegrationWarning:
        return 0.0


def hypersonic_aircraft_model(y, t, R_earth, g, T, alpha, m_total):
    V, theta, x, h = y

    dV = (T * np.cos(alpha)) / m_total - g * np.sin(theta)
    dtheta = (T * np.sin(alpha)) / (m_total * V) - (g * np.cos(theta)) / V + (V * np.cos(theta)) / (R_earth + h)
    dx = V * np.cos(theta) * R_earth / (R_earth + h)
    dy = V * np.sin(theta)

    return [dV, dtheta, dx, dy]


# Integration interval
t = 0.0  # initial time
y0 = [v, theta, x, h]  # initial conditions

# Arrays to store results
t_arr = [0.0]
x_arr = [x/1000]
h_arr = [h/1000]
speed_arr = [v]
m_arr = [m_ha + m_fuel]

while h > 0.0:

    m_not_burned = mass_after_fuel_burning(t, m_fuel)
    if m_not_burned > 0:
        m_total = m_ha + m_not_burned
    else:
        m_total = 1000
        T = 0


    # solve model
    sol = odeint(hypersonic_aircraft_model, y0, [t, t + dt], args=(R_earth, g, T, alpha, m_total))

    # Update y0
    y0 = sol[-1]
    V, theta, x, h = sol[-1]
    # Update time
    t += dt

    # Store results
    t_arr.append(t)
    x_arr.append(x/1000)
    h_arr.append(h/1000)
    speed_arr.append(V)
    m_arr.append(m_total)

# Plot of altitude versus range
plt.figure()
plt.plot(x_arr, h_arr)
plt.xlabel('Range, km')
plt.ylabel('Altitude, km')
plt.title('Altitude versus Range')

# Plot of speed versus time
plt.figure()
plt.plot(t_arr, speed_arr)
plt.xlabel('Time, s')
plt.ylabel('Speed, m/s')
plt.title('Speed versus Time')

# Plot of the dependence of mass on time
plt.figure()
mass_arr = [m_ha + mass_after_fuel_burning(t, m_fuel) for t in t_arr]
plt.plot(t_arr, mass_arr)
plt.xlabel('Time, s')
plt.ylabel('Mass, kg')
plt.title('Dependence of Mass on Time')

plt.show()