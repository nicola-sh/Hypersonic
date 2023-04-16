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
T = 100000.0  # thrust, N
m_ha = 1000.0  # initial full mass of aircraft, kg
m_fuel = 450.0  # initial full mass fuel of aircraft, kg
v = 1700.0  # initial velocity, m/s
x = 0.0  # initial Distance, m
h = 2000.0  # initial altitude, m
alpha = np.radians(30)  # initial angle of attack, radians
theta = np.radians(0)  # initial angle of inclination of the flight trajectory, radians
dt = 0.01  # time step, s
G_c = 0.015  # initial fuel burnout per 1 sec


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
v_arr = [v]
m_arr = [m_ha + m_fuel]


while h > 0.0:

    if m_fuel > 0:
        m_fuel = mass_after_fuel_burning(t, m_fuel)
        m_total = m_ha + m_fuel
    else:
        m_total = m_ha
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
    v_arr.append(V)
    m_arr.append(m_total)

# Set the style for the plot
sns.set_style("whitegrid")

# Create a figure with 2 subplots, arranged in a vertical layout
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

# Plot 1: Altitude versus Range
sns.lineplot(x=x_arr, y=h_arr, ax=ax1)
ax1.set_xlabel('Range, km')
ax1.set_ylabel('Height, km')
ax1.set_title('Height vs Range')

# Plot 2 and 3: Speed versus Time and Dependence of Mass on Time
sns.lineplot(x=t_arr, y=v_arr, ax=ax2, color='blue')
sns.lineplot(x=t_arr, y=m_arr, ax=ax2, color='orange')
ax2.set_xlabel('Time, s')
ax2.set_ylabel('Speed, m/s / Mass, kg')
ax2.set_title('Velocity and Mass by time')

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.5)

# Show the plot
plt.show()