import warnings
import datetime
import numpy as np
import scipy.integrate as spi
import seaborn as sns
from matplotlib import pyplot as plt, MatplotlibDeprecationWarning

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


# Constants
g = 9.8066                  # gravitational acceleration, m/s^2
R_earth = 6370000.0         # radius of the Earth, m

# Initial condition of Hypersonic Aircraft
T = 100000.0                # +initial thrust force, N
m_ha = 1000.0               # +initial mass of aircraft, kg
m_fuel = 450.0              # +initial mass fuel of aircraft, kg
m_total = m_ha + m_fuel

x = 0.0                     # initial Distance, m
h = 2 * 1000.0              # initial altitude, m
V = 5 * 340.0               # +initial velocity, m/s    1 mach = 340 m/s
alpha = np.radians(30)      # +initial angle of attack, radians
theta = np.radians(0)       # +initial angle of inclination of the flight trajectory, radians
dt = 0.01                   # time step, s
G_c = 0.02                  # +initial fuel burnout per dt



def hypersonic_aircraft_model(y, t, R_earth, g, T, m_total, G_c, alpha):
    V, theta, x, h, mfb = y

    dV = (T * np.cos(alpha)) / m_total - g * np.sin(theta)
    dtheta = (T * np.sin(alpha)) / (m_total * V) - (g * np.cos(theta)) / V + (V * np.cos(theta)) / (R_earth + h)
    dx = V * np.cos(theta) * R_earth / (R_earth + h)
    dh = V * np.sin(theta)
    mfb_new = G_c

    return [dV, dtheta, dx, dh, mfb_new]


# Integration interval
t = 0.0                     # initial time
mfb = 0                      # initial fuel burnout
y0 = [V, theta, x, h, mfb]       # initial conditions

# Arrays to store results
t_arr = [0.0]
x_arr = [x / 1000]
h_arr = [h / 1000]
v_arr = [V]
m_arr = [m_total]

while h > 0.0:

    # solve model using RK23
    sol = spi.solve_ivp(hypersonic_aircraft_model,
                        [t, t + dt], y0, method='RK23',
                        args=(R_earth, g, T, m_total, G_c, alpha))

    # Update y0
    y0 = sol.y[:, -1]
    V, theta, x, h, mfb_new = y0
    mfb = mfb_new
    m_total -= mfb

    # Update time
    t += dt

    # Store results
    t_arr.append(t)
    x_arr.append(x / 1000)
    h_arr.append(h / 1000)
    v_arr.append(V)
    m_arr.append(m_total)


# region Output plots
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
# Add current local time to the center of the bottom plot

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
ax2.text(0.5, -0.2, f"Local Time: {now}", ha='center', va='center', transform=ax2.transAxes)

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.5)

# Show the plot
plt.show()
# endregion