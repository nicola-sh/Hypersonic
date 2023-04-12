import numpy as np
from matplotlib import pyplot as plt, MatplotlibDeprecationWarning
from scipy.integrate import odeint
import scipy.integrate as spi
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


def integrate(y, t, R_earth, g, alpha, T, m):
    V, theta, x, h = y

    dV = (T * np.cos(alpha)) / m - g * np.sin(theta)
    dtheta = (T * np.sin(alpha)) / (m * V) - (g * np.cos(theta)) / V + (V * np.cos(theta)) / (R_earth + h)

    dx = V * np.cos(theta) * (R_earth/(R_earth + h))
    dh = V * np.sin(theta)

    return [dV, dtheta, dx, dh]


def integrand(t, G_c):
    return G_c

def m(t, m0, G_c):
    return m0 - spi.quad(integrand, 0, t, args=G_c)[0]


# Constants
g = 9.81                    # gravitational acceleration, m/s^2
R_earth = 6371000.0         # radius of the Earth, m

# Задание начальных условий
T = 200000.0                # thrust, N
m0 = 1400.0                  # initial mass, kg
m_fuel = 400.0              # initial mass of fuel, kg
v = 5000.0                  # initial velocity, m/s
x = 0.0                     # initial Distance, m
h = 2000.0                  # initial altitude, m
alpha = np.radians(30)      # initial angle of attack, radians
theta = np.radians(0)       # initial angle of inclination of the flight trajectory, radians
G_c = 125.0

# Интервал интегрирования
dt = 0.01                   # time step, s
t = 0                       # initial time
y0 = [v, theta, x, h]    # initial conditions

# Решение системы уравнений
V_list = []
theta_list = []
x_list = []
y_list = []
m_list = []
t_list = []

while y0[3] > 0:

    sol = odeint(integrate,
                 y0,
                 [t, t + dt],
                 args=(R_earth, g, alpha, T, m0))

    mass_total_after_burnout = m0 - spi.quad(integrand, 0, t, args=G_c)[0]
    m0 = mass_total_after_burnout

    # Update y0
    y0 = sol[-1]
    y0[4] = m0  # update mass in y0

    # Save variable
    V = sol[:, 0]
    theta = sol[:, 1]
    x = sol[:, 2]
    y = sol[:, 3]
    m = sol[:, 4]

    # Save it to lists
    V_list .append(V)
    theta_list.append(theta)
    x_list.append(x)
    y_list.append(y)
    m_list.append(m)
    t_list.append(t)

    # Update time
    t += dt

x_list_km = np.array(x_list) / 1000
y_list_km = np.array(y_list) / 1000

# Create a scatterplot of x and y
sns.set_style("whitegrid")
sns.set_palette("husl")
sns.lineplot(x=x_list_km, y=y_list_km)
plt.xlabel('Distance, km')
plt.ylabel('Altitude, km')

# Создаем графики скорости, массы и времени
fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(15, 15))
plt.subplots_adjust(hspace=0.4, wspace=0.4)

ax1.plot(t_list, V_list)
ax1.set_ylabel('Velocity, m/s')

ax2.plot(t_list, m_list)
ax2.set_ylabel('Mass, kg')

# ax3.plot(t_list)
# ax3.set_xlabel('Time, s')
# ax3.set_ylabel('Thrust, kN')

plt.show()
