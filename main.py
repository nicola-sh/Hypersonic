import numpy as np
from matplotlib import pyplot as plt, MatplotlibDeprecationWarning
from scipy.integrate import odeint
import pandas as pd
import seaborn as sns
import matplotlib
import warnings

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


# Задание функции правой части системы уравнений
def integrate(y, t, T, m, I_sp, h, R_earth, m_fuel, alpha):
    V, theta, x, y, m_fuel = y

    if m_fuel <= 0:
        T = 0

    m_total = m + m_fuel

    # Расчет производных
    dV = (T * np.cos(alpha)) / m_total - g * np.sin(theta)
    dtheta = ((T * np.sin(alpha)) / (m_total * V) - (g * np.cos(theta)) / V + (V * np.cos(theta)) / (R_earth + h))
    dx = V * np.cos(theta)
    dy = V * np.sin(theta)

    # Расчет изменения массы топлива
    if m_fuel > 0:
        dm_fuel = -T / (g * I_sp)
    else:
        dm_fuel = 0

    # Возвращаем производные в виде массива
    return [dV, dtheta, dx, dy, dm_fuel]


# Constants
g = 9.81  # gravitational acceleration, m/s^2
R_earth = 6371000.0  # radius of the Earth, m


# Задание начальных условий
T = 200000.0  # thrust, N
m = 1400.0  # initial mass, kg
m_fuel = 450.0  # initial mass of fuel, kg
v = 5000.0  # initial velocity, m/s
x = 0.0  # initial Distance, m
y = 2000.0  # initial altitude, m
alpha = np.radians(0)  # initial angle of attack, radians
theta = np.radians(0)  # initial angle of inclination of the flight trajectory, radians
I_sp = 325.0  # specific impulse

# Интервал интегрирования
dt = 0.01  # time step, s
t = 0  # initial time
y0 = [v, theta, x, y, m_fuel]  # initial conditions

# Решение системы уравнений
V_list = []
theta_list = []
x_list = []
y_list = []
m_fuel_list = []
t_list = []

while y0[3] > 0:

    sol = odeint(integrate, y0,
                 [t, t + dt],
                 args=(T, m, I_sp, y0[3], R_earth, m_fuel, alpha))
    y0 = sol[-1]

    # Извлечение решения
    V = sol[:, 0]
    theta = sol[:, 1]
    x = sol[:, 2]
    y = sol[:, 3]
    m_fuel = sol[:, 4]

    # Сохраняем результаты
    V_list += list(V[:-1])
    theta_list += list(theta[:-1])
    x_list += list(x[:-1])
    y_list += list(y[:-1])
    m_fuel_list += list(m_fuel[:-1])
    t_list += list(np.linspace(t, t+dt, len(V)-1))

    # Обновляем время
    t += dt

x_list_km = np.array(x_list) / 1000
y_list_km = np.array(y_list) / 1000

# Create a scatterplot of x and y
sns.set_style("whitegrid")
sns.set_palette("husl")
sns.lineplot(x=x_list_km, y=y_list_km)
plt.xlabel('Distance, km')
plt.ylabel('Altitude, km')
plt.show()


# def pressure(h):
#     "Calculates air pressure [Pa] at altitude [m]"
#     # from equations at
#     #   http://www.grc.nasa.gov/WWW/K-12/airplane/atmosmet.html
#
#     t = temperature(h)
#
#     if h <= 11000:
#         # troposphere
#         p = 101.29 * ((t + 273.1) / 288.08) ** 5.256
#     elif h <= 25000:
#         # lower stratosphere
#         p = 22.65 * np.exp(1.73 - .000157 * h)
#     elif h > 25000:
#         p = 2.488 * ((t + 273.1) / 288.08) ** -11.388
#     return p
#
#
# def temperature(h):
#     "Calculates air temperature [Celsius] at altitude [m]"
#     #from equations at
#     #   http://www.grc.nasa.gov/WWW/K-12/airplane/atmosmet.html
#     if h <= 11000:
#         #troposphere
#         t = 15.04 - .00649*h
#     elif h <= 25000:
#         #lower stratosphere
#         t = -56.46
#     elif h > 25000:
#         t = -131.21 + .00299*h
#     return t
