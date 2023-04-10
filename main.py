import numpy as np
import matplotlib.pyplot as plt

# Constants
g0 = 9.81  # m/s^2, gravitational acceleration
R_earth = 6371000  # m, radius of the Earth
I_sp = 3050  # s, specific impulse
m0 = 1000  # kg, initial mass of the aircraft
L0 = 0  # m, initial range of flight
H0 = 2000  # m, initial altitude of flight
V0 = 5000  # m/s, initial speed of flight
T0 = 100000  # N, initial thrust of the engine
alpha0 = np.radians(5)  # rad, initial angle between the thrust direction and the missile axis

# Time
t0 = 0  # s, initial time
tf = 200  # s, final time
dt = 0.1  # s, time step

# Arrays
t = np.arange(t0, tf, dt)
m = np.zeros(len(t))
V = np.zeros(len(t))
theta = np.zeros(len(t))
alpha = np.zeros(len(t))
L = np.zeros(len(t))
H = np.zeros(len(t))
T = np.zeros(len(t))
g = np.zeros(len(t))

# Initial values
m[0] = m0
V[0] = V0
theta[0] = np.radians(89)
alpha[0] = alpha0
L[0] = L0
H[0] = H0
T[0] = T0
g[0] = g0 * (R_earth / (R_earth + H[0])) ** 2

# Second-order Runge-Kutta method
for i in range(len(t) - 1):
    # Calculating k1
    k1_v = (T[i] * np.cos(alpha[i])) / m[i] - g[i] * np.sin(theta[i])
    k1_theta = (T[i] * np.sin(alpha[i])) / (m[i] * V[i]) - (g[i] * np.cos(theta[i])) / V[i] + (
                V[i] * np.cos(theta[i])) / (R_earth + H[i])
    k1_L = V[i] * np.cos(theta[i]) * R_earth / (R_earth + H[i])
    k1_H = V[i] * np.sin(theta[i])
    k1_m = -T[i] / (g0 * I_sp)
    k1_T = 0  # thrust is constant

    # Calculating intermediate values
    m_int = m[i] + k1_m * dt / 2
    V_int = V[i] + k1_v * dt / 2
    theta_int = theta[i] + k1_theta * dt / 2
    L_int = L[i] + k1_L * dt / 2
    H_int = H[i] + k1_H * dt / 2
    T_int = T[i] + k1_T * dt / 2
    g_int = g0 * (R_earth / (R_earth + H_int)) ** 2

    # Calculating k2
    k2_v = (T_int * np.cos(alpha[i])) / m_int - g_int * np.sin(theta_int)
    k2_theta = (T_int * np.sin(alpha[i])) / (m_int * V_int) - (g_int * np.cos(theta_int)) / V_int + (V_int * np.cos(theta_int)) / (R_earth + H_int)
    k2_L = V_int * np.cos(theta_int) * R_earth / (R_earth + H_int)
    k2_H = V_int * np.sin(theta_int)
    k2_m = -T_int / (g0 * I_sp)
    k2_T = 0 # thrust is constant

    # Updating values
    m[i + 1] = m[i] + k2_m * dt
    V[i + 1] = V[i] + k2_v * dt
    theta[i + 1] = theta[i] + k2_theta * dt
    L[i + 1] = L[i] + k2_L * dt
    H[i + 1] = H[i] + k2_H * dt
    T[i + 1] = T[i] + k2_T * dt
    g[i + 1] = g0 * (R_earth / (R_earth + H[i + 1])) ** 2
    alpha[i + 1] = alpha0 # angle of thrust is constant

    # Check for termination condition
    if H[i + 1] <= 0:
        break

# Plotting results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(L[:i+2], V[:i+2], H[:i+2])
ax.set_xlabel('L')
ax.set_ylabel('V')
ax.set_zlabel('H')
plt.show()
