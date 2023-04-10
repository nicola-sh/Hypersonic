import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81  # Gravitational acceleration (m/s^2)
R_earth = 6371000  # Radius of the Earth (m)
dt = 0.1  # Time step (s)
t_max = 600  # Maximum time (s)

#   y[0] = h
#   y[1] = v
#   y[2] = alpha
#   y[3] = m
#   y[4] = theta

def hypersonic_flight_range(T, m, H0, V0, alpha0):
    # Define initial conditions
    H = [H0]
    V = [V0]
    alpha = [alpha0]
    L = [0]
    t = [0]

    # Define the Runge-Kutta method of the 2nd order in dt
    def rk2(f, t, y, dt):
        k1 = dt * f(t, y)
        k2 = dt * f(t + dt / 2, [y[i] + k1[i] / 2 for i in range(5)])
        return k2

    # Define the functions for the differential equations
    def dV_dt(t, y):
        return (T * np.cos(y[2])) / m - g * np.sin(y[4])

    def dtheta_dt(t, y):
        return (T * np.sin(y[2])) / (m * y[1]) - (g * np.cos(y[4])) / y[1] + (y[1] * np.cos(y[4])) / (R_earth + y[0])

    def dL_dt(t, y):
        return y[1] * np.cos(y[4]) * R_earth / (R_earth + y[0])

    def dH_dt(t, y):
        return y[1] * np.sin(y[4])

    # Solve the differential equations using the Runge-Kutta method
    while t[-1] < t_max:
        # Calculate the next values of H, V, alpha, L, and t using the rk2
        H_next = H[-1] + rk2(dH_dt, t[-1],
                             [V[-1], alpha[-1]], dt)
        V_next = V[-1] + rk2(dV_dt, t[-1],
                             [t[-1], V[-1], alpha[-1], H[-1]], dt)
        alpha_next = alpha[-1] + rk2(dtheta_dt, t[-1],
                                     [t[-1], V[-1], alpha[-1], H[-1]], dt)
        L_next = L[-1] + rk2(dL_dt, t[-1],
                             [V[-1], alpha[-1], H[-1]], dt)
        t_next = t[-1] + dt

        # Append the next values to the lists
        H.append(H_next)
        V.append(V_next)
        alpha.append(alpha_next)
        L.append(L_next)
        t.append(t_next)

    return L, H, V, alpha, t


# Define the input parameters
T = 20000  # Thrust (N)
m = 1500  # Mass (kg)
H0 = 0  # Initial altitude (m)
V0 = 5000  # Initial speed (m/s)
alpha0 = 0  # Initial angle of inclination (rad)

# Call the function to solve the differential equations and calculate the flight range
L, H, V, alpha, t = hypersonic_flight_range(T, m, H0, V0, alpha0)

# Plot the flight range as a function of time=
plt.plot(t, L)
plt.xlabel('Time (s)')
plt.ylabel('Flight range (m)')
plt.show()