import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81  # Gravitational acceleration (m/s^2)
R_earth = 6371000  # Radius of the Earth (m)


#   y[0] = h
#   y[1] = v
#   y[2] = alpha angle attack
#   y[3] = m
#   y[4] = theta trajectory angle

def hypersonic_flight_range(T, m0, H0, V0, alpha0, dt, t_max):
    # Define initial conditions
    H = [H0]
    V = [V0]
    alpha = [alpha0]
    m = [m0]
    theta = [0]  # Assuming theta starts from 0
    L = [0]  # Assuming the initial position is (0,0)


    # Define the Runge-Kutta method of the 2nd order in dt
    def rk2(f, t, y, dt):
        k1 = dt * f(t, y)
        k2 = dt * f(t + dt / 2, [y[i] + k1[i] / 2 for i in range(len(y))])
        return k2

    # Define the functions for the differential equations
    def dV_dt(t, y):
        return (T * np.cos(y[2])) / y[3] - g * np.sin(y[4])

    def dtheta_dt(t, y):
        return (T * np.sin(y[2])) / (y[3] * y[1]) - (g * np.cos(y[4])) / y[1] + (y[1] * np.cos(y[4])) / (R_earth + y[0])

    def dL_dt(t, y):
        print(len(y))  # Check the length of y
        print(y)  # Print the values of y
        return y[1] * np.cos(y[4]) * R_earth / (R_earth + y[0])

    def dH_dt(t, y):
        return y[1] * np.sin(y[4])

    # Based on the assumption that density is constant
    def dm_dt(t, y):
        return -T / (y[3] * np.sqrt(y[0] ** 2 + 2 * R_earth * y[0]))

    # Calculate the total number of time steps
    N = int(np.ceil(t_max / dt))

    # Create a list of time steps
    t = np.linspace(0, t_max, N)

    # Solve the differential equations using the Runge-Kutta method
    while t[-1] < t_max:
        # Calculate the next values of H, V, alpha, L, and t using the rk2
        H_next = H[-1] + rk2(dH_dt, t[-1],[H[-1], V[-1], alpha[-1], m[-1], theta[-1]], dt)
        V_next = V[-1] + rk2(dV_dt, t[-1],[H[-1], V[-1], alpha[-1], m[-1], theta[-1]], dt)
        alpha_next = alpha[-1] + rk2(dtheta_dt, t[-1],[H[-1], V[-1], alpha[-1], m[-1]], dt)
        theta_next = theta[-1] + rk2(dtheta_dt, t[-1],[H[-1], V[-1], alpha[-1], m[-1]], dt)
        L_next = L[-1] + rk2(dL_dt, t[-1],[H[-1], V[-1], alpha[-1]], dt)
        t_next = t[-1] + dt
        dm = dm_dt(t[-1], [H[-1], V[-1], alpha[-1], m[-1]])

        # Append the next values to the lists
        H.append(H_next)
        V.append(V_next)
        alpha.append(alpha_next)
        L.append(L_next)
        theta.append(theta_next)
        t.append(t_next)

    # Compute the altitude and range as a function of time
    for i in range(len(t)):
        dL_dt = dL_dt(t[i], [H[i], V[i], alpha[i], 0, 0])
        dH_dt = dH_dt(t[i], [H[i], V[i], alpha[i], 0, 0])
        if i == 0:
            L[i] = 0
        else:
            L[i] = L[i - 1] + dL_dt * dt
        H[i] = H[i] + dH_dt * dt

    return L, H, V, alpha, t


# Define the input parameters
T = 20000  # Thrust (N)
m0 = 1500  # Mass (kg)
H0 = 10  # Initial altitude (m)
V0 = 5000  # Initial speed (m/s)
alpha0 = 20  # Initial angle of inclination (rad)

dt = 0.1  # Time step (s)
t_max = 600  # Maximum time (s)

# Call the function to solve the differential equations and calculate the flight range
L, H, V, alpha, t = hypersonic_flight_range(T, m0, H0, V0, alpha0, dt, t_max)

# Plot the flight range as a function of time
plt.plot(t, L)
plt.xlabel('Time (s)')
plt.ylabel('Flight range (m)')
plt.show()

# Plot the altitude as a function of time
plt.plot(t, H)
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.show()

# Plot the flight speed as a function of time
plt.plot(t, V)
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.show()

# Plot the trajectory angle as a function of time
plt.plot(t, alpha)
plt.xlabel('Time (s)')
plt.ylabel('Trajectory angle (rad)')
plt.show()
