import numpy as np
import matplotlib.pyplot as plt


# Define parameters and initial conditions
T = 50000  # thrust
m = 1000  # variable mass
g = 9.81  # gravitational acceleration
R_earth = 6371000  # radius of the Earth
V0 = 5000  # initial velocity
theta0 = 0.5  # initial angle of inclination (in radians)
L0 = 0  # initial range
H0 = 1000  # initial altitude

# Define the functions for the right-hand side of the differential equations
def dVdt(V, theta, H):
    return (T * np.cos(theta))/m - g * np.sin(H/(R_earth + H))

def dthetadt(V, theta, H):
    return (T * np.sin(theta))/(m*V) - (g*np.cos(H/(R_earth + H)))/V + (V*np.cos(H/(R_earth + H)))/(R_earth + H)

def dLdt(V, theta, H):
    return V*np.cos(theta)*(R_earth/(R_earth + H))

def dHdt(V, theta, H):
    return V*np.sin(theta)

# Define the time step and the number of time steps
dt = 0.6  # time step
N = 10000  # number of time steps

# Initialize arrays to store the values of V, theta, L, and H at each time step
V = np.zeros(N)
theta = np.zeros(N)
L = np.zeros(N)
H = np.zeros(N)

# Set the initial values
V[0] = V0
theta[0] = theta0
L[0] = L0
H[0] = H0

# Perform the numerical integration using the second-order Runge-Kutta method
for i in range(1, N):
    k1_V = dVdt(V[i-1], theta[i-1], H[i-1])
    k1_theta = dthetadt(V[i-1], theta[i-1], H[i-1])
    k1_L = dLdt(V[i-1], theta[i-1], H[i-1])
    k1_H = dHdt(V[i-1], theta[i-1], H[i-1])

    k2_V = dVdt(V[i - 1] + 0.5 * dt * k1_V, theta[i - 1] + 0.5 * dt * k1_theta, H[i - 1] + 0.5 * dt * k1_H)
    k2_theta = dthetadt(V[i - 1] + 0.5 * dt * k1_V, theta[i - 1] + 0.5 * dt * k1_theta, H[i - 1] + 0.5 * dt * k1_H)
    k2_L = dLdt(V[i - 1] + 0.5 * dt * k1_V, theta[i - 1] + 0.5 * dt * k1_theta, H[i - 1] + 0.5 * dt * k1_H)
    k2_H = dHdt(V[i - 1] + 0.5 * dt * k1_V, theta[i - 1] + 0.5 * dt * k1_theta, H[i - 1] + 0.5 * dt * k1_H)

    V[i] = V[i - 1] + dt * k2_V
    theta[i] = theta[i - 1] + dt * k2_theta
    L[i] = L[i - 1] + dt * k2_L
    H[i] = H[i - 1] + dt * k2_H

#Plot the results


plt.figure(figsize=(10,10))
plt.plot(L/1000, H/1000)
plt.xlabel('Range (km)')
plt.ylabel('Altitude (km)')
plt.title('Flight Trajectory')
plt.show()