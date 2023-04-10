import numpy as np
import matplotlib.pyplot as plt

# Define constants
g0 = 9.81   # m/s^2, gravitational acceleration at sea level
Re = 6371000   # m, radius of the Earth
Isp = 4500   # s, specific impulse of the engine
m0 = 1000   # kg, initial mass of the aircraft
S = 10   # m^2, cross-sectional area of the aircraft
Cd = 0.1   # drag coefficient of the aircraft
H0 = 2000   # m, initial altitude of the aircraft
L0 = 0   # m, initial range of the aircraft
v0 = 5000   # m/s, initial speed of the aircraft
alpha = np.deg2rad(5)   # rad, thrust angle
dt = 0.1   # s, time step
tf = 10000   # s, final time

# Define functions
def rho(H):
    """Calculate air density as a function of altitude"""
    return 1.225 * np.exp(-H/8000)

def drag(V, H):
    """Calculate drag force on the aircraft"""
    return 0.5 * rho(H) * V**2 * S * Cd

def thrust(m, V, H):
    """Calculate thrust force of the engine"""
    return T(m) - drag(V, H)

def T(m):
    """Calculate thrust as a function of mass"""
    return 50000 * (m/m0)**0.75

def m_dot(T):
    """Calculate mass flow rate"""
    return T / (g0 * Isp)

def V_dot(T, alpha, m, V, theta, H):
    """Calculate rate of change of speed"""
    return (T * np.cos(alpha) - drag(V, H) - m_dot(T) * V) / m - g0 * np.sin(theta)

def theta_dot(T, alpha, m, V, theta, H):
    """Calculate rate of change of inclination angle"""
    return (T * np.sin(alpha)) / (m * V) - (g0 * np.cos(theta)) / V + (V * np.cos(theta)) / (Re + H)

def L_dot(V, theta, H):
    """Calculate rate of change of range"""
    return V * np.cos(theta) * Re / (Re + H)

def H_dot(V, theta):
    """Calculate rate of change of altitude"""
    return V * np.sin(theta)

# Initialize arrays
t = np.arange(0, tf+dt, dt)
V = np.zeros_like(t)
theta = np.zeros_like(t)
H = np.zeros_like(t)
L = np.zeros_like(t)
m = np.zeros_like(t)

# Set initial conditions
V[0] = v0
theta[0] = np.pi/2
H[0] = H0
L[0] = L0
m[0] = m0

# Solve system of equations using second order Runge-Kutta method
i = 0
while H[i] > 0:
    # Calculate rate of change of variables
    V_k1 = dt * V_dot(T(m[i]), alpha, m[i], V[i], theta[i], H[i])
    theta_k1 = dt * theta_dot(T(m[i]), alpha, m[i], V[i], theta[i], H[i])
    H_k1 = dt * H_dot(V[i], theta[i])
    L_k1 = dt * L_dot(V[i], theta[i], H[i])
    m_k1 = dt * m_dot(T(m[i]))

    V_k2 = dt * V_dot(T(m[i]), alpha, m[i] + 0.5*m_k1, V[i] + 0.5*V_k1, theta[i] + 0.5*theta_k1, H[i] + 0.5*H_k1)
    theta_k2 = dt * theta_dot(T(m[i]), alpha, m[i] + 0.5*m_k1, V[i] + 0.5*V_k1, theta[i] + 0.5*theta_k1, H[i] + 0.5*H_k1)
    H_k2 = dt * H_dot(V[i] + 0.5*V_k1, theta[i] + 0.5*theta_k1)
    L_k2 = dt * L_dot(V[i] + 0.5*V_k1, theta[i] + 0.5*theta_k1, H[i] + 0.5*H_k1)
    m_k2 = dt * m_dot(T(m[i] + 0.5*m_k1))

    # Update variables
    V[i+1] = V[i] + V_k2
    theta[i+1] = theta[i] + theta_k2
    H[i+1] = H[i] + H_k2
    L[i+1] = L[i] + L_k2
    m[i+1] = m[i] - m_k2

    i += 1


# Plot results
plt.plot(L/1000, H/1000)
plt.xlabel('Range (km)')
plt.ylabel('Altitude (km)')
plt.title('Hypersonic Aircraft Trajectory')
plt.show()