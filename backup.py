import warnings
import pandas as pd
import numpy as np
import scipy.integrate as spi
import seaborn as sns
from matplotlib import pyplot as plt, MatplotlibDeprecationWarning
from scipy.integrate import odeint

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


def ballistic_trajectory(x, t):
    # TODO
    pass

def planning_trajectory(x, t):
    # TODO
    pass

def ricocheting_trajectory(x, t):
    # TODO
    pass

# ThrustEquation Thrust = dm/dt * g * I_sp
# where:
# dm/dt = mass flow rate
# g = acceleration due to gravity
# I_sp = specific impulse

def ThrustEquation(t, m, g, I_sp):
    return

# The drag coefficient Cd is equal to the drag D divided by the quantity:
# density r times half the velocity V squared times the reference area A.
# Drag = CoeffcientDrag * rho * v^2 * (reference area / 2)
def Drag(CoefficientDrag, rho, v, A):
    return CoefficientDrag * rho * v**2 * (A / 2)

# The lift coefficient Cl is equal to the lift L divided by the quantity:
# density r times half the velocity V squared times the reference area A.
# Lift = CoefficientLift * rho * v^2 * (reference area / 2)
def Lift(CoefficientLift, rho, v, L, d):
 return CoefficientLift * rho * v**2 * (L / 2)

def density(h):
    "Calculates air density at altitude"
    rho0 = 1.225 #[kg/m^3] air density at sea level
    if h < 19200:
        #use barometric formula, where 8420 is effective height of atmosphere [m]
        rho = rho0 * np.exp(-h/8420)
    elif h > 19200 and h < 47000:
        #use 1976 Standard Atmosphere model
        #http://modelweb.gsfc.nasa.gov/atmos/us_standard.html
        #from http://scipp.ucsc.edu/outreach/balloon/glost/environment3.html
        rho = rho0 * (.857003 + h/57947)**-13.201
    else:
        #vacuum
        rho = 1.e-6
    return rho

def temperature(h):
    "Calculates air temperature [Celsius] at altitude [m]"
    #from equations at
    #   http://www.grc.nasa.gov/WWW/K-12/airplane/atmosmet.html
    if h <= 11000:
        #troposphere
        t = 15.04 - .00649*h
    elif h <= 25000:
        #lower stratosphere
        t = -56.46
    elif h > 25000:
        t = -131.21 + .00299*h
    return t

def pressure(h):
    "Calculates air pressure [Pa] at altitude [m]"
    # from equations at
    #   http://www.grc.nasa.gov/WWW/K-12/airplane/atmosmet.html

    t = temperature(h)

    if h <= 11000:
        # troposphere
        p = 101.29 * ((t + 273.1) / 288.08) ** 5.256
    elif h <= 25000:
        # lower stratosphere
        p = 22.65 * np.exp(1.73 - .000157 * h)
    elif h > 25000:
        p = 2.488 * ((t + 273.1) / 288.08) ** -11.388
    return p


# Constants
g = 9.8066                  # gravitational acceleration, m/s^2
R_earth = 6370000.0         # radius of the Earth, m
CoefficientDrag = 0.5       # Coefficient of drag for Cone, 0.5
CoefficientLift = 0.5       # Coefficient of lift for Cone, 0.5

# Initial condition of Hypersonic Aircraft
L = 12.5                    # length of the Aircraft, m
d = 0.5                     # diameter of the Aircraft, m
gamma = np.radians(20)      # angle of the nose cone of  the Aircraft, radians

T = 100000.0                # +initial thrust force, N
m_ha = 1000.0               # +initial mass of aircraft, kg
m_fuel = 450.0              # +initial mass fuel of aircraft, kg
x = 0.0                     # initial Distance, m
h = 2 * 1000.0              # initial altitude, m
v = 5 * 340.0               # +initial velocity, m/s    1 mach = 340 m/s
alpha = np.radians(30)      # +initial angle of attack, radians
theta = np.radians(0)       # +initial angle of inclination of the flight trajectory, radians
dt = 0.01                   # time step, s
G_c = 0.02                  # +initial fuel burnout per dt

m_total = m_ha + m_fuel
m_arr = [m_total]

def hypersonic_aircraft_model(y, t, T, m_ha, m_fuel):
    V, theta, x, h, mfb = y

    mfb = G_c
    m_total = m_ha + m_fuel - mfb


    m_arr.append(m_total)


    dV = (T * np.cos(alpha)) / m_total - g * np.sin(theta)
    dtheta = (T * np.sin(alpha)) / (m_total * V) - (g * np.cos(theta)) / V + (V * np.cos(theta)) / (R_earth + h)
    dx = V * np.cos(theta) * R_earth / (R_earth + h)
    dy = V * np.sin(theta)

    return [dV, dtheta, dx, dy, mfb]


# Integration interval
t = 0.0                         # initial time
mfb = 0.0                      # initial fuel burnout
y0 = [v, theta, x, h, mfb]     # initial conditions

# Arrays to store results
t_arr = [0.0]
x_arr = [x / 1000]
h_arr = [h / 1000]
v_arr = [v]

while h > 0.0:
    # solve model
    sol = odeint(hypersonic_aircraft_model, y0, [t, t + dt], args=(T, m_ha, m_fuel))

    # Update y0
    y0 = sol[-1]
    V, theta, x, h, mafb = sol[-1]
    # Update time
    t += dt

    # Store results
    t_arr.append(t)
    x_arr.append(x / 1000)
    h_arr.append(h / 1000)
    v_arr.append(V)


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