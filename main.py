import datetime
import warnings
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, ticker, MatplotlibDeprecationWarning
import scipy.integrate as spi
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

# Initial condition of Hypersonic Aircraft
T = 100000.0                # +initial thrust force, N
m_fuel = 450.0              # +initial mass fuel of aircraft, kg
m_ha = 1440 - m_fuel        # +initial mass of aircraft, kg
m_total = m_ha + m_fuel     # total mass of aircraft, kg
x = 0.0                     # initial Distance, m
h = 2 * 1000.0              # 0-40.000 initial altitude, m
v = 3 * 340.0               # 2-6 +initial velocity, m/s    1 mach = 340 m/s
alpha = np.radians(0)       # 0-10 +initial angle of attack, radians
theta = np.radians(45)      # +initial angle of inclination of the flight trajectory, radians
dt = 0.01                   # time step, s
G_c = 0.02                  # +initial fuel burnout rate, N/s
I_sp = 15.0                 # +initial specific impulse


def f(t):
    return G_c

def mass_after_fuel_burning(t, m_fuel):
    try:
        return m_fuel - spi.quad(f, 0, t)[0]
    except spi.IntegrationWarning:
        return 0.0

def Thrust(m_fuel, g, I_sp):
    return m_fuel * g * I_sp # kg *m/s != кг*м / c^2

def hypersonic_aircraft_model(y, t, R_earth, g, T, alpha, m_total):
    V, theta, x, h = y

    dV = (T * np.cos(alpha)) / m_total - g * np.sin(theta)
    dtheta = (T * np.sin(alpha)) / (m_total * V) - (g * np.cos(theta)) / V + (V * np.cos(theta)) / (R_earth + h)
    dx = V * np.cos(theta) * R_earth / (R_earth + h)
    dy = V * np.sin(theta)

    return [dV, dtheta, dx, dy]


# Integration interval
t = 0.0                     # initial time
y0 = [v, theta, x, h]       # initial conditions

# Arrays to store results
t_arr = [0.0]
v_arr = [v / 340.0]
theta_arr = [np.degrees(theta)]
x_arr = [x / 1000]
h_arr = [h / 1000]
m_arr = [m_total]

while h > 0.0:

    if m_fuel > 0:
        m_fuel = mass_after_fuel_burning(t, m_fuel)
        m_total = m_ha + m_fuel
        T = Thrust(m_fuel, g, I_sp)
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
    v_arr.append(V / 340)
    theta_arr.append(np.degrees(theta))
    x_arr.append(x / 1000)
    h_arr.append(h / 1000)
    m_arr.append(m_total)


# region Output plots
# Set the style for the plot
sns.set_style("whitegrid")

# Create a figure and axis object
fig, ax1 = plt.subplots(figsize=(8, 8))

# Plot the line
sns.lineplot(x=x_arr, y=h_arr, ax=ax1, color='black', linewidth=2)

# Set the axis labels and title
ax1.set_xlabel('Дальность, км.')
ax1.set_ylabel('Высота, км.')
ax1.set_title('График зависимости дальности и высоты ЛА')

# Create a figure with 2 subplots, arranged in a vertical layout
fig, (ax2, ax3, ax4) = plt.subplots(nrows=3, ncols=1, figsize=(10,10), height_ratios=[2,1,1])

# Plot 2: Velocity on Time
sns.lineplot(x=t_arr, y=v_arr, ax=ax2, color='black', linewidth=1)
ax2.set_xlabel('Время, сек.')
ax2.set_ylabel('Скорость ЛА, мах')
ax2.set_title('График зависимости скорости ЛА от времени')

# Plot 3: Mass by Time
sns.lineplot(x=t_arr, y=m_arr, ax=ax3, color='black', linewidth=1)
ax3.set_xlabel('Время, сек.')
ax3.set_ylabel('Масса ЛА, кг.')
ax3.set_title('График зависимости  массы ЛА от времени')

# Plot 4: Theta versus Time
sns.lineplot(x=theta_arr, y=t_arr, ax=ax4, color='black', linewidth=1)
ax4.set_xlabel('Угол Тета, град.')
ax4.set_ylabel('Время, сек.')
ax4.set_title('График зависимости угла Тета(наклона траектории полета) от времени')


now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
ax4.text(0.5, -0.6, f"Данный график был сделан: {now}", ha='center', va='center', transform=ax4.transAxes)
# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.8)

# Show the plot
plt.show()
# endregion