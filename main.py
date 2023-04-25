import datetime
import warnings
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, MatplotlibDeprecationWarning
import scipy.integrate as spi
from scipy.integrate import odeint

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


# region data
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
#
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
# endregion

# Constants
g = 9.8066                  # gravitational acceleration, m/s^2
R_earth = 6370000.0         # radius of the Earth, m

# Initial condition of Hypersonic Aircraft
m_fuel = 450.0              # +initial mass fuel of aircraft, kg
m_ha = 1440 - m_fuel        # +initial mass of aircraft, kg
m_total = m_ha + m_fuel     # total mass of aircraft, kg
x = 0.0                     # initial Distance, m
y = 2 * 1000.0              # 0-40.000 initial altitude, m
V = 3 * 340.0               # 2-6 +initial velocity, m/s    1 mach = 340 m/s
alpha = np.radians(0)       # 0-10 +initial angle of attack, radians
theta = np.radians(43)      # +initial angle of inclination of the flight trajectory, radians
dt = 0.01                   # time step, s
G_c = 0.002                 # +initial fuel burnout rate per dt
I_sp = 15.0                 # +initial specific impulse

length = 12.5
diameter = 0.5
frontalArea = diameter**2/ (4 * np.pi)
lowerArea = length * diameter

CoefficientDrag = .4       # drag coefficient for cone 20 degrees
CoefficientLift = 1.2       # +lift coefficient for cone 20 degrees
def rhoDensity(h):
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

def Drag(CoefficientDrag, h, v, frontalArea):
    return CoefficientDrag * rhoDensity(h) * v**2 * (frontalArea / 2)

def Lift(CoefficientLift, h, v, lowerArea):
 return CoefficientLift * rhoDensity(h) * v**2 * (lowerArea / 2)

def f(t):
    return G_c

def mass_after_fuel_burning(t, m_fuel):
    try:
        return m_fuel - spi.quad(f, 0, t)[0]
    except spi.IntegrationWarning:
        return 0.0

def thrust(m_fuel, g, I_sp):
    return m_fuel * g * I_sp # kg *m/s != кг*м / c^2

def hypersonic_aircraft_model(y, t, R_earth, g, T, alpha, m_total, drag, lift):
    V, theta, x, y = y

    dV = (T * np.cos(alpha) - drag) / m_total - g * np.sin(theta)
    dtheta = (T * np.sin(alpha) + lift) / (m_total * V)  - (g * np.cos(theta)) / V + (V * np.cos(theta)) / (R_earth + y)
    dx = V * np.cos(theta) * R_earth / (R_earth + y)
    dy = V * np.sin(theta)

    return [dV, dtheta, dx, dy]


# Integration interval
t = 0.0                     # initial time
y0 = [V, theta, x, y]       # initial conditions

# Initial drag and lift
drag = Drag(CoefficientDrag, y, V, frontalArea)
lift = Lift(CoefficientLift, y, V, frontalArea)

# Arrays to store results
t_arr = [0.0]
v_arr = [V / 340]
theta_arr = [np.degrees(theta)]
x_arr = [x]
y_arr = [y / 1000]
m_arr = [m_total]
thrust_arr = [0.0]
lift_arr = [lift / 1000]
drag_arr = [drag / 1000]
alpha_arr = [np.degrees(alpha)]

is_Engine = True

is_ballistic_trajectory = False  # alpha = 0, горючее сгорает во время набора максимальной высоты и
                                 # дальше летит пока не достигнет точки апогея и не достигнет высоты h = 0

is_planning_trajectory = False   # набирает высоту => переходит к горизонтальному полету с исопльзованием lift,
                                 # alpha определяется изсходя из требования равенства нулю проекций действующих сил,
                                 # включая силу тяги двигаетля на вертикальную ось
# if lift + T * np.cos(alpha) == 0: то есть ALPHA = ARCCOS(-LIFT / T) ИЛИ alpha_degrees = 180/np.pi * np.arccos(-Lift / Thrust)

is_ricocheting_trajectory = False# При снижении ЛА с углом атаки alpha !=0 за счет возрастающего влияния аэродинамической подъемной силы
                                 # возникает эффект рикошетирования ,когда высота полета может начать увеличиваться.
                                 # И снова в момент набора высоты включить ненадолго двигатель и тогда получим несколько циклов рикошетирования


while y > 0.0:

    if theta < np.radians(19):
        is_Engine = True
    else:
        is_Engine = False


    if m_fuel > 0 and is_Engine == True:
        alpha = np.radians(6)
        m_fuel = mass_after_fuel_burning(t, m_fuel)
        m_total = m_ha + m_fuel
        T = thrust(m_fuel, g, I_sp)
    elif not is_Engine:
        m_total = m_ha + m_fuel
        T = 0
    else:
        m_total = m_ha
        T = 0


    drag = Drag(CoefficientDrag, y, V, frontalArea)
    lift = Lift(CoefficientLift, y, V, frontalArea)

    sol = odeint(hypersonic_aircraft_model, y0, [t, t + dt], args=(R_earth, g, T, alpha, m_total, drag, lift))

    # Update y0
    y0 = sol[-1]
    V, theta, x, y = sol[-1]

    # Update time
    t += dt

    # Store results
    t_arr.append(t)
    v_arr.append(V / 340)
    theta_arr.append(np.degrees(theta))
    alpha_arr.append(np.degrees(alpha))
    x_arr.append(x / 1000)
    y_arr.append(y / 1000)
    m_arr.append(m_total)
    thrust_arr.append(T / 1000)
    lift_arr.append(lift / 1000)
    drag_arr.append(drag / 1000)


# region Output plots
sns.set_style("whitegrid")
fig, (hx, thetaXalpha) = plt.subplots(nrows=2, ncols = 1, figsize=(12, 12), height_ratios=[3,2])
plt.subplots_adjust(hspace=0.3)

sns.lineplot(x=x_arr, y=y_arr, color='black', linewidth=3, ax=hx, label='Distance')
sns.lineplot(x=x_arr, y=lift_arr, color='#512DA8', linewidth=2, ax=hx, label='Lift')
sns.lineplot(x=x_arr, y=drag_arr, color='#f12e6d', linewidth=2, ax=hx, label='Drag')
hx.legend(loc='upper right')

plt.scatter(x=x_arr, y=theta_arr, s=1, color='#512DA8', label='Theta')
plt.scatter(x=x_arr, y=alpha_arr, s=1, color='#f12e6d', label='Alpha')
thetaXalpha.legend(loc='lower right')

hx.set(ylabel='Высота, км. // Сила тяги и споротивления, кН',
       xlabel='Дальность, км.',
       yticks=np.linspace(0, np.max(y_arr), 10),
       xticks=np.linspace(np.min(x_arr), np.max(x_arr), 10))
thetaXalpha.set(xlabel='Дальность, км.',
                ylabel='Угол Тета и Угол Альфа, град.',
                yticks=np.linspace(np.min([np.min(theta_arr), np.min(alpha_arr)]),
                                   np.max([np.max(theta_arr), np.max(alpha_arr)]), 8),
                xticks=np.linspace(np.min(x_arr), np.max(x_arr), 10))

thetaXalpha.invert_yaxis()


# fig, axld = plt.subplots(figsize=(8, 8))
# sns.lineplot(x=x_arr, y=lift_arr, color='blue', ax=axld)
# sns.lineplot(x=x_arr, y=drag_arr, color='red', ax=axld)
# axld.set_title('График сил: подьемная и сопротивления')
# axld.set_xlabel('Дальность, км')
# axld.set_ylabel('Сила в кН')


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
ax3.set_title('График зависимости массы ЛА от времени')

# Plot 4: Theta versus Time
sns.lineplot(x=t_arr, y=thrust_arr, ax=ax4, color='black', linewidth=2)
ax4.set_xlabel('Время, сек.')
ax4.set_ylabel('Сила тяги, кН.')
ax4.set_title('График')

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
ax4.text(0.5, -0.6, f"Данный график был сделан: {now}", ha='center', va='center', transform=ax4.transAxes)
plt.subplots_adjust(hspace=0.8)

plt.show()
# endregion