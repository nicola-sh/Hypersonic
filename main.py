import datetime
import warnings
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
import scipy.integrate as spi
from scipy.integrate import odeint

def hypersonic_aircraft_model(y, t, R_earth, g, T, alpha, m_total, drag, lift):
    V, theta, x, y = y

    dV = (T * np.cos(alpha) - drag) / m_total - g * np.sin(theta)
    dtheta = (T * np.sin(alpha) + lift) / (m_total * V)  - (g * np.cos(theta)) / V + (V * np.cos(theta)) / (R_earth + y)
    dx = V * np.cos(theta) * R_earth / (R_earth + y)
    dy = V * np.sin(theta)

    return [dV, dtheta, dx, dy]

def temperature(h):
    "Calculates air temperature [Celsius] at altitude [m]"
    # from equations at
    #   http://www.grc.nasa.gov/WWW/K-12/airplane/atmosmet.html
    if h <= 11000:
        # troposphere
        t = 15.04 - .00649 * h
    elif h <= 25000:
        # lower stratosphere
        t = -56.46
    elif h > 25000:
        t = -131.21 + .00299 * h
    return t

def pressure(h):
    "Calculates air pressure [Pa] at altitude [m]"
    # from equations at
    #   http://www.grc.nasa.gov/WWW/K-12/airplane/atmosmet.html

    t = temperature(h)

    if h <= 11000:
        # troposphere
        p = 101.29 * ((t + 273.1) / 288.15) ** 5.256
    elif h <= 25000:
        # lower stratosphere
        p = 22.65 * np.exp(1.73 - .000157 * h)
    elif h > 25000:
        # upper stratosphere
        p = 2.488 * ((t + 273.1) / 216.6) ** -11.388
    return p * 1000

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

def Drag(Cd, h, v, Area):
    return .5 * Cd * density(h) * v**2 * Area

def Lift(Cl, h, v, Area):
    return .5 * Cl * density(h) * v**2  * Area

def f(t):
    return G_c

def mass_after_fuel_burning(t, m_fuel):
    try:
        return m_fuel - spi.quad(f, 0, t)[0]
    except spi.IntegrationWarning:
        return 0.0

def mdot(Area, gamma, R, Mach, h, time):
    Pt = pressure(h)
    Tt = temperature(h) + 273.15
    return ((Area * Pt / np.sqrt(Tt)) * np.sqrt(gamma/R) * Mach * (1 + .5 * (gamma - 1) * Mach**2)**\
        -((gamma + 1)/(gamma - 1)/2)) * time

def massFlowRateVelocity(massFlowRate, h, Area):
    return massFlowRate / (density(h) * Area)

def thrust(massFlowRate, g, I_sp):
    return massFlowRate * g * I_sp # kg * (m/sec**2) * sec

# # bernuli eqution P + (1/2)ρV^2 = constant
# def pressureNozzle(h, V):
#     return density(h) * V**2 / 2
#
#
# def thrustRamjet(mdot, velocityInlet, velocityNozzle, beta, phiS, AreaInlet, Area, Cd, qInlet, h):
#     Pin = pressureNozzle(h, velocityInlet)
#     Pout = pressure(h)
#     return mdot * (beta * velocityInlet * phiS - velocityInlet) + Area * (Pout - Pin) - Cd * AreaInlet * qInlet




# Constants
g = 9.8066                  # gravitational acceleration, m/s^2
R_earth = 6370000.0         # radius of the Earth, m
R = 287                     # Gas constant of air [J/(kg*K)]
gamma = 1.4                 # Ratio of specific heats [K/J]

# Integration interval
t = 0.0                     # initial time, sec
tburnout = 0.0
dt = 0.01                   # time step, sec
tEND = 30000

# Initial condition of Hypersonic Aircraft
length = 12.5                       # length of the HA, m
diameter = 0.5                      # diameter of the HA, m
radius = diameter / 2               # radius of the HA, m
Area = np.pi * radius**2            # nozzle area of the ramjet, m^2
AreaInlet = 0.54 * Area             # inlet area of the ramjet, m^2

phiS = 0.97                         # nozzle speed coefficient
beta = 1                            # relative increase in the mass of the working fluid due to fuel supply
qInlet = 1
G_c = 0.12 * dt                        # fuel burnout rate per dt 0.0012
I_sp = 1200.0                         # specific impulse, sec ~1500-2000 sec for scramjet

m_fuel = 450.0                      # mass fuel of aircraft, kg
m_ha = 1440.0 - m_fuel              # mass of aircraft, kg
m_total = m_ha + m_fuel             # total mass of aircraft, kg
x = 0.0                             # Distance, m
y = 2 * 1000.0                      # altitude, m
V = 3 * 340.0                    # velocity, m/s
Mach = V / 340.0                         # Mach number, m/s
alpha = np.radians(0)               # angle of attack, radians
theta = np.radians(0)               # angle of inclination of the flight trajectory, radians
psi = np.radians(0)                 # pitch angle, radians

Cd = .3                             # drag coefficient for cone 20 degrees
Cl = .4                             # lift coefficient for cone 20 degrees
lift = 0.0                          # lift force, N
drag = 0.0                          # drag force, N

phiF = 1.0
co2 = 0.21
fSt = 0.125


# mDotAir = mdot(Area, gamma, R, Mach, y, 1)
mDotAir = density(y) * V * AreaInlet
mDotFuel = phiF * co2 * fSt * mDotAir
TForce = I_sp * mDotFuel * g
# airVelocityInlet = massFlowRateVelocity(mdt, y, AreaInlet)     # m/s
# airVelocityOut = (AreaInlet / Area) * airVelocityInlet                  # m/s
#
# thrustRamjet_test = thrustRamjet(mdt, airVelocityInlet,
#                     airVelocityOut, beta, phiS, AreaInlet, Area, Cd, qInlet, y)
#
# thrustSpecImpulse = thrustRamjet_test / mdt

is_Engine = True
is_ballistic_trajectory = True
is_planning_trajectory = False
is_ricocheting_trajectory = False
flag = False

if is_ballistic_trajectory:
    alpha = np.radians(0)
    theta = np.radians(29.7)
    drag = Drag(Cd, y, V, Area)
    lift = 0
elif is_planning_trajectory:
    alpha = np.radians(0)
    theta = np.radians(30)
    drag = Drag(Cd, y, V, Area)
    lift = Lift(Cl, y, V, Area)
elif is_ricocheting_trajectory:
    alpha = np.radians(0)
    theta = np.radians(30)
    drag = Drag(Cd, y, V, Area)
    lift = Lift(Cl, y, V, Area)



y0 = [V, theta, x, y]       # initial conditions
weight = m_total * g



# Arrays to store results
t_arr = [0.0]
v_arr = [V / 340]
theta_arr = [np.degrees(theta)]
psi_arr = [np.degrees(psi)]
alpha_arr = [np.degrees(alpha)]
x_arr = [x]
y_arr = [y / 1000]
m_arr = [m_total]
thrust_arr = [0.0]
lift_arr = [lift / 1000]
drag_arr = [drag / 1000]
weight_arr = [weight / 1000]

while y > 0.0 and t < tEND:

    # if V < 2 * 340.0:
    #     I_sp = 12.0
    # elif V > 3 * 340.0 or V < 4 * 340.0:
    #     I_sp = 20.0
    # elif V > 4 * 340.0 or V < 5 * 340.0:
    #     I_sp = 15.0
    # elif V > 5 * 340.0:
    #     I_sp = 10.0


    if is_ballistic_trajectory:
        alpha = np.radians(0)
        is_Engine = True

    elif is_planning_trajectory:
        # набирает высоту => переходит к горизонтальному полету с исопльзованием lift,
        # alpha определяется изсходя из требования равенства нулю проекций действующих сил,
        # включая силу тяги двигаетля на вертикальную ось

        if t < 5.0:
            is_Engine = True
        else:
            is_Engine = False

        if lift - weight < 0.1:
            is_Engine = False
            if lift - weight < 0.0:
                alpha = np.radians(2)
            elif lift - weight > 1.0:
                alpha = np.radians(10)
            else:
                alpha = np.radians(2)

        if abs(V) < 3 * 340 or V - drag < 0.0:
            is_Engine = True


    elif is_ricocheting_trajectory:
        # При снижении ЛА с углом атаки alpha !=0 за счет возрастающего влияния аэродинамической подъемной силы
        # возникает эффект рикошетирования ,когда высота полета может начать увеличиваться.
        # И снова в момент набора высоты включить ненадолго двигатель и тогда получим несколько циклов рикошетирования
        if theta < np.radians(19):
            is_Engine = True
            alpha = np.radians(6)
        else:
            is_Engine = False

    # Engine


    if m_fuel > 0 and is_Engine == True:
        # alpha = np.radians(6)
        # m_fuel = mass_after_fuel_burning(t, m_fuel)
        mDotAir = density(y) * V * AreaInlet
        mDotFuel = phiF * co2 * fSt * mDotAir

        m_fuel -= mDotFuel
        m_total = m_ha + m_fuel

        T = I_sp * mDotFuel * g

        # T = thrust(massFlowRate, g, I_sp)
    else:
        m_total = m_ha + m_fuel
        T = 0

    sol = odeint(hypersonic_aircraft_model, y0, [t, t + dt], args=(R_earth, g, T, alpha, m_total, drag, lift))

    if is_ballistic_trajectory:
        drag = Drag(Cd, y, V, Area)
        lift = 0
    elif is_planning_trajectory:
        drag = Drag(Cd, y, V, Area)
        lift = Lift(Cl, y, V, Area)
    elif is_ricocheting_trajectory:
        drag = Drag(Cd, y, V, Area)
        lift = Lift(Cl, y, V, Area)

    weight = m_total * g
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
    psi_arr.append(np.degrees(psi))
    x_arr.append(x / 1000)
    y_arr.append(y / 1000)
    m_arr.append(m_total)
    thrust_arr.append(T / 1000)
    lift_arr.append(lift / 1000)
    drag_arr.append(drag / 1000)
    weight_arr.append(weight / 1000)


# region Output plots
sns.set_style("whitegrid")
fig, (hx, thetaXalpha) = plt.subplots(nrows=2, ncols = 1, figsize=(12, 12), height_ratios=[3,2])
plt.subplots_adjust(hspace=0.3)

sns.lineplot(x=x_arr, y=y_arr, color='black', linewidth=3, ax=hx, label='Дальность полета')
sns.lineplot(x=x_arr, y=lift_arr, color='#512DA8', alpha =0.3, linewidth=2, ax=hx, label='Подъемная сила')
sns.lineplot(x=x_arr, y=drag_arr, color='#f12e6d', alpha =0.3, linewidth=2, ax=hx, label='Сила сопротивление')
sns.lineplot(x=x_arr, y=weight_arr, color='green', alpha =0.3, linewidth=2, ax=hx, label='Сила тяжести')
hx.legend(loc='upper right')

plt.scatter(x=x_arr, y=theta_arr, s=1, color='#512DA8', label='Тета')
plt.scatter(x=x_arr, y=alpha_arr, s=1, color='#f12e6d', label='Альфа')
thetaXalpha.legend(loc='lower right')

hx.set(ylabel='Высота, км. // Сила тяги и споротивления, кН',
       xlabel='Дальность, км.',
       yticks=np.linspace(np.min([np.min(y_arr), np.min(lift_arr), np.min(drag_arr)]),
                          np.max([np.max(y_arr), np.max(lift_arr), np.max(drag_arr)]),
                          12),
       xticks=np.linspace(np.min(x_arr), np.max(x_arr), 10))
thetaXalpha.set(xlabel='Дальность, км.',
                ylabel='Угол Тета и Угол Альфа, град.',
                yticks=np.linspace(np.min([np.min(theta_arr), np.min(alpha_arr)]),
                                   np.max([np.max(theta_arr), np.max(alpha_arr)]),
                                   12),
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
ax3.set(xlabel = 'Время, сек.',
        ylabel = 'Масса ЛА, кг.',
        title = 'График зависимости массы ЛА от времени',
        xticks = np.linspace(np.min(t_arr), np.max(t_arr), 10),
        yticks = np.linspace(np.min(m_arr), np.max(m_arr), 4))

# Plot 4: Theta versus Time
sns.scatterplot(x=t_arr, y=thrust_arr, ax=ax4, s=1, color='black')
ax4.set_xlabel('Время, сек.')
ax4.set_ylabel('Сила тяги, кН.')
ax4.set_title('График')

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
ax4.text(0.5, -0.6, f"Данный график был сделан: {now}", ha='center', va='center', transform=ax4.transAxes)
plt.subplots_adjust(hspace=0.8)

plt.show()
# endregion