import math

import numpy as np
import plotly.graph_objects as go
from isa import temperature, pressure, density

# Constants
pi = 3.141592653589793
g = 9.81  # acceleration due to gravity
Radius = 6371000  # radius of the Earth
a = 331  # speed of sound
R = 8.31446261815324  # gas law constant
atm = 101325

# Aircraft parameters
length = 12.5  # length HA,m
diameter = 0.5  # diameter HA,m
radius = diameter / 2  # radius HA,m

# Initial parameters
Ae = pi * radius ** 2   # Выходное сечение
Am = Ae                 # Сечение миделя ракеты
Akr = 0.47 * Ae
A0 = 0.54 * Ae          # Сечение воздухозаборник
Afin = 0.2              # Площадь крыльев



# print("Inlet area: " + str(A0) + " m^2")
# print("Midsection area: " + str(Am) + " m^2")
# print("Critical area: " + str(Akr) + " m^2")
# print("Exit area: " + str(Ae) + " m^2")

# Mass parameters
mf = 450  # initial fuel mass, kg
mha = 990  # mass of aircraft, kg
m = 1440  # initial total mass, kg


data = {
    "x": [],
    "y": [],
    "V": [],
    "theta": [],
    "alpha": [],
    "m": [],

    "Thrust": [],
    "Drag": [],
    "Lift": [],

    "AIRmassFlowRate": [],
    "FUELmassFlowRate": [],
    "specificImpulse": [],

    "t": []
}

CxData = {
    0: 0.32,
    0.5: 0.28,
    1: 0.24,
    1.5: 0.42,
    1.8: 0.3,
    2: 0.3,
    3: 0.26,
    4: 0.25,
    5: 0.23,
    6: 0.22,
    7: 0.21,
    8: 0.2
}


def get_Cd(mach):
    mach_floor = max([m for m in CxData.keys() if m <= mach])
    mach_ceil = min([m for m in CxData.keys() if m >= mach])

    if mach_floor == mach_ceil:
        return CxData.get(mach_floor)

    cd_floor = CxData[mach_floor]
    cd_ceil = CxData[mach_ceil]

    cd = cd_floor + (mach - mach_floor) * (cd_ceil - cd_floor) / (mach_ceil - mach_floor)
    return cd


def airMassFlowRate(altitude, velocityAircraft):
    Gair = density(altitude) * A0 * velocityAircraft
    return Gair


def fuelMassFlowRate(altitude, velocityAircraft):
    # L = 3.5 # Стехеометрический коэффицнет для керосин/кислород
    L = 14.7 # Стехеометрический коэффицнет для керосин/воздух
    Gfuel = airMassFlowRate(altitude, velocityAircraft) / L
    return Gfuel


def thrust(altitude, V, m, engine):

    Tref = 140000   # Сила тяги, Н
    Gref = 5.05      # Расход топлива, кг/с   1кг/с = 30кН

    MachRatio = (V / a) / 6.5

    if m > mha and engine and V > 2.5 * a and altitude < 99000:
        thrust = ((fuelMassFlowRate(altitude, V) * Tref) / Gref) * MachRatio
    else:
        thrust = 0
        engine = False

    return thrust


def specImp(y, V, m, engine):
    T = thrust(y, V, m, engine)
    Gfuel = fuelMassFlowRate(y, V)

    if T>0 and Gfuel>0:
        specImpulse = T / Gfuel
    else:
        specImpulse = 0

    return specImpulse


def drag(altitude, velocityAircraft):
    Mach = velocityAircraft / a
    Cd = get_Cd(Mach)
    return .5 * A0 * Cd * density(altitude) * velocityAircraft ** 2


def lift(altitude, velocityAircraft):
    return .5 * Afin * Cl * density(altitude) * velocityAircraft ** 2


# Function to calculate the derivatives using the Runge-Kutta method
def calculate_derivatives(t, x, y, V, theta, m, engine, alpha):

    P = thrust(y, V, m, engine)
    Drag = drag(y, V)
    Lift = lift(y, V)
    Gfuel = fuelMassFlowRate(y, V)

    dxdt = V * math.cos(theta) * Radius / (Radius + y)
    dydt = V * math.sin(theta)
    dVdt = (P * math.cos(alpha) - Drag - (m * g * math.sin(theta))) / m
    dthetadt = (P * math.sin(alpha) + Lift - m * g * math.cos(theta) +
                (m * V ** 2 * math.cos(theta)) / (Radius + y)) / (m * V)
    dmdt = -Gfuel

    return dxdt, dydt, dVdt, dthetadt, dmdt


# Function to perform one step of the Runge-Kutta method
def runge_kutta_step(t, dt, x, y, V, theta, m, engine, alpha):
    k1_x, k1_y, k1_V, k1_theta, k1_m = calculate_derivatives(t,
                                                             x,
                                                             y,
                                                             V,
                                                             theta,
                                                             m,
                                                             engine,
                                                             alpha)
    k2_x, k2_y, k2_V, k2_theta, k2_m = calculate_derivatives(t + dt / 2,
                                                             x + k1_x * dt / 2,
                                                             y + k1_y * dt / 2,
                                                             V + k1_V * dt / 2,
                                                             theta + k1_theta * dt / 2,
                                                             m + k1_m * dt / 2,
                                                             engine,
                                                             alpha)
    k3_x, k3_y, k3_V, k3_theta, k3_m = calculate_derivatives(t + dt / 2,
                                                             x + k2_x * dt / 2,
                                                             y + k2_y * dt / 2,
                                                             V + k2_V * dt / 2,
                                                             theta + k2_theta * dt / 2,
                                                             m + k2_m * dt / 2,
                                                             engine,
                                                             alpha)
    k4_x, k4_y, k4_V, k4_theta, k4_m = calculate_derivatives(t + dt,
                                                             x + k3_x * dt,
                                                             y + k3_y * dt,
                                                             V + k3_V * dt,
                                                             theta + k3_theta * dt,
                                                             m + k3_m * dt,
                                                             engine,
                                                             alpha)

    x = x + dt * (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
    y = y + dt * (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6
    V = V + dt * (k1_V + 2 * k2_V + 2 * k3_V + k4_V) / 6
    theta = theta + dt * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta) / 6

    if m > mha:
        m = m + dt * (k1_m + 2 * k2_m + 2 * k3_m + k4_m) / 6
    else:
        m = mha

    return x, y, V, theta, m


# Integration parameters
t = 0.0  # initial time, sec
t_end = 999  # end time, sec
dt = 0.01  # time step, sec

V = 3 * a  # initial velocity
x = 0  # initial x position
y = 2000  # initial y position

theta = math.radians(47)  # initial angle
alpha = math.radians(0)  # angle of attack

Cl = 0.35
engine = False

while y >= 0 and t < t_end:

    if 0.3 < t < 200:
        engine = True
    else:
        engine = False

    if t > 100:
        alpha = math.radians(6)  # angle of attack

    x, y, V, theta, m = runge_kutta_step(t, dt, x, y, V, theta, m, engine, alpha)

    # Append the current x and y values to the trajectory arrays
    data["x"].append(x)
    data["y"].append(y)
    data["V"].append(V)
    data["theta"].append(np.degrees(theta))
    data["alpha"].append(np.degrees(alpha))
    data["m"].append(m)

    data["Thrust"].append(thrust(y, V, m, engine))
    data["Drag"].append(drag(y, V))
    data["Lift"].append(lift(y, V))

    data["specificImpulse"].append(specImp(y, V, m, engine))
    data["AIRmassFlowRate"].append(airMassFlowRate(y, V))
    data["FUELmassFlowRate"].append(fuelMassFlowRate(y, V))
    data["t"].append(t)

    t += dt  # Update the time

last_x = data["x"][-1]
print("Последнее значение x:", last_x)

# Создание графика
fig = go.Figure()

# Добавление данных на график
fig.add_trace(go.Scatter(x=data["t"], y=data["x"], mode='lines', name="Положение ЛА по оси X", line=dict(width=2)))
fig.add_trace(go.Scatter(x=data["t"], y=data["y"], mode='lines', name="Положение ЛА по оси Y", line=dict(width=2)))
fig.add_trace(go.Scatter(x=data["t"], y=data["V"], mode='lines', name="Скорость ЛА", line=dict(width=2)))
fig.add_trace(go.Scatter(x=data["t"], y=data["m"], mode='lines', name="Масса ЛА", line=dict(width=2)))
fig.add_trace(go.Scatter(x=data["t"], y=data["theta"], mode='lines', name="Угол наклона траектории", line=dict(width=2)))
fig.add_trace(go.Scatter(x=data["t"], y=data["alpha"], mode='lines', name="Угол атаки", line=dict(width=2)))
fig.add_trace(go.Scatter(x=data["t"], y=data["Thrust"], mode='lines', name="Сила тяги", line=dict(width=2)))
fig.add_trace(go.Scatter(x=data["t"], y=data["Drag"], mode='lines', name="Сила сопротивление", line=dict(width=2)))
fig.add_trace(go.Scatter(x=data["t"], y=data["Lift"], mode='lines', name="Подъемная сила", line=dict(width=2)))
fig.add_trace(go.Scatter(x=data["t"], y=data["AIRmassFlowRate"], mode='lines', name="Массовый расход воздуха ПВРД", line=dict(width=2)))
fig.add_trace(go.Scatter(x=data["t"], y=data["FUELmassFlowRate"], mode='lines', name="Массовый расход топлива ПВРД", line=dict(width=2)))
fig.add_trace(go.Scatter(x=data["t"], y=data["specificImpulse"], mode='lines', name="Удельный импульс", line=dict(width=2)))
fig.add_trace(go.Scatter(x=data["x"], y=data["y"], mode='lines', name="Траектория полета ЛА", line=dict(width=2)))

# Настройка макета графика
fig.update_layout(
    title="График",
    xaxis=dict(
        title="Время",
        tickfont=dict(size=18, family='Arial'),
        title_font=dict(size=18, family='Arial')
    ),
    yaxis=dict(
        title="Значение",
        tickfont=dict(size=18, family='Arial'),
        title_font=dict(size=18, family='Arial')
    ),
    height=1000,
    width=1700,
    showlegend=True,
    legend=dict(font=dict(size=18, family='Arial')),
)

fig.show()