import math
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

print("Inlet area: " + str(A0) + " m^2")
print("Midsection area: " + str(Am) + " m^2")
print("Critical area: " + str(Akr) + " m^2")
print("Exit area: " + str(Ae) + " m^2")

# Mass parameters
mf = 450  # initial fuel mass, kg
mha = 990  # mass of aircraft, kg
m = 1440  # initial total mass, kg

engine = False

data = {
    "x": [],
    "y": [],
    "V": [],
    "theta": [],
    "m": [],

    "Thrust": [],
    "Drag": [],
    "Lift": [],

    "massFlowRate": [],
    "velocityExhaustGas": [],
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
    mfr = density(altitude) * A0 * velocityAircraft
    return mfr


def fuelMassFlowRate(altitude, velocityAircraft):
    # L = 3.5 # Стехеометрический коэффицнет для керосин/кислород
    L = 14.7 # Стехеометрический коэффицнет для керосин/воздух
    mfr = airMassFlowRate(altitude, velocityAircraft) / L
    return mfr


def velocityExhaustGas(altitude):
    global engine

    k = 1.221
    # mw = 21.40
    Tc = 3700  # Chamber Temp, Kelvin
    Pc = 80 * atm  # Chamber Pressure, Pa

    # Ve = math.sqrt(((2*g*k)/(k-1)) * R * Tc * (1 - (pressure(altitude) / Pc) ** ((k - 1) / k)))
    Ve = 1200
    if engine:
        return Ve
    else:
        return 0


def thrust(altitude, V, m):
    global engine

    if m > mha and engine and V > 3*a:
        thrust = airMassFlowRate(altitude, V) * (velocityExhaustGas(altitude) - V)\
                 # +fuelMassFlowRate(altitude, V)*velocityExhaustGas(altitude)
    else:
        # thrust = massFlowRate(velocityAircraft) * (a - velocityAircraft)
        thrust = 0
        engine = False

    return thrust


Cd = 0.3
Cl = 0.35


def drag(altitude, velocityAircraft):
    # Ma = velocityAircraft / a
    return .5 * A0 * Cd * density(altitude) * velocityAircraft ** 2


def lift(altitude, velocityAircraft):
    return .5 * Afin * Cl * density(altitude) * velocityAircraft ** 2

def specific_impulse(altitude, velocityAircraft, m):
    if airMassFlowRate(altitude, velocityAircraft) > 0:
        fuel_mass_flow_rate = airMassFlowRate(altitude, velocityAircraft) / 3
        specific_imp = thrust(altitude, velocityAircraft, m) / fuel_mass_flow_rate
    else:
        specific_imp = 0
    return specific_imp


# Function to calculate the derivatives using the Runge-Kutta method
def calculate_derivatives(t, x, y, V, theta, m):
    P = thrust(y, V, m)
    Drag = drag(y, V)
    Lift = lift(y, V)

    dxdt = V * math.cos(theta) * Radius / (Radius + y)
    dydt = V * math.sin(theta)
    dVdt = (P * math.cos(alpha) - Drag - (m * g * math.sin(theta))) / m
    dthetadt = (P * math.sin(alpha) + Lift - m * g * math.cos(theta) +
                (m * V ** 2 * math.cos(theta)) / (Radius + y)) / (m * V)
    dmdt = -Gc

    return dxdt, dydt, dVdt, dthetadt, dmdt


# Function to perform one step of the Runge-Kutta method
def runge_kutta_step(t, dt, x, y, V, theta, m):
    k1_x, k1_y, k1_V, k1_theta, k1_m = calculate_derivatives(t,
                                                             x,
                                                             y,
                                                             V,
                                                             theta,
                                                             m)
    k2_x, k2_y, k2_V, k2_theta, k2_m = calculate_derivatives(t + dt / 2,
                                                             x + k1_x * dt / 2,
                                                             y + k1_y * dt / 2,
                                                             V + k1_V * dt / 2,
                                                             theta + k1_theta * dt / 2,
                                                             m + k1_m * dt / 2)
    k3_x, k3_y, k3_V, k3_theta, k3_m = calculate_derivatives(t + dt / 2,
                                                             x + k2_x * dt / 2,
                                                             y + k2_y * dt / 2,
                                                             V + k2_V * dt / 2,
                                                             theta + k2_theta * dt / 2,
                                                             m + k2_m * dt / 2)
    k4_x, k4_y, k4_V, k4_theta, k4_m = calculate_derivatives(t + dt,
                                                             x + k3_x * dt,
                                                             y + k3_y * dt,
                                                             V + k3_V * dt,
                                                             theta + k3_theta * dt,
                                                             m + k3_m * dt)

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
t_end = 1200  # end time, sec
dt = 0.01  # time step, sec

V = 4 * a  # initial velocity
x = 0  # initial x position
y = 2000  # initial y position

Gc = 3.5  # fuel consumption rate, kg/s

theta = math.radians(30)  # initial angle
alpha = math.radians(0)  # angle of attack

while y >= 0 and t < t_end:
    if t > 0.3:
        engine = True

    x, y, V, theta, m = runge_kutta_step(t, dt, x, y, V, theta, m)

    # Append the current x and y values to the trajectory arrays
    data["x"].append(x)
    data["y"].append(y)
    data["V"].append(V)
    data["theta"].append(theta)
    data["m"].append(m)

    data["Thrust"].append(thrust(y, V, m))
    data["Drag"].append(drag(y, V))
    data["Lift"].append(lift(y, V))

    data["massFlowRate"].append(airMassFlowRate(y, V))
    data["velocityExhaustGas"].append(velocityExhaustGas(y))
    data["specificImpulse"].append(specific_impulse(y, V, m))

    data["t"].append(t)

    t += dt  # Update the time

last_x = data["x"][-1]
print("Последнее значение x:", last_x)


# Создание графика
fig = go.Figure()

# # Добавление данных на график
# for key, name in [("x", "Положение ЛА по оси X"),
#                   ("y", "Положение ЛА по оси Y"),
#                   ("V", "Скорость ЛА"),
#                   ("theta", "Угол наклона траектории"),
#                   ("m", "Масса ЛА"),
#                   ("Thrust", "Тяга"),
#                   ("Drag", "Сила сопротивление"),
#                   ("Lift", "Подъемная сила"),
#                   # ("temperature", "Температура на данной высоте"),
#                   # ("pressure", "Давление на данной высоте"),
#                   # ("density", "Плотность на данной высоте"),
#                   ("massFlowRate", "Массовый расход воздуха ПВРД"),
#                   ("velocityExhaustGas", "Скорость выхлопных газов")
#                   ]:
#     fig.add_trace(go.Scatter(x=data["t"], y=data[key], mode='lines', name=name, line=dict(width=2)))


# Добавление графика траектории (X vs Y)
fig.add_trace(go.Scatter(x=data["t"], y=data["x"], mode='lines', name="Положение ЛА по оси X", line=dict(width=2)))
fig.add_trace(go.Scatter(x=data["t"], y=data["y"], mode='lines', name="Положение ЛА по оси Y", line=dict(width=2)))

fig.add_trace(go.Scatter(x=data["t"], y=data["V"], mode='lines', name="Скорость ЛА", line=dict(width=2)))
fig.add_trace(go.Scatter(x=data["t"], y=data["m"], mode='lines', name="Масса ЛА", line=dict(width=2)))
fig.add_trace(
    go.Scatter(x=data["t"], y=data["theta"], mode='lines', name="Угол наклона траектории", line=dict(width=2)))

fig.add_trace(go.Scatter(x=data["t"], y=data["Thrust"], mode='lines', name="Сила тяги", line=dict(width=2)))
fig.add_trace(go.Scatter(x=data["t"], y=data["Drag"], mode='lines', name="Сила сопротивление", line=dict(width=2)))
fig.add_trace(go.Scatter(x=data["t"], y=data["Lift"], mode='lines', name="Подъемная сила", line=dict(width=2)))

fig.add_trace(go.Scatter(x=data["t"], y=data["massFlowRate"], mode='lines', name="Массовый расход воздуха ПВРД",
                         line=dict(width=2)))
fig.add_trace(go.Scatter(x=data["t"], y=data["velocityExhaustGas"], mode='lines', name="Скорость выхлопных газов",
                         line=dict(width=2)))

fig.add_trace(go.Scatter(x=data["t"], y=data["specificImpulse"], mode='lines', name="Удельный импульс",
                         line=dict(width=2)))

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

# print(fig)
fig.show()

# # Plot x vs y
# plt.figure(figsize=(8, 6))
# plt.plot(data["x"], data["y"])
# plt.xlabel('x (km)')
# plt.ylabel('y (km)')
# plt.title('Rocket Trajectory (x vs y)')
# plt.grid(True)
# plt.show()
#
# # Plot mass against time
# plt.figure(figsize=(10, 6))
# plt.plot(data["t"], data["m"])
# plt.xlabel('Time (s)')
# plt.ylabel('Mass (kg)')
# plt.title('Rocket Mass vs Time')
# plt.grid(True)
# plt.show()
#
# # Plot velocity (V), lift, and drag against time (t)
# plt.figure(figsize=(10, 6))
# plt.plot(data["t"], data["Thrust"], color='b', linewidth=2, label='Velocity')
# plt.plot(data["t"], data["Lift"], color='g', linestyle='--', linewidth=2, label='Lift')
# plt.plot(data["t"], data["Drag"], color='r', linestyle='-.', linewidth=2, label='Drag')
# plt.xlabel('Time (s)')
# plt.ylabel('Force (kN) / Velocity (m/s)')
# plt.title('Rocket Velocity vs Time and Forces')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
#
# # Создание нового окна с графиками
# plt.figure(figsize=(10, 6))
#
# # График температуры
# plt.plot(data["t"], data["temperature"], label='Temperature')
#
# # График давления
# plt.plot(data["t"], data["pressure"], label='Pressure')
#
# # График плотности
# plt.plot(data["t"], data["density"], label='Density')
#
# # Добавление заголовка и меток осей
# plt.title('Environmental Conditions vs Time')
# plt.xlabel('Time (s)')
# plt.ylabel('Value')
#
# # Добавление легенды
# plt.legend()
#
# # Отображение графика
# plt.grid(True)
# plt.show()
